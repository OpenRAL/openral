#!/usr/bin/env bash
# The ceiling battery — both arms at once, on spark.
#
# Answers PLAN.md §4: what does XR-1 complete on these four scenes with the
# world-voxel gate OFF, against the same commit and host with it ON?
#
# EIGHT WORKERS, 4 scenes x 2 arms, running SIMULTANEOUSLY. That is the design,
# not a shortcut: both arms then experience identical host conditions at the
# same moment, so CPU contention, thermal drift and whatever else the box is
# doing load onto both arms equally and cannot masquerade as an effect. It also
# halves wall time. Ten rounds each gives 40 runs per arm — 0.91 power against
# the 8% -> 40% case the fork turns on (`tools/round_power.py`).
#
# Each worker gets:
#   * its own ROS_DOMAIN_ID — a shared DDS graph is not merely noisy here;
#     SimSensorBridge reads `count_publishers` to decide (tests/sim/conftest.py).
#   * its own XR-1 sidecar port, via its own rskill dir copy. NOT shared: the
#     sidecar server holds a resettable policy object, so concurrent clients
#     could corrupt each other's episode state silently.
#
# Usage:  ROUNDS=10 bash tools/ceiling_battery.sh
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export OPENRAL_ALLOW_REMOTE_CODE=1
# Required for parallel workers: `deploy sim` reaps "orphan" graph processes
# by argv signature at startup and cannot distinguish a concurrent sibling
# from a crash leftover, so without this the second worker kills the first
# (measured: worker 1 never got its action server, 626 s to the timeout).
export OPENRAL_SKIP_ORPHAN_REAP=1

# ROS + this worktree's OWN overlay. Without it every scene dies at launch with
# "ros2 not found on PATH" and reports a harness-error in seconds. `set +u`
# because /opt/ros/*/setup.bash reads $AMENT_TRACE_SETUP_FILES unguarded and
# dies under nounset.
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

ROUNDS="${ROUNDS:-10}"
OUT="${OUT:-outputs/ceiling/$(date +%F)}"
SCENES=(baguette sink_cup fridge utensil)
mkdir -p "$OUT"

# ONE SHARED XR-1 SIDECAR, deliberately. The server is stateless:
# `tools/_xr1_server.py`'s `reset()` is literally `return`, and every piece of
# episode state (the image/state history deques, the action queue) lives
# client-side in `openral_sim/policies/xr1.py`. `get_action` is a pure function
# of the observation the client sends. ZMQ REP serialises requests and replies
# to the sender, so N concurrent sims cannot corrupt each other's episodes --
# they only queue. At replan_steps=16 and ~0.8 s per chunk that queueing costs
# a few seconds per replan against a 420 s deadline.
#
# Per-worker rskill copies (to pin distinct ports) were tried and abandoned:
# the runner node resolves through `rSkill.from_pretrained`, whose repo root
# differs from the launching shell's, so path- and hub-shaped copy ids both
# 404'd against the real Hub and produced policy-free runs in ~35 s.
RSKILL="OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"

echo "=== staging 8 workers ==="
i=0; pids=(); labels=()
for gate in off on; do
  for scene in "${SCENES[@]}"; do
    domain=$((60 + i))
    log="$OUT/worker-${scene}-${gate}.log"
    echo "  w$i  $scene/$gate  domain=$domain  -> $log"
    ROS_DOMAIN_ID=$domain nohup uv run --no-sync python tools/_ceiling_probe.py \
      --scene "$scene" --gate "$gate" --rounds "$ROUNDS" \
      --rskill "$RSKILL" --out "$OUT" > "$log" 2>&1 &
    pids+=($!); labels+=("$scene/$gate")
    i=$((i+1))
    # The FIRST worker gets a long head start: it is the one that spawns the
    # shared sidecar, and a second client that pings before the model has
    # loaded will try to spawn its own on the same port, lose the bind, and die
    # with "xr1 sidecar process exited with code 1 during boot" — the same error
    # seen on q-laptop. After it is up the rest only need graph-bring-up spacing.
    if [ "$i" -eq 1 ]; then sleep 210; else sleep 30; fi
  done
done

echo "=== $((i)) workers up; waiting ==="
fail=0
for n in "${!pids[@]}"; do
  wait "${pids[$n]}" || { echo "!!! ${labels[$n]} exited non-zero"; fail=$((fail+1)); }
done
echo "=== CEILING_BATTERY_DONE (${fail} workers non-zero) ==="
