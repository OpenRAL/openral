#!/usr/bin/env bash
# The ceiling battery — both arms at once, on spark.
#
# Answers PLAN.md §4: what does XR-1 complete on these four scenes with the
# world-voxel gate OFF vs ON, same commit and host.
#
# EIGHT WORKERS, 4 scenes x 2 arms, SIMULTANEOUSLY: both arms see identical
# host conditions (CPU contention, thermal drift) at the same moment, so
# neither can masquerade as an effect, and wall time halves. Ten rounds each
# gives 40 runs per arm — 0.91 power against the 8% -> 40% case the fork turns
# on (`tools/round_power.py`).
#
# Each worker gets its own ROS_DOMAIN_ID (SimSensorBridge reads
# `count_publishers` to decide, see tests/sim/conftest.py); all workers share
# ONE XR-1 sidecar (see below).
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

# ONE SHARED XR-1 SIDECAR, deliberately: the server is stateless
# (`tools/_xr1_server.py` `reset()` is `return`; episode state lives
# client-side in `openral_sim/policies/xr1.py`; `get_action` is pure). ZMQ REP
# serialises requests, so concurrent sims queue rather than corrupt each
# other — at replan_steps=16, ~0.8 s/chunk, costing seconds against a 420 s
# deadline. Per-worker rskill copies (distinct ports) were tried and
# abandoned: `rSkill.from_pretrained`'s repo root differs from the launching
# shell's, so path/hub copy ids 404'd against the real Hub, producing
# policy-free runs in ~35 s.
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
