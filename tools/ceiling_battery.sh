#!/usr/bin/env bash
# The ceiling battery — both arms at once, on spark.
#
# Answers PLAN.md §4: what does XR-1 complete on these four scenes with the
# world-voxel gate OFF vs ON, same commit and host.
#
# 4 scenes x 2 arms; WORKERS (default 2) run at once, both arms interleaved
# so they see identical
# host conditions (CPU contention, thermal drift) at the same moment, so
# neither can masquerade as an effect, and wall time halves. Ten rounds each
# gives 40 runs per arm — 0.91 power against the 8% -> 40% case the fork turns
# on (`tools/round_power.py`).
#
# Each worker gets its own ROS_DOMAIN_ID (SimSensorBridge reads
# `count_publishers` to decide, see tests/sim/conftest.py); all workers share
# ONE XR-1 sidecar (see below).
#
# Usage:  ROUNDS=10 WORKERS=2 bash tools/ceiling_battery.sh
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

# WORKERS caps how many of the 8 run at once. It exists because running all 8
# was what made the 2026-09-06 battery unreadable (issue #256): with the stack
# that heavily oversubscribed, a Nav2 server would miss its bond heartbeat,
# `lifecycle_manager_navigation` would tear the whole navigation stack down,
# and the run would idle out its 420 s deadline doing nothing — scored as the
# policy failing to grasp. 31 of 89 valid runs died that way. Raising
# `BOND_TIMEOUT_S` (packages/openral_nav2_bringup/launch/nav2.launch.py) is the
# fix for the cascade; keeping the host un-oversubscribed is the belt to that
# braces, and is also what makes the ABSOLUTE completion rates mean anything.
WORKERS="${WORKERS:-2}"

run_worker() {  # $1 = index, $2 = scene, $3 = gate
  local domain=$((60 + $1))
  local log="$OUT/worker-${2}-${3}.log"
  echo "  w${1}  ${2}/${3}  domain=$domain  -> $log"
  ROS_DOMAIN_ID=$domain uv run --no-sync python tools/_ceiling_probe.py \
    --scene "$2" --gate "$3" --rounds "$ROUNDS" \
    --rskill "$RSKILL" --out "$OUT" > "$log" 2>&1
}
export -f run_worker
export OUT ROUNDS RSKILL

echo "=== 8 workers, $WORKERS at a time ==="
# Interleaved BY SCENE, not grouped by arm: with WORKERS=2 the two live lanes
# are the same scene's off and on arm, so the arms stay paired under identical
# host conditions — the property the whole battery design rests on. Grouping by
# gate would run all of OFF, then all of ON, and any drift in host state
# between the two halves would masquerade as the gate's effect.
i=0
for scene in "${SCENES[@]}"; do
  for gate in off on; do
    echo "$i $scene $gate"
    i=$((i+1))
  done
done > "$OUT/worklist.txt"

# Worker 0 goes first and ALONE for 210 s: it is the one that spawns the shared
# XR-1 sidecar, and a second client that pings before the model has loaded will
# try to spawn its own on the same port, lose the bind, and die with "xr1
# sidecar process exited with code 1 during boot".
read -r i0 s0 g0 < "$OUT/worklist.txt"
run_worker "$i0" "$s0" "$g0" &
first=$!
sleep 210

tail -n +2 "$OUT/worklist.txt" \
  | xargs -P "$((WORKERS > 1 ? WORKERS - 1 : 1))" -L 1 bash -c 'run_worker "$@"' _
fail=0
wait "$first" || { echo "!!! ${s0}/${g0} exited non-zero"; fail=1; }
echo "=== CEILING_BATTERY_DONE (${fail} non-zero on the lead worker) ==="
