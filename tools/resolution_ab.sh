#!/usr/bin/env bash
# The 25 mm vs 15 mm world-voxel A/B — the measurement #253 lists as missing.
#
# PRIMARY ENDPOINT IS NOT COMPLETION RATE. #253 asks for a live A/B of
# completion rate, and that endpoint is unaffordable: against the predicted
# 2.7% -> 10.8% it needs **200 runs per arm** for 80% power
# (`tools/round_power.py --baseline 0.027 --alternative 0.108`), i.e. ~33 h on
# one host. At the 40/arm a normal battery gives, power is **0.11** — it can
# only report a null, which is the trap `docs/reference/collision-validation-
# evidence.md` warns about ("accept in advance that it can only report a null").
#
# So the primary endpoint is the PAIRED, CONTINUOUS one the mechanism actually
# predicts: for each stop, the over-approximation (certified mesh gap minus the
# kernel's reported depth). Shrinking the cell from 25 mm to 15 mm takes the
# half-diagonal from 21.65 mm to 12.99 mm, so the prediction is a **-8.66 mm
# shift**, paired by scene. That is a large effect on a quantity measured at
# 9.5-19.5 mm in the 2026-09-10 battery, and it is answerable in tens of runs
# rather than hundreds. Stop count and completion count are recorded as
# secondary and are stated as under-powered.
#
# Both arms are gate-ON (the shipped configuration). The ONLY difference is
# `OPENRAL_OCTOMAP_RESOLUTION_M`, which `deploy_e2e.launch.py` validates and
# `validation_matrix.py` records, so a round that changed it can never be
# mistaken afterwards for one that did not.
#
# CONCURRENCY IS FIXED AT ONE SCENE, BOTH ARMS. The paired endpoint above only
# means anything if the two arms met the same host, so the two live lanes must
# be the same scene's 25 mm and 15 mm, start together and finish together. The
# first version of this script wrote a scene-interleaved worklist and then fed
# it to `xargs -P`, which is not the same thing: each worker holds its slot for
# all ROUNDS rounds, so only the FIRST pair ever overlapped and every later
# lane drifted by about one lane's duration. In the 2026-09-10 run `baguette`
# was paired and `sink_cup` was not — its 25 mm arm ran 18:50-19:19 and its
# 15 mm arm 19:19-19:37, alone, after ~30 rounds of host wear. That lane then
# showed 7/10 unreadable runs against 0/10 and read as a resolution effect
# until the order was checked. So: launch a scene's two arms together, barrier,
# next scene. No `xargs`, no WORKERS knob — two graphs is also q-laptop's
# ceiling. If a bigger host makes several scenes at once worth it, add whole
# SCENES to the inner group, never a bare parallelism count.
#
# Usage:  ROUNDS=10 bash tools/resolution_ab.sh
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export OPENRAL_ALLOW_REMOTE_CODE=1
export OPENRAL_SKIP_ORPHAN_REAP=1

set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

ROUNDS="${ROUNDS:-10}"
# Only the first lane pays this; a scheduling test overrides it to keep short.
SIDECAR_BOOT_S="${SIDECAR_BOOT_S:-210}"
OUT="${OUT:-outputs/resolution-ab/$(date +%F)}"
SCENES=(baguette sink_cup fridge utensil)
RSKILL="OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"
mkdir -p "$OUT"

run_worker() {  # $1 = index, $2 = scene, $3 = resolution in metres
  local domain=$((80 + $1))
  local tag="${2}-${3}"
  local log="$OUT/worker-${tag}.log"
  echo "  w${1}  ${2} @ ${3} m  domain=$domain  -> $log"
  # Test seam: a harness can substitute the lane body to exercise the
  # scheduling below without 10 h of GPU. Unset in every real run.
  if [ -n "${RUN_WORKER_CMD:-}" ]; then
    OPENRAL_OCTOMAP_RESOLUTION_M="$3" ROS_DOMAIN_ID=$domain \
      $RUN_WORKER_CMD "$1" "$2" "$3" > "$log" 2>&1
    return
  fi
  OPENRAL_OCTOMAP_RESOLUTION_M="$3" ROS_DOMAIN_ID=$domain \
    uv run --no-sync python tools/_ceiling_probe.py \
      --scene "$2" --gate on --rounds "$ROUNDS" \
      --rskill "$RSKILL" --out "$OUT/$3" > "$log" 2>&1
}

: > "$OUT/worklist.txt"
echo "=== ${#SCENES[@]} scenes x 2 arms, ROUNDS=$ROUNDS, both arms of a scene concurrent"

i=0
fail=0
booted=0
for scene in "${SCENES[@]}"; do
  pids=()
  lanes=()
  for res in 0.025 0.015; do
    echo "$i $scene $res" >> "$OUT/worklist.txt"
    run_worker "$i" "$scene" "$res" &
    pids+=("$!")
    lanes+=("${scene}@${res}")
    if [ "$booted" -eq 0 ]; then
      # The very first lane boots the XR-1 sidecar the whole battery shares.
      # Its partner has to wait that out ONCE; every later pair starts together.
      sleep "$SIDECAR_BOOT_S"
      booted=1
    fi
    i=$((i + 1))
  done
  # Barrier: the next scene does not start until BOTH arms of this one are done,
  # which is what keeps each pair's host conditions common.
  for n in "${!pids[@]}"; do
    wait "${pids[$n]}" || { echo "!!! ${lanes[$n]} exited non-zero"; fail=$((fail + 1)); }
  done
  echo "--- ${scene}: both arms done"
done

echo "=== RESOLUTION_AB_DONE (${fail} lane(s) non-zero) ==="
