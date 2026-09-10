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
# Usage:  ROUNDS=10 WORKERS=2 bash tools/resolution_ab.sh
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
WORKERS="${WORKERS:-2}"
OUT="${OUT:-outputs/resolution-ab/$(date +%F)}"
SCENES=(baguette sink_cup fridge utensil)
RSKILL="OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"
mkdir -p "$OUT"

run_worker() {  # $1 = index, $2 = scene, $3 = resolution in metres
  local domain=$((80 + $1))
  local tag="${2}-${3}"
  local log="$OUT/worker-${tag}.log"
  echo "  w${1}  ${2} @ ${3} m  domain=$domain  -> $log"
  OPENRAL_OCTOMAP_RESOLUTION_M="$3" ROS_DOMAIN_ID=$domain \
    uv run --no-sync python tools/_ceiling_probe.py \
      --scene "$2" --gate on --rounds "$ROUNDS" \
      --rskill "$RSKILL" --out "$OUT/$3" > "$log" 2>&1
}
export -f run_worker
export OUT ROUNDS RSKILL

# Interleaved by scene so the two live lanes are the SAME scene at both
# resolutions — the arms stay paired under identical host conditions, which is
# what the paired endpoint rests on.
i=0
for scene in "${SCENES[@]}"; do
  for res in 0.025 0.015; do
    echo "$i $scene $res"
    i=$((i+1))
  done
done > "$OUT/worklist.txt"

echo "=== $i workers, $WORKERS at a time, ROUNDS=$ROUNDS"
read -r i0 s0 r0 < "$OUT/worklist.txt"
run_worker "$i0" "$s0" "$r0" &
first=$!
sleep 210   # worker 0 boots the shared XR-1 sidecar alone

tail -n +2 "$OUT/worklist.txt" \
  | xargs -P "$((WORKERS > 1 ? WORKERS - 1 : 1))" -L 1 bash -c 'run_worker "$@"' _
fail=0
wait "$first" || { echo "!!! ${s0}@${r0} exited non-zero"; fail=1; }
echo "=== RESOLUTION_AB_DONE (${fail} non-zero on the lead worker) ==="
