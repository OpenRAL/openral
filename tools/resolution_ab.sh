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
# ONE GRAPH AT A TIME, ARMS ALTERNATING ROUND BY ROUND. The paired endpoint is
# void unless a scene's 25 mm and 15 mm rounds met the same host, and there are
# two ways to get that wrong. The first version fed a scene-interleaved worklist
# to `xargs -P`, which does not interleave anything: a worker holds its slot for
# all ROUNDS rounds, so only the first pair overlapped. On 2026-09-10 `sink_cup`
# ran its arms 29 minutes apart and its 7/10-vs-0/10 unreadable-run gap read as
# a resolution effect; `baguette`, the one lane that stayed paired, was 3/10 vs
# 3/10, p = 1.0.
#
# Running the two lanes genuinely concurrently fixes the pairing and breaks the
# host. Measured on q-laptop the same evening: baseline occupancy is ~9.1 GB of
# 15.4 GB (browser, editor sessions, the shared XR-1 sidecar) and ONE deploy
# graph adds ~4.9 GB — HAL/MuJoCo 3.2, runtime_node 0.9, the Nav2 stack ~0.8.
# Two graphs is ~18.9 GB against 15.4. The second attempt exhausted all 4 GB of
# swap, load averages hit 225, and 9 of 20 `baguette` rounds came back
# unreadable — worse than the run it was replacing.
#
# So: one graph live at any instant, and the arms alternate round by round —
# r01@25mm, r01@15mm, r02@25mm, ... Adjacent rounds are minutes apart, which is
# TIGHTER pairing than concurrent lanes ever gave (those only share a window,
# not an instant), and nothing contends. It costs wall-clock, not validity:
# ~4.5 h for 4 scenes x 10 rounds instead of a ~2.5 h run that produces
# nothing usable. A host with headroom (spark, 121 GB) could run whole SCENES
# in parallel — never the two arms of one scene against each other.
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

# One round of one arm. Serial by construction: the caller waits for it.
run_round() {  # $1 = scene, $2 = resolution in metres, $3 = round number
  local domain
  case "$2" in 0.025) domain=80 ;; 0.015) domain=81 ;; *) echo "bad arm $2" >&2; return 2 ;; esac
  local log="$OUT/worker-${1}-${2}.log"
  if [ -n "${RUN_ROUND_CMD:-}" ]; then
    OPENRAL_OCTOMAP_RESOLUTION_M="$2" ROS_DOMAIN_ID=$domain \
      $RUN_ROUND_CMD "$1" "$2" "$3" >> "$log" 2>&1
    return
  fi
  OPENRAL_OCTOMAP_RESOLUTION_M="$2" ROS_DOMAIN_ID=$domain \
    uv run --no-sync python tools/_ceiling_probe.py \
      --scene "$1" --gate on --rounds 1 --start-round "$3" \
      --rskill "$RSKILL" --out "$OUT/$2" >> "$log" 2>&1
}

echo "=== ${#SCENES[@]} scenes x 2 arms x $ROUNDS rounds, ONE graph at a time"
fail=0
booted=0
for scene in "${SCENES[@]}"; do
  for n in $(seq 1 "$ROUNDS"); do
    for res in 0.025 0.015; do
      printf '  %s r%02d @ %s m ... ' "$scene" "$n" "$res"
      if run_round "$scene" "$res" "$n"; then echo "ok"; else
        echo "NON-ZERO"; fail=$((fail + 1))
      fi
    done
    if [ "$booted" -eq 0 ]; then
      # The first PAIR boots the XR-1 sidecar the whole battery then shares.
      # This wait must not fall BETWEEN the two arms of a round: when it did,
      # baguette r01's arms were 210 s apart and the report flagged that pair
      # unpaired. Between rounds it costs nothing — pairing is within a round.
      sleep "$SIDECAR_BOOT_S"
      booted=1
    fi
  done
  echo "--- ${scene}: both arms done"
done

echo "=== RESOLUTION_AB_DONE (${fail} round(s) non-zero) ==="
