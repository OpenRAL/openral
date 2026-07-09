#!/usr/bin/env bash
# record_demo.sh — thin, reusable screen recorder for sim + dashboard demo clips.
#
# Captures the full X11 display (:1 by default) to recordings/<name>.mp4 using
# ffmpeg x11grab + libx264 (CPU encode, to leave the 8 GB GPU for the policy and
# MuJoCo/Isaac render). Optionally tiles the sim viewer + the chromium dashboard
# window side-by-side first (best-effort via xdotool; never fatal).
#
# It records until EITHER a stop sentinel file appears OR the duration elapses,
# then sends SIGINT to ffmpeg so the MP4 is finalised cleanly.
#
# Usage:
#   tools/record_demo.sh <name> [duration_s] [--tile]
#
#   <name>        output basename -> recordings/<name>.mp4
#   [duration_s]  max record seconds (default 300 = 5 min)
#   --tile        best-effort: tile a MuJoCo/Isaac viewer + chromium side-by-side
#
# Stop early from another shell:
#   touch recordings/.<name>.stop
#
# Env:
#   REC_DISPLAY   X display to grab (default :1)
#   REC_FPS       capture framerate (default 15 — sim is slow, keeps CPU sane)
set -euo pipefail

NAME="${1:?usage: record_demo.sh <name> [duration_s] [--tile]}"
DURATION="${2:-300}"
case "${2:-}" in ''|*[!0-9]*) DURATION=300 ;; esac
TILE=0
for a in "$@"; do [ "$a" = "--tile" ] && TILE=1; done

DISPLAY_GRAB="${REC_DISPLAY:-:1}"
FPS="${REC_FPS:-15}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/recordings"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$NAME.mp4"
STOP="$OUT_DIR/.$NAME.stop"
rm -f "$STOP"

# Capture region. Default = the DP-4-3 external monitor (2560x1440 at +2560+0),
# which is where the MuJoCo/Isaac viewer opens on this host. Override with
# REC_REGION="WxH+X+Y" (e.g. the laptop panel "2560x1600+0+940").
REGION="${REC_REGION:-2560x1440+2560+0}"
WH="${REGION%%+*}"; OFF="+${REGION#*+}"
W="${WH%x*}"; H="${WH#*x}"
GRAB_INPUT="${DISPLAY_GRAB}.0${OFF}"

tile_windows() {
  # Best-effort: left half = viewer (MuJoCo/Isaac/Kit), right half = chromium.
  command -v xdotool >/dev/null || return 0
  local half=$(( W / 2 ))
  local vid cid
  vid="$(DISPLAY="$DISPLAY_GRAB" xdotool search --name 'MuJoCo\|Isaac\|Omniverse\|Kit\|mujoco\|viewer' 2>/dev/null | head -1 || true)"
  cid="$(DISPLAY="$DISPLAY_GRAB" xdotool search --class 'Chromium\|chromium\|chrome' 2>/dev/null | head -1 || true)"
  if [ -n "$vid" ]; then
    DISPLAY="$DISPLAY_GRAB" xdotool windowsize "$vid" "$half" "$H" windowmove "$vid" 0 0 2>/dev/null || true
  fi
  if [ -n "$cid" ]; then
    DISPLAY="$DISPLAY_GRAB" xdotool windowsize "$cid" "$half" "$H" windowmove "$cid" "$half" 0 2>/dev/null || true
  fi
}

[ "$TILE" = "1" ] && tile_windows

echo "[record_demo] grabbing ${W}x${H} at ${GRAB_INPUT} @ ${FPS}fps -> $OUT (max ${DURATION}s)"
ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -framerate "$FPS" -video_size "${W}x${H}" -i "${GRAB_INPUT}" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -crf 23 \
  "$OUT" &
FF_PID=$!

cleanup() {
  if kill -0 "$FF_PID" 2>/dev/null; then
    kill -INT "$FF_PID" 2>/dev/null || true
    wait "$FF_PID" 2>/dev/null || true
  fi
  rm -f "$STOP"
  echo "[record_demo] finalised $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
}
trap cleanup INT TERM EXIT

elapsed=0
while [ "$elapsed" -lt "$DURATION" ]; do
  kill -0 "$FF_PID" 2>/dev/null || { echo "[record_demo] ffmpeg exited early"; break; }
  [ -f "$STOP" ] && { echo "[record_demo] stop sentinel seen"; break; }
  sleep 1
  elapsed=$(( elapsed + 1 ))
done
# cleanup() runs via trap on EXIT.
