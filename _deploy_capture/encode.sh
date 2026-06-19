#!/usr/bin/env bash
# Compose grabbed per-window frame pairs into a side-by-side MP4.
# Usage: encode.sh <frames_dir> <out.mp4> <fps> [label]
DIR="$1"; OUT="$2"; FPS="${3:-10}"; LABEL="${4:-}"
COMP="$DIR/comp"; mkdir -p "$COMP"
i=0
for vf in "$DIR"/view/f_*.jpg; do
  base=$(basename "$vf"); df="$DIR/dash/$base"
  [ -f "$df" ] || continue
  n=$(printf "%05d" "$i")
  # Pad both to identical height (960), hstack, label optional.
  convert "$vf" -resize 1280x960 -background black -gravity center -extent 1280x960 /tmp/_cl.jpg
  convert "$df" -resize 1280x960 -background black -gravity center -extent 1280x960 /tmp/_cr.jpg
  convert /tmp/_cl.jpg /tmp/_cr.jpg +append "$COMP/c_$n.jpg"
  i=$((i+1))
done
echo "composed $i frames"
ffmpeg -y -framerate "$FPS" -i "$COMP/c_%05d.jpg" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT" -loglevel error
echo "wrote $OUT ($(stat -c %s "$OUT" 2>/dev/null) bytes)"
