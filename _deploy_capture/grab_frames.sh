#!/usr/bin/env bash
# Per-window frame grabber for side-by-side sim+dashboard capture.
# The mutter/XWayland compositor redirects windows offscreen, so ffmpeg
# x11grab of the root is black — but `import -window <id>` composites each
# window correctly. We grab both windows (in parallel) into numbered JPGs,
# then hstack+encode in post (grab_encode.sh).
#
# Usage: grab_frames.sh <view_win_id> <dash_win_id> <out_dir> <n_frames> <pace_s>
VIEW="$1"; DASH="$2"; OUT="$3"; N="${4:-300}"; PACE="${5:-0}"
mkdir -p "$OUT/view" "$OUT/dash"
export DISPLAY=:1
for i in $(seq -w 1 "$N"); do
  import -silent -window "$VIEW" -quality 90 "$OUT/view/f_$i.jpg" 2>/dev/null &
  p1=$!
  import -silent -window "$DASH" -quality 90 "$OUT/dash/f_$i.jpg" 2>/dev/null &
  p2=$!
  wait $p1 $p2
  [ "$PACE" != "0" ] && sleep "$PACE"
done
echo "grabbed $N frame pairs into $OUT"
