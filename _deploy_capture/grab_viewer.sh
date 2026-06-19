#!/usr/bin/env bash
# Grab the MuJoCo viewer (GL window) into numbered JPGs, with per-frame retries
# so transient "Resource temporarily unavailable" (window mid-render) don't drop
# frames. DO NOT open/move/kill other windows while this runs — desktop churn
# unmaps the GLFW viewer. Usage: grab_viewer.sh <view_win_id> <out_dir> <n>
VIEW="$1"; OUT="$2"; N="${3:-300}"
mkdir -p "$OUT"
export DISPLAY=:1
ok=0
for i in $(seq -w 1 "$N"); do
  for try in 1 2 3; do
    if import -silent -window "$VIEW" -quality 90 "$OUT/f_$i.jpg" 2>/dev/null; then
      ok=$((ok+1)); break
    fi
    sleep 0.05
  done
done
echo "viewer: grabbed $ok / $N frames"
