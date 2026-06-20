#!/usr/bin/env bash
# Capture the FULL dashboard page via the bundled playwright chromium (the only
# reliable full-page grab on this compositor). Loops M full-page screenshots.
# Usage: grab_dash_pw.sh <url> <out_dir> <m_frames>
URL="${1:-http://127.0.0.1:4318/}"; OUT="$2"; M="${3:-14}"
mkdir -p "$OUT"
cd "$OUT"
for i in $(seq -w 1 "$M"); do
  npx playwright screenshot --browser chromium --full-page --wait-for-timeout 900 \
    "$URL" "d_$i.png" >/dev/null 2>&1
done
echo "dashboard: grabbed $(ls "$OUT"/d_*.png 2>/dev/null | wc -l) full-page frames"
