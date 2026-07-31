#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

if [[ -z "${A1_SDK_ROOT:-}" || ! -f "${A1_SDK_ROOT}/install/setup.bash" ]]; then
  echo "A1_SDK_ROOT must point to a mounted, built Galaxea A1 SDK" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${A1_SDK_ROOT}/install/setup.bash"
exec "$@"
