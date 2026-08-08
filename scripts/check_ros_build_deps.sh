#!/usr/bin/env bash
# Preflight for `just ros2-build`: fail early, and legibly, when the ROS 2
# apt dependencies the colcon graph needs are not installed.
#
# Without this the build runs for ~45 s, dies inside CMake on the first
# package whose find_package() misses, and aborts every remaining package —
# so the operator sees a wall of "Aborted <<< openral_hal_*" with the single
# real cause (one missing apt package) buried in a per-package log file:
#
#   CMake Error at CMakeLists.txt:20 (find_package):
#     Could not find a package configuration file provided by "octomap_msgs"
#
# The required set is DERIVED from the package.xml files rather than
# hardcoded, so it stays correct as packages come and go. In-tree packages
# (anything declaring its own <name> under packages/ or cpp/) are excluded,
# as are non-ament apt deps that resolve to system libraries or Python
# distributions rather than an ament share directory.
#
# Advisory by default: exits 0 with a warning so an operator who knows a
# given package is unnecessary for their build can proceed. Set
# OPENRAL_ROS_DEPS_STRICT=1 to make missing deps a hard failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "==> ROS 2 is not sourced (ROS_DISTRO unset) — skipping the apt dep check."
  echo "    source /opt/ros/jazzy/setup.bash   # then re-run"
  exit 0
fi

ROS_SHARE="/opt/ros/${ROS_DISTRO}/share"

# Package names provided by this repo — never apt deps.
mapfile -t IN_TREE < <(
  grep -hoP '(?<=<name>)[^<]+' packages/*/package.xml cpp/*/package.xml 2>/dev/null | sort -u
)

# Deps that are NOT ament packages: they resolve to system libraries or to
# Python distributions installed in the uv venv, so an ament share directory
# is the wrong thing to look for.
NOT_AMENT=" curl nlohmann-json-dev "

missing=()
while IFS= read -r dep; do
  [[ -z "${dep}" ]] && continue
  [[ "${dep}" == python3-* ]] && continue
  [[ "${NOT_AMENT}" == *" ${dep} "* ]] && continue
  printf '%s\n' "${IN_TREE[@]}" | grep -qxF "${dep}" && continue
  [[ -d "${ROS_SHARE}/${dep}" ]] && continue
  missing+=("${dep}")
done < <(
  grep -hoP '(?<=<depend>|<build_depend>|<exec_depend>|<buildtool_depend>)[^<]+' \
    packages/*/package.xml cpp/*/package.xml 2>/dev/null | sort -u
)

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "==> ROS 2 build deps present (${ROS_DISTRO})."
  exit 0
fi

echo "==> Missing ${#missing[@]} ROS 2 apt package(s) required by the colcon graph:" >&2
for dep in "${missing[@]}"; do
  echo "      ${dep}" >&2
done
echo "" >&2
echo "    Install them with:" >&2
# ament package names are underscored; apt names are hyphenated.
apt_names=""
for dep in "${missing[@]}"; do
  apt_names+=" ros-${ROS_DISTRO}-${dep//_/-}"
done
echo "      sudo apt-get install -y${apt_names}" >&2
echo "" >&2
echo "    or re-run the full bootstrap:  just bootstrap" >&2

if [[ "${OPENRAL_ROS_DEPS_STRICT:-0}" == "1" ]]; then
  exit 1
fi
echo "" >&2
echo "    (continuing anyway — set OPENRAL_ROS_DEPS_STRICT=1 to make this fatal)" >&2
