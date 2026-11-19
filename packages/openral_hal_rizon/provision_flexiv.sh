#!/usr/bin/env bash
# Build the upstream Flexiv stack the dual-Rizon bringup loads, against system ROS 2 Jazzy,
# into one colcon workspace (default ~/flexiv_ws). Idempotent: re-run after a re-pin. The
# slow prebuilt products are stamped with what they were built for (the RDK commit; the
# private DDS tags), so a re-pin rebuilds them and an unchanged pin skips them.
#
#   bash packages/openral_hal_rizon/provision_flexiv.sh [workspace]
#   source <workspace>/install/setup.bash   # before `openral deploy run`
#
# Steps, all upstream sources at the deps.repos pins:
#   1. vcs-import flexiv_rdk / flexiv_ros2 / flexiv_description.
#   2. flexiv_rdk's own dependency script for the deps ROS lacks (Boost 1.83, SpaceVecAlg,
#      RBDyn) into <ws>/rdk_install.
#   3. A private Fast-DDS v2.14.5 (its bundled Fast-CDR v2.2.5) + foonathan_memory v0.7-3,
#      static, PIC, hidden visibility, into <ws>/dds_install. The vendor-precompiled RDK
#      `.a` is compiled against Fast-CDR 2.2.5; Fast-CDR 2.2.6 grew `Cdr` (216 -> 296
#      bytes), so linking it against Jazzy's current Fast-CDR (2.2.8 on q5090, 2026-11-19)
#      smashes the stack in the RDK's first deserialize ("*** stack smashing detected ***"
#      in rdk_msgs RealtimeStatesPubSubType::deserialize). Hidden + static keeps this copy
#      inside libflexiv_hardware.so, away from the RMW's own Fast-DDS in ros2_control_node.
#   4. flexiv_rdk (downloads + SHA256-checks the vendor ros2-jazzy `.a`) into rdk_install.
#   5. colcon: flexiv_msgs, flexiv_hardware, flexiv_description.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${1:-$HOME/flexiv_ws}"
WS=$(cd "${1:-$HOME/flexiv_ws}" && pwd)  # absolute: the RDK's installer runs from its own dir
RDK=$WS/rdk_install
DDS=$WS/dds_install
JOBS=$(nproc)
# The Fast-CDR the vendor RDK `.a` is compiled against (2.2.5, bundled by Fast-DDS 2.14.5).
# Re-check both whenever deps.repos moves the RDK to another release.
FASTDDS_TAG=v2.14.5
MEMORY_TAG=v0.7-3
mkdir -p "$WS/src" "$WS/third"
# shellcheck source=/dev/null
source /opt/ros/jazzy/setup.bash

# No --skip-existing: an existing checkout is fetched and moved to the pin; vcs refuses a
# different remote URL. Git would carry non-conflicting local edits across the checkout,
# so refuse any (bar this script's own flexiv_rdk/COLCON_IGNORE) rather than build them.
for repo in "$WS"/src/*/; do
  [ -d "$repo/.git" ] || continue
  dirty=$(git -C "$repo" status --porcelain)
  [ "$(basename "$repo")" = flexiv_rdk ] && dirty=$(grep -v '^?? COLCON_IGNORE$' <<<"$dirty" || true)
  if [ -n "$dirty" ]; then
    echo "refusing: local changes in $repo (commit, stash or remove them):" >&2
    echo "$dirty" >&2
    exit 1
  fi
done
vcs import "$WS/src" < "$HERE/deps.repos"
touch "$WS/src/flexiv_rdk/COLCON_IGNORE"

built_for() { [ "$(cat "$1" 2>/dev/null)" = "$2" ]; }  # stamp file, expected content

RDK_SHA=$(git -C "$WS/src/flexiv_rdk" rev-parse HEAD)
if ! built_for "$RDK/.openral-deps" "$RDK_SHA"; then
  rm -rf "$RDK"
  # flexiv_rdk/third*/ is upstream's third-party dir (the glob keeps codespell off its name).
  (cd "$WS"/src/flexiv_rdk/third*/ && bash build_and_install_dependencies_not_in_ros2.sh "$RDK" "$JOBS")
  echo "$RDK_SHA" > "$RDK/.openral-deps"
fi

STATIC=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$DDS" -DCMAKE_PREFIX_PATH="$DDS"
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_VISIBILITY_PRESET=hidden
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON -DBUILD_TESTING=OFF)
DDS_KEY="fastdds=$FASTDDS_TAG memory=$MEMORY_TAG"
if ! built_for "$DDS/.openral-build" "$DDS_KEY"; then
  rm -rf "$DDS" "$WS/third/memory" "$WS/third/Fast-DDS"
  git clone -q --depth 1 --branch "$MEMORY_TAG" https://github.com/foonathan/memory.git "$WS/third/memory"
  cmake -S "$WS/third/memory" -B "$WS/third/memory/build" "${STATIC[@]}" \
    -DFOONATHAN_MEMORY_BUILD_EXAMPLES=OFF -DFOONATHAN_MEMORY_BUILD_TESTS=OFF
  cmake --build "$WS/third/memory/build" --target install -j"$JOBS"
  git clone -q --depth 1 --branch "$FASTDDS_TAG" --recurse-submodules \
    --shallow-submodules https://github.com/eProsima/Fast-DDS.git "$WS/third/Fast-DDS"
  cmake -S "$WS/third/Fast-DDS" -B "$WS/third/Fast-DDS/build" "${STATIC[@]}" -DTHIRDPARTY=FORCE \
    -DCOMPILE_EXAMPLES=OFF -DSECURITY=OFF -DBUILD_DOCUMENTATION=OFF
  cmake --build "$WS/third/Fast-DDS/build" --target install -j"$JOBS"
  echo "$DDS_KEY" > "$DDS/.openral-build"
fi

cmake -S "$WS/src/flexiv_rdk" -B "$WS/build_rdk" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$RDK" -DCMAKE_PREFIX_PATH="$DDS;$RDK" -DRDK_SUPPORT_ROS2_JAZZY=ON
cmake --build "$WS/build_rdk" --target install -j"$JOBS"

# The private DDS prefix first, so flexiv_rdk's find_dependency(fastrtps/fastcdr) resolves
# there for flexiv_hardware; flexiv_msgs keeps ROS's own (its type supports must match the RMW).
cd "$WS"
colcon build --packages-select flexiv_msgs flexiv_description --cmake-args -DBUILD_TESTING=OFF
CMAKE_PREFIX_PATH="$DDS:$RDK:$CMAKE_PREFIX_PATH" colcon build --packages-select flexiv_hardware \
  --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
echo "Flexiv stack ready: source $WS/install/setup.bash"
