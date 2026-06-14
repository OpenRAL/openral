#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 22.04 (Humble) or 24.04 (Jazzy) bootstrap.
# Usage: ./scripts/bootstrap_ubuntu.sh

if [[ ! -f /etc/os-release ]] || ! grep -qiE 'ubuntu' /etc/os-release; then
  echo "This script is for Ubuntu only." >&2
  exit 1
fi

VERSION_ID=$(. /etc/os-release && echo "${VERSION_ID}")
case "${VERSION_ID}" in
  22.04) ROS_DISTRO="humble" ;;
  24.04) ROS_DISTRO="jazzy"  ;;
  *) echo "Unsupported Ubuntu ${VERSION_ID}"; exit 1 ;;
esac
echo "==> ROS 2 distro: ${ROS_DISTRO}"

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git curl jq \
  python3-dev python3-pip python3-venv \
  clang clang-format clang-tidy cppcheck \
  libusb-1.0-0-dev libudev-dev udev usbutils \
  software-properties-common locales

# python3.12-dev — required by the workspace Python pin (pyproject.toml
# requires-python = ">=3.12,<3.13") for any sdist that needs pyconfig.h.
# Triggered most often by `uv sync --group maniskill3`, which pulls
# mplib → toppra, both of which compile against the Python headers.
# Ships in 24.04 (jazzy) main; on 22.04 (humble) it comes from the
# deadsnakes PPA. We make this best-effort so older hosts that
# already have a 3.12 from another source don't fail bootstrap.
sudo apt-get install -y python3.12-dev || \
  echo "(python3.12-dev not in apt — add the deadsnakes PPA: " \
       "sudo add-apt-repository ppa:deadsnakes/ppa && " \
       "sudo apt-get install python3.12-dev)"

sudo locale-gen en_US.UTF-8 || true

# ROS 2 apt repository
sudo add-apt-repository -y universe
sudo apt-get install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update
sudo apt-get install -y "ros-${ROS_DISTRO}-ros-base" python3-colcon-common-extensions \
  "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" || true
sudo apt-get install -y "ros-${ROS_DISTRO}-rmw-zenoh-cpp" || \
  echo "(rmw_zenoh may not be in the apt repo for this distro yet — install from source if needed)"

# ADR-0024 / ADR-0025 — wrapped-ROS rSkills + the ADR-0025 mobile-base demo
# need these upstream packages. Mirror the Dockerfile.dev / Dockerfile.x86
# install lists (commit d814f4a) so a fresh-machine `just quickstart`
# produces the same graph as the container images:
#
#   * moveit                              — openral/rskill-moveit-joints
#   * moveit-msgs                         — IDL for the MoveGroup action
#   * moveit-resources-panda-moveit-config — tests/integration/test_moveit_*
#   * nav2-bringup + nav2-msgs            — openral/rskill-nav2-navigate-to-pose
#   * slam-toolbox                        — ADR-0025 lifecycle peer
#   * control-msgs                        — control_msgs/GripperCommand (ALOHA HIL gripper transport)
#   * nav-msgs                            — Odometry/OccupancyGrid (panda_mobile HAL + slam bridge)
#
# `|| true` because some of these may not be in the apt repo for every
# combination of (Ubuntu, ROS distro); the missing ones surface as
# rSkill resolution errors at runtime rather than a hard bootstrap
# failure on hosts that don't need them.
sudo apt-get install -y \
  "ros-${ROS_DISTRO}-moveit" \
  "ros-${ROS_DISTRO}-moveit-msgs" \
  "ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config" \
  "ros-${ROS_DISTRO}-nav2-bringup" \
  "ros-${ROS_DISTRO}-nav2-msgs" \
  "ros-${ROS_DISTRO}-slam-toolbox" \
  "ros-${ROS_DISTRO}-control-msgs" \
  "ros-${ROS_DISTRO}-nav-msgs" || \
  echo "(one or more wrapped-ROS rSkill apt deps unavailable on this distro — " \
       "rSkills that need them will surface a typed runtime error)"

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"

# just
if ! command -v just >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | sudo bash -s -- --to /usr/local/bin
fi

echo ""
echo "==> System bootstrap complete."
echo "==> Source ROS 2 with: source /opt/ros/${ROS_DISTRO}/setup.bash"
