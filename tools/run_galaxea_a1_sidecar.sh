#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --image IMAGE --sdk-root ABS_PATH [--serial /dev/a1] [--port 46011] [--check-only]" >&2
}

image=""
sdk_root=""
serial="/dev/a1"
port="46011"
check_only="false"
while (( $# > 0 )); do
  case "$1" in
    --image) image="${2:-}"; shift 2 ;;
    --sdk-root) sdk_root="${2:-}"; shift 2 ;;
    --serial) serial="${2:-}"; shift 2 ;;
    --port) port="${2:-}"; shift 2 ;;
    --check-only) check_only="true"; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${image}" || "${sdk_root}" != /* || ! -f "${sdk_root}/install/setup.bash" ]]; then
  usage
  exit 2
fi
if [[ ! -e "${serial}" ]]; then
  echo "serial device does not exist: ${serial}" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "sidecar port must be an integer in [1, 65535]: ${port}" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH" >&2
  exit 1
fi
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "sidecar image is not available locally: ${image}" >&2
  exit 1
fi
if [[ ! -f "${sdk_root}/install/share/signal_arm/launch/single_arm_node.launch" ]]; then
  echo "A1 SDK is missing signal_arm/single_arm_node.launch: ${sdk_root}" >&2
  exit 1
fi
tracker_binary="${sdk_root}/install/lib/mobiman/jointTracker_demo_node"
tracker_share="${sdk_root}/install/share/mobiman"
tracker_source="${tracker_share}/auto_generated/x1_robot"
for required in \
  "${tracker_binary}" \
  "${tracker_share}/config/task.info" \
  "${tracker_share}/urdf/A1/urdf/A1_URDF_0607_0028.urdf" \
  "${tracker_source}"; do
  if [[ ! -e "${required}" ]]; then
    echo "A1 SDK tracker dependency is missing: ${required}" >&2
    exit 1
  fi
done
if command -v fuser >/dev/null 2>&1 && fuser "${serial}" >/dev/null 2>&1; then
  echo "serial device is already owned by another process: ${serial}" >&2
  exit 1
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${port}" | grep -q .; then
  echo "sidecar TCP port is already listening on the host: ${port}" >&2
  exit 1
fi
if docker container inspect openral-galaxea-a1-sidecar >/dev/null 2>&1; then
  echo "container name is already in use: openral-galaxea-a1-sidecar" >&2
  exit 1
fi

lock_name="${serial//\//_}"
lock_path="/tmp/openral-galaxea-a1-${lock_name}.lock"
exec 9>"${lock_path}"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "another OpenRAL A1 sidecar launcher holds ${lock_path}" >&2
  exit 1
fi

cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
if [[ "${cache_base}" != /* ]]; then
  echo "XDG_CACHE_HOME must be absolute when set: ${cache_base}" >&2
  exit 2
fi
tracker_cache_parent="${cache_base}/openral/galaxea-a1"
tracker_cache="${tracker_cache_parent}/x1_robot"
cache_probe="${tracker_cache_parent}"
while [[ ! -e "${cache_probe}" ]]; do
  cache_probe="$(dirname "${cache_probe}")"
done
if [[ ! -d "${cache_probe}" || ! -w "${cache_probe}" ]]; then
  echo "tracker cache parent is not writable: ${cache_probe}" >&2
  exit 1
fi

if [[ "${check_only}" == "true" ]]; then
  echo "Galaxea A1 sidecar preflight passed"
  echo "  image: ${image}"
  echo "  SDK: ${sdk_root}"
  echo "  tracker cache: ${tracker_cache}"
  echo "  serial: ${serial}"
  echo "  TCP: 127.0.0.1:${port}"
  exit 0
fi

if [[ ! -d "${tracker_cache}" ]]; then
  mkdir -p "${tracker_cache_parent}"
  tracker_cache_seed="$(mktemp -d "${tracker_cache_parent}/.x1_robot.seed.XXXXXX")"
  cp -a "${tracker_source}/." "${tracker_cache_seed}/"
  mv "${tracker_cache_seed}" "${tracker_cache}"
fi
if [[ ! -w "${tracker_cache}" ]]; then
  echo "tracker cache is not writable: ${tracker_cache}" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker run --rm --name openral-galaxea-a1-sidecar \
  --network bridge \
  --publish "127.0.0.1:${port}:${port}" \
  --device "${serial}:${serial}" \
  -e A1_SDK_ROOT=/opt/galaxea/A1_SDK \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e ROS_HOSTNAME=127.0.0.1 \
  -v "${sdk_root}:/opt/galaxea/A1_SDK:ro" \
  -v "${tracker_cache}:/opt/galaxea/A1_SDK/install/share/mobiman/auto_generated/x1_robot:rw" \
  -v "${repo_root}/tools/galaxea_a1_ros1_sidecar.py:/opt/openral/galaxea_a1_ros1_sidecar.py:ro" \
  "${image}" \
  python3 /opt/openral/galaxea_a1_ros1_sidecar.py \
    --bind 0.0.0.0 \
    --serial "${serial}" \
    --port "${port}"
