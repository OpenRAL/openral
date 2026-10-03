#!/usr/bin/env bash
# Launch the REAL OpenArm cell with the kernel's world-voxel check ON — for a human at the cell.
#
# Runbook: docs/tutorials/deploy/openarm-real-world-voxel-check.md (step 4).
#
# `openral deploy run` checks none of the OpenArm motion gates, and bringing the graph up
# MOVES the arms (on_activate steps all 16 motors to zero, unramped). This wrapper is the
# only sanctioned way to launch scenes/deploy/openarm_real_world_voxels.yaml and, with
# `--autonomous`, scenes/deploy/openarm_real_autonomous.yaml (the same cell with the LLM reasoner,
# open-vocab detector and spatial memory on). Those two committed scenes are the whole allow-list.
# It refuses unless, in this order:
#   1. OPENRAL_OPENARM_ALLOW_MOTION=1 and OPENRAL_OPENARM_ATTENDED=1 (the HIL tier's gates),
#      and with --autonomous OPENRAL_REASONER_MODEL naming the planner;
#   2. OPENRAL_ROBOT_UNIT names this cell (robots/openarm/units/<unit>.yaml), and that unit's
#      overlay declares head_zed's calibrated mount (static_transform_xyz_rpy): the camera is
#      bolted to the robot, so its pose is robot geometry, measured per cell by the operator,
#      never the manifest's nominal value. `openral deploy run` applies the same gate; this
#      copy only refuses earlier;
#   3. it runs in an interactive terminal and the operator types the confirmation.
# `--scene <path>` swaps in a local copy of the selected committed scene (runbook step 4: one carrying a
# grasp_declaration) under the same gates. It must be a .yaml under scenes/deploy/local/
# (gitignored: operator copies never enter the tracked scene registry) that validates as a DeployScene and, parsed, is IDENTICAL to the committed scene except for the
# top-level `grasp_declaration` key: any other difference (octomap off, a wider self-filter
# padding, other drivers or HAL, an allowance turned on, ...) is refused, naming the differing
# key paths. Other extra arguments pass through to `openral deploy run` only from an
# allow-list of observability flags (--foxglove, --dataset-out <dir>, ...). Anything else — a --config, a --no-enable-octomap-kernel-check
# — could swap or weaken the graph the gates just verified, so it is refused.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
committed_scene="${root}/scenes/deploy/openarm_real_world_voxels.yaml"
scene="${committed_scene}"
robot="${root}/robots/openarm/robot.yaml"

refuse() {
  echo "REFUSED: $*" >&2
  exit 2
}

# Allow-listed pass-through only. Flags taking a value consume the next argument.
passthrough=()
while (($#)); do
  case "$1" in
    --foxglove | --no-dashboard)
      passthrough+=("$1")
      shift
      ;;
    --dataset-out | --dataset-repo-id | --dataset-license | --dashboard-port | --foxglove-port)
      (($# >= 2)) || refuse "$1 needs a value."
      passthrough+=("$1" "$2")
      shift 2
      ;;
    --autonomous)
      committed_scene="${root}/scenes/deploy/openarm_real_autonomous.yaml"
      autonomous=1
      shift
      ;;
    --scene)
      (($# >= 2)) || refuse "--scene needs a path."
      scene_copy=1
      scene="$(readlink -f -- "$2")" || refuse "--scene $2 does not resolve."
      [[ "${scene}" == "${root}/scenes/deploy/local/"*.yaml && -f "${scene}" ]] ||
        refuse "--scene must be an existing .yaml file under ${root}/scenes/deploy/local/."
      shift 2
      ;;
    *)
      refuse "argument '$1' is not allowed here: only observability flags pass through; the scene, robot and safety posture are fixed by this wrapper."
      ;;
  esac
done
# --autonomous may come after --scene, so the scene is settled only once parsing is done.
[[ "${scene_copy:-}" ]] || scene="${committed_scene}"

[[ "${OPENRAL_OPENARM_ALLOW_MOTION:-}" == "1" ]] ||
  refuse "OPENRAL_OPENARM_ALLOW_MOTION is not 1 — this graph moves the arms at bringup."
[[ "${OPENRAL_OPENARM_ATTENDED:-}" == "1" ]] ||
  refuse "OPENRAL_OPENARM_ATTENDED is not 1 — export it only while you are at the E-stop."
[[ -z "${autonomous:-}" || -n "${OPENRAL_REASONER_MODEL:-}" ]] ||
  refuse "OPENRAL_REASONER_MODEL is not set — the autonomous scene's reasoner needs a planner (openral_core.REASONER_MODELS)."
unit="${OPENRAL_ROBOT_UNIT:-}"
[[ -n "${unit}" ]] ||
  refuse "OPENRAL_ROBOT_UNIT is not set — name this cell (robots/openarm/units/<unit>.yaml); its ZED mount is per unit."
[[ -n "${ROS_DISTRO:-}" ]] || refuse "ROS 2 is not sourced (and source the ZED overlay too)."
command -v openral >/dev/null || refuse "openral is not on PATH (activate the venv)."

# Verify the manifest `deploy run` will actually load, not merely this checkout's copy: the
# installed CLI resolves $OPENRAL_ROBOTS_DIR first, then its OWN repo root, which may be a
# different worktree. Ask that resolver with the same interpreter `openral` runs under.
openral_python="$(dirname "$(readlink -f "$(command -v openral)")")/python"
[[ -x "${openral_python}" ]] || openral_python="python"
deployed_robot="$("${openral_python}" - <<'PY'
from pathlib import Path

import openral_cli.deploy_sim as deploy_sim
from openral_sim.policies.robots import resolve_robot_manifest

root = deploy_sim._repo_root_from(Path(deploy_sim.__file__))
print(Path(resolve_robot_manifest("openarm", repo_root=root)).resolve())
PY
)" || refuse "could not resolve the robot manifest openral deploy run would load."
[[ "${deployed_robot}" == "$(readlink -f "${robot}")" ]] ||
  refuse "openral deploy run would load ${deployed_robot}, not ${robot}: run from the checkout whose openral you are using, and unset OPENRAL_ROBOTS_DIR."

# `openral deploy run` re-applies this gate itself (any robot, world-voxel check on); checking
# here too refuses before the operator is asked to confirm, not after.
"${openral_python}" - "${robot}" "${unit}" <<'PY' ||
import sys
from pathlib import Path

from openral_core import RobotDescription, apply_sensor_overlays, load_robot_unit
from openral_core.depth_extrinsic import depth_extrinsic_problems

robot, unit = Path(sys.argv[1]), sys.argv[2]
desc = RobotDescription.from_yaml(str(robot))
overlays = load_robot_unit(robot, unit).sensors
desc = desc.model_copy(update={"sensors": apply_sensor_overlays(desc.sensors, overlays)})
problems = depth_extrinsic_problems(desc, overlays)
print("\n".join(problems), file=sys.stderr)
sys.exit(1 if problems else 0)
PY
  refuse "head_zed's mount is not declared for unit ${unit} (runbook step 2)."
# The scene `deploy run` gets: OpenArm's, world-voxel check still on, unit from the env, and
# for a --scene copy the committed scene exactly, bar grasp_declaration.
"${openral_python}" - "${scene}" "${unit}" "${committed_scene}" <<'PY' ||
import sys

from openral_core import DeployScene

import yaml

scene = DeployScene.from_yaml(sys.argv[1])
problems = []
# A copy may add a grasp target and nothing else: everything else the gates verified must be
# the committed scene's. Compared after parsing, so comments and key order do not matter.
_ALLOWED = {"grasp_declaration"}


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if not path and key in _ALLOWED:
                continue
            if key not in a or key not in b:
                yield sub
            else:
                yield from _diff(a[key], b[key], sub)
    elif a != b:
        yield path or "<root>"


with open(sys.argv[1], encoding="utf-8") as fh:
    copy = yaml.safe_load(fh)
with open(sys.argv[3], encoding="utf-8") as fh:
    committed = yaml.safe_load(fh)
problems += [f"differs from the committed scene at {p}" for p in _diff(committed, copy)]
if scene.robot_id != "openarm":
    problems.append(f"robot_id is {scene.robot_id!r}, not 'openarm'")
if scene.runtime is None or scene.runtime.enable_octomap_kernel_check is not True:
    problems.append("runtime.enable_octomap_kernel_check is not true")
if scene.robot_unit not in (None, sys.argv[2]):
    problems.append(f"robot_unit {scene.robot_unit!r} is not OPENRAL_ROBOT_UNIT {sys.argv[2]!r}")
print("\n".join(problems), file=sys.stderr)
sys.exit(1 if problems else 0)
PY
  refuse "scene ${scene} is not an OpenArm world-voxel scene (see above)."
[[ -t 0 ]] || refuse "not an interactive terminal; a person at the cell launches this."

echo "Bringup steps all 16 motors to zero UNRAMPED. Arms parked near zero, cell clear,"
echo "hardware E-stop in your hand, second person on the stop?"
read -r -p "Type ESTOP IN HAND to launch: " answer
[[ "${answer}" == "ESTOP IN HAND" ]] || refuse "not confirmed."

exec openral deploy run --config "${scene}" "${passthrough[@]}"
