"""The deploy refuses an MJCF self-collision lowering whose joint columns are wrong.

``lower_collision_params`` numbers links by movable-joint ordinal, one joint per body.
OpenAMRobot 2.0's MJCF puts its planar base (x, y, yaw) on one body, so every later link
read the joint two columns before its own in the kernel's FK (``openarm_left_link1``
read ``base_yaw``); with the lift and arms then counted as base dofs, the voxel-map
coverage ball came out infinite and the launch crashed with the voxel map on. The deploy
keeps the manifest's model (joints by name) whenever the lowering disagrees by name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_ROBOTS = Path(__file__).resolve().parents[3] / "robots"


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    path = Path(__file__).resolve().parents[1] / "launch" / "deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_mjcf_dof_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lowered(robot: str) -> tuple[object, dict[str, object], list[str]]:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("openral_safety")
    from openral_core import RobotDescription
    from openral_core.assets import resolve_asset
    from openral_core.exceptions import ROSConfigError
    from openral_safety.mjcf_lowering import lower_collision_params

    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    assert desc.assets.mjcf
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf", manifest_dir=_ROBOTS / robot)
    except (ROSConfigError, OSError) as exc:  # asset clone unreachable on this host
        pytest.skip(f"{robot} MJCF unavailable: {exc}")
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [j.name for j in desc.joints]
    return model, lower_collision_params(model, names, margin_m=0.0), names


def test_a_planar_base_on_one_body_is_refused() -> None:
    mod = _import_launch_module()
    model, lowered, names = _lowered("openamrobot_v2")
    mismatches = mod._mjcf_dof_mismatches(model, lowered, names)
    assert any("base_x" in m and "one body" in m for m in mismatches)


def test_a_consistent_lowering_is_kept() -> None:
    mod = _import_launch_module()
    model, lowered, names = _lowered("franka_panda")
    assert mod._mjcf_dof_mismatches(model, lowered, names) == []


def test_the_manifest_model_gives_openamrobot_a_finite_voxel_map() -> None:
    mod = _import_launch_module()
    pytest.importorskip("openral_safety")
    from openral_core import RobotDescription
    from openral_safety.envelope_loader import collision_params_from_description

    desc = RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
    base = [i for i, j in enumerate(desc.joints) if j.name in set(desc.base_joints or [])]
    ball = mod._coverage_ball(
        collision_params_from_description(desc),
        [j.position_limits for j in desc.joints],
        base,
        mod._octomap_resolution("sim"),
    )
    assert 0.3 < ball.radius < 2.0
    assert 0.5 < ball.centre[2] < 2.0  # the arms hang from the lifted shoulders
