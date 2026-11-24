"""The WidowX MuJoCo twin's world map: posed in ``base_frame``, sized by the real finger.

Two gaps kept ``/openral/world_voxels`` empty on ``scenes/deploy/widowx_twin.yaml`` with the
octomap leg on (2026-11-24):

* the robot self-filter posed the MJCF-lowered model in its root link's NAME,
  ``wx250s/base_link``, which no TF tree publishes (the manifest has no URDF), so every
  cloud was dropped ("source_frame does not exist"). It now gets the manifest
  ``base_frame`` on a fixed arm too, the frame the kernel checks the model in;
* the MJCF lowering posed ``left_finger`` (a 0.015-0.037 m slide) straight from the
  normalised [0, 1] ``gripper`` column: a finger reaching 1 m, which inflated the coverage
  ball to 1.6 m (9.8 M cells, past the bridge's 4 M cap at the 0.015 m default). The
  twin's own affine is now the joint's curve (``with_twin_gripper_curves``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ROBOTS = Path(__file__).resolve().parents[3] / "robots"


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    path = Path(__file__).resolve().parents[1] / "launch" / "deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_widowx_world_map", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mjcf(robot: str) -> tuple[Any, Any]:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("openral_safety")
    from openral_core import RobotDescription
    from openral_core.assets import resolve_asset
    from openral_core.exceptions import ROSConfigError

    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    assert desc.assets.mjcf
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf", manifest_dir=_ROBOTS / robot)
    except (ROSConfigError, OSError) as exc:  # asset clone unreachable on this host
        pytest.skip(f"{robot} MJCF unavailable: {exc}")
    return desc, mujoco.MjModel.from_xml_path(str(path))


def test_the_twin_gripper_curve_is_the_twins_ctrl_range() -> None:
    from openral_safety.urdf_lowering import with_twin_gripper_curves

    desc, model = _mjcf("widowx")
    joints = {j.name: j for j in with_twin_gripper_curves(desc, model).joints}
    assert joints["gripper"].urdf_curve == ((0.0, 0.015), (1.0, 0.037))
    assert all(joints[j.name].urdf_curve is None for j in desc.joints if j.name != "gripper")


def test_a_declared_curve_is_kept() -> None:
    """The MakerMods Metal declares its own (normalised) jaw curve: it wins."""
    from openral_safety.urdf_lowering import with_twin_gripper_curves

    desc, model = _mjcf("makermods_metal")
    declared = {j.name: j.urdf_curve for j in desc.joints}
    assert {j.name: j.urdf_curve for j in with_twin_gripper_curves(desc, model).joints} == (
        declared
    )


def test_the_widowx_world_map_fits_the_bridge_at_the_default_resolution() -> None:
    mod = _import_launch_module()
    from openral_safety.envelope_loader import with_joint_curves
    from openral_safety.mjcf_lowering import lower_collision_params
    from openral_safety.urdf_lowering import with_twin_gripper_curves

    desc, model = _mjcf("widowx")
    lowered = lower_collision_params(model, [j.name for j in desc.joints])
    limits = [j.position_limits for j in desc.joints]
    resolution = mod._octomap_resolution("sim")
    # Without the curve the 1 m phantom finger needs more cells than the bridge publishes.
    with pytest.raises(Exception, match="more than the voxel bridge publishes"):
        mod._coverage_ball(lowered, limits, [], resolution)
    ball = mod._coverage_ball(
        with_joint_curves(lowered, with_twin_gripper_curves(desc, model)), limits, [], resolution
    )
    # WidowX-250s reach is 0.65 m (data sheet); the ball adds the finger, link radii, margin.
    assert 0.65 < ball.radius < 0.85


def test_a_fixed_arm_self_filter_is_posed_in_base_frame() -> None:
    mod = _import_launch_module()
    from openral_safety.mjcf_lowering import lower_collision_params

    desc, model = _mjcf("widowx")
    lowered = lower_collision_params(model, [j.name for j in desc.joints])
    assert lowered["collision_link_names"] == [
        f"wx250s/{n}"
        for n in (
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "upper_forearm_link",
            "lower_forearm_link",
            "wrist_link",
            "gripper_link",
            "left_finger_link",
            "right_finger_link",
        )
    ]  # an MJCF body name, in no TF tree
    params = mod._self_filter_params(lowered, desc, "/joint_states", 0.02)
    assert params["base_frame"] == desc.base_frame == "base_link"
    assert "collision_base_dofs" not in params
    # A peer is posed the same way (peer_voxels looks up world -> its base_frame).
    peer = mod._peer_params([("b", 34, str(_ROBOTS / "widowx" / "robot.yaml"))])
    assert peer["peer_b.base_frame"] == "base_link"
    # The camera the twin publishes hangs off that frame, so TF joins cloud and model.
    (top,) = desc.sensors
    assert top.parent_frame == desc.base_frame
