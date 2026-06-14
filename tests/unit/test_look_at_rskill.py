"""ADR-0044 Phase 3 — LookAtRskill goal lowering, camera resolution, manifest.

The constraint-lowering math is tested against the real franka robot.yaml
(LIBERO eye-in-hand on ``panda_hand``) and against a synthetic so101-style
offset mount; the live MoveGroup dispatch is exercised in
``tests/integration/test_look_at_franka.py`` against the real MoveIt panda
demo (skip-gated, never faked).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from openral_core import RobotDescription, SensorSpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.look_at_rskill import (
    build_look_at_constraints,
    resolve_camera_sensor,
)
from openral_world_state.geometry import compute_gaze_pose
from openral_world_state.object_lift import homogeneous_from_quat_xyz

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "rskills" / "rskill-moveit-look-at" / "rskill.yaml"
_FRANKA_YAML = _REPO / "robots" / "franka_panda" / "robot.yaml"


def test_manifest_validates_and_selects_look_at_builder() -> None:
    from openral_core.schemas import RSkillManifest

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    assert m.kind == "ros_action"
    assert m.ros_integration is not None
    assert m.ros_integration.goal_builder == "look_at"
    assert m.chunk_size == 1
    assert "look" in [a.value for a in m.actions]
    schema = m.goal_params_schema
    assert schema is not None
    assert schema["properties"]["look_at"]["properties"]["target_xyz"]["minItems"] == 3


def test_resolve_camera_sensor_against_real_franka_manifest() -> None:
    desc = RobotDescription.from_yaml(str(_FRANKA_YAML))
    sensor = resolve_camera_sensor(desc, "wrist")
    assert sensor.frame_id == "panda_hand"
    # No declared static mount: the camera frame IS the constrained link.
    assert sensor.parent_frame is None


def test_resolve_camera_sensor_lists_available_on_miss() -> None:
    desc = RobotDescription.from_yaml(str(_FRANKA_YAML))
    with pytest.raises(ROSConfigError, match=r"available sensors"):
        resolve_camera_sensor(desc, "head")
    with pytest.raises(ROSConfigError, match="RobotDescription"):
        resolve_camera_sensor(None, "wrist")


def _rotate(quat_xyzw: tuple[float, float, float, float], vec: tuple[float, float, float]):
    m = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), quat_xyzw)
    return m[:3, :3] @ np.asarray(vec, dtype=np.float64)


def test_constraints_identity_mount_aim_at_target() -> None:
    """Camera frame == constrained link: constraint pose is the gaze pose itself."""
    camera_goal = compute_gaze_pose((0.3, 0.0, 0.6), (0.5, 0.0, 0.2), frame_id="panda_link0")
    entry = build_look_at_constraints(camera_goal=camera_goal, link_name="panda_hand")

    pc = entry["position_constraints"][0]
    assert pc["link_name"] == "panda_hand"
    assert pc["header"]["frame_id"] == "panda_link0"
    region = pc["constraint_region"]
    assert region["primitives"][0]["type"] == 2  # shape_msgs SolidPrimitive.SPHERE
    pos = region["primitive_poses"][0]["position"]
    assert (pos["x"], pos["y"], pos["z"]) == pytest.approx((0.3, 0.0, 0.6))

    oc = entry["orientation_constraints"][0]
    q = oc["orientation"]
    # The constrained link's +Z (optical axis) must point at the target.
    pointed = _rotate((q["x"], q["y"], q["z"], q["w"]), (0.0, 0.0, 1.0))
    expected = np.asarray((0.2, 0.0, -0.4))
    expected /= np.linalg.norm(expected)
    assert np.allclose(pointed, expected, atol=1e-9)
    # Roll about the optical axis is free; the other two axes are tight.
    assert oc["absolute_z_axis_tolerance"] == pytest.approx(math.pi)
    assert oc["absolute_x_axis_tolerance"] < 0.5


def test_constraints_offset_mount_compensates_link_pose() -> None:
    """An so101-style static mount: the LINK goal must place the CAMERA at the gaze pose."""
    # Camera mounted 5 cm forward of the link, pitched 90° down (camera +Z = link -Z... a
    # representative non-trivial mount).
    sensor = SensorSpec(
        name="wrist",
        modality="rgb",
        frame_id="wrist_cam_optical",
        parent_frame="wrist_roll",
        static_transform_xyz_rpy=(0.05, 0.0, 0.0, 0.0, math.pi / 2.0, 0.0),
        rate_hz=30.0,
    )
    from openral_rskill.look_at_rskill import (
        _camera_mount,  # reason: validating the mount math through the module's own path
    )

    link_name, link_t_cam = _camera_mount(sensor)
    assert link_name == "wrist_roll"
    assert link_t_cam is not None

    camera_goal = compute_gaze_pose((0.2, 0.1, 0.5), (0.6, 0.1, 0.1), frame_id="map")
    entry = build_look_at_constraints(
        camera_goal=camera_goal, link_name=link_name, link_t_cam=link_t_cam
    )
    # Recompose: goal_link ∘ link_T_cam must reproduce the camera gaze pose.
    pc = entry["position_constraints"][0]["constraint_region"]["primitive_poses"][0]["position"]
    oc = entry["orientation_constraints"][0]["orientation"]
    goal_link = homogeneous_from_quat_xyz(
        (pc["x"], pc["y"], pc["z"]), (oc["x"], oc["y"], oc["z"], oc["w"])
    )
    recomposed_cam = goal_link @ link_t_cam
    assert np.allclose(recomposed_cam[:3, 3], camera_goal.xyz, atol=1e-9)
    expected_cam_rot = homogeneous_from_quat_xyz((0, 0, 0), camera_goal.quat_xyzw)[:3, :3]
    assert np.allclose(recomposed_cam[:3, :3], expected_cam_rot, atol=1e-9)
