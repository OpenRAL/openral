"""ADR-0054 — the generic Cartesian-pose MoveGroup builder (`goal_builder: "pose"`).

Pins the pure pieces of :mod:`openral_rskill.pose_goal_rskill`:

* ``build_pose_constraints`` — pose → MoveGroup position + orientation
  constraints, the shared lowering ``LookAtRskill`` also uses (look-at being a
  gaze *specialisation* that leaves optical roll free).
* ``pose_from_block`` — parse a ``pose`` goal block, honouring the
  manifest-declared ``quaternion_order`` (Q2).

Pure — no ROS. The full dispatch path is exercised by the gated MoveGroup
integration tests, exactly like the look_at adapter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_core import Pose6D
from openral_core.exceptions import ROSConfigError
from openral_rskill.pose_goal_rskill import build_pose_constraints, pose_from_block


def test_pose_constraints_constrain_all_three_axes() -> None:
    # A generic EEF pose constrains orientation fully (unlike look-at's free roll).
    pose = Pose6D(xyz=(0.4, 0.1, 0.5), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="panda_link0")
    entry = build_pose_constraints(
        pose=pose,
        link_name="panda_hand",
        position_tolerance_m=0.01,
        orientation_axis_tolerances_rad=(0.05, 0.05, 0.05),
    )
    pc = entry["position_constraints"][0]
    assert pc["link_name"] == "panda_hand"
    assert pc["header"]["frame_id"] == "panda_link0"
    pos = pc["constraint_region"]["primitive_poses"][0]["position"]
    assert (pos["x"], pos["y"], pos["z"]) == pytest.approx((0.4, 0.1, 0.5))

    oc = entry["orientation_constraints"][0]
    assert oc["link_name"] == "panda_hand"
    assert oc["absolute_x_axis_tolerance"] == pytest.approx(0.05)
    assert oc["absolute_y_axis_tolerance"] == pytest.approx(0.05)
    # Full Cartesian goal: the third axis is NOT free (contrast look_at = π).
    assert oc["absolute_z_axis_tolerance"] == pytest.approx(0.05)
    q = oc["orientation"]
    assert (q["x"], q["y"], q["z"], q["w"]) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_pose_constraints_offset_compensates_link_pose() -> None:
    # With a tool offset, the constrained-link pose @ offset recovers the target.
    from openral_world_state.object_lift import homogeneous_from_quat_xyz

    pose = Pose6D(xyz=(0.5, 0.0, 0.4), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="base")
    link_t_target = homogeneous_from_quat_xyz((0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))
    entry = build_pose_constraints(
        pose=pose,
        link_name="tcp",
        link_t_target=link_t_target,
        position_tolerance_m=0.01,
        orientation_axis_tolerances_rad=(0.05, 0.05, 0.05),
    )
    p = entry["position_constraints"][0]["constraint_region"]["primitive_poses"][0]["position"]
    o = entry["orientation_constraints"][0]["orientation"]
    goal_link = homogeneous_from_quat_xyz(
        (p["x"], p["y"], p["z"]), (o["x"], o["y"], o["z"], o["w"])
    )
    recomposed = goal_link @ link_t_target
    assert np.allclose(recomposed[:3, 3], pose.xyz, atol=1e-9)


def test_pose_from_block_xyzw_default() -> None:
    pose, link_name, tool_frame, pos_tol, orient_tol = pose_from_block(
        {
            "frame_id": "panda_link0",
            "link_name": "panda_hand",
            "position": [0.4, 0.1, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "position_tolerance_m": 0.02,
            "orientation_tolerance_rad": 0.1,
        }
    )
    assert pose.xyz == pytest.approx((0.4, 0.1, 0.5))
    assert pose.quat_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert pose.frame_id == "panda_link0"
    assert link_name == "panda_hand"
    assert tool_frame is None  # no offset declared → constrain link directly
    assert (pos_tol, orient_tol) == pytest.approx((0.02, 0.1))


def test_pose_from_block_tool_frame_is_parsed() -> None:
    # ADR-0054 Q3 — an explicit tool/TCP frame is carried for TF-lookup.
    _, link_name, tool_frame, _, _ = pose_from_block(
        {
            "frame_id": "panda_link0",
            "link_name": "panda_hand",
            "tool_frame": "panda_grasp_tcp",
            "position": [0.4, 0.1, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
    )
    assert link_name == "panda_hand"
    assert tool_frame == "panda_grasp_tcp"


def test_pose_from_block_wxyz_reorders_to_xyzw() -> None:
    # Manifest declares wxyz: [w, x, y, z] -> quat_xyzw (x, y, z, w).
    pose, _, _, _, _ = pose_from_block(
        {
            "frame_id": "base",
            "link_name": "tcp",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.7071, 0.7071, 0.0, 0.0],
            "quaternion_order": "wxyz",
        }
    )
    assert pose.quat_xyzw == pytest.approx((0.7071, 0.0, 0.0, 0.7071))


def test_pose_from_block_rejects_unknown_quaternion_order() -> None:
    with pytest.raises(ROSConfigError, match="quaternion_order"):
        pose_from_block(
            {
                "frame_id": "base",
                "link_name": "tcp",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "quaternion_order": "abcd",
            }
        )


def test_look_at_still_leaves_optical_roll_free() -> None:
    # Equivalence guard: look_at delegates to build_pose_constraints but keeps
    # the free roll about the optical (z) axis.
    from openral_rskill.look_at_rskill import build_look_at_constraints

    cam = Pose6D(xyz=(0.3, 0.0, 0.6), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="panda_link0")
    entry = build_look_at_constraints(camera_goal=cam, link_name="panda_hand")
    oc = entry["orientation_constraints"][0]
    assert oc["absolute_z_axis_tolerance"] == pytest.approx(math.pi)
    assert oc["absolute_x_axis_tolerance"] < 0.5
