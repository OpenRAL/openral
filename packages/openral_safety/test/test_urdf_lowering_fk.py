"""Joint-FK lowering + chain-scoping for onboarding a robot onto self-collision.

The kernel places each link's capsule via its parent joint's FK (origin + axis);
``lower_joint_fk`` reads those from the URDF, matched to manifest joints by
``child_link``. ``lower_robot`` scopes generated geometry to the manifest's
kinematic chain (no orphan links). Real franka manifest + URDF, no mocks (§1.11).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

from openral_core import RobotDescription
from openral_core.urdf_resolve import resolve_urdf_path
from openral_safety.urdf_lowering import lower_joint_fk, lower_robot

_FRANKA = "robots/franka_panda/robot.yaml"


def test_joint_fk_matches_urdf_origins() -> None:
    """Each lowered joint FK equals the URDF joint origin (matched by child_link)."""
    import yourdfpy
    from openral_safety.urdf_lowering import _mat_to_rpy

    robot = RobotDescription.from_yaml(_FRANKA)
    fk = lower_joint_fk(robot, robot.urdf_path)  # type: ignore[arg-type]  # urdf_path is set
    assert {"panda_joint1", "panda_joint7"} <= set(fk)  # the arm joints matched

    um = yourdfpy.URDF.load(resolve_urdf_path(robot.urdf_path), load_meshes=False)
    by_child = {j.child: j for j in um.robot.joints}
    for joint in robot.joints:
        if joint.name not in fk:
            continue
        xyz, rpy, _axis = fk[joint.name]
        uj = by_child[joint.child_link]
        utf = np.eye(4) if uj.origin is None else np.asarray(uj.origin)
        assert np.allclose(xyz, utf[:3, 3], atol=1e-9)
        assert np.allclose(rpy, _mat_to_rpy(utf[:3, :3]), atol=1e-9)


def test_unmatched_joints_are_omitted() -> None:
    """A synthetic joint with no URDF child match (panda_gripper) is not lowered."""
    robot = RobotDescription.from_yaml(_FRANKA)
    fk = lower_joint_fk(robot, robot.urdf_path)  # type: ignore[arg-type]
    # panda_gripper's child (panda_finger_pair) is not a URDF link → omitted.
    assert "panda_gripper" not in fk


def test_lower_robot_scopes_geometry_to_chain_drops_orphans() -> None:
    """Generated geometry only covers links in the manifest's kinematic chain."""
    robot = RobotDescription.from_yaml(_FRANKA)
    model = lower_robot(robot)
    chain = {j.parent_link for j in robot.joints} | {j.child_link for j in robot.joints}
    geom_links = {g.link_name for g in model.collision_geometry}
    assert geom_links <= chain, f"orphan geometry links: {geom_links - chain}"
    # The URDF's panda_leftfinger/rightfinger are NOT in the franka manifest chain.
    assert "panda_leftfinger" not in geom_links
    assert "panda_link7" in geom_links  # a real arm link IS covered
    # joint_fk is populated for the arm joints.
    assert "panda_joint4" in model.joint_fk


def test_franka_acm_uses_srdf_when_srdf_path_set() -> None:
    """With the vendored Franka SRDF, the ACM is mesh-authoritative (source=srdf)."""
    robot = RobotDescription.from_yaml(_FRANKA)
    model = lower_robot(robot)
    assert model.acm_source == "srdf"
    # link1↔link4 (a Franka "Never" pair) is disabled.
    assert ("panda_link1", "panda_link4") in model.allowed_collision_pairs
