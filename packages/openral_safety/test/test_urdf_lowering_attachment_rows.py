# SPDX-License-Identifier: Apache-2.0
"""A ``fixed_attachments`` row must be something the kernel can hold rigid.

The kernel poses a row's child rigidly at the row's origin, and the always-colliding
proof holds the joints the row welds at zero. So a row must be the URDF's own
parent -> child transform at zero, and must not span a joint the manifest keeps as
a kernel DoF: the kernel would move one side while holding the other still, and a
pair the proof certified rigid could separate (its self-collision check removed).

A real URDF file and real manifests, built here because no shipped robot has the
refused shape; ``_manifest_welded_joints`` is the lowering's own entry point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yourdfpy")

from openral_core import (
    ControlMode,
    EmbodimentKind,
    FixedAttachment,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError
from openral_safety.urdf_lowering import _manifest_welded_joints

# base -j1-> arm -spacer-> hub -j2-> tool
_URDF = """<?xml version="1.0"?>
<robot name="chain">
  <link name="base"/><link name="arm"/><link name="hub"/><link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="spacer" type="fixed">
    <parent link="arm"/><child link="hub"/><origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="hub"/><child link="tool"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _robot(rows: list[FixedAttachment]) -> RobotDescription:
    return RobotDescription(
        name="chain",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        base_frame="base",
        joints=[
            JointSpec(
                name="j1",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="arm",
                origin_xyz=(0.0, 0.0, 0.1),
                axis_xyz=(0.0, 0.0, 1.0),
                position_limits=(-1.0, 1.0),
            )
        ],
        fixed_attachments=rows,
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


@pytest.fixture
def urdf(tmp_path: Path) -> str:
    path = tmp_path / "chain.urdf"
    path.write_text(_URDF, encoding="utf-8")
    return str(path)


def test_a_row_spanning_a_manifest_joint_is_refused(urdf: str) -> None:
    # tool welded straight to base, across j1 (a kernel DoF): base -> tool at zero is
    # (0.3, 0, 0.1), but the kernel moves `arm` (and, in the URDF, `tool`) with j1.
    row = FixedAttachment(
        name="tool_weld", parent_link="base", child_link="tool", origin_xyz=(0.3, 0.0, 0.1)
    )
    with pytest.raises(ROSConfigError, match=r"spans manifest joints \['j1'\]"):
        _manifest_welded_joints(_robot([row]), urdf)


def test_a_row_below_the_manifest_joint_welds_its_own_chain(urdf: str) -> None:
    # tool welded to arm (below j1): the spacer and j2 are welded; the pair can be certified.
    row = FixedAttachment(
        name="tool_weld", parent_link="arm", child_link="tool", origin_xyz=(0.3, 0.0, 0.0)
    )
    assert _manifest_welded_joints(_robot([row]), urdf) == frozenset({"j2"})


def test_a_row_off_its_urdf_zero_pose_is_refused(urdf: str) -> None:
    row = FixedAttachment(
        name="tool_weld", parent_link="arm", child_link="tool", origin_xyz=(0.31, 0.0, 0.0)
    )
    with pytest.raises(ROSConfigError, match="transform at zero"):
        _manifest_welded_joints(_robot([row]), urdf)
