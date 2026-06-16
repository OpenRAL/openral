"""Tests for the unified ``assets:`` block on :class:`RobotDescription`.

Covers the schema-level ref-string grammar (ADR-0058 §4). The validator here
only checks the ref *string* format; file resolution lives in
:mod:`openral_core.assets`. The two must agree on the accepted schemes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openral_core.schemas import AssetRefs, RobotDescription, SimDescription, UrdfAsset


def test_assets_block_parses_and_validates() -> None:
    a = AssetRefs(
        urdf=UrdfAsset(
            ref="rd:panda_description",
            root_frame="panda_link0",
            base_to_root_xyz_rpy=(0, 0, 0, 0, 0, 0),
        ),
        mjcf="rd:panda_mj_description",
        srdf="file:franka_panda.srdf",
    )
    assert a.urdf is not None
    assert a.urdf.ref == "rd:panda_description"
    assert a.urdf.root_frame == "panda_link0"
    assert a.mjcf == "rd:panda_mj_description"
    assert a.srdf == "file:franka_panda.srdf"


def test_ros2_dynamic_ref_accepted() -> None:
    a = AssetRefs(urdf=UrdfAsset(ref="ros2://robot_description"))
    assert a.urdf is not None
    assert a.urdf.ref == "ros2://robot_description"


def test_malformed_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        UrdfAsset(ref="python:robot_descriptions.panda_description:URDF_PATH")


def test_malformed_mjcf_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        AssetRefs(mjcf="robot_descriptions:panda_mj_description")


def test_old_fields_are_gone() -> None:
    for gone in (
        "urdf_path",
        "srdf_path",
        "urdf_root_frame",
        "static_base_to_urdf_root_xyz_rpy",
    ):
        assert gone not in RobotDescription.model_fields
    assert "mjcf_uri" not in SimDescription.model_fields


def test_assets_default_is_empty() -> None:
    desc = RobotDescription(
        name="smoke_robot",
        embodiment_kind="manipulator",
        joints=[
            {
                "name": "j1",
                "joint_type": "revolute",
                "parent_link": "base_link",
                "child_link": "link_1",
            }
        ],
        capabilities={
            "supported_control_modes": ["joint_position"],
            "embodiment_tags": ["smoke"],
        },
        safety={},
    )
    assert desc.assets.urdf is None
    assert desc.assets.mjcf is None
    assert desc.assets.srdf is None
