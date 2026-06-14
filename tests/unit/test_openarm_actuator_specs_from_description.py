"""Unit tests for ``actuator_specs_from_description`` in the openarm composer.

The openarm_robosuite scene composer used to carry its own copy of the
16-row ``_JOINT_SPECS`` table — the file's own comment admitted it
"Mirrors robots/openarm/robot.yaml::joints[*].effort_limit". Two
sources of truth. The table is now derived at use time from a loaded
:class:`openral_core.RobotDescription`, so the manifest is the only
place a joint limit can drift.

CLAUDE.md §1.11: real schemas, real fixture under ``robots/openarm/``,
no mocks.
"""

from __future__ import annotations

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.openarm_robosuite._assets import (
    actuator_specs_from_description,
    load_openarm_description,
    motor_actuator_names_from_description,
)


def _openarm_desc_from_yaml() -> RobotDescription:
    return RobotDescription.from_yaml("robots/openarm/robot.yaml")


def test_actuator_specs_table_has_16_entries_for_openarm() -> None:
    """The derived table must enumerate exactly the 16 driven joints."""
    desc = _openarm_desc_from_yaml()
    specs = actuator_specs_from_description(desc)
    assert len(specs) == 16


def test_actuator_specs_table_matches_expected_openarm_inventory() -> None:
    """The derived ``(actuator_name, mjcf_joint, lo, hi, effort)`` rows
    match the OpenArm v2 inventory the previous module-level constant
    enumerated by hand.

    Pins the naming mapping (logical ``left_joint1`` ↔ MJCF
    ``openarm_left_joint1`` ↔ actuator ``left_joint1_ctrl``; logical
    ``left_gripper`` ↔ MJCF ``openarm_left_finger_joint1`` ↔ actuator
    ``left_finger1_ctrl``).
    """
    desc = _openarm_desc_from_yaml()
    specs = actuator_specs_from_description(desc)

    expected_first = ("left_joint1_ctrl", "openarm_left_joint1", -3.49066, 1.39626, 40.0)
    expected_left_gripper = (
        "left_finger1_ctrl",
        "openarm_left_finger_joint1",
        0.0,
        0.7854,
        333.0,
    )
    expected_right_gripper = (
        "right_finger1_ctrl",
        "openarm_right_finger_joint1",
        -0.7854,
        0.0,
        333.0,
    )

    assert specs[0] == expected_first
    # Index 7 is the left gripper (after 7 arm joints).
    assert specs[7] == expected_left_gripper
    # Index 15 is the right gripper (after 7 left + 8 right = 15).
    assert specs[15] == expected_right_gripper


def test_motor_actuator_names_match_actuator_specs_order() -> None:
    """``motor_actuator_names_from_description`` is the bare-names view."""
    desc = _openarm_desc_from_yaml()
    names = motor_actuator_names_from_description(desc)
    specs = actuator_specs_from_description(desc)
    assert names == [s[0] for s in specs]
    # Sanity: every actuator name ends in ``_ctrl`` (the upstream MJCF
    # convention) and contains either ``joint`` or ``finger``.
    for n in names:
        assert n.endswith("_ctrl")
        assert ("joint" in n) or ("finger" in n)


def test_load_openarm_description_returns_in_code_constant() -> None:
    """``load_openarm_description`` returns the HAL's ``OPENARM_DESCRIPTION``.

    This is what guarantees the env factory can derive its actuator
    table without going through disk I/O on every rollout setup.
    """
    pytest.importorskip("openral_hal")
    from openral_hal.openarm import OPENARM_DESCRIPTION

    assert load_openarm_description() is OPENARM_DESCRIPTION


def test_actuator_specs_rejects_joint_without_position_limits() -> None:
    """A description missing ``position_limits`` on a joint is rejected loudly."""
    from openral_core import (
        ControlMode,
        EmbodimentKind,
        JointSpec,
        JointType,
        RobotCapabilities,
        SafetyEnvelope,
    )

    desc = RobotDescription(
        name="broken_openarm_fixture",
        embodiment_kind=EmbodimentKind.BIMANUAL,
        joints=[
            JointSpec(
                name="left_joint1",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link1",
                # position_limits intentionally omitted
                effort_limit=40.0,
            ),
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["openarm"],
        ),
        safety=SafetyEnvelope(),
    )
    with pytest.raises(ROSConfigError, match="position_limits"):
        actuator_specs_from_description(desc)


def test_actuator_specs_rejects_joint_without_effort_limit() -> None:
    """A description missing ``effort_limit`` on a joint is rejected loudly."""
    from openral_core import (
        ControlMode,
        EmbodimentKind,
        JointSpec,
        JointType,
        RobotCapabilities,
        SafetyEnvelope,
    )

    desc = RobotDescription(
        name="broken_openarm_fixture",
        embodiment_kind=EmbodimentKind.BIMANUAL,
        joints=[
            JointSpec(
                name="left_joint1",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link1",
                position_limits=(-1.0, 1.0),
                # effort_limit intentionally omitted
            ),
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["openarm"],
        ),
        safety=SafetyEnvelope(),
    )
    with pytest.raises(ROSConfigError, match="effort_limit"):
        actuator_specs_from_description(desc)
