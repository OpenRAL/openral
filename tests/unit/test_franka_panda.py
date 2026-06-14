"""Unit tests for the Franka Panda HAL **description** — no MuJoCo required.

The full sim-driven coverage lives in ``tests/sim/test_franka_panda_hal_mujoco.py`` and
requires ``mujoco`` + ``robot_descriptions``.  This file pins the manifest's
data-sheet invariants (joint count, position / velocity / effort limits,
gripper normalisation, capability + safety envelope, embodiment tags) so a
breaking change to ``FRANKA_PANDA_DESCRIPTION`` fails the fast unit lane
instead of the slow sim lane.

Per CLAUDE.md §5.4 unit tests must mock all I/O.  This file does that by
exercising only the pure-Python ``RobotDescription`` and module-level
constants — never importing ``mujoco`` or instantiating ``FrankaPandaHAL``.
"""

from __future__ import annotations

import pytest
from openral_core.schemas import (
    ControlMode,
    EmbodimentKind,
    Hand,
    JointType,
)
from openral_hal.franka_panda import FRANKA_PANDA_DESCRIPTION

# ── Joint inventory ──────────────────────────────────────────────────────────


_EXPECTED_ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
_EXPECTED_GRIPPER_JOINT = "panda_gripper"
_EXPECTED_TOTAL_JOINTS = 8


def test_description_has_seven_arm_joints_plus_gripper() -> None:
    names = [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]
    assert names == [*_EXPECTED_ARM_JOINTS, _EXPECTED_GRIPPER_JOINT]


def test_total_joint_count_is_8() -> None:
    assert len(FRANKA_PANDA_DESCRIPTION.joints) == _EXPECTED_TOTAL_JOINTS


def test_arm_joints_are_revolute() -> None:
    arm = [j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name in _EXPECTED_ARM_JOINTS]
    assert all(j.joint_type is JointType.REVOLUTE for j in arm)


def test_gripper_joint_is_prismatic_synthetic_channel() -> None:
    g = next(j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name == _EXPECTED_GRIPPER_JOINT)
    assert g.joint_type is JointType.PRISMATIC
    # Synthetic normalised [0, 1] channel — closed = 0, open = 1
    assert g.position_limits == (0.0, 1.0)


# ── Data-sheet limits ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        ("panda_joint1", -2.8973, 2.8973),
        ("panda_joint2", -1.7628, 1.7628),
        ("panda_joint3", -2.8973, 2.8973),
        ("panda_joint4", -3.0718, -0.0698),
        ("panda_joint5", -2.8973, 2.8973),
        ("panda_joint6", -0.0175, 3.7525),
        ("panda_joint7", -2.8973, 2.8973),
    ],
)
def test_arm_position_limits_match_franka_datasheet(name: str, lo: float, hi: float) -> None:
    j = next(j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name == name)
    assert j.position_limits == (lo, hi)


@pytest.mark.parametrize(
    ("name", "vmax"),
    [
        ("panda_joint1", 2.175),
        ("panda_joint2", 2.175),
        ("panda_joint3", 2.175),
        ("panda_joint4", 2.175),
        ("panda_joint5", 2.610),
        ("panda_joint6", 2.610),
        ("panda_joint7", 2.610),
    ],
)
def test_arm_velocity_limits_match_franka_datasheet(name: str, vmax: float) -> None:
    j = next(j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name == name)
    assert j.velocity_limit == vmax


@pytest.mark.parametrize(
    ("name", "tmax"),
    [
        ("panda_joint1", 87.0),
        ("panda_joint2", 87.0),
        ("panda_joint3", 87.0),
        ("panda_joint4", 87.0),
        ("panda_joint5", 12.0),
        ("panda_joint6", 12.0),
        ("panda_joint7", 12.0),
    ],
)
def test_arm_effort_limits_match_franka_datasheet(name: str, tmax: float) -> None:
    j = next(j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name == name)
    assert j.effort_limit == tmax


def test_arm_joints_declare_torque_sensors() -> None:
    arm = [j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name in _EXPECTED_ARM_JOINTS]
    assert all(j.has_torque_sensor for j in arm)


def test_gripper_does_not_declare_torque_sensor() -> None:
    g = next(j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name == _EXPECTED_GRIPPER_JOINT)
    assert g.has_torque_sensor is False


def test_arm_actuator_kind_is_bldc() -> None:
    arm = [j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name in _EXPECTED_ARM_JOINTS]
    assert all(j.actuator_kind == "bldc" for j in arm)


# ── Kinematic chain ──────────────────────────────────────────────────────────


def test_arm_chain_is_serial_panda_link0_to_link7() -> None:
    arm = [j for j in FRANKA_PANDA_DESCRIPTION.joints if j.name in _EXPECTED_ARM_JOINTS]
    parents = [j.parent_link for j in arm]
    children = [j.child_link for j in arm]
    assert parents == [
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
    ]
    assert children == [
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
    ]


def test_base_frame_is_panda_link0() -> None:
    assert FRANKA_PANDA_DESCRIPTION.base_frame == "panda_link0"


# ── End-effector ─────────────────────────────────────────────────────────────


def test_end_effector_is_parallel_gripper_with_3kg_payload_and_70n_grip() -> None:
    ees = FRANKA_PANDA_DESCRIPTION.end_effectors
    assert ees is not None
    assert len(ees) == 1
    ee = ees[0]
    assert ee.name == "panda_hand"
    assert ee.kind == "parallel_gripper"
    assert ee.hand is Hand.NA
    assert ee.n_dof == 1
    assert ee.max_grip_force_n == 70.0
    assert ee.max_payload_kg == 3.0
    # Workspace radius matches Franka FR3 / Panda data sheet (855 mm reach).
    assert ee.workspace_radius_m == pytest.approx(0.855)


# ── Capabilities + safety envelope ───────────────────────────────────────────


def test_embodiment_kind_is_manipulator() -> None:
    assert FRANKA_PANDA_DESCRIPTION.embodiment_kind is EmbodimentKind.MANIPULATOR


def test_capabilities_declare_joint_position_control_only() -> None:
    cap = FRANKA_PANDA_DESCRIPTION.capabilities
    assert ControlMode.JOINT_POSITION in cap.supported_control_modes
    # Franka FCI does support velocity / torque modes, but the Panda HAL
    # currently exposes only joint position; if we add modes we must pin them.
    assert len(cap.supported_control_modes) == 1


def test_capabilities_declare_force_control_and_3kg_lift() -> None:
    cap = FRANKA_PANDA_DESCRIPTION.capabilities
    assert cap.has_force_control is True
    assert cap.can_lift_kg == 3.0


def test_embodiment_tags_include_franka_aliases() -> None:
    tags = set(FRANKA_PANDA_DESCRIPTION.capabilities.embodiment_tags)
    # Skill manifests target any of these aliases interchangeably.
    assert {"franka_panda", "franka", "panda"} <= tags


def test_safety_envelope_pins_known_limits() -> None:
    env = FRANKA_PANDA_DESCRIPTION.safety
    assert env.max_ee_speed_m_s == 1.0
    assert env.max_joint_speed_factor == 0.5
    assert env.max_force_n == 100.0
    assert env.max_torque_nm == 87.0
    # Per CLAUDE.md §1: deadman is required for the Panda.
    assert env.deadman_required is True


# ── SDK pointer ──────────────────────────────────────────────────────────────


def test_sim_sdk_pointer_resolves_to_franka_panda_hal() -> None:
    """``FRANKA_PANDA_DESCRIPTION`` is the *sim* baseline; its sdk pointer
    resolves to the MuJoCo adapter.

    The manifest carries two pointers: a sim baseline (this constant) and a
    real-HW companion (``FRANKA_PANDA_REAL_DESCRIPTION`` in
    ``franka_panda_real.py``) derived via
    :func:`openral_hal._real_description.make_real_description`.  The
    pattern matches PR #60's UR adapters
    (``UR5e_DESCRIPTION`` / ``UR5e_REAL_DESCRIPTION``).
    """
    assert FRANKA_PANDA_DESCRIPTION.sdk_kind == "open"
    assert FRANKA_PANDA_DESCRIPTION.hal.sim == "openral_hal.franka_panda:FrankaPandaHAL"


def test_real_sdk_pointer_resolves_to_franka_panda_real_hal() -> None:
    """``FRANKA_PANDA_REAL_DESCRIPTION`` is what ``robots/franka_panda/robot.yaml``
    pins to (closed-with-api → :class:`FrankaPandaRealHAL`, issue #56).

    Kinematics + safety envelope + capabilities + ``hal`` entrypoints are
    inherited from the sim baseline via ``model_copy``; only ``sdk_kind``
    differs (ADR-0031).
    """
    from openral_hal.franka_panda_real import FRANKA_PANDA_REAL_DESCRIPTION

    assert FRANKA_PANDA_REAL_DESCRIPTION.sdk_kind == "closed_with_api"
    assert (
        FRANKA_PANDA_REAL_DESCRIPTION.hal.real == "openral_hal.franka_panda_real:FrankaPandaRealHAL"
    )
    # Round-trip: kinematics and safety are shared verbatim with the sim baseline.
    sim_dump = FRANKA_PANDA_DESCRIPTION.model_dump()
    real_dump = FRANKA_PANDA_REAL_DESCRIPTION.model_dump()
    for shared_field in ("name", "joints", "end_effectors", "capabilities", "safety"):
        assert sim_dump[shared_field] == real_dump[shared_field], (
            f"{shared_field} drifted between FRANKA_PANDA_DESCRIPTION (sim) and "
            f"FRANKA_PANDA_REAL_DESCRIPTION (real-HW)"
        )


# ── Round-trip ───────────────────────────────────────────────────────────────


def test_description_round_trip_through_json() -> None:
    """The description must survive JSON serialisation → re-validation unchanged."""
    raw = FRANKA_PANDA_DESCRIPTION.model_dump_json()
    reloaded = FRANKA_PANDA_DESCRIPTION.model_validate_json(raw)
    assert reloaded.model_dump() == FRANKA_PANDA_DESCRIPTION.model_dump()
