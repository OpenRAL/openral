# SPDX-License-Identifier: Apache-2.0
"""A manifest that names a real HAL must declare every value the real path consumes.

Issue #303 follow-up. Schema defaults are numbers nobody measured on a rig, and
they used to reach the runner (tick rate), the transport (trajectory deadline)
and the safety kernel (force / torque / contact thresholds) silently. The
`RobotDescription` validator refuses such a manifest at load time — before
`deploy validate`, `build_hal` or any node — and names every gap at once.
Sim-only manifests are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import (
    ActionSpec,
    ControlMode,
    EmbodimentKind,
    HalEntrypoints,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = sorted(REPO_ROOT.glob("robots/*/robot.yaml"))

_REAL = "openral_hal.ur_real:UR5eRealHAL"
_FULL_SAFETY = SafetyEnvelope(
    max_ee_speed_m_s=1.0,
    max_ee_accel_m_s2=1.0,
    max_joint_speed_factor=0.5,
    max_force_n=50.0,
    max_torque_nm=10.0,
    contact_force_threshold_n=30.0,
    deadman_required=True,
    self_collision_margin_m=0.0,
    starting_pose_max_joint_speed_rad_s=0.5,
    starting_pose_tolerance_rad=0.05,
    joint_state_staleness_limit_s=0.5,
)


def _joints(n: int = 2, *, velocity_limit: float | None = 3.0) -> list[JointSpec]:
    return [
        JointSpec(
            name=f"j{i}",
            joint_type=JointType.REVOLUTE,
            parent_link="base" if i == 0 else f"link{i}",
            child_link=f"link{i + 1}",
            velocity_limit=velocity_limit,
        )
        for i in range(n)
    ]


def _description(**overrides: object) -> RobotDescription:
    fields: dict[str, object] = {
        "name": "contract_arm",
        "embodiment_kind": EmbodimentKind.MANIPULATOR,
        "joints": _joints(),
        "capabilities": RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        "safety": _FULL_SAFETY,
        "hal": HalEntrypoints(real=_REAL),
        "action_spec": ActionSpec(dim=2, control_freq_hz=30.0),
    }
    fields.update(overrides)
    return RobotDescription(**fields)  # type: ignore[arg-type]  # reason: test builder over a typed mapping


# ── Every committed manifest already honours it ───────────────────────────────


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.parent.name for m in MANIFESTS])
def test_every_committed_manifest_loads(manifest: Path) -> None:
    desc = RobotDescription.from_yaml(str(manifest))
    if desc.hal.real:
        assert desc.control_rate_hz is not None
        assert set(RobotDescription.REAL_HARDWARE_SAFETY_FIELDS) <= desc.safety.model_fields_set


def test_a_real_manifest_survives_a_json_round_trip() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    again = RobotDescription.model_validate(desc.model_dump(mode="json"))
    assert again.control_rate_hz == 30.0
    assert again.safety == desc.safety


# ── The contract ──────────────────────────────────────────────────────────────


def test_a_complete_real_manifest_is_accepted() -> None:
    desc = _description()
    assert desc.control_rate_hz == 30.0
    assert desc.safety.starting_pose_max_joint_speed_rad_s == 0.5


def test_a_sim_only_manifest_keeps_the_schema_defaults() -> None:
    desc = _description(hal=HalEntrypoints(real=None), safety=SafetyEnvelope(), action_spec=None)
    assert desc.control_rate_hz is None
    assert desc.safety.contact_force_threshold_n == 30.0


def test_every_gap_is_named_in_one_error() -> None:
    """One load reports everything the rig still has to declare, not the first miss."""
    with pytest.raises(ValidationError) as excinfo:
        _description(
            safety=SafetyEnvelope(),
            action_spec=None,
            joints=_joints(velocity_limit=None),
        )
    message = str(excinfo.value)
    assert "action_spec.control_freq_hz" in message
    for name in RobotDescription.REAL_HARDWARE_SAFETY_FIELDS:
        assert f"safety.{name}" in message
    assert "safety.starting_pose_max_joint_speed_rad_s" in message
    assert "safety.starting_pose_tolerance_rad" in message


@pytest.mark.parametrize("rate", [None, 0.0, -30.0])
def test_the_control_rate_must_be_positive(rate: float | None) -> None:
    with pytest.raises(ValidationError, match=r"action_spec\.control_freq_hz"):
        _description(action_spec=ActionSpec(dim=2, control_freq_hz=rate))


def test_a_missing_action_spec_is_a_missing_control_rate() -> None:
    with pytest.raises(ValidationError, match=r"action_spec\.control_freq_hz"):
        _description(action_spec=None)


@pytest.mark.parametrize("field", RobotDescription.REAL_HARDWARE_SAFETY_FIELDS)
def test_each_kernel_read_safety_field_must_be_declared(field: str) -> None:
    """Leaving one field to its schema default is refused, and the error names it."""
    declared = {k: getattr(_FULL_SAFETY, k) for k in RobotDescription.REAL_HARDWARE_SAFETY_FIELDS}
    del declared[field]
    with pytest.raises(ValidationError, match=rf"safety\.{field}") as excinfo:
        _description(safety=SafetyEnvelope(**declared))  # type: ignore[arg-type]  # reason: mapping of the model's own fields
    others = set(RobotDescription.REAL_HARDWARE_SAFETY_FIELDS) - {field}
    assert not any(f"safety.{o}" in str(excinfo.value) for o in others)


def test_an_explicit_value_equal_to_the_default_counts_as_declared() -> None:
    """What is refused is inheriting a default silently, not the number itself."""
    explicit = SafetyEnvelope(
        starting_pose_max_joint_speed_rad_s=0.5,
        starting_pose_tolerance_rad=0.05,
        joint_state_staleness_limit_s=0.5,
        **{k: getattr(SafetyEnvelope(), k) for k in RobotDescription.REAL_HARDWARE_SAFETY_FIELDS},  # type: ignore[arg-type]  # reason: mapping of the model's own fields
    )
    assert _description(safety=explicit).safety.contact_force_threshold_n == 30.0


@pytest.mark.parametrize(
    "field",
    [
        "starting_pose_max_joint_speed_rad_s",
        "starting_pose_tolerance_rad",
        "joint_state_staleness_limit_s",
    ],
)
def test_the_approach_speed_and_tolerance_must_be_declared(field: str) -> None:
    """They have no default at all: the runner refuses rather than guess an approach speed."""
    with pytest.raises(ValidationError, match=rf"safety\.{field}"):
        _description(safety=_FULL_SAFETY.model_copy(update={field: None}))
    with pytest.raises(ValidationError):
        _FULL_SAFETY.model_copy(update={field: 0.0}).model_validate(
            _FULL_SAFETY.model_copy(update={field: 0.0}).model_dump()
        )


def test_joints_without_velocity_limits_are_no_longer_a_contract_failure() -> None:
    """The limit was only ever consumed by the ramp derivation, which is gone."""
    assert _description(joints=_joints(velocity_limit=None)).control_rate_hz == 30.0


def test_the_contract_is_only_held_against_real_hals() -> None:
    """The same gaps on a sim-only manifest are not an error."""
    desc = _description(
        hal=HalEntrypoints(sim="openral_hal.ur:UR5eMujocoHAL", real=None),
        safety=SafetyEnvelope(),
        action_spec=None,
    )
    assert desc.hal.real is None
    assert desc.safety.starting_pose_max_joint_speed_rad_s is None
