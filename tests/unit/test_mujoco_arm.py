"""Unit tests for ``openral_hal._mujoco_arm.MujocoArmHAL``.

The full closed-loop coverage lives in ``tests/sim/test_ur5e_hal_mujoco.py``,
``tests/sim/test_ur10e_hal_mujoco.py``, and ``tests/sim/test_franka_panda_hal_mujoco.py``
(all three need ``mujoco`` + ``robot_descriptions``).  This file pins the
**pre-connect** behaviour of the shared base — constructor validation,
not-connected error paths, and structural checks — so refactors of the
shared base fail the unit lane (<2 s) instead of the slow sim lane.

Per CLAUDE.md §5.4 unit tests must "mock all I/O".  ``MujocoArmHAL`` only
imports ``mujoco`` inside :meth:`connect`, so every test in this file
constructs the HAL but never connects.
"""

from __future__ import annotations

import pytest
from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_core.schemas import Action
from openral_hal._mujoco_arm import MujocoArmHAL

# ── Description fixtures ─────────────────────────────────────────────────────


def _arm_description(n_joints: int = 3) -> RobotDescription:
    return RobotDescription(
        name="mujoco_arm_under_test",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=f"j{i}",
                joint_type=JointType.REVOLUTE,
                parent_link="base" if i == 0 else f"link_{i - 1}",
                child_link=f"link_{i}",
            )
            for i in range(n_joints)
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )


def _make_arm(*, n_joints: int = 3, **overrides: object) -> MujocoArmHAL:
    desc = _arm_description(n_joints)
    kwargs: dict[str, object] = {
        "mjcf_path": "/nonexistent/path.xml",  # never loaded — connect() not called
        "joint_qpos_addr": {f"j{i}": i for i in range(n_joints)},
        "actuator_index": {f"j{i}": i for i in range(n_joints)},
    }
    kwargs.update(overrides)
    return MujocoArmHAL(desc, **kwargs)  # type: ignore[arg-type]


# ── Constructor invariants ──────────────────────────────────────────────────


def test_constructor_rejects_empty_joints() -> None:
    desc = RobotDescription(
        name="empty",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )
    with pytest.raises(ROSConfigError, match="no joints"):
        MujocoArmHAL(
            desc,
            mjcf_path="/dev/null",
            joint_qpos_addr={},
            actuator_index={},
        )


def test_constructor_rejects_gripper_pointing_at_unknown_joint() -> None:
    """ADR-0023: a ``SimGripperDescription`` referring to a joint not in
    ``description.joints`` must be rejected at HAL construction time.
    """
    from openral_core import GripperReadMode, SimGripperDescription

    with pytest.raises(ROSConfigError, match=r"not present in description\.joints"):
        _make_arm(
            grippers=[
                SimGripperDescription(
                    joint="not_a_joint",
                    ctrl_range=(0.0, 1.0),
                    qpos_addrs=(0,),
                    qpos_scale=1.0,
                    read_mode=GripperReadMode.SUM_OVER_SCALE,
                ),
            ],
        )


def test_constructor_rejects_duplicate_gripper_joints() -> None:
    """Two ``SimGripperDescription`` entries with the same joint name are
    a config bug — would double-apply the write in ``send_action``.
    """
    from openral_core import SimGripperDescription

    g = SimGripperDescription(
        joint="j2",
        ctrl_range=(0.0, 1.0),
        qpos_addrs=(2,),
        qpos_scale=1.0,
    )
    with pytest.raises(ROSConfigError, match="duplicate joint names"):
        _make_arm(grippers=[g, g])


def test_constructor_accepts_full_gripper_config() -> None:
    """ADR-0023: ``grippers=[SimGripperDescription(...)]`` replaces the old
    flat ``gripper_*`` kwargs.  A complete entry passes validation."""
    from openral_core import SimGripperDescription

    arm = _make_arm(
        grippers=[
            SimGripperDescription(
                joint="j2",
                ctrl_range=(0.0, 255.0),
                qpos_addrs=(2, 3),
                qpos_scale=0.08,
            ),
        ],
    )
    assert arm.description.name == "mujoco_arm_under_test"


def test_constructor_stores_description_and_joint_names_in_order() -> None:
    arm = _make_arm(n_joints=4)
    assert arm.description.name == "mujoco_arm_under_test"
    # The HAL's internal joint name ordering must match description.joints order;
    # this is what read_state() uses to align positions with names.
    assert [j.name for j in arm.description.joints] == ["j0", "j1", "j2", "j3"]


# ── Not-connected error paths ───────────────────────────────────────────────


def test_read_state_before_connect_raises_rosruntimeerror() -> None:
    arm = _make_arm()
    with pytest.raises(ROSRuntimeError, match="not connected"):
        arm.read_state()


def test_send_action_before_connect_raises_rosruntimeerror() -> None:
    arm = _make_arm(n_joints=3)
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0, 0.0, 0.0]],
    )
    with pytest.raises(ROSRuntimeError, match="not connected"):
        arm.send_action(action)


def test_disconnect_when_never_connected_is_noop() -> None:
    """Idempotency contract: ``disconnect`` on a fresh HAL must not raise."""
    arm = _make_arm()
    arm.disconnect()  # must not raise
    arm.disconnect()  # second call — also a no-op


def test_estop_always_raises_estoprequested_even_if_not_connected() -> None:
    """``estop()`` is the safety-supervisor boundary; it raises unconditionally.

    Per :class:`HAL` Protocol docstring, ``estop`` raises ``ROSEStopRequested``
    every time.  This test exercises the not-connected branch (``self._data
    is None``) — that branch must still raise so a panicked caller's E-stop
    propagates regardless of HAL state.
    """
    arm = _make_arm()
    with pytest.raises(ROSEStopRequested):
        arm.estop()


# ── Connect-time error paths (no mujoco install required) ───────────────────


def test_connect_with_missing_mjcf_raises_rosconfigerror() -> None:
    """``connect()`` reports a clean ``ROSConfigError`` for a missing MJCF path.

    Skipped on hosts without ``mujoco`` installed because the same code path
    raises a different ``ROSConfigError`` ("mujoco not installed") that the
    sim-level tests already cover.  Here we want to pin the *MJCF-not-found*
    branch specifically.
    """
    pytest.importorskip("mujoco")
    arm = _make_arm()
    with pytest.raises(ROSConfigError, match=r"Could not (read|load) MJCF"):
        arm.connect()


# ── Constructor wiring of optional parameters ───────────────────────────────


def test_constructor_settle_steps_default_is_one() -> None:
    arm = _make_arm()
    assert arm._settle_steps == 1  # type: ignore[attr-defined]  # reason: pinning private invariant


def test_constructor_settle_steps_override_is_stored() -> None:
    arm = _make_arm(settle_steps=42)
    assert arm._settle_steps == 42  # type: ignore[attr-defined]


def test_constructor_gravity_enabled_default_is_true() -> None:
    arm = _make_arm()
    assert arm._gravity_enabled is True  # type: ignore[attr-defined]


def test_constructor_gravity_enabled_override_is_stored() -> None:
    arm = _make_arm(gravity_enabled=False)
    assert arm._gravity_enabled is False  # type: ignore[attr-defined]


def test_constructor_staleness_limit_default_is_half_second() -> None:
    arm = _make_arm()
    assert arm._staleness_limit_s == 0.5  # type: ignore[attr-defined]
