"""Unit tests for ``SawyerRealHAL`` — the real-hardware Sawyer adapter.

The adapter wraps ``RosControlHAL``; the heavy hot-path logic is
already covered by ``tests/unit/test_hal.py``.  This file pins:

- the ``SAWYER_DESCRIPTION`` joint inventory + capability surface;
- the manifest pointer (closed_with_api → ``SawyerRealHAL``);
- the closed-loop ``send_action`` / ``read_state`` path against a real
  ``SimTransport`` (no mocks, per CLAUDE.md §1.11 / §5.4).
"""

from __future__ import annotations

import time

import pytest
from openral_core import (
    Action,
    ControlMode,
    JointType,
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_core.schemas import EmbodimentKind, JointState
from openral_hal.sawyer_real import (
    SAWYER_DESCRIPTION,
    SAWYER_REAL_DESCRIPTION,
    SawyerRealHAL,
)
from openral_hal.sim_transport import SimTransport

_N_JOINTS = len(SAWYER_DESCRIPTION.joints)


@pytest.fixture()
def transport() -> SimTransport:
    return SimTransport(n_joints=_N_JOINTS)


@pytest.fixture()
def hal(transport: SimTransport) -> SawyerRealHAL:
    return SawyerRealHAL(
        hostname="sawyer.local",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )


def _hold_action() -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * _N_JOINTS],
        stamp_ns=time.time_ns(),
    )


# ── SAWYER_DESCRIPTION ────────────────────────────────────────────────────────


class TestSawyerDescription:
    def test_name(self) -> None:
        assert SAWYER_DESCRIPTION.name == "sawyer"

    def test_seven_arm_joints_plus_gripper(self) -> None:
        # 7 arm joints + 1 ``right_gripper`` prismatic DoF.
        # The public ``SAWYER_DESCRIPTION.joints`` includes the gripper
        # so VLA action contracts emitting a gripper channel match the
        # robot's declared joint count.
        assert _N_JOINTS == 8
        arm = [j for j in SAWYER_DESCRIPTION.joints if j.role == "arm"]
        gripper = [j for j in SAWYER_DESCRIPTION.joints if j.role == "gripper"]
        assert len(arm) == 7
        assert len(gripper) == 1

    def test_joint_names_match_intera_sdk_convention(self) -> None:
        names = [j.name for j in SAWYER_DESCRIPTION.joints]
        assert names == [*[f"right_j{i}" for i in range(7)], "right_gripper"]

    def test_arm_joints_are_revolute_and_gripper_is_prismatic(self) -> None:
        # Arm DoFs revolute; the parallel-gripper width is a
        # single prismatic abstraction over the per-finger mimic.
        arm = SAWYER_DESCRIPTION.joints[:7]
        gripper = SAWYER_DESCRIPTION.joints[7]
        assert all(j.joint_type is JointType.REVOLUTE for j in arm)
        assert gripper.joint_type is JointType.PRISMATIC
        assert gripper.role == "gripper"

    def test_embodiment_kind_is_manipulator(self) -> None:
        assert SAWYER_DESCRIPTION.embodiment_kind is EmbodimentKind.MANIPULATOR

    def test_supports_joint_position(self) -> None:
        assert ControlMode.JOINT_POSITION in SAWYER_DESCRIPTION.capabilities.supported_control_modes

    def test_embodiment_tags_include_sawyer_aliases(self) -> None:
        tags = set(SAWYER_DESCRIPTION.capabilities.embodiment_tags)
        assert {"sawyer", "rethink"} <= tags

    def test_safety_envelope_pins_known_limits(self) -> None:
        env = SAWYER_DESCRIPTION.safety
        assert env.max_force_n == 80.0
        assert env.max_torque_nm == 80.0
        assert env.deadman_required is True

    def test_sim_baseline_sdk_pointer(self) -> None:
        """The sim baseline keeps ``sdk_kind: open`` and has no sim HAL
        (``hal.sim is None``) because Sawyer has no MuJoCo HAL adapter today;
        ``hal.real`` already points at the real adapter.
        """
        assert SAWYER_DESCRIPTION.sdk_kind == "open"
        assert SAWYER_DESCRIPTION.hal.sim is None
        assert SAWYER_DESCRIPTION.hal.real == "openral_hal.sawyer_real:SawyerRealHAL"

    def test_real_sdk_pointer(self) -> None:
        """``SAWYER_REAL_DESCRIPTION`` is what ``robots/sawyer/robot.yaml`` pins to."""
        assert SAWYER_REAL_DESCRIPTION.sdk_kind == "closed_with_api"
        assert SAWYER_REAL_DESCRIPTION.hal.real == "openral_hal.sawyer_real:SawyerRealHAL"

    def test_real_description_inherits_kinematics_from_sim(self) -> None:
        sim = SAWYER_DESCRIPTION.model_dump()
        real = SAWYER_REAL_DESCRIPTION.model_dump()
        for shared_field in ("name", "joints", "end_effectors", "capabilities", "safety"):
            assert sim[shared_field] == real[shared_field]
        assert sim["sdk_kind"] != real["sdk_kind"]
        # The hal entrypoints are shared; only sdk_kind differs.
        assert sim["hal"] == real["hal"]

    def test_description_round_trip_through_json(self) -> None:
        raw = SAWYER_DESCRIPTION.model_dump_json()
        reloaded = SAWYER_DESCRIPTION.model_validate_json(raw)
        assert reloaded.model_dump() == SAWYER_DESCRIPTION.model_dump()


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_empty_hostname_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            SawyerRealHAL(hostname="")

    def test_whitespace_hostname_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            SawyerRealHAL(hostname="   ")

    def test_default_controller_name(self, hal: SawyerRealHAL) -> None:
        assert hal.controller_name == "sawyer_arm_controller"

    def test_hostname_stored(self, hal: SawyerRealHAL) -> None:
        assert hal.hostname == "sawyer.local"


# ── HAL Protocol conformance ──────────────────────────────────────────────────


class TestProtocolConformance:
    # test_satisfies_hal_protocol moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_satisfies_runtime_checkable_protocol
    # (parametrized over HAL_BUILDERS, "SawyerRealHAL" included).
    # test_read_state_before_connect_raises moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_read_state_before_connect_raises
    # (parametrized over HAL_BUILDERS, "SawyerRealHAL" included).

    def test_send_action_before_connect_raises(self, hal: SawyerRealHAL) -> None:
        with pytest.raises(ROSRuntimeError):
            hal.send_action(_hold_action())

    def test_connect_then_read_state(self, hal: SawyerRealHAL) -> None:
        hal.connect()
        try:
            state = hal.read_state()
            assert isinstance(state, JointState)
            assert state.name == [j.name for j in SAWYER_DESCRIPTION.joints]
        finally:
            hal.disconnect()

    # test_disconnect_idempotent moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_disconnect_is_idempotent
    # (parametrized over HAL_BUILDERS, "SawyerRealHAL" included).

    def test_send_action_publishes_to_sawyer_controller(
        self, hal: SawyerRealHAL, transport: SimTransport
    ) -> None:
        hal.connect()
        try:
            hal.send_action(_hold_action())
        finally:
            hal.disconnect()
        topic, _msg = transport.calls[-1]
        assert topic == "/sawyer_arm_controller/joint_trajectory"


# ── Safety ────────────────────────────────────────────────────────────────────


class TestSafety:
    # test_estop_always_raises moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_estop_always_raises_estoprequested
    # (parametrized over HAL_BUILDERS, "SawyerRealHAL" included).
    # test_estop_is_safety_violation moved to
    # tests/unit/test_hal_protocol_conformance.py::test_estoprequested_is_safety_violation_subclass
    # (structural check that ROSEStopRequested subclasses ROSSafetyViolation).

    def test_estop_deactivates_the_controller_then_publishes_the_super_stop(
        self, hal: SawyerRealHAL, transport: SimTransport
    ) -> None:
        """Controller deactivation first, then intera's super stop as std_msgs/Empty."""
        hal.attach_controller_stop(transport)
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert transport.switch_calls == [("deactivate", ("sawyer_arm_controller",))]
        assert transport.controller_state("sawyer_arm_controller") == "inactive"
        assert transport.empty_publishes == ["/robot/set_super_stop"]
        assert hal.vendor_stop_topics() == ["/robot/set_super_stop"]
        report = hal.last_stop_report
        assert report is not None and report.stopped
        assert report.vendor_stop == "/robot/set_super_stop"

    def test_recovery_policy_is_restart_required(self, hal: SawyerRealHAL) -> None:
        from openral_hal.protocol import EStopRecovery, LifecycleEStopHAL

        assert isinstance(hal, LifecycleEStopHAL)
        assert hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
        with pytest.raises(ROSRuntimeError, match="in-process reset is forbidden"):
            hal.reset_estop()

    # test_after_estop_send_action_fails moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_send_action_after_estop_fails
    # (parametrized over "FrankaPandaRealHAL" / "SawyerRealHAL").
