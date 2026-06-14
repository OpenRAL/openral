"""Unit tests for :class:`AlohaHAL` — the Trossen ALOHA bimanual adapter.

The ALOHA exposes a 14-DoF action vector (left arm 6 + left gripper 1 +
right arm 6 + right gripper 1) split across four ros2_control controllers.
This file pins:

- the :data:`ALOHA_DESCRIPTION` joint inventory + bimanual capability;
- the manifest pointer (closed_with_api → ``AlohaHAL``);
- the per-arm + per-gripper command split in :meth:`AlohaHAL.send_action`,
  exercised against a real :class:`SimTransport` (no mocks per CLAUDE.md
  §1.11 / §5.4);
- the e-stop semantics.
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
    ROSPerceptionStale,
    ROSRuntimeError,
    ROSSafetyViolation,
)
from openral_core.schemas import EmbodimentKind, Hand, JointState
from openral_hal.aloha import ALOHA_DESCRIPTION, ALOHA_REAL_DESCRIPTION, AlohaHAL
from openral_hal.protocol import HAL
from openral_hal.sim_transport import SimTransport

_N_JOINTS = len(ALOHA_DESCRIPTION.joints)


@pytest.fixture()
def transport() -> SimTransport:
    return SimTransport(n_joints=_N_JOINTS)


@pytest.fixture()
def hal(transport: SimTransport) -> AlohaHAL:
    return AlohaHAL(publish_fn=transport.publish, state_fn=transport.state)


def _zero_action() -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * _N_JOINTS],
        stamp_ns=time.time_ns(),
    )


# ── ALOHA_DESCRIPTION ─────────────────────────────────────────────────────────


class TestAlohaDescription:
    def test_name(self) -> None:
        assert ALOHA_DESCRIPTION.name == "aloha_bimanual"

    def test_fourteen_joints(self) -> None:
        assert _N_JOINTS == 14

    def test_embodiment_kind_is_bimanual(self) -> None:
        assert ALOHA_DESCRIPTION.embodiment_kind is EmbodimentKind.BIMANUAL

    def test_capabilities_declare_bimanual(self) -> None:
        assert ALOHA_DESCRIPTION.capabilities.bimanual is True

    def test_two_grippers_one_per_hand(self) -> None:
        ees = ALOHA_DESCRIPTION.end_effectors
        hands = {ee.hand for ee in ees}
        assert {Hand.LEFT, Hand.RIGHT} <= hands

    def test_joint_order_left_arm_then_left_gripper_then_right_arm_then_right_gripper(
        self,
    ) -> None:
        names = [j.name for j in ALOHA_DESCRIPTION.joints]
        assert names == [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper",
        ]

    def test_grippers_are_prismatic(self) -> None:
        for j in ALOHA_DESCRIPTION.joints:
            if j.name.endswith("gripper"):
                assert j.joint_type is JointType.PRISMATIC

    def test_arm_joints_are_revolute(self) -> None:
        for j in ALOHA_DESCRIPTION.joints:
            if not j.name.endswith("gripper"):
                assert j.joint_type is JointType.REVOLUTE

    def test_supports_joint_position(self) -> None:
        assert ControlMode.JOINT_POSITION in ALOHA_DESCRIPTION.capabilities.supported_control_modes

    def test_embodiment_tags(self) -> None:
        tags = set(ALOHA_DESCRIPTION.capabilities.embodiment_tags)
        assert "aloha" in tags

    def test_sim_baseline_sdk_pointer(self) -> None:
        """The sim baseline keeps ``sdk_kind: open``; its ``hal`` block names
        both the sim HAL (AlohaMujocoHAL) and real HAL (AlohaHAL) — ADR-0031.
        """
        assert ALOHA_DESCRIPTION.sdk_kind == "open"
        assert ALOHA_DESCRIPTION.hal.sim == "openral_hal.aloha:AlohaMujocoHAL"
        assert ALOHA_DESCRIPTION.hal.real == "openral_hal.aloha:AlohaHAL"

    def test_real_sdk_pointer(self) -> None:
        """``ALOHA_REAL_DESCRIPTION`` is what ``robots/aloha_bimanual/robot.yaml`` pins to."""
        assert ALOHA_REAL_DESCRIPTION.sdk_kind == "closed_with_api"
        assert ALOHA_REAL_DESCRIPTION.hal.real == "openral_hal.aloha:AlohaHAL"

    def test_real_description_inherits_kinematics_from_sim(self) -> None:
        sim = ALOHA_DESCRIPTION.model_dump()
        real = ALOHA_REAL_DESCRIPTION.model_dump()
        for shared_field in ("name", "joints", "end_effectors", "capabilities", "safety"):
            assert sim[shared_field] == real[shared_field]
        assert sim["sdk_kind"] != real["sdk_kind"]
        # ADR-0031: the hal entrypoints are shared; only sdk_kind differs.
        assert sim["hal"] == real["hal"]

    def test_description_round_trip_through_json(self) -> None:
        raw = ALOHA_DESCRIPTION.model_dump_json()
        reloaded = ALOHA_DESCRIPTION.model_validate_json(raw)
        assert reloaded.model_dump() == ALOHA_DESCRIPTION.model_dump()


# ── HAL Protocol conformance ──────────────────────────────────────────────────


class TestProtocolConformance:
    def test_satisfies_hal_protocol(self, hal: AlohaHAL) -> None:
        assert isinstance(hal, HAL)

    def test_read_state_before_connect_raises(self, hal: AlohaHAL) -> None:
        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    def test_send_action_before_connect_raises(self, hal: AlohaHAL) -> None:
        with pytest.raises(ROSRuntimeError):
            hal.send_action(_zero_action())

    def test_double_connect_raises(self, hal: AlohaHAL) -> None:
        hal.connect()
        try:
            with pytest.raises(ROSRuntimeError):
                hal.connect()
        finally:
            hal.disconnect()

    def test_connect_then_read_state(self, hal: AlohaHAL) -> None:
        hal.connect()
        try:
            state = hal.read_state()
            assert isinstance(state, JointState)
            assert state.name == [j.name for j in ALOHA_DESCRIPTION.joints]
            assert len(state.position) == _N_JOINTS
        finally:
            hal.disconnect()

    def test_disconnect_idempotent(self, hal: AlohaHAL) -> None:
        hal.connect()
        hal.disconnect()
        hal.disconnect()


# ── send_action: 4-way command split ──────────────────────────────────────────


class TestSendActionSplit:
    def test_publishes_to_all_four_controllers(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        hal.connect()
        try:
            hal.send_action(_zero_action())
        finally:
            hal.disconnect()
        topics = [topic for topic, _ in transport.calls]
        assert "/left_arm/arm_controller/joint_trajectory" in topics
        assert "/right_arm/arm_controller/joint_trajectory" in topics
        assert "/left_arm/gripper_controller/command" in topics
        assert "/right_arm/gripper_controller/command" in topics

    def test_left_arm_gets_indices_0_to_5(self, hal: AlohaHAL, transport: SimTransport) -> None:
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[float(i) for i in range(_N_JOINTS)]],
            stamp_ns=time.time_ns(),
        )
        hal.connect()
        try:
            hal.send_action(action)
        finally:
            hal.disconnect()
        for topic, msg in transport.calls:
            if topic == "/left_arm/arm_controller/joint_trajectory":
                targets = msg["joint_targets"]
                assert isinstance(targets, list)
                assert targets == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
                return
        pytest.fail("left arm trajectory was not published")

    def test_right_arm_gets_indices_7_to_12(self, hal: AlohaHAL, transport: SimTransport) -> None:
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[float(i) for i in range(_N_JOINTS)]],
            stamp_ns=time.time_ns(),
        )
        hal.connect()
        try:
            hal.send_action(action)
        finally:
            hal.disconnect()
        for topic, msg in transport.calls:
            if topic == "/right_arm/arm_controller/joint_trajectory":
                targets = msg["joint_targets"]
                assert isinstance(targets, list)
                assert targets == [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
                return
        pytest.fail("right arm trajectory was not published")

    def test_grippers_get_singleton_positions(self, hal: AlohaHAL, transport: SimTransport) -> None:
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[float(i) for i in range(_N_JOINTS)]],
            stamp_ns=time.time_ns(),
        )
        hal.connect()
        try:
            hal.send_action(action)
        finally:
            hal.disconnect()
        gripper_msgs = {
            topic: msg
            for topic, msg in transport.calls
            if topic.endswith("/gripper_controller/command")
        }
        assert gripper_msgs["/left_arm/gripper_controller/command"]["position"] == 6.0
        assert gripper_msgs["/right_arm/gripper_controller/command"]["position"] == 13.0

    def test_wrong_action_dim_rejected(self, hal: AlohaHAL) -> None:
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 8],  # not 14
        )
        hal.connect()
        try:
            with pytest.raises(ROSConfigError):
                hal.send_action(bad)
        finally:
            hal.disconnect()

    def test_unsupported_control_mode_rejected(self, hal: AlohaHAL) -> None:
        bad = Action(
            control_mode=ControlMode.CARTESIAN_DELTA,
            horizon=1,
            joint_targets=[[0.0] * _N_JOINTS],
        )
        hal.connect()
        try:
            with pytest.raises(ROSConfigError):
                hal.send_action(bad)
        finally:
            hal.disconnect()


# ── Safety ────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_estop_always_raises(self, hal: AlohaHAL) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()

    def test_estop_is_safety_violation(self, hal: AlohaHAL) -> None:
        hal.connect()
        with pytest.raises(ROSSafetyViolation):
            hal.estop()

    def test_estop_publishes_to_estop_topic(self, hal: AlohaHAL, transport: SimTransport) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert any(topic == "/aloha/estop" for topic, _ in transport.calls)


# ── Staleness guard ───────────────────────────────────────────────────────────


class TestStaleness:
    def test_read_state_raises_when_stale(self, transport: SimTransport) -> None:
        hal = AlohaHAL(
            publish_fn=transport.publish,
            state_fn=transport.state,
            staleness_limit_s=0.0,
        )
        hal.connect()
        time.sleep(0.001)
        try:
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
        finally:
            hal.disconnect()
