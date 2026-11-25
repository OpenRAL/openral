"""Unit tests for ``AlohaHAL`` — the Trossen ALOHA bimanual adapter.

The ALOHA exposes a 14-DoF action vector (left arm 6 + left gripper 1 +
right arm 6 + right gripper 1) split into the ``xs_sdk`` command contract
(issue #250): each arm as a ``JointGroupCommand`` payload on
``/<ns>/commands/joint_group``, each gripper as a ``JointSingleCommand``
payload on ``/<ns>/commands/joint_single``, for the stock follower
namespaces. Under ``hal_mode:=real`` the lifecycle node attaches the
production ``InterbotixXSTransport`` before ``connect()``, which refuses
without a transport. This file pins:

- the ``ALOHA_DESCRIPTION`` joint inventory + bimanual capability;
- the manifest pointer (closed_with_api → ``AlohaHAL``);
- the per-arm + per-gripper command split in ``AlohaHAL.send_action``,
  exercised against a real ``SimTransport`` (no mocks per CLAUDE.md
  §1.11 / §2);
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
)
from openral_core.schemas import EmbodimentKind, Hand
from openral_hal.aloha import ALOHA_DESCRIPTION, ALOHA_REAL_DESCRIPTION, AlohaHAL
from openral_hal.sim_transport import SimTorqueSeam, SimTransport

_N_JOINTS = len(ALOHA_DESCRIPTION.joints)


@pytest.fixture()
def transport() -> SimTransport:
    return SimTransport(n_joints=_N_JOINTS)


def _seam(**kwargs: object) -> SimTorqueSeam:
    """A torque/mode seam for both stock follower arms (``kwargs`` override)."""
    kwargs.setdefault("arms", ["follower_left", "follower_right"])
    return SimTorqueSeam(**kwargs)  # type: ignore[arg-type]  # reason: test kwargs pass-through


@pytest.fixture()
def hal(transport: SimTransport) -> AlohaHAL:
    # connect() reads both arms' operating modes, so a seam is part of the rig.
    h = AlohaHAL(publish_fn=transport.publish, state_fn=transport.state)
    h.attach_torque_stop(_seam())
    return h


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
        the real HAL (AlohaHAL); ``hal.sim`` is null so ``build_hal`` derives
        ``MujocoArmHAL`` from the ``sim:`` block.
        """
        assert ALOHA_DESCRIPTION.sdk_kind == "open"
        assert ALOHA_DESCRIPTION.hal.sim is None
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
        # The hal entrypoints are shared; only sdk_kind differs.
        assert sim["hal"] == real["hal"]

    def test_description_round_trip_through_json(self) -> None:
        raw = ALOHA_DESCRIPTION.model_dump_json()
        reloaded = ALOHA_DESCRIPTION.model_validate_json(raw)
        assert reloaded.model_dump() == ALOHA_DESCRIPTION.model_dump()


# ── HAL Protocol conformance ──────────────────────────────────────────────────


class TestProtocolConformance:
    # test_satisfies_hal_protocol moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_satisfies_runtime_checkable_protocol
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).
    # test_read_state_before_connect_raises moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_read_state_before_connect_raises
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).

    # test_send_action_before_connect_raises moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_send_action_before_connect_raises
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).

    def test_double_connect_raises(self, hal: AlohaHAL) -> None:
        hal.connect()
        try:
            with pytest.raises(ROSRuntimeError):
                hal.connect()
        finally:
            hal.disconnect()

    # test_connect_then_read_state moved to tests/unit/test_hal_protocol_conformance.py
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).

    # test_disconnect_idempotent moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_disconnect_is_idempotent
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).


# ── send_action: 4-way command split ──────────────────────────────────────────


def _indexed_action(steps: int = 1) -> Action:
    """Step ``k`` carries ``k * 100 + joint_index`` so slices and steps are both visible."""
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=steps,
        joint_targets=[[float(k * 100 + i) for i in range(_N_JOINTS)] for k in range(steps)],
        stamp_ns=time.time_ns(),
    )


def _sent(hal: AlohaHAL, transport: SimTransport, action: Action) -> dict[str, dict[str, object]]:
    # The first command reads both arms' operating modes back; the default
    # seam reports what upstream setup_follower_bot leaves a follower in.
    hal.attach_torque_stop(_seam())
    hal.connect()
    try:
        hal.send_action(action)
    finally:
        hal.disconnect()
    by_topic = {topic: msg for topic, msg in transport.calls}
    assert len(by_topic) == len(transport.calls) == 4, transport.calls
    return by_topic


class TestSendActionSplit:
    def test_publishes_the_xs_sdk_contract(self, hal: AlohaHAL, transport: SimTransport) -> None:
        sent = _sent(hal, transport, _indexed_action())
        assert set(sent) == {
            "/follower_left/commands/joint_group",
            "/follower_left/commands/joint_single",
            "/follower_right/commands/joint_group",
            "/follower_right/commands/joint_single",
        }
        for topic, msg in sent.items():
            assert isinstance(msg, dict)
            assert msg["name"] == ("arm" if topic.endswith("joint_group") else "gripper")

    def test_arms_get_indices_0_to_5_and_7_to_12(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        sent = _sent(hal, transport, _indexed_action())
        assert sent["/follower_left/commands/joint_group"]["cmd"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert sent["/follower_right/commands/joint_group"]["cmd"] == [
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ]

    def test_a_chunk_sends_its_first_step_not_its_last(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """One-target messages: jumping to the chunk's end would skip the planned path."""
        sent = _sent(hal, transport, _indexed_action(steps=3))
        assert sent["/follower_left/commands/joint_group"]["cmd"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_gripper_maps_normalised_onto_the_command_range(self, transport: SimTransport) -> None:
        hal = AlohaHAL(
            publish_fn=transport.publish, state_fn=transport.state, gripper_cmd_range=(0.5, 1.5)
        )
        targets = [0.0] * _N_JOINTS
        targets[6], targets[13] = 1.0, 0.25  # left fully open, right a quarter
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[targets])
        sent = _sent(hal, transport, action)
        assert sent["/follower_left/commands/joint_single"]["cmd"] == pytest.approx(1.5)
        assert sent["/follower_right/commands/joint_single"]["cmd"] == pytest.approx(0.75)

    def test_default_gripper_range_is_upstream_aloha(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """Interbotix aloha/constants.py FOLLOWER_GRIPPER_JOINT_{CLOSE,OPEN}."""
        sent = _sent(hal, transport, _zero_action())
        assert sent["/follower_left/commands/joint_single"]["cmd"] == pytest.approx(0.6197)

    def test_connect_refuses_without_a_transport(self, transport: SimTransport) -> None:
        """``build_hal(mode="real")`` injects nothing: commands would go nowhere (#250)."""
        for kwargs in ({}, {"publish_fn": transport.publish}, {"state_fn": transport.state}):
            hal = AlohaHAL(**kwargs)
            with pytest.raises(ROSConfigError, match="commands would go nowhere"):
                hal.connect()

    def test_the_manifest_built_real_hal_refuses_to_connect(self) -> None:
        """The real ``deploy run`` path: manifest params reach the HAL, no transport does."""
        from pathlib import Path

        from openral_core import RobotDescription
        from openral_hal import build_hal

        manifest = Path(__file__).resolve().parents[2] / "robots/aloha_bimanual/robot.yaml"
        hal = build_hal(RobotDescription.from_yaml(str(manifest)), mode="real")
        assert isinstance(hal, AlohaHAL)
        assert hal.arm_namespaces() == ["follower_left", "follower_right"]
        with pytest.raises(ROSConfigError, match="publish_fn / state_fn"):
            hal.connect()

    def test_the_manifest_built_real_hal_drives_once_a_transport_is_attached(
        self, transport: SimTransport
    ) -> None:
        """What the lifecycle node does under ``hal_mode:=real``: attach, then connect."""
        from pathlib import Path

        from openral_core import RobotDescription
        from openral_hal import build_hal
        from openral_hal.aloha import InterbotixStoppable

        manifest = Path(__file__).resolve().parents[2] / "robots/aloha_bimanual/robot.yaml"
        hal = build_hal(RobotDescription.from_yaml(str(manifest)), mode="real")
        assert isinstance(hal, AlohaHAL)
        assert isinstance(hal, InterbotixStoppable)
        assert hal.joint_names() == [j.name for j in ALOHA_DESCRIPTION.joints]
        assert hal.gripper_cmd_range == pytest.approx((0.6197, 1.6214))
        hal.attach_transport(transport.publish, transport.state, time.monotonic)
        sent = _sent(hal, transport, _zero_action())
        assert set(sent) == {
            f"/{ns}/commands/{kind}"
            for ns in ("follower_left", "follower_right")
            for kind in ("joint_group", "joint_single")
        }

    def test_attach_transport_refuses_after_connect_and_non_callables(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        with pytest.raises(ROSConfigError, match="needs a callable"):
            hal.attach_transport(transport.publish, transport.state, 0.0)  # type: ignore[arg-type]  # reason: the guard under test
        hal.connect()
        with pytest.raises(ROSRuntimeError, match="before connect"):
            hal.attach_transport(transport.publish, transport.state)

    def test_bad_gripper_range_rejected(self) -> None:
        for bad in ((1.0,), (1.0, 1.0), (0.5, float("nan")), (float("inf"), 1.0)):
            with pytest.raises(ROSConfigError, match="closed, open"):
                AlohaHAL(gripper_cmd_range=bad)

    def test_gripper_overshoot_is_clamped_to_the_calibrated_stops(
        self, transport: SimTransport
    ) -> None:
        hal = AlohaHAL(
            publish_fn=transport.publish, state_fn=transport.state, gripper_cmd_range=(0.5, 1.5)
        )
        targets = [0.0] * _N_JOINTS
        targets[6], targets[13] = 1.3, -0.2
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[targets])
        sent = _sent(hal, transport, action)
        assert sent["/follower_left/commands/joint_single"]["cmd"] == pytest.approx(1.5)
        assert sent["/follower_right/commands/joint_single"]["cmd"] == pytest.approx(0.5)

    def test_stock_linear_position_gripper_refuses_at_connect(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """Bring-up yaml mode: radians read as metres put a NaN on the gripper (flag 1)."""
        hal.attach_torque_stop(_seam(modes={"arm": "position", "gripper": "linear_position"}))
        with pytest.raises(ROSConfigError, match="current_based_position"):
            hal.connect()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(_zero_action())
        assert transport.calls == []

    def test_unconfirmed_modes_refuse_at_connect_without_a_seam(
        self, transport: SimTransport
    ) -> None:
        hal = AlohaHAL(publish_fn=transport.publish, state_fn=transport.state)
        with pytest.raises(ROSConfigError, match="cannot confirm"):
            hal.connect()
        assert transport.calls == []

    def test_unreachable_mode_service_refuses_at_connect(self, hal: AlohaHAL) -> None:
        """Checked at configure, so a down xs_sdk never blocks a command tick."""
        hal.attach_torque_stop(_seam(arms=["follower_left"]))  # right xs_sdk down
        with pytest.raises(ROSRuntimeError, match="follower_right/get_robot_info"):
            hal.connect()

    def test_modes_are_read_once_at_connect_not_per_command(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """Commands make no service calls: a wrong mode found later cannot stall ticks."""
        seam = _seam()
        hal.attach_torque_stop(seam)
        hal.connect()
        seam._modes["gripper"] = "linear_position"  # would refuse, if it were re-read
        try:
            hal.send_action(_zero_action())
            hal.send_action(_zero_action())
        finally:
            hal.disconnect()
        assert len(transport.calls) == 8

    def test_modes_are_reread_on_every_connect(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """A relaunched xs_sdk comes back in the bring-up yaml's modes."""
        sent = _sent(hal, transport, _zero_action())
        assert len(sent) == 4
        hal.attach_torque_stop(_seam(modes={"arm": "position", "gripper": "linear_position"}))
        with pytest.raises(ROSConfigError, match="linear_position"):
            hal.connect()

    def test_a_stream_that_went_stale_voids_the_mode_check_until_reconnect(
        self, transport: SimTransport
    ) -> None:
        """xs_sdk relaunch: the stream drops, resumes in bring-up modes; commands stay refused."""
        stamps = [time.monotonic()]
        hal = AlohaHAL(
            publish_fn=transport.publish,
            state_fn=transport.state,
            stamp_fn=lambda: stamps[0],
            staleness_limit_s=0.05,
        )
        hal.attach_torque_stop(_seam())
        hal.connect()
        try:
            hal.read_state()  # fresh
            stamps[0] = time.monotonic() - 1.0  # the driver went away
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
            stamps[0] = time.monotonic()  # ... and came back
            hal.read_state()
            with pytest.raises(ROSRuntimeError, match="went stale"):
                hal.send_action(_zero_action())
            assert transport.calls == []
        finally:
            hal.disconnect()
        hal.connect()  # re-reads the modes
        try:
            hal.send_action(_zero_action())
        finally:
            hal.disconnect()
        assert len(transport.calls) == 4

    def test_never_fresh_at_startup_does_not_void_the_mode_check(
        self, transport: SimTransport
    ) -> None:
        """Before the first sample the stream is stale by definition; that is not a relaunch."""
        stamps = [0.0]
        hal = AlohaHAL(
            publish_fn=transport.publish, state_fn=transport.state, stamp_fn=lambda: stamps[0]
        )
        hal.attach_torque_stop(_seam())
        hal.connect()
        try:
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
            stamps[0] = time.monotonic()
            hal.send_action(_zero_action())
        finally:
            hal.disconnect()
        assert len(transport.calls) == 4

    def test_non_finite_target_rejected(self, hal: AlohaHAL) -> None:
        targets = [0.0] * _N_JOINTS
        targets[13] = float("nan")
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[targets])
        hal.connect()
        try:
            with pytest.raises(ROSConfigError, match="non-finite"):
                hal.send_action(action)
        finally:
            hal.disconnect()

    def test_single_namespace_rejected(self) -> None:
        """One namespace would collide both arms onto one command topic."""
        for bad in (
            ["follower_left"],
            ["follower_left", "follower_left"],
            [],
            "lr",
            ["", "/"],
            ["/follower_left", "follower_left"],  # one namespace, spelled twice
        ):
            with pytest.raises(ROSConfigError, match="two distinct"):
                AlohaHAL(arm_namespaces=bad)

    def test_namespaces_are_stored_without_slashes(self) -> None:
        """A slash left on would build ``//ns/commands/...``, an invalid topic name."""
        hal = AlohaHAL(arm_namespaces=["/follower_left", "follower_right/"])
        assert hal.arm_namespaces() == ["follower_left", "follower_right"]

    def test_bad_gripper_position_range_rejected(self) -> None:
        for bad in ((0.04,), (0.05, 0.05), (0.04, float("inf"))):
            with pytest.raises(ROSConfigError, match="gripper_position_range"):
                AlohaHAL(gripper_position_range=bad)

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
    # test_estop_always_raises moved to
    # tests/unit/test_hal_protocol_conformance.py::test_hal_estop_always_raises_estoprequested
    # (parametrized over HAL_BUILDERS, "AlohaHAL" included).
    # test_estop_is_safety_violation moved to
    # tests/unit/test_hal_protocol_conformance.py::test_estoprequested_is_safety_violation_subclass
    # (structural check that ROSEStopRequested subclasses ROSSafetyViolation).

    def test_estop_cuts_torque_on_both_arms_and_publishes_nothing(
        self, hal: AlohaHAL, transport: SimTransport
    ) -> None:
        """The downstream stop is xs_sdk's torque_enable(false) per arm, not a topic.

        The earlier adapter published on ``/aloha/estop`` for a watchdog node that does not
        exist in this repo — a stop request nothing consumed. Now every arm namespace gets
        a ``torque_enable`` call for the ``all`` group and the report is acknowledged only
        when every arm answered.
        """
        from openral_hal.sim_transport import SimTorqueSeam

        seam = SimTorqueSeam(arms=["follower_left", "follower_right"])
        hal.attach_torque_stop(seam)
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert seam.calls == [
            ("follower_left", "all", False),
            ("follower_right", "all", False),
        ]
        assert not seam.torque("follower_left") and not seam.torque("follower_right")
        assert transport.calls == []
        report = hal.last_stop_report
        assert report is not None and report.stopped
        assert report.controller_states == {
            "follower_left": "torque_off",
            "follower_right": "torque_off",
        }
        with pytest.raises(ROSRuntimeError):
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.0] * 14]
                )
            )

    def test_one_arm_refusing_torque_off_is_reported_not_hidden(self, hal: AlohaHAL) -> None:
        """One arm refusing torque off is reported not hidden."""
        from openral_hal.sim_transport import SimTorqueSeam

        seam = SimTorqueSeam(arms=["follower_left", "follower_right"], failing=["follower_right"])
        hal.attach_torque_stop(seam)
        hal.connect()
        with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
            hal.estop()
        # The left arm was still torqued off — a refusal on one side never skips the other.
        assert not seam.torque("follower_left")
        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "follower_right" in report.detail

    def test_an_exception_on_one_arm_still_torques_off_the_other(self, hal: AlohaHAL) -> None:
        """An rclpy-shaped plain Exception on one arm is recorded, never lets the loop escape."""
        from openral_hal.sim_transport import SimTorqueSeam

        seam = SimTorqueSeam(
            arms=["follower_left", "follower_right"],
            faults={"follower_left": RuntimeError("context has been shut down")},
        )
        hal.attach_torque_stop(seam)
        hal.connect()
        with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
            hal.estop()
        assert not seam.torque("follower_right")
        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "follower_left: RuntimeError: context has been shut down" in report.detail
        assert report.controller_states["follower_right"] == "torque_off"

    def test_without_a_seam_the_stop_is_reported_unproven(self) -> None:
        """Without a seam the stop is reported unproven."""
        hal = AlohaHAL()
        with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
            hal.estop()
        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "no torque stop seam" in report.detail

    def test_recovery_policy_is_restart_required(self, hal: AlohaHAL) -> None:
        """Recovery policy is restart required."""
        from openral_hal.protocol import EStopRecovery, LifecycleEStopHAL

        assert isinstance(hal, LifecycleEStopHAL)
        assert hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
        assert hal.arm_namespaces() == ["follower_left", "follower_right"]


# ── Staleness guard ───────────────────────────────────────────────────────────


class TestStaleness:
    def test_read_state_raises_when_stale(self, transport: SimTransport) -> None:
        hal = AlohaHAL(
            publish_fn=transport.publish,
            state_fn=transport.state,
            staleness_limit_s=0.0,
        )
        hal.attach_torque_stop(_seam())
        hal.connect()
        time.sleep(0.001)
        try:
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
        finally:
            hal.disconnect()

    def test_never_received_is_stale_not_fresh_at_connect(self, transport: SimTransport) -> None:
        """``stamp_fn`` 0.0 (no arm reported yet) must not read as a fresh pose of zeros."""
        hal = AlohaHAL(publish_fn=transport.publish, state_fn=transport.state, stamp_fn=lambda: 0.0)
        hal.attach_torque_stop(_seam())
        hal.connect()
        try:
            with pytest.raises(ROSPerceptionStale, match="inf"):
                hal.read_state()
        finally:
            hal.disconnect()

    def test_age_follows_the_newest_sample_not_connect(self, transport: SimTransport) -> None:
        """Freshness is the data's, so a live stream stays fresh long after connect()."""
        stamps = [time.monotonic()]
        hal = AlohaHAL(
            publish_fn=transport.publish,
            state_fn=transport.state,
            stamp_fn=lambda: stamps[0],
            staleness_limit_s=0.05,
        )
        hal.attach_torque_stop(_seam())
        hal.connect()
        try:
            time.sleep(0.1)
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()  # the stream stopped: stale
            stamps[0] = time.monotonic() - 0.02
            state = hal.read_state()  # a sample arrived 20 ms ago: fresh
            assert time.time_ns() - state.stamp_ns >= 20_000_000  # stamped at arrival
        finally:
            hal.disconnect()
