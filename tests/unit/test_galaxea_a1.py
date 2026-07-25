"""Galaxea A1 HAL contract tests against a process-boundary fake transport."""

from __future__ import annotations

import json
import runpy
import signal
import socket
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import (
    Action,
    ControlMode,
    RobotDescription,
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
    RSkillManifest,
    SensorReaderBackend,
    SensorReaderConfig,
    VLASpec,
)
from openral_hal.galaxea_a1 import (
    GALAXEA_A1_DESCRIPTION,
    GalaxeaA1HAL,
    _Snapshot,
    _SocketTransport,
)
from openral_hal.protocol import HAL, HALHealthProvider, LifecycleEStopHAL
from openral_runner.factory import SENSOR_BACKEND_REGISTRY
from openral_sim.policies.lingbot_va_a1 import (
    _bounded_joint_substep,
    _joint_limits_from_description,
    _LingBotVaA1Adapter,
    _runtime_paths,
    _validated_joint_substep_limit,
)

_SIDECAR = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "galaxea_a1_ros1_sidecar.py")
)


class FakeA1Transport:
    def __init__(self, *, status_codes: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)) -> None:
        self.connected = False
        self.hello: dict[str, object] | None = None
        self.messages: list[dict[str, object]] = []
        self.estopped = False
        now = time.monotonic()
        self.current = _Snapshot(
            position=(0.0, 1.0, -1.0, 0.0, 0.0, 0.0),
            velocity=(0.0,) * 6,
            effort=(0.0,) * 6,
            stamp_ns=time.time_ns(),
            received_monotonic=now,
            status_codes=status_codes,
            status_received_monotonic=now,
            gripper_norm=0.5,
        )

    def connect(self, hello: dict[str, object], timeout_s: float) -> None:
        del timeout_s
        self.connected = True
        self.hello = hello

    def close(self) -> None:
        self.connected = False

    def snapshot(self) -> _Snapshot:
        return self.current

    def send(self, message: dict[str, object]) -> None:
        self.messages.append(message)

    def estop(self, timeout_s: float) -> None:
        del timeout_s
        self.estopped = True
        self.connected = False


def _joint_action(delta: float = 0.0) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        joint_targets=[[delta, 1.0, -1.0, 0.0, 0.0, 0.0]],
        stamp_ns=time.time_ns(),
    )


def test_description_and_protocol_surface() -> None:
    transport = FakeA1Transport()
    hal = GalaxeaA1HAL(transport=transport)
    assert isinstance(hal, HAL)
    assert isinstance(hal, HALHealthProvider)
    assert isinstance(hal, LifecycleEStopHAL)
    assert hal.description is GALAXEA_A1_DESCRIPTION
    assert hal.description.sdk_kind == "closed_with_api"
    assert hal.description.hal.sim is None
    assert hal.description.capabilities.has_vision
    assert [sensor.name for sensor in hal.description.sensors] == ["front", "wrist"]
    assert "policy_substep_rad" not in hal.description.hal.parameters.defaults


def test_lingbot_manifest_owns_policy_joint_substep() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "rskills"
        / "lingbot-va-galaxea-a1-fruit-placement"
        / "rskill.yaml"
    )
    manifest = RSkillManifest.from_yaml(str(manifest_path))

    assert manifest.policy_extras["max_joint_substep_rad"] == 0.045
    assert (
        manifest.policy_extras["runtime_config"]
        == "configs/deployments/lingbot/fruit_placement_eef.toml"
    )


def test_lingbot_runtime_config_resolves_inside_runtime_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs" / "deployments" / "lingbot" / "fruit_placement_eef.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[deployment]\n", encoding="utf-8")
    monkeypatch.setenv("OPENRAL_GALAXEA_A1_RUNTIME_ROOT", str(tmp_path))
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
        extra={"runtime_config": str(config.relative_to(tmp_path))},
    )

    assert _runtime_paths(spec) == (tmp_path.resolve(), config.resolve())


def test_lingbot_runtime_config_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENRAL_GALAXEA_A1_RUNTIME_ROOT", str(tmp_path))
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
    )

    with pytest.raises(ROSConfigError, match=r"policy_extras\.runtime_config"):
        _runtime_paths(spec)


@pytest.mark.parametrize("runtime_config", ["/tmp/outside.toml", "../outside.toml"])
def test_lingbot_runtime_config_cannot_escape_runtime_checkout(
    runtime_config: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENRAL_GALAXEA_A1_RUNTIME_ROOT", str(tmp_path))
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
        extra={"runtime_config": runtime_config},
    )

    with pytest.raises(ROSConfigError, match=r"must be relative|escapes"):
        _runtime_paths(spec)


def test_lingbot_ik_uses_openral_manifest_joint_limits() -> None:
    names = tuple(joint.name for joint in GALAXEA_A1_DESCRIPTION.joints)

    lower, upper = _joint_limits_from_description(
        GALAXEA_A1_DESCRIPTION,
        expected_joint_names=names,
    )

    assert lower == pytest.approx([-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798])
    assert upper == pytest.approx([2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798])


def test_lingbot_ik_rejects_description_without_command_limits() -> None:
    joints = list(GALAXEA_A1_DESCRIPTION.joints)
    joints[1] = joints[1].model_copy(update={"position_limits": None})
    description = GALAXEA_A1_DESCRIPTION.model_copy(update={"joints": joints})

    with pytest.raises(ValueError, match="finite position limits"):
        _joint_limits_from_description(
            description,
            expected_joint_names=tuple(joint.name for joint in joints),
        )


def test_manifest_and_hal_pin_official_a1_joint_transforms() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "robots" / "galaxea_a1" / "robot.yaml"
    manifest = RobotDescription.from_yaml(str(manifest_path))
    expected = [
        ((-0.0011147, 0.0, 0.0892), (0.0, 0.0, 3.1416), (0.0, 0.0, 1.0)),
        ((0.0, -0.00004, 0.0615), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.34928, 0.02, 0.0), (0.0, 0.0, 1.5708), (0.0, 0.0, 1.0)),
        ((0.07, -0.00395, -0.00004), (-1.5708, 1.5708, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.2776), (-1.5708, 0.0, 3.1416), (0.0, 0.0, 1.0)),
        ((0.0, -0.1575, -0.00023266), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]

    for description in (manifest, GALAXEA_A1_DESCRIPTION):
        assert [
            (joint.origin_xyz, joint.origin_rpy, joint.axis_xyz) for joint in description.joints
        ] == expected


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "sidecar.example", "::1"])
def test_hal_rejects_non_loopback_sidecar_hosts(host: str) -> None:
    with pytest.raises(ROSConfigError, match="IPv4 loopback"):
        GalaxeaA1HAL(host=host)


@pytest.mark.parametrize("port", [True, "46011", 46011.5, 0, 65536])
def test_hal_rejects_invalid_sidecar_ports(port: object) -> None:
    with pytest.raises(ROSConfigError, match="TCP port"):
        GalaxeaA1HAL(port=port)  # type: ignore[arg-type] # reason: runtime validation contract


def test_camera_bridge_backend_requires_explicit_a1_runtime_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={
            "camera": "front",
            "runtime_root_env": "OPENRAL_TEST_A1_RUNTIME_ROOT",
            "deployment_config": "configs/deployments/lingbot/fruit_placement_eef.toml",
        },
        max_age_ms=500,
    )
    reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
    assert reader.sensor_id == "front"
    assert not reader.is_open
    monkeypatch.delenv("OPENRAL_TEST_A1_RUNTIME_ROOT", raising=False)
    with pytest.raises(ROSConfigError, match="OPENRAL_TEST_A1_RUNTIME_ROOT"):
        reader.open()


def test_camera_bridge_backend_requires_explicit_deployment_config() -> None:
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={"camera": "front"},
    )

    with pytest.raises(ROSConfigError, match=r"backend_params\.deployment_config"):
        SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)


def test_camera_bridge_backend_rejects_unknown_camera() -> None:
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={"camera": "side"},
    )
    with pytest.raises(ROSConfigError, match="'front' or 'wrist'"):
        SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)


def test_lingbot_joint_lowering_respects_manifest_step_limit() -> None:
    current = np.zeros(6, dtype=np.float64)
    target = np.array([0.16, -0.08, 0.04, 0.0, 0.0, 0.0], dtype=np.float64)
    step_limit = _validated_joint_substep_limit(
        policy_substep_rad=0.04,
        max_target_step_rad=0.04,
        initial_alignment_tolerance_rad=0.05,
    )
    first, reached = _bounded_joint_substep(
        current,
        target,
        max_step_rad=step_limit,
    )
    assert not reached
    assert float(np.max(np.abs(first - current))) < 0.04
    second, reached = _bounded_joint_substep(
        first,
        target,
        max_step_rad=step_limit,
    )
    assert not reached
    assert float(np.max(np.abs(second - first))) < 0.04
    third, reached = _bounded_joint_substep(second, target, max_step_rad=step_limit)
    assert not reached
    fourth, reached = _bounded_joint_substep(third, target, max_step_rad=step_limit)
    assert not reached
    final, reached = _bounded_joint_substep(fourth, target, max_step_rad=step_limit)
    assert reached
    assert final == pytest.approx(target)


def test_lingbot_joint_lowering_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="matching finite vectors"):
        _bounded_joint_substep(
            np.zeros(6, dtype=np.float64),
            np.array([np.nan, 0, 0, 0, 0, 0], dtype=np.float64),
            max_step_rad=0.08,
        )


def test_lingbot_joint_lowering_rejects_policy_bound_above_hal_phase_limits() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _validated_joint_substep_limit(
            policy_substep_rad=0.06,
            max_target_step_rad=0.08,
            initial_alignment_tolerance_rad=0.05,
        )


def test_lingbot_boundary_inference_replays_last_safe_target_without_blocking() -> None:
    adapter = _LingBotVaA1Adapter.__new__(_LingBotVaA1Adapter)
    adapter._boundary_future = Future()
    adapter._boundary_deadline = time.monotonic() + 1.0
    adapter._inference_timeout_s = 1.0
    adapter._last_joint_target = np.arange(6, dtype=np.float64) / 10.0
    adapter._last_gripper_target = 0.25

    started = time.monotonic()
    action = adapter._poll_chunk_boundary()

    assert time.monotonic() - started < 0.05
    assert action == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.25])


def test_lingbot_boundary_inference_fails_at_manifest_latency_budget() -> None:
    adapter = _LingBotVaA1Adapter.__new__(_LingBotVaA1Adapter)
    adapter._boundary_future = Future()
    adapter._boundary_deadline = time.monotonic() - 0.001
    adapter._inference_timeout_s = 6.0

    with pytest.raises(ROSRuntimeError, match=r"6\.000 s per-chunk latency budget"):
        adapter._poll_chunk_boundary()


def test_lingbot_action_advances_only_after_reaching_solved_target() -> None:
    adapter = _LingBotVaA1Adapter.__new__(_LingBotVaA1Adapter)
    adapter._active_joint_target = np.array(
        [0.12, -0.06, 0.03, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    adapter._active_gripper = 0.25
    adapter._max_joint_step_rad = 0.03
    adapter._origin_pose7 = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    adapter._solver = SimpleNamespace(
        forward=lambda joints: (
            np.asarray(joints[:3], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )
    adapter._config = SimpleNamespace(action=SimpleNamespace(pose_mode="absolute"))
    adapter._action_config = SimpleNamespace(min_quat_norm=1e-6)
    adapter._chunk = SimpleNamespace(
        cache_state=np.zeros((8, 1, 1), dtype=np.float64),
        needs_observation_after=lambda _index: True,
    )
    adapter._active_cache_frame_index = 0
    adapter._active_action_index = 0
    adapter._step_index = 0
    adapter._pending_observation = False
    adapter._last_joint_target = None
    adapter._last_gripper_target = None

    first = adapter._step_active_action(np.zeros(6, dtype=np.float64))

    assert first == pytest.approx([0.03, -0.015, 0.0075, 0.0, 0.0, 0.0, 0.25])
    assert adapter._step_index == 0
    assert adapter._pending_observation is False
    assert adapter._active_joint_target is not None
    assert adapter._chunk.cache_state == pytest.approx(np.zeros((8, 1, 1)))

    final_feedback = np.array(
        [0.11, -0.055, 0.0275, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    final = adapter._step_active_action(final_feedback)

    assert final == pytest.approx([0.12, -0.06, 0.03, 0.0, 0.0, 0.0, 0.25])
    assert adapter._step_index == 1
    assert adapter._pending_observation is True
    assert adapter._active_joint_target is None
    assert adapter._chunk.cache_state[:, 0, 0] == pytest.approx(
        [0.12, -0.06, 0.03, 0.0, 0.0, 0.0, 1.0, 0.25]
    )


def test_connect_reads_state_and_sends_aligned_joint_target() -> None:
    transport = FakeA1Transport()
    hal = GalaxeaA1HAL(transport=transport)
    hal.connect()
    assert hal.read_state().position == [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    hal.send_action(_joint_action(0.01))
    assert transport.messages[-1]["joint_targets"] == [0.01, 1.0, -1.0, 0.0, 0.0, 0.0]
    assert transport.hello is not None
    assert transport.hello["robot"] == "galaxea_a1"
    assert transport.hello["command_lease_s"] == 0.5


def test_sidecar_cross_checks_identity_limits_and_mask_types() -> None:
    hello = {
        "op": "hello",
        "protocol": 1,
        "robot": "galaxea_a1",
        "joint_names": [f"arm_joint{i}" for i in range(1, 7)],
        "joint_position_min": [-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798],
        "joint_position_max": [2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798],
        "initial_alignment_tolerance_rad": 0.05,
        "tracker_alignment_timeout_s": 5.0,
        "max_target_step_rad": 0.08,
        "command_lease_s": 0.5,
        "state_timeout_s": 0.5,
        "status_timeout_s": 1.0,
        "feedback_limit_tolerance_rad": 0.01,
        "idle_timeout_error_mask": 64,
        "gripper_ignored_error_mask": 8,
        "gripper_stroke_min_mm": 0.0,
        "gripper_stroke_max_mm": 104.0,
    }
    decode_hello = _SIDECAR["_hello_config"]

    assert decode_hello(hello)["joint_position_min"][1] == 0.0
    with pytest.raises(ValueError, match="official A1 limits"):
        decode_hello(
            {**hello, "joint_position_min": [-3.0, 0.0, -3.3161, -2.8798, -1.6581, -2.8798]}
        )
    with pytest.raises(ValueError, match="integers"):
        decode_hello({**hello, "idle_timeout_error_mask": True})


def test_sidecar_pins_every_vendor_topic_in_launch_commands() -> None:
    driver = _SIDECAR["_driver_argv"]("/dev/a1")
    tracker = _SIDECAR["_tracker_argv"]()
    tracker_parameters = _SIDECAR["_tracker_parameters"]("/opt/galaxea/A1_SDK")

    assert "single_arm_serial_port_path:=/dev/a1" in driver
    assert "single_arm_joint_states_topic:=/joint_states_host" in driver
    assert "single_arm_control_topic:=/arm_joint_command_host" in driver
    assert "single_arm_gripper_position_control_topic:=/gripper_position_control_host" in driver
    assert tracker == [
        "rosrun",
        "mobiman",
        "jointTracker_demo_node",
        "__name:=openral_a1_joint_tracker",
    ]
    assert (
        tracker_parameters["/openral_a1_joint_tracker/joint_states_sub_topic"]
        == "/joint_states_host"
    )
    assert (
        tracker_parameters["/openral_a1_joint_tracker/arm_joint_command_topic"]
        == "/openral/arm_joint_command_staged"
    )
    assert (
        tracker_parameters["/openral_a1_joint_tracker/arm_joint_target_position"]
        == "/arm_joint_target_position"
    )
    assert not any("eePose" in key for key in tracker_parameters)


def test_sidecar_relay_never_forwards_unaligned_tracker_startup() -> None:
    class FakePublisher:
        def __init__(self, topic: str) -> None:
            self.topic = topic
            self.messages: list[object] = []

        def publish(self, message: object) -> None:
            self.messages.append(message)

        def get_num_connections(self) -> int:
            return 1

    class FakeSubscriber:
        def get_num_connections(self) -> int:
            return 1

    class FakeRospy:
        class Time:
            @staticmethod
            def now() -> int:
                return 123

        def __init__(self) -> None:
            self.publishers: dict[str, FakePublisher] = {}

        def Publisher(  # noqa: N802 - mirrors rospy's public API
            self, topic: str, _message_type: object, *, queue_size: int
        ) -> FakePublisher:
            assert queue_size == 1
            publisher = FakePublisher(topic)
            self.publishers[topic] = publisher
            return publisher

        def Subscriber(  # noqa: N802 - mirrors rospy's public API
            self,
            _topic: str,
            _message_type: object,
            _callback: object,
            *,
            queue_size: int,
        ) -> FakeSubscriber:
            assert queue_size == 1
            return FakeSubscriber()

        def logerr(self, *_args: object) -> None:
            pass

        def logwarn(self, *_args: object) -> None:
            pass

    def joint_state() -> SimpleNamespace:
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(to_nsec=time.time_ns),
                frame_id="",
            ),
            name=[],
            position=[],
            velocity=[],
            effort=[],
        )

    def arm_command(position: tuple[float, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            header=SimpleNamespace(stamp=0),
            p_des=list(position),
            v_des=[0.0] * 6,
            kp=[1.0] * 6,
            kd=[1.0] * 6,
            t_ff=[0.0] * 6,
            mode=0,
        )

    def gripper_command() -> SimpleNamespace:
        return SimpleNamespace(header=SimpleNamespace(stamp=0), gripper_stroke=0.0)

    rospy = FakeRospy()
    bridge = _SIDECAR["Bridge"](
        rospy,
        joint_state,
        object,
        gripper_command,
        SimpleNamespace,
    )
    config = _SIDECAR["_hello_config"](
        {
            "op": "hello",
            "protocol": 1,
            "robot": "galaxea_a1",
            "joint_names": [f"arm_joint{i}" for i in range(1, 7)],
            "joint_position_min": [-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798],
            "joint_position_max": [2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798],
            "initial_alignment_tolerance_rad": 0.05,
            "tracker_alignment_timeout_s": 5.0,
            "max_target_step_rad": 0.08,
            "command_lease_s": 0.5,
            "state_timeout_s": 0.5,
            "status_timeout_s": 1.0,
            "feedback_limit_tolerance_rad": 0.01,
            "idle_timeout_error_mask": 64,
            "gripper_ignored_error_mask": 8,
            "gripper_stroke_min_mm": 0.0,
            "gripper_stroke_max_mm": 104.0,
        }
    )
    # Encoder zero/quantization can leave feedback just beyond a nominal
    # endpoint. The requested hold is projected to the exact legal endpoint.
    current = (0.0, -0.001, -1.0, 0.0, 0.0, 0.0)
    hold_target = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    joint_feedback = joint_state()
    joint_feedback.name = [f"arm_joint{i}" for i in range(1, 7)]
    joint_feedback.position = list(current)
    joint_feedback.velocity = [0.0] * 6
    joint_feedback.effort = [0.0] * 6
    bridge._on_joint(joint_feedback)
    bridge._on_gripper(SimpleNamespace(position=[52.0]))
    bridge._on_status(
        SimpleNamespace(
            data=SimpleNamespace(motor_errors=[SimpleNamespace(error_code=64) for _ in range(7)])
        )
    )
    bridge.configure(config)

    assert not bridge.apply({"joint_targets": list(hold_target)}, config, True)
    target_publisher = rospy.publishers["/arm_joint_target_position"]
    assert len(target_publisher.messages) == 1
    host = rospy.publishers["/arm_joint_command_host"]
    bridge._on_staged(arm_command((0.105, 0.889, -0.623, 1.919, 0.312, 1.245)))
    assert bridge.relay_state() == "ARMING"
    assert host.messages == []
    updated_target = (0.01, 0.0, -1.0, 0.0, 0.0, 0.0)
    assert not bridge.apply({"joint_targets": list(updated_target)}, config, False)
    assert len(target_publisher.messages) == 2
    assert host.messages == []

    # The tracker begins at measured feedback. This near-endpoint sample is
    # tolerated only while staged and must never reach the host command topic.
    bridge._on_staged(arm_command(current))
    assert bridge.relay_state() == "ARMING"
    assert host.messages == []

    bridge._on_staged(arm_command(updated_target))
    assert bridge.relay_state() == "ACTIVE"
    assert len(host.messages) == 1

    gripper_publisher = rospy.publishers["/gripper_position_control_host"]
    assert not bridge.apply({"gripper": 0.75}, config, False)
    assert not bridge.apply({"gripper": 0.75}, config, False)
    assert len(gripper_publisher.messages) == 1
    assert gripper_publisher.messages[0].gripper_stroke == 78.0


def test_sidecar_stop_waits_for_sigkill_completion() -> None:
    stack = _SIDECAR["Stack"](shutdown_grace_s=0.05, kill_wait_s=1.0)
    child = stack.start(
        [
            sys.executable,
            "-c",
            ("import signal,time;signal.signal(signal.SIGINT, signal.SIG_IGN);time.sleep(30)"),
        ]
    )
    time.sleep(0.1)

    stack.stop()

    assert child.poll() is not None
    assert child.returncode == -signal.SIGKILL


def test_sidecar_drains_buffered_action_before_active_lease_check() -> None:
    """A tracker callback stall must not hide a fresh buffered hold."""

    class FakeRospy:
        def __init__(self, bridge: FakeBridge) -> None:
            self._bridge = bridge

        def loginfo(self, *_args: object) -> None:
            pass

        def is_shutdown(self) -> bool:
            return self._bridge.apply_count >= 2

    class FakeBridge:
        def __init__(self) -> None:
            self.apply_count = 0
            self.active = False
            self.state_entered = threading.Event()
            self.release_state = threading.Event()

        def configure(self, _config: dict[str, object]) -> None:
            pass

        def apply(
            self,
            _packet: dict[str, object],
            _config: dict[str, object],
            _first_joint_command: bool,
        ) -> bool:
            self.apply_count += 1
            return False

        def relay_state(self) -> str:
            return "ACTIVE" if self.active else "ARMING"

        def state(self, _config: dict[str, object]) -> dict[str, object]:
            if self.apply_count == 1:
                self.state_entered.set()
                assert self.release_state.wait(1.0)
            return {"op": "state"}

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=1.0)
    bridge = FakeBridge()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            _SIDECAR["_serve"](listener, bridge, FakeRospy(bridge), lambda: None)
        except BaseException as exc:
            failures.append(exc)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    hello = {
        "op": "hello",
        "protocol": 1,
        "robot": "galaxea_a1",
        "joint_names": [f"arm_joint{i}" for i in range(1, 7)],
        "joint_position_min": [-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798],
        "joint_position_max": [2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798],
        "initial_alignment_tolerance_rad": 0.05,
        "tracker_alignment_timeout_s": 5.0,
        "max_target_step_rad": 0.08,
        "command_lease_s": 0.05,
        "state_timeout_s": 0.5,
        "status_timeout_s": 1.0,
        "feedback_limit_tolerance_rad": 0.01,
        "idle_timeout_error_mask": 64,
        "gripper_ignored_error_mask": 8,
        "gripper_stroke_min_mm": 0.0,
        "gripper_stroke_max_mm": 104.0,
    }
    client.sendall(json.dumps(hello).encode() + b"\n")
    client.sendall(json.dumps({"op": "action", "joint_targets": [0.0] * 6}).encode() + b"\n")
    assert bridge.state_entered.wait(1.0)

    # Queue a fresh hold while the server is starved in bridge.state(), then
    # let the relay become ACTIVE after the old command is already expired.
    client.sendall(json.dumps({"op": "action", "joint_targets": [0.0] * 6}).encode() + b"\n")
    time.sleep(0.08)
    bridge.active = True
    bridge.release_state.set()
    server.join(timeout=1.0)
    client.close()
    listener.close()

    assert not failures
    assert bridge.apply_count == 2


def test_real_socket_transport_round_trip() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    action_received = threading.Event()
    records: list[dict[str, object]] = []

    def _server() -> None:
        with listener:
            conn, _ = listener.accept()
            with conn, conn.makefile("rb") as stream:
                records.append(json.loads(stream.readline()))
                conn.sendall(
                    json.dumps({"op": "hello", "protocol": 1, "robot": "galaxea_a1"}).encode()
                    + b"\n"
                )
                state = {
                    "op": "state",
                    "name": [f"arm_joint{i}" for i in range(1, 7)],
                    "position": [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0] * 6,
                    "effort": [0.0] * 6,
                    "stamp_ns": time.time_ns(),
                    "status_codes": [0] * 7,
                    "gripper_norm": 0.5,
                    "relay_state": "LOCKED",
                }
                conn.sendall(json.dumps(state).encode() + b"\n")
                records.append(json.loads(stream.readline()))
                action_received.set()

    server = threading.Thread(target=_server, daemon=True)
    server.start()
    hal = GalaxeaA1HAL(host="127.0.0.1", port=port, connect_timeout_s=2.0)
    hal.connect()
    hal.send_action(_joint_action(0.01))
    assert action_received.wait(2.0)
    hal.disconnect()
    server.join(timeout=2.0)
    assert records[0]["op"] == "hello"
    assert records[1]["joint_targets"] == [0.01, 1.0, -1.0, 0.0, 0.0, 0.0]


def test_socket_transport_coalesces_adjacent_arm_and_gripper_modes() -> None:
    transport = _SocketTransport("127.0.0.1", 46011)
    transport.send({"op": "action", "stamp_ns": 10, "joint_targets": [0.0] * 6})
    transport.send({"op": "action", "stamp_ns": 11, "gripper": 0.75})

    message = transport._take_pending_message()

    assert message == {
        "op": "action",
        "stamp_ns": 11,
        "joint_targets": [0.0] * 6,
        "gripper": 0.75,
    }


def test_socket_transport_writer_is_not_blocked_by_state_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _SocketTransport("127.0.0.1", 46011)
    client, server = socket.socketpair()
    reader_entered = threading.Event()
    release_reader = threading.Event()

    def blocked_handle(_raw: object) -> None:
        reader_entered.set()
        assert release_reader.wait(1.0)

    monkeypatch.setattr(transport, "_handle", blocked_handle)
    transport._sock = client
    transport._reader_thread = threading.Thread(target=transport._read_loop, daemon=True)
    transport._writer_thread = threading.Thread(target=transport._write_loop, daemon=True)
    transport._reader_thread.start()
    transport._writer_thread.start()
    try:
        server.sendall(b"{}\n")
        assert reader_entered.wait(1.0)
        transport.send({"op": "action", "stamp_ns": 10, "joint_targets": [0.0] * 6})
        server.settimeout(1.0)
        assert json.loads(server.recv(4096))["joint_targets"] == [0.0] * 6
    finally:
        release_reader.set()
        transport.close()
        server.close()


def test_first_joint_target_must_align_with_feedback() -> None:
    hal = GalaxeaA1HAL(transport=FakeA1Transport())
    hal.connect()
    with pytest.raises(ROSConfigError, match="initial alignment"):
        hal.send_action(_joint_action(0.06))


def test_small_feedback_endpoint_error_is_tolerated_but_larger_error_fails() -> None:
    transport = FakeA1Transport()
    transport.current = _Snapshot(
        position=(0.0, -0.003, -1.0, 0.0, 0.0, 0.0),
        velocity=(0.0,) * 6,
        effort=(0.0,) * 6,
        stamp_ns=time.time_ns(),
        received_monotonic=time.monotonic(),
        status_codes=(0,) * 7,
        status_received_monotonic=time.monotonic(),
        gripper_norm=0.5,
    )
    GalaxeaA1HAL(transport=transport).connect()

    transport.current = _Snapshot(
        position=(0.0, -0.02, -1.0, 0.0, 0.0, 0.0),
        velocity=(0.0,) * 6,
        effort=(0.0,) * 6,
        stamp_ns=time.time_ns(),
        received_monotonic=time.monotonic(),
        status_codes=(0,) * 7,
        status_received_monotonic=time.monotonic(),
        gripper_norm=0.5,
    )
    with pytest.raises(ROSRuntimeError, match="feedback arm_joint2"):
        GalaxeaA1HAL(transport=transport).connect()


def test_motor_fault_blocks_command_but_idle_timeout_is_allowed() -> None:
    bad_transport = FakeA1Transport(status_codes=(1, 0, 0, 0, 0, 0, 0))
    bad = GalaxeaA1HAL(transport=bad_transport)
    with pytest.raises(ROSRuntimeError, match="error code 1"):
        bad.connect()
    assert not bad_transport.connected

    idle = FakeA1Transport(status_codes=(64, 64, 64, 64, 64, 64, 72))
    hal = GalaxeaA1HAL(transport=idle)
    hal.connect()
    hal.send_action(_joint_action())
    assert len(idle.messages) == 1


def test_absolute_joint_limit_is_enforced_inside_hal() -> None:
    transport = FakeA1Transport()
    transport.current = _Snapshot(
        position=(2.87, 1.0, -1.0, 0.0, 0.0, 0.0),
        velocity=(0.0,) * 6,
        effort=(0.0,) * 6,
        stamp_ns=time.time_ns(),
        received_monotonic=time.monotonic(),
        status_codes=(0,) * 7,
        status_received_monotonic=time.monotonic(),
        gripper_norm=0.5,
    )
    hal = GalaxeaA1HAL(transport=transport)
    hal.connect()
    with pytest.raises(ROSConfigError, match="outside"):
        hal.send_action(_joint_action(2.89))


def test_health_report_exposes_motor_codes_without_io() -> None:
    hal = GalaxeaA1HAL(transport=FakeA1Transport(status_codes=(64,) * 7))
    hal.connect()

    report = hal.health()

    assert report.message == "A1 feedback and motor status healthy"
    assert report.fields["motor_status_codes"] == "64,64,64,64,64,64,64"
    assert report.fields["gripper_position_normalized"] == "0.500000"


def test_gripper_action_uses_normalized_contract() -> None:
    transport = FakeA1Transport()
    hal = GalaxeaA1HAL(transport=transport)
    hal.connect()
    hal.send_action(
        Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.75], stamp_ns=time.time_ns())
    )
    assert transport.messages[-1]["gripper"] == 0.75


def test_estop_stops_transport_and_always_raises() -> None:
    transport = FakeA1Transport()
    hal = GalaxeaA1HAL(transport=transport)
    hal.connect()
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert transport.estopped
    with pytest.raises(ROSRuntimeError):
        hal.read_state()
