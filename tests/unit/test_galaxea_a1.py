"""Galaxea A1 HAL contract tests against a process-boundary fake transport."""

from __future__ import annotations

import json
import runpy
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from openral_runner.backends.galaxea_a1_ipc import RuntimeLocalClient, runtime_socket_path
from openral_runner.factory import SENSOR_BACKEND_REGISTRY, make_sensor_readers
from openral_sim.policies.lingbot_va_a1 import (
    _joint_contract,
    _model_identity,
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
    assert "runtime_config" not in manifest.policy_extras


def test_lingbot_rskill_pins_the_gateway_model_identity() -> None:
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
    )

    repo_id, revision = _model_identity(spec)

    assert repo_id == "pengyue-polaron/lingbot-va-galaxea-a1-fruit-placement-eef"
    assert revision == "90e017bdbc6afac2e441b4634c9192776bbcb8b7"


def test_lingbot_joint_contract_uses_manifest_limits_and_policy_bound() -> None:
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
        extra={"max_joint_substep_rad": 0.045},
    )

    names, lower, upper, step = _joint_contract(
        spec,
        GALAXEA_A1_DESCRIPTION,
    )

    assert names == tuple(f"arm_joint{index}" for index in range(1, 7))
    assert lower == pytest.approx([-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798])
    assert upper == pytest.approx([2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798])
    assert step == 0.045


def test_lingbot_joint_contract_rejects_bound_above_hal_phase_limit() -> None:
    spec = VLASpec(
        id="lingbot_va_a1",
        weights_uri="rskills/lingbot-va-galaxea-a1-fruit-placement",
        extra={"max_joint_substep_rad": 0.06},
    )

    with pytest.raises(ROSConfigError, match="no greater than"):
        _joint_contract(spec, GALAXEA_A1_DESCRIPTION)


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


def test_camera_bridge_backend_uses_runtime_process_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A1_PROCESS_STATE_ROOT", str(tmp_path))
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={"camera": "front"},
        max_age_ms=500,
    )
    reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
    assert reader.sensor_id == "front"
    assert not reader.is_open
    assert runtime_socket_path("a1-camera-bridge.sock") == (tmp_path / "a1-camera-bridge.sock")
    with pytest.raises(ROSConfigError, match="corresponding A1 Runtime service"):
        reader.open()


def test_camera_bridge_backend_rejects_checkout_parameters() -> None:
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={
            "camera": "front",
            "runtime_root_env": "OPENRAL_GALAXEA_A1_RUNTIME_ROOT",
        },
    )

    with pytest.raises(ROSConfigError, match="unknown params"):
        SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)


def test_camera_bridge_backend_rejects_unknown_camera() -> None:
    cfg = SensorReaderConfig(
        sensor_id="front",
        backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
        backend_params={"camera": "side"},
    )
    with pytest.raises(ROSConfigError, match="'front' or 'wrist'"):
        SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)


def test_runtime_local_client_closes_a_broken_transport() -> None:
    client_socket, peer_socket = socket.socketpair()
    client = RuntimeLocalClient(
        "test.sock",
        label="test Runtime service",
        max_response_bytes=1024,
    )
    client._socket = client_socket
    peer_socket.close()

    with pytest.raises(ROSRuntimeError, match="transport failed"):
        client.call({"op": "describe"}, timeout_s=0.1)

    assert client._socket is None


def test_camera_bridge_batch_shares_one_paired_session() -> None:
    configs = [
        SensorReaderConfig(
            sensor_id=camera,
            backend=SensorReaderBackend.GALAXEA_A1_CAMERA_BRIDGE,
            backend_params={"camera": camera},
            max_age_ms=500,
        )
        for camera in ("front", "wrist")
    ]

    front, wrist = make_sensor_readers(configs)

    assert front._session is wrist._session


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


class _FakeSidecarPublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)

    def get_num_connections(self) -> int:
        return 1


class _FakeSidecarSubscriber:
    def get_num_connections(self) -> int:
        return 1


class _FakeSidecarRospy:
    class Time:
        @staticmethod
        def now() -> int:
            return 123

    def __init__(self) -> None:
        self.publishers: dict[str, _FakeSidecarPublisher] = {}

    def Publisher(  # noqa: N802 - mirrors rospy's public API
        self, topic: str, _message_type: object, *, queue_size: int
    ) -> _FakeSidecarPublisher:
        assert queue_size == 1
        publisher = _FakeSidecarPublisher(topic)
        self.publishers[topic] = publisher
        return publisher

    def Subscriber(  # noqa: N802 - mirrors rospy's public API
        self,
        _topic: str,
        _message_type: object,
        _callback: object,
        *,
        queue_size: int,
    ) -> _FakeSidecarSubscriber:
        assert queue_size == 1
        return _FakeSidecarSubscriber()

    def logerr(self, *_args: object) -> None:
        pass

    def logwarn(self, *_args: object) -> None:
        pass


def _sidecar_joint_state() -> SimpleNamespace:
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


def _sidecar_arm_command(position: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=0),
        p_des=list(position),
        v_des=[0.0] * 6,
        kp=[1.0] * 6,
        kd=[1.0] * 6,
        t_ff=[0.0] * 6,
        mode=0,
    )


def _sidecar_gripper_command() -> SimpleNamespace:
    return SimpleNamespace(header=SimpleNamespace(stamp=0), gripper_stroke=0.0)


_SIDECAR_HELLO: dict[str, object] = {
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


def _primed_sidecar_bridge(
    current: tuple[float, ...],
) -> tuple[_FakeSidecarRospy, Any, dict[str, object]]:
    """Build a configured Bridge with joint/gripper/status feedback already latched."""
    rospy = _FakeSidecarRospy()
    bridge = _SIDECAR["Bridge"](
        rospy,
        _sidecar_joint_state,
        object,
        _sidecar_gripper_command,
        SimpleNamespace,
    )
    config = _SIDECAR["_hello_config"](dict(_SIDECAR_HELLO))
    joint_feedback = _sidecar_joint_state()
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
    return rospy, bridge, config


def test_sidecar_relay_never_forwards_unaligned_tracker_startup() -> None:
    arm_command = _sidecar_arm_command
    # Encoder zero/quantization can leave feedback just beyond a nominal
    # endpoint. The requested hold is projected to the exact legal endpoint.
    current = (0.0, -0.001, -1.0, 0.0, 0.0, 0.0)
    hold_target = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    rospy, bridge, config = _primed_sidecar_bridge(current)

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


def test_sidecar_gripper_requires_active_relay() -> None:
    """Gripper actuation is gated by the same relay machine as arm joints.

    The gripper bypasses the tracker's staged-hold interpolation, so a
    setpoint arriving while the relay is still ``LOCKED`` or ``ARMING`` must
    fail closed instead of moving the physical gripper before alignment.
    """
    current = (0.0, -0.001, -1.0, 0.0, 0.0, 0.0)
    hold_target = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    rospy, bridge, config = _primed_sidecar_bridge(current)
    gripper_publisher = rospy.publishers["/gripper_position_control_host"]

    # LOCKED: a gripper-only action is refused and nothing is published.
    with pytest.raises(RuntimeError, match="not ACTIVE: LOCKED"):
        bridge.apply({"gripper": 0.5}, config, True)
    assert gripper_publisher.messages == []

    # First joint command stages the relay (ARMING); gripper stays refused,
    # even when it rides in the same packet as a valid joint target.
    assert not bridge.apply({"joint_targets": list(hold_target)}, config, True)
    assert bridge.relay_state() == "ARMING"
    with pytest.raises(RuntimeError, match="not ACTIVE: ARMING"):
        bridge.apply({"gripper": 0.5, "joint_targets": list(hold_target)}, config, False)
    assert gripper_publisher.messages == []

    # Tracker aligns with the staged hold -> ACTIVE: the setpoint now forwards.
    bridge._on_staged(_sidecar_arm_command(hold_target))
    assert bridge.relay_state() == "ACTIVE"
    assert not bridge.apply({"gripper": 0.5}, config, False)
    assert len(gripper_publisher.messages) == 1
    assert gripper_publisher.messages[0].gripper_stroke == 52.0


def test_deploy_sim_of_real_only_robot_raises_typed_errors() -> None:
    """`deploy sim` of the hardware-only A1 fails with typed errors, never a TypeError.

    The bare-twin path reports the robot as real-hardware-only; attaching the
    deploy bench scene (which registers no sim scene id) reports the unknown
    scene id — the latter used to crash with ``TypeError: '_Registry' object
    is not iterable`` while formatting its own error message.
    """
    from openral_core.exceptions import ROSCapabilityMismatch
    from openral_hal.resolver import build_hal

    repo_root = Path(__file__).resolve().parents[2]
    robot_yaml = repo_root / "robots" / "galaxea_a1" / "robot.yaml"
    description = RobotDescription.from_yaml(str(robot_yaml))
    with pytest.raises(ROSCapabilityMismatch, match="real-hardware-only"):
        build_hal(description, mode="sim")
    with pytest.raises(ROSConfigError, match="not registered"):
        build_hal(
            description,
            mode="sim",
            sim_env_yaml=str(repo_root / "scenes" / "deploy" / "galaxea_a1_bench.yaml"),
        )


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
