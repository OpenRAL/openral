"""ADR-0020 PR-F — end-to-end: the C++ kernel emits safety.check spans.

Brings up the real ``safety_kernel_node`` as a subprocess pointed at a
loopback FastAPI server that decodes OTLP/HTTP protobuf on
``/v1/traces``. The test then publishes one valid and one invalid
``ActionChunk`` and asserts the receiver collected two ``safety.check``
spans with ``safety.kernel="cpp"`` and the expected severities.

This is the contract test that closes the gap with the dashboard: the
dashboard's TelemetryStore consumes exactly the same OTLP/HTTP payload
on the same route, so a passing test here means the dashboard's Safety
card will populate when the kernel runs against ``openral dashboard``.

No mocks (CLAUDE.md §1.11): real rclpy, real colcon-built openral_msgs,
real opentelemetry-cpp exporter, real FastAPI ASGI receiver. Gated on
``rclpy`` / ``openral_msgs`` / ``opentelemetry-proto`` being importable
— without a sourced ROS env or without the otel proto deps the test
skips.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import uuid
from typing import Any

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

# ``Request`` must live at module scope: with ``from __future__ import
# annotations`` the route's ``request: Request`` hint is a string that
# FastAPI resolves against the function's globals (the module namespace),
# not the local ``__init__`` scope. Imported only inside ``__init__`` it
# resolves to nothing on FastAPI ≥0.116 and the param is mistaken for a
# query field → every OTLP POST 422s and no spans are recorded.
from fastapi import Request

# Kernel envelope as ROS parameters (ADR-0020 PR-K).
_KERNEL_PARAM_ARGS: list[str] = [
    "-p",
    "n_dof:=3",
    "-p",
    "robot_name:=launchtest",
    "-p",
    "joint_position_min:=[-1.0, -1.0, -1.0]",
    "-p",
    "joint_position_max:=[1.0, 1.0, 1.0]",
    "-p",
    "joint_velocity_max:=[3.15, 3.15, 3.15]",
    "-p",
    "joint_torque_max:=[5.0, 5.0, 5.0]",
    "-p",
    "max_ee_speed_m_s:=0.5",
    "-p",
    "max_ee_accel_m_s2:=2.0",
    "-p",
    "max_force_n:=10.0",
    "-p",
    "max_torque_nm:=3.0",
    "-p",
    "contact_force_threshold_n:=5.0",
    "-p",
    "deadman_required:=false",
    "-p",
    "estop_reset_cooldown_s:=0.1",
]


def _free_port() -> int:
    """Pick an unused loopback port for the OTLP receiver."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _OtlpHttpReceiver:
    """Minimal FastAPI app that decodes OTLP/HTTP protobuf trace exports.

    Mirrors the route shape of the in-tree dashboard
    (python/observability/.../dashboard/app.py:60-86) so a span that
    lands here would also land on the dashboard's Safety card.
    """

    def __init__(self) -> None:
        from fastapi import FastAPI
        from opentelemetry.proto.collector.trace.v1 import (
            trace_service_pb2,
        )

        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._app = FastAPI()

        @self._app.post("/v1/traces")
        async def ingest(request: Request) -> dict[str, Any]:
            req = trace_service_pb2.ExportTraceServiceRequest()
            req.ParseFromString(await request.body())
            with self._lock:
                for rs in req.resource_spans:
                    for ss in rs.scope_spans:
                        for span in ss.spans:
                            attrs: dict[str, Any] = {}
                            for kv in span.attributes:
                                value_field = kv.value.WhichOneof("value")
                                if value_field == "string_value":
                                    attrs[kv.key] = kv.value.string_value
                                elif value_field == "bool_value":
                                    attrs[kv.key] = kv.value.bool_value
                                elif value_field == "int_value":
                                    attrs[kv.key] = kv.value.int_value
                                elif value_field == "double_value":
                                    attrs[kv.key] = kv.value.double_value
                            event_names = [e.name for e in span.events]
                            self.spans.append(
                                {
                                    "name": span.name,
                                    "attrs": attrs,
                                    "events": event_names,
                                }
                            )
            return {}

    @property
    def app(self) -> Any:
        return self._app


def _start_receiver(port: int) -> tuple[_OtlpHttpReceiver, Any, threading.Thread]:
    """Start the receiver in a background uvicorn server on ``port``."""
    import uvicorn

    receiver = _OtlpHttpReceiver()
    config = uvicorn.Config(receiver.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for the socket to accept connections.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return receiver, server, thread
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("OTLP receiver failed to start")


def _start_kernel(
    node_name: str,
    domain_id: int,
    otlp_endpoint: str,
) -> Any:
    """Spawn the C++ safety kernel pointed at our loopback OTLP receiver."""
    import shutil

    if shutil.which("ros2") is None:
        pytest.skip("ros2 binary not on PATH; source install/setup.bash first")

    env = {
        **os.environ,
        "ROS_DOMAIN_ID": str(domain_id),
        # The C++ otel.cpp reads this; match what configure_observability
        # consumes on the Python side (_sdk.py:48-49).
        "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint,
    }
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "openral_safety_kernel",
            "safety_kernel_node",
            "--ros-args",
            "-r",
            f"__node:={node_name}",
            *_KERNEL_PARAM_ARGS,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _activate_lifecycle(node_name: str, helper: Any) -> bool:
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = helper.create_client(ChangeState, f"/{node_name}/change_state")
    if not client.wait_for_service(timeout_sec=10.0):
        return False
    for t in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        req = ChangeState.Request()
        req.transition.id = t
        fut = client.call_async(req)
        deadline = time.time() + 5.0
        while time.time() < deadline and not fut.done():
            rclpy.spin_once(helper, timeout_sec=0.05)
        if not fut.done() or not fut.result().success:  # type: ignore[union-attr]
            return False
    return True


def test_kernel_emits_safety_check_spans_to_otlp_receiver() -> None:
    """One pass + one violation → at least two safety.check spans on the wire."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.executors import SingleThreadedExecutor

    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    receiver, _server, _thread = _start_receiver(port)

    node_name = f"safety_kernel_otel_test_{uuid.uuid4().hex[:8]}"
    domain_id = 60 + (os.getpid() % 40)
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    proc = _start_kernel(node_name, domain_id, endpoint)
    if True:  # preserve indent of original tempfile context
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("safety_kernel_otel_helper")
                assert _activate_lifecycle(node_name, helper), "kernel failed to activate"

                pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                # Discovery settle.
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)

                # One pass — joint 0 inside [-1.0, 1.0].
                pass_chunk = ActionChunk()
                pass_chunk.control_mode = 0
                pass_chunk.horizon = 1
                pass_chunk.n_dof = 3
                pass_chunk.flat = [0.1, 0.2, -0.1]
                pass_chunk.rskill_id = "openral/rskill-otel-pass"
                pass_chunk.trace_id = ""
                pub.publish(pass_chunk)

                # One violation — joint 0 = 5.0 > 1.0.
                bad_chunk = ActionChunk()
                bad_chunk.control_mode = 0
                bad_chunk.horizon = 1
                bad_chunk.n_dof = 3
                bad_chunk.flat = [5.0, 0.0, 0.0]
                bad_chunk.rskill_id = "openral/rskill-otel-bad"
                bad_chunk.trace_id = ""
                pub.publish(bad_chunk)

                # Wait for BatchSpanProcessor to flush.
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    executor.spin_once(timeout_sec=0.05)
                    if sum(1 for s in receiver.spans if s["name"] == "safety.check") >= 2:
                        break

                safety_spans = [s for s in receiver.spans if s["name"] == "safety.check"]
                assert len(safety_spans) >= 2, (
                    f"expected ≥2 safety.check spans, got {len(safety_spans)}: "
                    f"{[s['name'] for s in receiver.spans]}"
                )
                # Every span carries the kernel identity that latches the
                # dashboard's Identity row.
                for s in safety_spans:
                    assert s["attrs"].get("safety.kernel") == "cpp", s["attrs"]
                    assert s["attrs"].get("safety.check_name") == "envelope", s["attrs"]

                severities = {s["attrs"].get("safety.severity") for s in safety_spans}
                assert "info" in severities, severities
                assert "violation" in severities, severities

                violation = next(
                    s for s in safety_spans if s["attrs"].get("safety.severity") == "violation"
                )
                assert "openral.event.safety_violation" in violation["events"], violation["events"]
            finally:
                rclpy.shutdown()
        finally:
            import signal

            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=2)
