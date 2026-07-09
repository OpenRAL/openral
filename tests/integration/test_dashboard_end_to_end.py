"""End-to-end integration test for ``openral dashboard``.

Spins up the dashboard on a real local socket, configures a real OTel
SDK to export spans over OTLP/HTTP at that socket, emits a span, and
asserts the dashboard's ``/api/state`` reflects it. Real components
end-to-end per CLAUDE.md §1.11 and §5.4.

The test is in ``tests/integration/`` (not ``tests/unit/``) because it
spins a uvicorn thread; per CLAUDE.md §5.4 integration tests are
allowed to take a few seconds.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from contextlib import closing
from typing import Any

import pytest
import uvicorn
from openral_observability.dashboard import TelemetryStore, create_app
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5).read()
            return
        except Exception as exc:
            last_err = exc
            time.sleep(0.05)
    msg = f"dashboard not ready on :{port}: {last_err!r}"
    raise RuntimeError(msg)


def _fetch_state(port: int) -> dict[str, Any]:
    import json

    raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2.0).read()
    return json.loads(raw)


def _pynvml_available() -> bool:
    """Return ``True`` iff a live NVIDIA driver answers via pynvml.

    The system_metrics sampler runs the same probe internally; we replicate
    it here so the integration test can decide whether to additionally
    assert against the GPU bucket. Failing imports or a missing driver both
    legitimately yield ``False`` and the assertion path is skipped.
    """
    try:
        import pynvml  # type: ignore[import-not-found]  # reason: optional probe
    except ImportError:
        return False
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        return False
    return True


def test_dashboard_receives_real_otlp_http_span() -> None:
    store = TelemetryStore()
    app = create_app(store)
    port = _free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn")
    thread.start()
    try:
        _wait_ready(port)

        # Real OTel SDK + real OTLP/HTTP exporter — no mocks.
        exporter = OTLPSpanExporter(endpoint=f"http://127.0.0.1:{port}/v1/traces")
        provider = TracerProvider(resource=Resource.create({"service.name": "ral-itest"}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("ral-itest")

        with tracer.start_as_current_span("rskill.execute") as span:
            span.set_attribute("rskill.id", "smolvla-libero")
            span.set_attribute("openral.tick.idx", 7)

        provider.shutdown()

        deadline = time.monotonic() + 3.0
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            state = _fetch_state(port)
            if "rskill_execute" in state.get("cards", {}):
                break
            time.sleep(0.05)
        assert state["service_name"] == "ral-itest", state
        card = state["cards"].get("rskill_execute")
        assert card is not None, state
        assert card["attrs"]["rskill.id"] == "smolvla-libero"
        assert card["attrs"]["openral.tick.idx"] == 7
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        # Reset the global tracer provider so other tests aren't poisoned.
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset


def test_dashboard_receives_real_otlp_http_debug_log() -> None:
    """A real OTLP/HTTP DEBUG log line lands in the dashboard event log (issue #318).

    Drives the same path production uses — a real OTel ``LoggerProvider`` +
    OTLP/HTTP log exporter + stdlib ``LoggingHandler`` (the structlog→OTel
    bridge's transport) — and asserts the dashboard surfaces it as a
    ``debug``-severity event keyed by the logger (scope) name. No mocks per
    CLAUDE.md §1.11.
    """
    import logging

    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

    store = TelemetryStore()
    app = create_app(store)
    port = _free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn-logs")
    thread.start()
    logger_name = "openral.world_state.itest"
    logger = logging.getLogger(logger_name)
    provider = LoggerProvider(resource=Resource.create({"service.name": "ral-logs-itest"}))
    try:
        _wait_ready(port)

        exporter = OTLPLogExporter(endpoint=f"http://127.0.0.1:{port}/v1/logs")
        provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        handler = LoggingHandler(level=logging.DEBUG, logger_provider=provider)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)

        logger.debug("world_state.detected_objects count=0")
        provider.shutdown()

        deadline = time.monotonic() + 3.0
        debug_events: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            events = _fetch_state(port).get("events", [])
            debug_events = [ev for ev in events if ev.get("severity") == "debug"]
            if debug_events:
                break
            time.sleep(0.05)
        assert debug_events, "no debug event reached the dashboard"
        ev = debug_events[0]
        assert ev["kind"] == logger_name, ev
        assert ev["title"] == "world_state.detected_objects count=0", ev
    finally:
        logger.removeHandler(handler)
        server.should_exit = True
        thread.join(timeout=5.0)


def test_dashboard_system_health_card_receives_gpu_cpu_ram() -> None:
    """`configure_observability` must boot the host sampler so the dashboard's
    System health card surfaces CPU / RAM (and GPU when available).

    Regression for "System health in dashboard not showing GPU usage" — the
    sampler used to be defined but never invoked outside its own unit test,
    so the topic bucket the UI reads stayed empty for the lifetime of the
    process.
    """
    pytest.importorskip("psutil")  # sampler no-ops without psutil/pynvml.
    from openral_observability import (
        configure_observability,
        shutdown_observability,
        system_metrics,
    )
    from openral_observability._sdk import _ENV_METRIC_INTERVAL_MS

    # Isolate from any prior observability test that installed an OTel meter
    # provider and left OTel's set-once guard tripped. A stale global provider
    # makes the `configure_observability` call below silently no-op its
    # `set_meter_provider` (the SDK warns and keeps the old provider), so the
    # host sampler's gauges never reach *this* test's dashboard exporter and the
    # System card stays empty — the exact intermittent CI failure this test hit
    # when an earlier observability test ran first. The teardown already resets
    # this for the *next* test; reset up front too so we aren't the victim of
    # the *previous* one.
    from opentelemetry.metrics import _internal as metrics_internal

    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset

    store = TelemetryStore()
    app = create_app(store)
    port = _free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn")
    thread.start()
    import os

    prior_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    prior_protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    prior_interval = os.environ.get(_ENV_METRIC_INTERVAL_MS)
    try:
        _wait_ready(port)

        # Drive the workload through the real SDK entry point — the same
        # call site every `openral` subcommand uses. A short export interval
        # keeps the test wall-time tight without changing production code.
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://127.0.0.1:{port}"
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
        os.environ[_ENV_METRIC_INTERVAL_MS] = "500"

        # Detect whether the GPU path is exercisable here so we can assert
        # it landed in the topic bucket when it is. We probe before the
        # sampler starts; the sampler has its own guard and will skip GPU
        # if the driver is absent.
        gpu_available = _pynvml_available()

        assert configure_observability(service_name="ral-system-health-itest") is True
        try:
            # 1× export interval (500 ms) + sampler interval (1 s) + buffer.
            deadline = time.monotonic() + 5.0
            sys_topic: dict[str, Any] = {}
            while time.monotonic() < deadline:
                sys_topic = _fetch_state(port).get("topics", {}).get("system", {})
                have_cpu_ram = "ram_total_mb" in sys_topic and "cpu_util_pct" in sys_topic
                have_gpu = not gpu_available or bool(sys_topic.get("gpus"))
                if have_cpu_ram and have_gpu:
                    break
                time.sleep(0.1)
            assert "ram_total_mb" in sys_topic, sys_topic
            assert sys_topic["ram_total_mb"] > 0.0
            assert "cpu_util_pct" in sys_topic, sys_topic
            if gpu_available:
                # Regression: the original report was specifically "GPU usage
                # not showing in dashboard". Without the nvidia-ml-py direct
                # dep, pynvml wasn't importable and the GPU bucket stayed
                # empty even on a host with a real NVIDIA driver.
                gpus = sys_topic.get("gpus") or {}
                assert gpus, f"GPU bucket empty despite live driver: {sys_topic}"
                first = next(iter(gpus.values()))
                assert "memory_total_mb" in first and first["memory_total_mb"] > 0.0, first
        finally:
            shutdown_observability()
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        for key, value in (
            ("OTEL_EXPORTER_OTLP_ENDPOINT", prior_endpoint),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", prior_protocol),
            (_ENV_METRIC_INTERVAL_MS, prior_interval),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Reset the globals other tests rely on.
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
        from opentelemetry.metrics import _internal as metrics_internal

        metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
        metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
        # Make sure the daemon sampler isn't left running between test files.
        assert system_metrics._thread is None, "shutdown should have stopped the sampler"


def test_dashboard_robot_state_card_receives_hal_read_state_joint_arrays() -> None:
    """``hal.read_state`` spans with array attributes must populate ``topics.robot_state``.

    Regression for "joint states not displayed on dashboard":  the OTel SDK
    encodes list attributes as OTLP ``array_value`` and the store must decode
    them back into Python lists that the frontend's ``renderRobotState`` can
    iterate.  This end-to-end test also covers the fix for the overwrite bug —
    an error-path span (no joint attrs) must not blank previously stored data.
    """
    store = TelemetryStore()
    app = create_app(store)
    port = _free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="dashboard-uvicorn-joints")
    thread.start()
    try:
        _wait_ready(port)

        exporter = OTLPSpanExporter(endpoint=f"http://127.0.0.1:{port}/v1/traces")
        provider = TracerProvider(resource=Resource.create({"service.name": "ral-joints-itest"}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("ral-joints-itest")

        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # 1. Successful hal.read_state span with full joint data.
        with tracer.start_as_current_span("hal.read_state") as span:
            span.set_attribute("openral.hal.adapter", "so100followerhal")
            span.set_attribute("openral.hal.robot.model", "so100_follower")
            span.set_attribute("openral.tick.idx", 0)
            span.set_attribute("openral.hal.joint.names", joint_names)
            span.set_attribute("openral.hal.joint.positions", joint_positions)
            span.set_attribute("openral.hal.joint.velocities", [0.0] * 6)
            span.set_attribute("openral.hal.joint.position_limits_lo", [-3.14] * 6)
            span.set_attribute("openral.hal.joint.position_limits_hi", [3.14] * 6)

        # 2. Error-path hal.read_state span — no joint attributes.
        with tracer.start_as_current_span("hal.read_state") as span:
            span.set_attribute("openral.hal.adapter", "so100followerhal")
            span.set_attribute("openral.tick.idx", 1)

        provider.shutdown()

        deadline = time.monotonic() + 3.0
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            state = _fetch_state(port)
            rs = state.get("topics", {}).get("robot_state", {})
            if rs.get("names"):
                break
            time.sleep(0.05)

        rs = state.get("topics", {}).get("robot_state", {})
        # Joint arrays survive the OTLP/HTTP round-trip as Python lists.
        assert rs.get("names") == joint_names, rs
        assert rs.get("positions") == pytest.approx(joint_positions, abs=1e-3), rs
        # Error-path span must not have blanked the data.
        assert rs.get("names") is not None, "error-path span cleared joint names"
        assert rs.get("positions") is not None, "error-path span cleared joint positions"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
