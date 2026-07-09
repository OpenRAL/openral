"""Unit tests for :class:`openral_observability.DiagnosticsHeartbeat`.

Two test tiers (mirrors ``tests/unit/test_sensor_ros_publisher.py``):

* **Construction / validation** — no rclpy required. Asserts the helper
  rejects bad inputs and stays inert until ``create_publisher`` runs.
* **Live publish/subscribe** — gated on rclpy via
  ``pytest.importorskip``. Drives a real ``DiagnosticsHeartbeat``
  attached to a real ``rclpy.lifecycle.LifecycleNode``, opens an rclpy
  subscriber in the same process, and asserts the round-trip arrives
  with the expected ``hardware_id`` / ``component_name`` / level.

Per CLAUDE.md §1.11 — no mocks. The status callback is a real
function; the LifecycleNode is a real ``rclpy`` lifecycle node; the
subscriber is a real ``rclpy`` subscriber.
"""

from __future__ import annotations

import time

import pytest
from openral_observability.diagnostics import DiagnosticsHeartbeat, Level


def _ok_status() -> tuple[int, str, dict[str, str]]:
    """Real status_fn — not a mock."""
    return Level.OK, "healthy", {"tick": "42", "robot": "so100"}


# ── Construction / validation (no rclpy) ─────────────────────────────────────


def test_heartbeat_rejects_empty_hardware_id() -> None:
    """``hardware_id`` must be a non-empty string."""

    class _Stub:
        """Minimal node stand-in for the constructor path that never touches it."""

    with pytest.raises(ValueError, match=r"hardware_id"):
        DiagnosticsHeartbeat(
            _Stub(),  # type: ignore[arg-type]
            hardware_id="",
            component_name="safety",
            status_fn=_ok_status,
        )


def test_heartbeat_rejects_empty_component_name() -> None:
    """``component_name`` must be a non-empty string."""

    class _Stub:
        pass

    with pytest.raises(ValueError, match=r"component_name"):
        DiagnosticsHeartbeat(
            _Stub(),  # type: ignore[arg-type]
            hardware_id="openral_safety:so100",
            component_name="",
            status_fn=_ok_status,
        )


def test_heartbeat_rejects_non_positive_rate() -> None:
    """``rate_hz`` must be strictly positive."""

    class _Stub:
        pass

    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match=r"rate_hz"):
            DiagnosticsHeartbeat(
                _Stub(),  # type: ignore[arg-type]
                hardware_id="openral_safety:so100",
                component_name="safety",
                status_fn=_ok_status,
                rate_hz=bad,
            )


def test_heartbeat_construction_does_not_touch_ros() -> None:
    """Constructor never imports ``rclpy`` / ``diagnostic_msgs``."""

    class _Stub:
        """Constructor MUST NOT touch this stub."""

    hb = DiagnosticsHeartbeat(
        _Stub(),  # type: ignore[arg-type]
        hardware_id="openral_safety:so100",
        component_name="safety",
        status_fn=_ok_status,
    )
    assert hb.hardware_id == "openral_safety:so100"
    assert hb.component_name == "safety"


def test_heartbeat_start_before_create_publisher_raises() -> None:
    """``start()`` without ``create_publisher()`` is a programmer error."""

    class _Stub:
        pass

    hb = DiagnosticsHeartbeat(
        _Stub(),  # type: ignore[arg-type]
        hardware_id="openral_safety:so100",
        component_name="safety",
        status_fn=_ok_status,
    )
    with pytest.raises(RuntimeError, match=r"create_publisher"):
        hb.start()


# ── Live publish/subscribe (rclpy-gated) ─────────────────────────────────────


def _rclpy_available() -> bool:
    """True iff rclpy + diagnostic_msgs are importable in this venv."""
    try:
        import diagnostic_msgs.msg  # noqa: F401
        import rclpy
        import rclpy.lifecycle  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / diagnostic_msgs / lifecycle not on PYTHONPATH; "
    "source a ROS 2 install to run live tests",
)
def test_heartbeat_publishes_diagnostic_array_to_real_subscriber() -> None:
    """Full round-trip: real LifecycleNode → /diagnostics → real subscriber."""
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.lifecycle import LifecycleNode
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    received: list[DiagnosticArray] = []
    publisher_node: LifecycleNode | None = None
    sub_node = None
    try:
        publisher_node = LifecycleNode("openral_diag_test_publisher")
        # Track call count so we can assert status_fn actually ran.
        calls: list[int] = []

        def status_fn() -> tuple[int, str, dict[str, str]]:
            calls.append(1)
            return Level.WARN, "test heartbeat", {"tick": str(len(calls))}

        hb = DiagnosticsHeartbeat(
            publisher_node,
            hardware_id="openral_diag_test:unit",
            component_name="diag_test",
            status_fn=status_fn,
            rate_hz=10.0,
        )
        hb.create_publisher()
        hb.start()

        sub_node = rclpy.create_node("openral_diag_test_subscriber")
        sub_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        sub_node.create_subscription(DiagnosticArray, "/diagnostics", received.append, sub_qos)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            rclpy.spin_once(publisher_node, timeout_sec=0.02)
            rclpy.spin_once(sub_node, timeout_sec=0.02)

        assert received, "no DiagnosticArray received within 2 s"
        msg = received[0]
        assert len(msg.status) == 1
        status = msg.status[0]
        # ``level`` is a byte string of length 1 in rosidl Python bindings.
        assert int.from_bytes(status.level, "little") == Level.WARN
        assert status.name == "diag_test"
        assert status.hardware_id == "openral_diag_test:unit"
        assert status.message == "test heartbeat"
        kv = {pair.key: pair.value for pair in status.values}
        assert "tick" in kv
        assert int(kv["tick"]) >= 1
        assert calls, "status_fn was never called"

        hb.stop()
        hb.destroy()
    finally:
        if sub_node is not None:
            sub_node.destroy_node()
        if publisher_node is not None:
            publisher_node.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / diagnostic_msgs / lifecycle not on PYTHONPATH",
)
def test_heartbeat_status_fn_exception_publishes_error_level() -> None:
    """A raising ``status_fn`` must produce an ERROR-level diagnostic, not crash."""
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    received: list[DiagnosticArray] = []
    publisher_node: LifecycleNode | None = None
    sub_node = None
    try:
        publisher_node = LifecycleNode("openral_diag_test_publisher_exc")

        def bad_status() -> tuple[int, str, dict[str, str]]:
            raise RuntimeError("synthetic failure")

        hb = DiagnosticsHeartbeat(
            publisher_node,
            hardware_id="openral_diag_test:exc",
            component_name="diag_test_exc",
            status_fn=bad_status,
            rate_hz=10.0,
        )
        hb.create_publisher()
        hb.start()

        sub_node = rclpy.create_node("openral_diag_test_subscriber_exc")
        sub_node.create_subscription(DiagnosticArray, "/diagnostics", received.append, 10)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            rclpy.spin_once(publisher_node, timeout_sec=0.02)
            rclpy.spin_once(sub_node, timeout_sec=0.02)

        assert received, "no DiagnosticArray received within 2 s"
        status = received[0].status[0]
        assert int.from_bytes(status.level, "little") == Level.ERROR
        assert "synthetic failure" in status.message
        assert status.hardware_id == "openral_diag_test:exc"
    finally:
        if sub_node is not None:
            sub_node.destroy_node()
        if publisher_node is not None:
            publisher_node.destroy_node()
        rclpy.shutdown()
