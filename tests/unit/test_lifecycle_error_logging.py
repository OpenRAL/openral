"""Unit tests for :func:`openral_observability.log_lifecycle_errors`.

Regression guard for the opaque ``runtime_node`` exit-code-4 failure: when a
``LifecycleNode`` transition callback (``on_configure`` / ``on_activate`` / …)
raises, rclpy's ``LifecycleNodeMixin.__execute_callback`` catches the exception
and returns ``TransitionCallbackReturn.ERROR`` **without logging it** (see the
literal ``# TODO(ivanpauno): log sth here`` in rclpy). The host then prints only
the return code, so the real traceback is lost and the operator sees nothing but
``exit code 4``.

:func:`log_lifecycle_errors` closes that gap: it wraps the callback so any
uncaught exception is logged with its full traceback via the node's ROS logger
(``get_logger()`` → ``/rosout`` → the launch console) and converted to a clean
``TransitionCallbackReturn.FAILURE``.

Per CLAUDE.md §1.11 — no mocks. The decorated callbacks run on a real
``rclpy.lifecycle.LifecycleNode`` driven through a real ``trigger_configure``.
"""

from __future__ import annotations

import pytest


def _rclpy_available() -> bool:
    """True iff rclpy + lifecycle are importable in this venv."""
    try:
        import rclpy
        import rclpy.lifecycle  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / lifecycle not on PYTHONPATH; source a ROS 2 install to run",
)


def test_raising_configure_logs_traceback_and_returns_failure(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A decorated ``on_configure`` that raises → FAILURE (not ERROR) + logged traceback."""
    import rclpy
    from openral_observability import log_lifecycle_errors
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

    class _RaisingNode(LifecycleNode):
        @log_lifecycle_errors
        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            raise ValueError("synthetic configure failure")

    rclpy.init()
    node: _RaisingNode | None = None
    try:
        node = _RaisingNode("openral_lifecycle_err_test_raise")
        rc = node.trigger_configure()

        # rclpy would have returned ERROR (99) for an uncaught exception; the
        # decorator must downgrade it to FAILURE so the host can report a clean
        # transition failure rather than the opaque ERROR sentinel.
        assert rc == TransitionCallbackReturn.FAILURE

        err = capfd.readouterr().err
        assert "synthetic configure failure" in err
        assert "Traceback" in err
        assert "ValueError" in err
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_successful_configure_passes_through_unchanged(capfd: pytest.CaptureFixture[str]) -> None:
    """The decorator is transparent on success — SUCCESS returned, nothing logged as error."""
    import rclpy
    from openral_observability import log_lifecycle_errors
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

    class _OkNode(LifecycleNode):
        @log_lifecycle_errors
        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            return TransitionCallbackReturn.SUCCESS

    rclpy.init()
    node: _OkNode | None = None
    try:
        node = _OkNode("openral_lifecycle_err_test_ok")
        rc = node.trigger_configure()
        assert rc == TransitionCallbackReturn.SUCCESS

        err = capfd.readouterr().err
        assert "Traceback" not in err
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


# ── bringup span (deploy.bringup) ────────────────────────────────────────────


def test_transition_emits_a_bringup_span_with_node_and_transition() -> None:
    """The decorator times every transition, not just failures.

    Nothing measured bringup before this: the "HAL ``on_configure`` takes
    ~6 s, or ~27 s on a cold robocasa kitchen" figures that justify the
    300 s lifecycle-autostart timeouts lived only in launch-file comments,
    so they could neither be verified nor detected when they regressed.
    Instrumenting the shared decorator covers every node that already uses
    it without touching a single call site.
    """
    import rclpy
    from openral_observability import log_lifecycle_errors, semconv
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)

    class _TimedNode(LifecycleNode):
        @log_lifecycle_errors
        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            return TransitionCallbackReturn.SUCCESS

    rclpy.init()
    node: _TimedNode | None = None
    try:
        node = _TimedNode("openral_lifecycle_bringup_span_test")
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS

        spans = [s for s in exporter.get_finished_spans() if s.name == semconv.SPAN_DEPLOY_BRINGUP]
        assert spans, "the transition must leave a deploy.bringup span"
        attrs = dict(spans[-1].attributes or {})
        assert attrs[semconv.BRINGUP_TRANSITION] == "on_configure"
        assert attrs[semconv.BRINGUP_NODE] == "openral_lifecycle_bringup_span_test"
        assert attrs[semconv.BRINGUP_RESULT] == "SUCCESS"
        assert spans[-1].end_time is not None and spans[-1].start_time is not None
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        exporter.clear()


def test_a_failed_transition_marks_the_bringup_span_error() -> None:
    """A clean FAILURE return is still a failed bringup and must show red."""
    import rclpy
    from openral_observability import log_lifecycle_errors, semconv
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)

    class _FailingNode(LifecycleNode):
        @log_lifecycle_errors
        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            return TransitionCallbackReturn.FAILURE

    rclpy.init()
    node: _FailingNode | None = None
    try:
        node = _FailingNode("openral_lifecycle_bringup_fail_test")
        node.trigger_configure()

        spans = [s for s in exporter.get_finished_spans() if s.name == semconv.SPAN_DEPLOY_BRINGUP]
        assert spans
        assert spans[-1].status.status_code == StatusCode.ERROR
        assert dict(spans[-1].attributes or {})[semconv.BRINGUP_RESULT] == "FAILURE"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        exporter.clear()
