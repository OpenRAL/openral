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
