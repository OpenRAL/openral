"""Regression: HAL lifecycle configure failures must surface, not vanish.

``HALLifecycleNodeBase.on_configure`` already catches *typed* errors
(``ROSConfigError`` / ``ROSRuntimeError``) and returns ``FAILURE``. But an
*unexpected* exception (e.g. the ``ModuleNotFoundError`` that broke
``openral_rskill_ros``'s ``runtime_node``) would otherwise escape into rclpy's
``__execute_callback``, which converts it to ``TransitionCallbackReturn.ERROR``
**without logging** — so the lifecycle-autostart driver sees only an opaque
failure and the traceback is lost.

The :func:`openral_observability.log_lifecycle_errors` decorator on the base's
``on_configure`` closes that gap for every HAL (UR5e / Franka / SO-100 /
OpenArm / panda_mobile / …) at once, since they all share this base. This test
drives a minimal real subclass whose ``_create_hal`` raises an untyped error
through a real ``trigger_configure``.

Per CLAUDE.md §1.11 — no mocks. The node is a real ``HALLifecycleNodeBase``
subclass on a real ``rclpy`` context.
"""

from __future__ import annotations

import pytest


def _rclpy_available() -> bool:
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


def test_hal_unexpected_configure_error_logs_traceback_and_returns_failure(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """An untyped exception in HAL configure → FAILURE (not ERROR) + logged traceback."""
    import rclpy
    from openral_hal.lifecycle import HALLifecycleNodeBase
    from rclpy.lifecycle import TransitionCallbackReturn

    class _BoomHAL(HALLifecycleNodeBase):  # type: ignore[misc, valid-type]
        def _create_hal(self) -> object:
            # Untyped error — NOT ROSConfigError / ROSRuntimeError, so it bypasses
            # the base's typed except and would otherwise reach rclpy's silent ERROR.
            raise ValueError("synthetic HAL construction failure")

    rclpy.init()
    node: _BoomHAL | None = None
    try:
        node = _BoomHAL("openral_hal_err_test")
        rc = node.trigger_configure()

        assert rc == TransitionCallbackReturn.FAILURE

        err = capfd.readouterr().err
        assert "synthetic HAL construction failure" in err
        assert "Traceback" in err
        assert "ValueError" in err
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
