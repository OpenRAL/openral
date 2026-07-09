"""Per-backend :class:`SensorReader` implementations.

The default :class:`OpenCVThreadSensorReader` is always available
(``opencv-python`` is declared as the ``opencv`` optional-extra on
``openral-runner``). The :class:`GStreamerSensorReader` (PR I) and
:class:`Ros2ImageSensorReader` (PR D follow-up) gate on their respective
optional deps and import lazily so the package stays importable on hosts
without GStreamer / rclpy.

OpenCVThreadSensorReader is exposed via PEP 562 ``__getattr__`` so that
``import openral_runner.backends.gstreamer`` does NOT eagerly pull
in ``cv2`` — ``cv2`` initialises its own gstreamer / glib state that
segfaults a subsequent ``rclpy.init()`` / ``Node()`` inside the
x86-ros Docker image (observed in the ROS-tee smoke test, PR I/8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_runner.backends.opencv_thread import OpenCVThreadSensorReader

__all__ = ["OpenCVThreadSensorReader"]


def __getattr__(name: str) -> Any:  # noqa: ANN401  # reason: PEP 562 attribute hook
    """Lazy-import OpenCVThreadSensorReader so cv2 stays out of paths that don't need it."""
    if name == "OpenCVThreadSensorReader":
        from openral_runner.backends.opencv_thread import (  # noqa: PLC0415
            OpenCVThreadSensorReader,
        )

        return OpenCVThreadSensorReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
