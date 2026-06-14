"""SensorReader Protocol.

The :class:`SensorReader` Protocol is the seam between a sensor's physical
capture backend (OpenCV / ROS image topic / GStreamer pipeline) and the
inference runner. Each tick the runner asks every configured reader for
its freshest frame; the reader returns a :class:`SensorFrame` whose
``data | topic | handle`` carry-mode reflects the backend's transport.

The Protocol is intentionally narrow: ``open / close / read_latest``.
``open`` may start background capture threads or pipelines; ``close`` is
idempotent. ``read_latest`` is non-blocking — it returns the most recent
frame the backend has buffered, or raises
:class:`~openral_core.exceptions.ROSPerceptionStale` when the
freshest frame is older than the caller's ``max_age_ms`` budget.

See ``docs/adr/0010-inference-runner.md`` §SensorReader for the design.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core import SensorFrame

__all__ = ["SensorReader"]


@runtime_checkable
class SensorReader(Protocol):
    """Structural protocol every per-sensor reader backend satisfies.

    Three concrete backends live under ``openral_runner.backends``:

    - :class:`OpenCVThreadSensorReader` — default, per-camera background
      thread on top of ``cv2.VideoCapture`` (mirrors lerobot's pattern).
    - :class:`Ros2ImageSensorReader` (planned) — subscribes to a ROS 2
      image topic published by a vendor driver.
    - :class:`GStreamerSensorReader` (planned, ADR-0010 PR I) — pipeline
      string from config; appsink delivers frames. NVMM / DMA-BUF
      zero-copy on Jetson when ``nvv4l2decoder`` is present.

    Attributes:
        sensor_id: Sensor name; matches :attr:`SensorReaderConfig.sensor_id`
            in the :class:`~openral_core.RobotEnvironment`.
        is_open: ``True`` between :meth:`open` and :meth:`close`.
    """

    sensor_id: str
    is_open: bool

    def open(self) -> None:
        """Acquire the capture device and start any background workers.

        Idempotent: calling ``open`` on an already-open reader is a no-op.
        After this returns the backend may not yet have a frame
        available; :meth:`read_latest` will raise until one arrives.
        """
        ...

    def close(self) -> None:
        """Release the capture device and join any background workers.

        Idempotent. After ``close`` the reader can be re-opened.
        """
        ...

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the most recent buffered frame.

        Non-blocking. Returns whatever the backend has cached as ``latest``.

        Args:
            max_age_ms: Maximum acceptable frame age in milliseconds,
                measured from ``time.monotonic_ns()`` at capture time. When
                ``None`` the reader's configured default
                (:attr:`SensorReaderConfig.max_age_ms`, default 100 ms) is
                applied.

        Returns:
            A populated :class:`~openral_core.SensorFrame`.

        Raises:
            ROSPerceptionStale: When no frame has been captured yet, or the
                freshest frame is older than ``max_age_ms``.
            RuntimeError: When called on a closed reader.
        """
        ...
