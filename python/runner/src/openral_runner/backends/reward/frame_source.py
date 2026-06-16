"""Transport-agnostic rolling frame buffer for the reward monitor (ADR-0057).

The reward sidecar is stateless — it scores a clip on demand. State lives here,
node-side: a time-indexed ring of recent frames fed by whatever publishes the
co-active VLA's camera (the GStreamer perception tee on real hardware, the sim
HAL camera publisher in ``deploy-sim``). Pure Python + stdlib only, so it unit-
tests without ROS, torch, or a GPU.

Frames are stored as opaque BGR byte payloads with their ``(width, height)`` so
the buffer never depends on numpy; conversion to the model's RGB tensor happens
in the sidecar.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Frame:
    """One buffered camera frame.

    Attributes:
        stamp_ns: Capture time in integer nanoseconds (monotonic within a run).
        bgr: Raw BGR888 bytes, length ``width * height * 3``.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    stamp_ns: int
    bgr: bytes
    width: int
    height: int


class RollingFrameBuffer:
    """Keep the most recent ``window_s`` seconds of frames.

    Eviction is relative to the **newest** frame seen (not wall clock), so the
    buffer behaves identically against a sim clock and a real clock. A
    ``max_frames`` cap bounds memory when frames arrive faster than expected.

    Args:
        window_s: Retention horizon in seconds. Must be > 0.
        max_frames: Hard cap on buffered frames (oldest dropped first).
        stale_after_s: A query is ``stale`` if the newest frame is older than
            this, relative to the query's ``now_ns``.

    Example:
        >>> buf = RollingFrameBuffer(window_s=2.0)
        >>> for i in range(4):
        ...     buf.push(Frame(stamp_ns=i * 1_000_000_000, bgr=b"x", width=1, height=1))
        >>> # newest stamp = 3s; window 2s keeps stamps >= 1s -> frames at 1,2,3
        >>> len(buf.window(2.0))
        3
    """

    def __init__(
        self,
        *,
        window_s: float,
        max_frames: int = 256,
        stale_after_s: float = 3.0,
    ) -> None:
        """Build an empty buffer retaining ``window_s`` seconds / ``max_frames`` frames."""
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}")
        self._window_s = window_s
        self._max_frames = max_frames
        self._stale_after_ns = int(stale_after_s * _NS_PER_S)
        self._frames: collections.deque[Frame] = collections.deque()

    def push(self, frame: Frame) -> None:
        """Append a frame and evict anything older than the window / over cap."""
        self._frames.append(frame)
        horizon = frame.stamp_ns - int(self._window_s * _NS_PER_S)
        while self._frames and self._frames[0].stamp_ns < horizon:
            self._frames.popleft()
        while len(self._frames) > self._max_frames:
            self._frames.popleft()

    def window(self, seconds: float) -> list[Frame]:
        """Return frames within the last ``seconds`` relative to the newest frame.

        Capped to the buffer's retention window. Returns ``[]`` if empty.
        """
        if not self._frames:
            return []
        span_ns = int(min(seconds, self._window_s) * _NS_PER_S)
        horizon = self._frames[-1].stamp_ns - span_ns
        return [f for f in self._frames if f.stamp_ns >= horizon]

    def is_stale(self, now_ns: int) -> bool:
        """True if no fresh frame arrived within ``stale_after_s`` of ``now_ns``."""
        if not self._frames:
            return True
        return (now_ns - self._frames[-1].stamp_ns) > self._stale_after_ns

    def __len__(self) -> int:
        """Number of frames currently buffered."""
        return len(self._frames)


# A least-squares slope needs at least two points; fewer reads as "no trend".
_MIN_POINTS_FOR_SLOPE = 2


def trend(series: list[float]) -> float:
    """Least-squares slope per sample of ``series`` (0.0 for < 2 points).

    Used to report whether progress/success is rising, flat (``stalled``), or
    falling over the queried window — no numpy dependency.
    """
    n = len(series)
    if n < _MIN_POINTS_FOR_SLOPE:
        return 0.0
    xs = range(n)
    mean_x = (n - 1) / 2.0
    mean_y = sum(series) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0
