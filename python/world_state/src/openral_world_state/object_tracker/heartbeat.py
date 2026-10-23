"""Keep the tracker's streams alive while its executor is busy (issue #418, WP C).

The tracker node runs its fits, cover tests and segmenter callbacks in one executor thread.
A long callback (a 960 x 540 camera frame, a fit that grows a track) delays the 2 Hz
heartbeat timer behind it, and the kernel judges the tracked-object stream by the age of
``/openral/attachment_state`` against a sim-clock deadline: gaps of 2 s wall in
``r10rs02_camlong`` made the stream stale for the stop (``tracked_grasp_eligible n=0 of=0``).

``StarvationFill`` is the ROS-free core of the fix: the node remembers the last message it
published on each stream, and a second timer on its own callback group republishes that
message when the main heartbeat is late. It never invents state, and it never papers over a
hung tracker: it republishes only while the main thread has shown activity within
``silent_s`` and, for the attachment stream, joint states within ``silent_s``. Past that the
streams go quiet and the kernel's own freshness gate trips, as before.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

M = TypeVar("M")


class StarvationFill(Generic[M]):
    """Cache of the last published tracked / attachment messages and the due-ness rule.

    Example:
        >>> f = StarvationFill[str](gap_s=0.4, silent_s=5.0)
        >>> f.touch(0.0)
        >>> f.joints(0.0)
        >>> f.published(0.0, tracked="set", attach="att")
        >>> f.due(0.2)
        (None, None)
        >>> f.due(0.5)
        ('set', 'att')
        >>> f.due(6.0)  # the main thread has been silent for 6 s: say nothing
        (None, None)
    """

    def __init__(self, *, gap_s: float = 0.4, silent_s: float = 5.0) -> None:
        """Republish after ``gap_s`` of main-thread silence; give up after ``silent_s``."""
        if not (0.0 < gap_s < silent_s):
            raise ValueError("StarvationFill needs 0 < gap_s < silent_s.")
        self._gap, self._silent = gap_s, silent_s
        self._lock = threading.Lock()
        self._tracked: M | None = None
        self._attach: M | None = None
        self._t_tracked = self._t_attach = self._t_touch = self._t_joints = float("-inf")

    def touch(self, now_s: float) -> None:
        """The main thread entered or left a callback at ``now_s`` (monotonic)."""
        with self._lock:
            self._t_touch = now_s

    def joints(self, now_s: float) -> None:
        """A joint state arrived at ``now_s``: the arm is being watched."""
        with self._lock:
            self._t_joints = now_s

    def published(self, now_s: float, *, tracked: M | None = None, attach: M | None = None) -> None:
        """The main thread published (and so remembers) these messages at ``now_s``."""
        with self._lock:
            if tracked is not None:
                self._tracked, self._t_tracked = tracked, now_s
            if attach is not None:
                self._attach, self._t_attach = attach, now_s

    def due(self, now_s: float) -> tuple[M | None, M | None]:
        """The cached (tracked, attachment) messages to republish now, or ``None`` each."""
        with self._lock:
            if now_s - self._t_touch > self._silent:
                return None, None
            tracked = self._tracked if now_s - self._t_tracked >= self._gap else None
            attach_live = now_s - self._t_joints <= self._silent
            attach = self._attach if attach_live and now_s - self._t_attach >= self._gap else None
            if tracked is not None:
                self._t_tracked = now_s
            if attach is not None:
                self._t_attach = now_s
            return tracked, attach


__all__ = ["StarvationFill"]
