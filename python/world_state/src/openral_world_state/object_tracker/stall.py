"""The tracker's own grasp event: a stalled jaw read robustly (issue #418, bind/on, sim only).

The HAL's position-stall trigger (``openral_hal._grasp_trigger``) gives ATTACH only for a command
within ``close_band`` of closed and a jaw whose span over a short window is within
``settle_tolerance``. Two real traces missed a hold that way (2026-10-17):

* DROID r7dr_on1: a banana carried 3 s with no ATTACH -- the jaw held 0.43 with the command near
  0.19, chattering (spans 0.015-0.032 per 0.4 s, 0.06 over the settle window) and the policy's
  command drifting past the band;
* the SO-101 PickOrange runs: the opposite case, a jaw that tracks its command (0.2 -> 0.2, 0.38 ->
  0.38) is closed on nothing and must never bind.

This detector asks the same physical question with a robust read: the commanded jaw is further
from closed than the *median* measured jaw by at least ``closed_rest_offset + stall_gap`` (the
manifest's numbers, like the release gap), the median is not moving (the medians of the window's
two halves agree within ``settle_tolerance``), and both hold for ``dwell_s``. Chatter resets nothing
a median does not see; a closing jaw (a large half-to-half drift) or a jaw at its command (no gap)
never fires. No command band: the gap alone says "something is between the pads".

Pure: no ROS, no clock, no numpy. Source stamps only (``JointState`` stamps, command stamps).
"""

from __future__ import annotations

from collections import deque
from statistics import median

__all__ = ["JawStallDetector"]

_NS = 1_000_000_000


class JawStallDetector:
    """One gripper joint's stall read; ``update`` returns ``True`` once per hold.

    Args:
        closed: The jaw's closed end (``closure_calibration.closed_position``).
        gap: ``closed_rest_offset + stall_gap``: how much further from closed than the command a
            stalled jaw sits.
        settle: ``settle_tolerance``; the Theil-Sen drift allowed is ``max(settle, gap / 2)``.
        window_s: Source seconds of samples the median reads. *Calibration point.*
        dwell_s: Source seconds the stall must hold before it fires. *Calibration point.*
        min_samples: Fewest distinct samples a window needs. *Calibration point.*

    Example:
        >>> d = JawStallDetector(0.0, 0.15, 0.02)
        >>> d.command(0.2, 0)
        >>> fired = [d.update(0.45, int(i * 0.1 * 1e9)) for i in range(14)]
        >>> fired.index(True) < 10, sum(fired)
        (True, 1)
    """

    def __init__(
        self,
        closed: float,
        gap: float,
        settle: float,
        *,
        window_s: float = 0.6,
        dwell_s: float = 0.3,
        min_samples: int = 5,
    ) -> None:
        """Read nothing yet."""
        self.closed, self.gap = closed, gap
        self.settle = max(settle, gap / 2.0)
        self.window_ns, self.dwell_ns = int(window_s * _NS), int(dwell_s * _NS)
        self.min_samples = min_samples
        self._cmd: float | None = None
        self._samples: deque[tuple[int, float]] = deque()
        self._since_ns: int | None = None
        self._latched = False

    @property
    def held_median(self) -> float:
        """The median measured jaw over the window (the hold a fired stall reads)."""
        return float(median(p for _, p in self._samples))

    @property
    def hold_since_ns(self) -> int | None:
        """Source time the stall condition first held for the current hold, or ``None``.

        The moment the jaw settled on whatever is between the pads: a late ATTACH (the HAL's
        trigger or this detector's dwell) still names the hand pose the object was grasped at.
        """
        return self._since_ns

    @property
    def latched(self) -> bool:
        """Whether a stall already fired (or a real ATTACH took over) for this hold."""
        return self._latched

    def command(self, target: float, stamp_ns: int) -> None:
        """Fold the commanded jaw in (the approved chunk's last row)."""
        del stamp_ns
        if self._cmd is not None and self._latched and self._opening(target):
            self._latched = False  # the command left the hold: the next close is a new hold
        self._cmd = target

    def suppress(self) -> None:
        """A real ATTACH covers this hold: do not fire for it."""
        self._latched = True

    def reset(self) -> None:
        """Forget the window (a release, a sim reset, a source clock that went back)."""
        self._samples.clear()
        self._since_ns = None
        self._latched = False

    def _opening(self, target: float) -> bool:
        """Is ``target`` clearly further from closed than the jaw holding at the last median?"""
        if not self._samples or self._cmd is None:
            cmd = self._cmd if self._cmd is not None else target
            return abs(target - self.closed) > abs(cmd - self.closed) + self.gap
        held = median(p for _, p in self._samples)
        return abs(target - self.closed) >= abs(held - self.closed) - self.gap / 2.0

    def _drift(self) -> float:
        """How far the window's jaw moved, robustly: |Theil-Sen slope| x the window."""
        pts = list(self._samples)
        min_dt = 0.2 * self.window_ns
        slopes = [
            (pts[j][1] - pts[i][1]) / ((pts[j][0] - pts[i][0]) / _NS)
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
            if pts[j][0] - pts[i][0] >= min_dt
        ]
        return abs(median(slopes)) * self.window_ns / _NS if slopes else 0.0

    def update(self, position: float, stamp_ns: int) -> bool:  # noqa: PLR0911  # reason: each gate
        """Fold one measured jaw sample in; ``True`` exactly once, when a hold is read."""
        if self._samples and stamp_ns < self._samples[-1][0]:
            self.reset()  # the source clock went back
        if self._samples and stamp_ns == self._samples[-1][0]:
            return False  # a repeated joint state
        self._samples.append((stamp_ns, position))
        while self._samples[0][0] < stamp_ns - self.window_ns:
            self._samples.popleft()
        if self._cmd is None or self._latched:
            return False
        if len(self._samples) < self.min_samples:
            return False
        if self._samples[-1][0] - self._samples[0][0] < 0.5 * self.window_ns:
            return False
        vals = [p for _, p in self._samples]
        short = abs(median(vals) - self.closed) - abs(self._cmd - self.closed)
        drift = self._drift()
        if short < self.gap or drift > self.settle:
            self._since_ns = None
            return False
        if self._since_ns is None:
            self._since_ns = stamp_ns
        if stamp_ns - self._since_ns >= self.dwell_ns:
            self._latched = True
            return True
        return False
