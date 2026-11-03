"""The tracker's own grasp event: a stalled jaw read robustly (issue #418, tracking/on, sim only).

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
manifest's numbers, like the release gap), the jaw is not moving (the last ``dwell_s`` has a
Theil-Sen drift within the allowance), the jaw got there by closing (some sample in the last
``approach_s`` sits further from closed than the median by the drift allowance), and both the gap
and the hold have lasted ``dwell_s`` (the hold: the run of samples within twice the allowance of
that median, chatter of that amplitude). Chatter resets nothing a median does not see; a closing
jaw (a large drift), a jaw at its command (no gap) or a jaw that has not started to close never
fires. No command band: the gap alone says "something is between the pads".

The hold starts where the jaw entered the band it sits in, not where a fixed window first stopped
seeing it close: a policy lifts within a fraction of a second of the close, and under the lift a
simulated jaw squeezing at its effort cap wobbles a few mrad (Spark restock bench 2026-10-31). A
0.6 s window that still held the closing samples dated the hold up to 36 cm into the lift and fired
at +4 to +37 cm (or never), and on the 6 cm cartons the late bind refused 7 of 9 grasps (hand
travelled, stale fit, pinch outside the box). Dated from the run, the hold sits at the grasp (the
carton within 5 mm of where it lay, 30 of 30 bench grasps) and the read fires at most 9 cm into
the lift.

The command is the approved chunk's last row, up to a chunk ahead of the row the jaw executes:
a fully open jaw under a chunk that closes at its end is short of that command and not moving
(2026-10-23, Spark banana R322: a derived ATTACH at jaw 0.9996, 9 s before the real grasp, bound
the banana where it lay). Only a jaw that closed and stopped is held. The gap must also have lasted
``dwell_s``, not just hold now: a jaw that stopped part-closed (a pre-shape) and then starts a real
close lags its command by more than the gap at once, while the run still reaches back into the
pre-shape (restock bench clean2_02 / clean3_08: a hold read 0.3-0.4 s before the real close).
Each jaw sample is read against the command in force at its own stamp (``command``'s
``stamp_ns``, the node's clock at receipt), not the newest one folded in: the approved chunk
carries no stamp, and a node that reads joint states late (an executor held by a fit) would
otherwise judge a backlog of still, part-closed samples against a newer closing command they
were never under. Offline, jaw samples 0.4 s behind their commands fired the 0.6 s-window read
1.5 s before the close (restock clean3_06), and this read on three bench grasps; a receipt stamp
errs late either way (a lagging jaw stream pairs with older commands, a lagging command stream
stamps them late), which only delays a hold.

Pure: no ROS, no clock, no numpy. Source stamps only (``JointState`` stamps, command stamps).
"""

from __future__ import annotations

from collections import deque
from statistics import median

from openral_core.tracking_thresholds import load_tracking_thresholds

__all__ = ["JawStallDetector"]

_NS = 1_000_000_000
_S = load_tracking_thresholds().stall
_FLOAT_SLACK = 1e-9


class JawStallDetector:
    """One gripper joint's stall read; ``update`` returns ``True`` once per hold.

    Args:
        closed: The jaw's closed end (``closure_calibration.closed_position``).
        gap: ``closed_rest_offset + stall_gap``: how much further from closed than the command a
            stalled jaw sits.
        settle: ``settle_tolerance``; the drift allowance is ``max(settle, gap / 2)``: the
            Theil-Sen drift allowed over ``window_s``; a held jaw's samples stay within twice it
            of their median.
        window_s: Source seconds the drift is scaled to: the jaw moves at most the allowance per
            ``window_s``. *Calibration point.*
        dwell_s: Source seconds the gap and the run must both last before it fires; the median
            reads the last ``dwell_s`` of samples. *Calibration point.*
        min_samples: Fewest distinct samples a window needs. *Calibration point.*
        approach_s: Source seconds back the closing motion into the hold may lie. *Calibration
            point.*

    Example:
        >>> d = JawStallDetector(0.0, 0.15, 0.02)
        >>> d.command(0.2, 0)
        >>> d.update(0.9, 0)
        False
        >>> fired = [d.update(0.45, int(i * 0.1 * 1e9)) for i in range(1, 15)]
        >>> fired.index(True) < 10, sum(fired)
        (True, 1)
        >>> still = JawStallDetector(0.0, 0.15, 0.02)  # open under a closing command: no hold
        >>> still.command(0.2, 0)
        >>> sum(still.update(0.9996, int(i * 0.1 * 1e9)) for i in range(30))
        0
    """

    def __init__(
        self,
        closed: float,
        gap: float,
        settle: float,
        *,
        window_s: float = _S.window_s,
        dwell_s: float = _S.dwell_s,
        min_samples: int = _S.min_samples,
        approach_s: float = _S.approach_s,
    ) -> None:
        """Read nothing yet."""
        self.closed, self.gap = closed, gap
        self.settle = max(settle, gap / 2.0)
        self.window_ns, self.dwell_ns = int(window_s * _NS), int(dwell_s * _NS)
        self.min_samples = min_samples
        self.approach_ns = int(approach_s * _NS)
        self._recent: deque[tuple[int, float]] = deque()  # approach_s of samples
        self._cmd: float | None = None  # the newest command (the latch reads it)
        self._cmds: deque[tuple[int, float]] = deque()  # (stamp_ns, target), stamps ascending
        self._samples: deque[tuple[int, float]] = deque()  # dwell_s of samples
        self._since_ns: int | None = None
        self._gap_since_ns: int | None = None
        self._latched = False

    @property
    def held_median(self) -> float:
        """The median measured jaw over the last ``dwell_s`` (the hold a fired stall reads)."""
        return float(median(p for _, p in self._samples))

    @property
    def hold_since_ns(self) -> int | None:
        """Source time the jaw entered the band it holds in, or ``None`` while no hold is read.

        The moment the jaw settled on whatever is between the pads: a late ATTACH (the HAL's
        trigger or this detector's dwell) still names the hand pose the object was grasped at.
        """
        return self._since_ns

    @property
    def latched(self) -> bool:
        """Whether a stall already fired (or a real ATTACH took over) for this hold."""
        return self._latched

    def command(self, target: float, stamp_ns: int) -> None:
        """Fold the commanded jaw in (the approved chunk's last row), in force from ``stamp_ns``.

        ``stamp_ns`` is the source time the command was received at; one stamped before the
        last is taken as in force from the last's stamp (never retroactively).
        """
        if self._cmd is not None and self._latched and self._opening(target):
            self._latched = False  # the command left the hold: the next close is a new hold
        self._cmd = target
        if self._cmds and stamp_ns < self._cmds[-1][0]:
            stamp_ns = self._cmds[-1][0]
        self._cmds.append((stamp_ns, target))

    def suppress(self) -> None:
        """A real ATTACH covers this hold: do not fire for it."""
        self._latched = True

    def reset(self) -> None:
        """Forget the window (a release, a sim reset, a source clock that went back)."""
        self._samples.clear()
        self._recent.clear()
        self._cmds.clear()
        self._since_ns = None
        self._gap_since_ns = None
        self._latched = False

    def _opening(self, target: float) -> bool:
        """Is ``target`` clearly further from closed than the jaw holding at the last median?"""
        if not self._samples or self._cmd is None:
            cmd = self._cmd if self._cmd is not None else target
            return abs(target - self.closed) > abs(cmd - self.closed) + self.gap
        held = median(p for _, p in self._samples)
        return abs(target - self.closed) >= abs(held - self.closed) - self.gap / 2.0

    def _drift(self, run: list[tuple[int, float]]) -> float:
        """How far ``run``'s jaw moves per ``window_s``: |Theil-Sen slope| x the window."""
        min_dt = _S.min_dt_window_fraction * self.window_ns
        slopes = [
            (run[j][1] - run[i][1]) / ((run[j][0] - run[i][0]) / _NS)
            for i in range(len(run))
            for j in range(i + 1, len(run))
            if run[j][0] - run[i][0] >= min_dt
        ]
        return abs(median(slopes)) * self.window_ns / _NS if slopes else 0.0

    def _run(self, held: float) -> list[tuple[int, float]]:
        """The newest samples, oldest first, back to the first one off the band around ``held``."""
        run: list[tuple[int, float]] = []
        for sample in reversed(self._recent):
            if abs(sample[1] - held) > 2.0 * self.settle + _FLOAT_SLACK:  # at the band is in it
                break
            run.append(sample)
        run.reverse()
        return run

    def _command_at(self, stamp_ns: int) -> float | None:
        """The command in force at ``stamp_ns``: the newest one stamped at or before it."""
        while len(self._cmds) > 1 and self._cmds[1][0] <= stamp_ns - self.approach_ns:
            self._cmds.popleft()  # superseded long before any sample still read
        for t, target in reversed(self._cmds):
            if t <= stamp_ns:
                return target
        return None

    def update(self, position: float, stamp_ns: int) -> bool:
        """Fold one measured jaw sample in; ``True`` exactly once, when a hold is read."""
        if self._recent and stamp_ns < self._recent[-1][0]:
            self.reset()  # the source clock went back
        if self._recent and stamp_ns == self._recent[-1][0]:
            return False  # a repeated joint state
        self._samples.append((stamp_ns, position))
        while self._samples[0][0] < stamp_ns - self.dwell_ns:
            self._samples.popleft()
        self._recent.append((stamp_ns, position))
        while self._recent[0][0] < stamp_ns - self.approach_ns:
            self._recent.popleft()
        cmd = self._command_at(stamp_ns)
        if cmd is None or self._latched:
            return False
        held = median(p for _, p in self._samples)
        if abs(held - self.closed) - abs(cmd - self.closed) < self.gap:
            self._since_ns = self._gap_since_ns = None
            return False
        if self._gap_since_ns is None:
            self._gap_since_ns = stamp_ns
        run = self._run(held)
        closed_in = max(abs(p - self.closed) for _, p in self._recent) - abs(held - self.closed)
        drift = self._drift(list(self._samples))
        if len(run) < self.min_samples or closed_in < self.settle or drift > self.settle:
            self._since_ns = None
            return False
        self._since_ns = run[0][0]
        if stamp_ns - max(self._since_ns, self._gap_since_ns) >= self.dwell_ns:
            self._latched = True
            return True
        return False
