"""Which gripper command a safety-approved chunk carries (issue #418, tracking/on, sim only).

The tracker's own stalled-jaw read (``stall.JawStallDetector``) needs the commanded jaw. The
approved chunk carries it three ways, and the node read only the first:

* a full-dof row, the gripper at its manifest index (DROID, 8 dof; SO-101, 6 dof): positional;
* a compact row with ``joint_names``;
* an ADR-0102 slot group (``tick_group_size > 1``, the OpenArm's four slots per tick): the arm
  slots are joint rows, a gripper slot is one value (``n_dof == 1``, no names) that names its
  gripper by ``ee_name`` (``left_gripper`` is the gripper joint's own name).

Ignoring every slot group meant the restock tracker never held a command, so the stalled-jaw
read never fired in any restock run (r12rs02_on1/on2/camlong_b: ATTACH came from the HAL's
trigger 0.3 to 0.7 s after the carton had started to lift).

Pure: no ROS.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence

__all__ = ["SourceClock", "jaw_commands"]


def jaw_commands(
    joints: Collection[str],
    joint_order: Sequence[str],
    *,
    n_dof: int,
    flat: Sequence[float],
    joint_names: Sequence[str],
    tick_group_size: int,
    ee_name: str,
) -> dict[str, float]:
    """The gripper targets of one approved chunk, by gripper joint name.

    Args:
        joints: The gripper joints the tracker models.
        joint_order: The manifest's joint order (a full-dof row's layout).
        n_dof: The chunk's row width.
        flat: The chunk's rows, row-major; the last row is the applied command.
        joint_names: The chunk's joint names (empty for a full-dof or gripper-slot row).
        tick_group_size: ``> 1`` for one slot of an ADR-0102 group.
        ee_name: The end-effector a gripper slot names.

    Returns:
        ``{joint: target}`` for each modelled gripper the chunk commands, else ``{}``.

    Example:
        >>> jaw_commands(
        ...     {"left_gripper"},
        ...     ["a", "left_gripper"],
        ...     n_dof=1,
        ...     flat=[0.4],
        ...     joint_names=[],
        ...     tick_group_size=4,
        ...     ee_name="left_gripper",
        ... )
        {'left_gripper': 0.4}
        >>> jaw_commands(
        ...     {"left_gripper"},
        ...     ["a", "left_gripper"],
        ...     n_dof=2,
        ...     flat=[0.1, 0.2],
        ...     joint_names=[],
        ...     tick_group_size=1,
        ...     ee_name="",
        ... )
        {'left_gripper': 0.2}
    """
    if n_dof <= 0 or len(flat) < n_dof:
        return {}
    row, names = list(flat[-n_dof:]), list(joint_names)
    if tick_group_size > 1:  # a slot of a group: only a one-value gripper slot is a jaw command
        return {ee_name: float(row[0])} if n_dof == 1 and ee_name in joints else {}
    out: dict[str, float] = {}
    for joint in joints:
        if names:
            index = names.index(joint) if joint in names and len(names) == n_dof else None
        elif n_dof == len(joint_order) and joint in joint_order:
            index = list(joint_order).index(joint)
        else:
            index = None
        if index is not None:
            out[joint] = float(row[index])
    return out


class SourceClock:
    """The joint states' source time at a moment on the node's clock, for stamping a command.

    The approved chunk carries no stamp, and the node's clock need not share the joint states'
    time base (a bag replayed with ``--clock`` runs on its recorded wall times; the joint stamps
    are sim time). Each joint state gives ``clock - stamp``: the transport offset, plus the time
    it sat in the executor's queue. The smallest offset over the last ``window_s`` of clock is
    the transport alone, so ``source_at(clock)`` is the source time *now*, never earlier: a
    backlog of joint states only raises the offsets, it cannot make a command look older than
    the samples queued behind it. The estimate is held to at most ``max_lead_s`` past the newest
    stamp seen: a node clock that runs faster than the source (a wall-clock replay of a sim
    running below real time: offsets grow, the window's minimum is stale, and a restock replay
    dated commands 7 s ahead of every jaw sample, so none was ever in force) must cost at most
    that much delay, never the read.

    Example:
        >>> c = SourceClock()
        >>> c.source_at(5_000) is None  # no joint state yet
        True
        >>> c.observe(clock_ns=1_000_000_100, stamp_ns=100)
        >>> c.observe(clock_ns=1_000_000_900, stamp_ns=150)  # read 750 ns late
        >>> c.source_at(1_000_001_000)
        1000
    """

    def __init__(self, window_s: float = 10.0, max_lead_s: float = 0.5) -> None:
        """Hold no offset yet."""
        self._window_ns = int(window_s * 1e9)
        self._max_lead_ns = int(max_lead_s * 1e9)
        # (clock_ns, clock_ns - stamp_ns), offsets strictly increasing front to back.
        self._offsets: deque[tuple[int, int]] = deque()
        self._newest_ns = 0

    def observe(self, *, clock_ns: int, stamp_ns: int) -> None:
        """Fold in one joint state: its stamp and the node's clock when it was handled."""
        if self._offsets and clock_ns < self._offsets[-1][0]:
            self._offsets.clear()  # the node's clock went back (a sim reset)
            self._newest_ns = 0
        offset = clock_ns - stamp_ns
        # A monotonic min-deque: an offset no smaller than a newer one can never be the
        # window's minimum again, so the front is always the minimum (O(1) amortised at the
        # joint states' 750 Hz, where a scan of the window was ~7,500 entries per read).
        while self._offsets and self._offsets[-1][1] >= offset:
            self._offsets.pop()
        self._offsets.append((clock_ns, offset))
        self._newest_ns = max(self._newest_ns, stamp_ns)
        while self._offsets[0][0] < clock_ns - self._window_ns:
            self._offsets.popleft()

    def source_at(self, clock_ns: int) -> int | None:
        """The joint states' source time at ``clock_ns``, or ``None`` before any joint state."""
        if not self._offsets:
            return None
        estimate = clock_ns - self._offsets[0][1]
        return min(estimate, self._newest_ns + self._max_lead_ns)
