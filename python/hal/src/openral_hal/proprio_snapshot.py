"""Thread-safe plain-data proprio snapshot (ADR-0049).

Decouples the HAL's control-critical publishers (``odom`` / ``joint_state``)
from its *sim* callback group so they publish at their configured rate without
head-of-line blocking behind ``env.step`` / camera render / scan-raycast on the
HAL's multi-threaded executor.

Contract (ADR-0049):

* A :class:`ProprioFrame` is **captured only from the sim callback group** —
  right after an ``env.step`` — where reading the HAL's proprio is safe (no
  concurrent step). Capture builds *plain data* (lists / tuples / a frozen
  :class:`~openral_core.schemas.JointState`); it never hands the simulator's
  live state to another thread.
* The frame is **stored and read** through this holder. The lock guards only
  the single reference swap / read — never an ``env.step`` / render / raycast —
  so a reader on the control thread is never blocked by the expensive sim work
  and always observes a *complete* frame (immutable, swapped atomically), never
  a torn one.

The holder is deliberately HAL-agnostic: it stores whatever the sim group
captured. The HAL-reading lives in the lifecycle node (the sim group), which is
where the "only the sim group touches the simulator" invariant is enforced.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from openral_core.schemas import JointState

# (x, y, z), (qx, qy, qz, qw)
Pose6Dof = tuple[tuple[float, float, float], tuple[float, float, float, float]]


@dataclass(frozen=True)
class ProprioFrame:
    """One coherent proprio sample — all fields read from the same ``env.step``.

    Plain, immutable data only (no live simulator handles), so it is safe to
    publish from a different executor thread than the one that stepped the sim.

    Attributes:
        state: Joint state (a frozen-ish Pydantic model of plain lists).
        base_pose: Planar base pose ``(x, y, yaw)`` — what Nav2 / SLAM consume.
        base_pose_6dof: Full base pose ``((x, y, z), (qx, qy, qz, qw))`` or
            ``None`` when the HAL does not expose it (fixed-base arms).
        base_twist: Base body twist ``(vx, vy, vz, wx, wy, wz)`` (REP-105 child
            frame), the latched commanded velocity Nav2's controller reads back.
        sim_time_ns: Cross-reset-monotonic elapsed simulation time in ns at this
            step (`SimAttachedHAL.sim_time_ns`, ADR-0048), or ``None`` for a
            clock-less / wall-clock HAL. Captured here so the ADR-0049 publisher
            thread can emit ``/clock`` without touching the simulator.
    """

    state: JointState
    base_pose: tuple[float, float, float]
    base_pose_6dof: Pose6Dof | None
    base_twist: tuple[float, ...]
    sim_time_ns: int | None = None


class ProprioSnapshot:
    """Lock-guarded holder for the latest :class:`ProprioFrame`.

    One writer (the sim callback group, after each step) calls :meth:`set`; any
    number of readers (the control callback group's publishers) call
    :meth:`latest`. Because :class:`ProprioFrame` is immutable and the reference
    is swapped under the lock, a reader never sees a partially-updated frame.

    Example:
        >>> snap = ProprioSnapshot()
        >>> snap.latest() is None
        True
        >>> frame = ProprioFrame(
        ...     state=JointState(name=["j1"], position=[0.0], stamp_ns=1),
        ...     base_pose=(1.0, 2.0, 0.5),
        ...     base_pose_6dof=None,
        ...     base_twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ... )
        >>> snap.set(frame)
        >>> snap.latest().base_pose
        (1.0, 2.0, 0.5)
    """

    def __init__(self) -> None:
        """Initialise an empty snapshot (no frame published until :meth:`set`)."""
        self._lock = threading.Lock()
        self._frame: ProprioFrame | None = None

    def set(self, frame: ProprioFrame) -> None:
        """Atomically publish ``frame`` as the latest sample (sim group only)."""
        with self._lock:
            self._frame = frame

    def latest(self) -> ProprioFrame | None:
        """Return the most recent frame, or ``None`` before the first capture."""
        with self._lock:
            return self._frame
