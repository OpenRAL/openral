"""Unit tests for the ADR-0049 proprio snapshot holder.

These exercise the thread-safety contract that lets the HAL publish odom /
joint_state from a different executor thread than the one stepping the sim:
the holder must hand readers a *complete* frame (never torn) and must never
require touching the simulator from the reader thread (it stores plain data).

No mocks — :class:`ProprioFrame` carries real ``JointState`` Pydantic models and
plain tuples (CLAUDE.md §1.11).
"""

from __future__ import annotations

import threading

from openral_core.schemas import JointState
from openral_hal.proprio_snapshot import ProprioFrame, ProprioSnapshot


def _frame(i: int) -> ProprioFrame:
    """A coherent frame whose fields all encode the same index ``i``."""
    return ProprioFrame(
        state=JointState(name=["base", "j1"], position=[float(i), float(i)], stamp_ns=i),
        base_pose=(float(i), float(i), 0.0),
        base_pose_6dof=((float(i), float(i), 0.7), (0.0, 0.0, 0.0, 1.0)),
        base_twist=(float(i), 0.0, 0.0, 0.0, 0.0, 0.0),
        sim_time_ns=i,
    )


def test_empty_before_first_set() -> None:
    assert ProprioSnapshot().latest() is None


def test_set_then_latest_returns_same_frame() -> None:
    snap = ProprioSnapshot()
    f = _frame(7)
    snap.set(f)
    got = snap.latest()
    assert got is f
    assert got.base_pose == (7.0, 7.0, 0.0)
    assert got.base_pose_6dof == ((7.0, 7.0, 0.7), (0.0, 0.0, 0.0, 1.0))
    assert got.state.stamp_ns == 7
    assert got.sim_time_ns == 7  # ADR-0048 Phase 2 — carried for the /clock publisher


def test_latest_reflects_most_recent_set() -> None:
    snap = ProprioSnapshot()
    snap.set(_frame(1))
    snap.set(_frame(2))
    assert snap.latest().state.stamp_ns == 2


def test_concurrent_reads_never_see_a_torn_frame() -> None:
    """A writer thread swaps frames while readers spin; every observed frame
    must be internally consistent (all fields encode the same index) — proving
    the reference swap is atomic and readers never block on / corrupt the sim.
    """
    snap = ProprioSnapshot()
    snap.set(_frame(0))
    stop = threading.Event()
    torn: list[ProprioFrame] = []
    n_iters = 20_000

    def writer() -> None:
        for i in range(n_iters):
            snap.set(_frame(i))
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            f = snap.latest()
            assert f is not None
            # All fields derive from the same index i -> they must agree.
            i = f.state.stamp_ns
            if not (
                f.base_pose == (float(i), float(i), 0.0)
                and f.base_twist[0] == float(i)
                and f.state.position == [float(i), float(i)]
            ):
                torn.append(f)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not torn, f"observed {len(torn)} torn frames"
    assert snap.latest().state.stamp_ns == n_iters - 1
