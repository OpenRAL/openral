"""Unit tests for the reward-monitor rolling frame buffer (ADR-0057).

Pure-Python, no ROS / torch / GPU — exercises the node-side windowing the
stateless Robometer sidecar relies on.

Run with:
    uv run pytest tests/unit/test_reward_frame_source.py -v
"""

from __future__ import annotations

import pytest
from openral_runner.backends.reward.frame_source import Frame, RollingFrameBuffer, trend

_NS = 1_000_000_000


def _frame(stamp_s: float) -> Frame:
    return Frame(stamp_ns=int(stamp_s * _NS), bgr=b"\x00\x00\x00", width=1, height=1)


def test_window_retains_only_recent_frames() -> None:
    """Frames older than window_s relative to the newest are evicted."""
    buf = RollingFrameBuffer(window_s=2.0)
    for s in range(4):  # stamps 0,1,2,3 s
        buf.push(_frame(float(s)))
    # newest = 3 s; window 2 s keeps stamps >= 1 s -> 1,2,3
    win = buf.window(2.0)
    assert len(win) == 3
    assert [f.stamp_ns for f in win] == [1 * _NS, 2 * _NS, 3 * _NS]


def test_push_evicts_beyond_window() -> None:
    """The buffer itself never holds frames older than its window."""
    buf = RollingFrameBuffer(window_s=1.5)
    for s in range(6):
        buf.push(_frame(float(s)))
    # newest = 5 s; horizon = 3.5 s -> frames at 4, 5 retained
    assert len(buf) == 2


def test_window_capped_to_buffer_window() -> None:
    """Requesting a window longer than the buffer's retention returns at most all."""
    buf = RollingFrameBuffer(window_s=2.0)
    for s in range(4):
        buf.push(_frame(float(s)))
    # asking for 30 s can only return what the 2 s buffer kept (3 frames)
    assert len(buf.window(30.0)) == 3


def test_max_frames_cap() -> None:
    """max_frames bounds memory even within the time window."""
    buf = RollingFrameBuffer(window_s=100.0, max_frames=3)
    for s in range(10):
        buf.push(_frame(float(s)))
    assert len(buf) == 3


def test_is_stale() -> None:
    """is_stale flips once no fresh frame arrives within stale_after_s."""
    buf = RollingFrameBuffer(window_s=10.0, stale_after_s=2.0)
    assert buf.is_stale(now_ns=0)  # empty -> stale
    buf.push(_frame(5.0))
    assert not buf.is_stale(now_ns=int(6.0 * _NS))  # 1 s old
    assert buf.is_stale(now_ns=int(8.0 * _NS))  # 3 s old > 2 s


def test_empty_window_is_empty_list() -> None:
    buf = RollingFrameBuffer(window_s=2.0)
    assert buf.window(2.0) == []


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window_s must be > 0"):
        RollingFrameBuffer(window_s=0.0)


def test_trend_rising_flat_falling() -> None:
    """trend() returns a positive/zero/negative slope per sample."""
    assert trend([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.25)
    assert trend([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert trend([1.0, 0.5, 0.0]) == pytest.approx(-0.5)
    assert trend([0.7]) == 0.0  # < 2 points
    assert trend([]) == 0.0
