"""Unit tests for ``SimTransport`` — in-memory ros2_control transport for HAL tests.

The transport is small (~120 LOC) but is used as the backbone of every
``RosControlHAL`` unit test in ``tests/unit/test_hal.py``.  Pinning its
public contract here lets future refactors of ``SimTransport`` fail loudly
instead of silently changing how the HAL suite behaves.

Coverage
--------
- ``__init__``                — zeroed state for ``n_joints``.
- ``state()``                 — returns a fresh dict with the three keys.
- ``publish()``               — records the call; applies the **last** trajectory step.
- Drop policy                  — non-list / empty trajectories don't crash; state untouched.
- ``call_count`` / ``last_call`` / ``calls``  — introspection helpers.
"""

from __future__ import annotations

import pytest
from openral_hal.sim_transport import SimTransport

# ── __init__ ─────────────────────────────────────────────────────────────────


def test_init_zeroes_state_for_n_joints() -> None:
    t = SimTransport(n_joints=4)
    s = t.state()
    assert s["position"] == [0.0, 0.0, 0.0, 0.0]
    assert s["velocity"] == [0.0, 0.0, 0.0, 0.0]
    assert s["effort"] == [0.0, 0.0, 0.0, 0.0]


def test_init_zero_joints_returns_empty_lists() -> None:
    t = SimTransport(n_joints=0)
    s = t.state()
    assert s == {"position": [], "velocity": [], "effort": []}


# ── state() ──────────────────────────────────────────────────────────────────


def test_state_returns_independent_copies() -> None:
    """Mutating the returned dict must not corrupt internal state."""
    t = SimTransport(n_joints=2)
    s = t.state()
    s["position"][0] = 9.0  # type: ignore[index]
    s["position"].append(99.0)  # type: ignore[union-attr]
    # Internal state is unchanged
    s2 = t.state()
    assert s2["position"] == [0.0, 0.0]


def test_state_keys_are_complete() -> None:
    t = SimTransport(n_joints=1)
    s = t.state()
    assert set(s.keys()) == {"position", "velocity", "effort"}


# ── publish() — happy path ───────────────────────────────────────────────────


def test_publish_applies_last_trajectory_step_to_position() -> None:
    """A 3-step trajectory: only the *final* waypoint becomes the new position."""
    t = SimTransport(n_joints=3)
    t.publish(
        "/ctrl/traj",
        {
            "joint_targets": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            "control_mode": "joint_position",
            "horizon": 3,
            "stamp_ns": 0,
        },
    )
    assert t.state()["position"] == [0.7, 0.8, 0.9]


def test_publish_with_single_step_applies_step() -> None:
    t = SimTransport(n_joints=2)
    t.publish(
        "/ctrl/x",
        {
            "joint_targets": [[1.0, -1.0]],
            "control_mode": "joint_position",
            "horizon": 1,
            "stamp_ns": 0,
        },
    )
    assert t.state()["position"] == [1.0, -1.0]


def test_publish_coerces_int_targets_to_float() -> None:
    """Targets supplied as ints (e.g. from JSON) are stored as floats."""
    t = SimTransport(n_joints=2)
    t.publish(
        "/ctrl/x",
        {
            "joint_targets": [[1, -1]],
            "control_mode": "joint_position",
            "horizon": 1,
            "stamp_ns": 0,
        },
    )
    pos = t.state()["position"]
    assert pos == [1.0, -1.0]
    assert all(isinstance(v, float) for v in pos)  # type: ignore[union-attr]


# ── publish() — drop / no-op cases ───────────────────────────────────────────


def test_publish_without_joint_targets_records_call_but_no_state_change() -> None:
    t = SimTransport(n_joints=2)
    t.publish("/ctrl/x", {"control_mode": "joint_position", "horizon": 1, "stamp_ns": 0})
    assert t.call_count == 1
    assert t.state()["position"] == [0.0, 0.0]


def test_publish_with_empty_trajectory_is_noop_on_state() -> None:
    t = SimTransport(n_joints=2)
    t.publish(
        "/ctrl/x",
        {"joint_targets": [], "control_mode": "joint_position", "horizon": 0, "stamp_ns": 0},
    )
    assert t.call_count == 1
    assert t.state()["position"] == [0.0, 0.0]


def test_publish_with_non_list_targets_is_noop_on_state() -> None:
    """Defensive: a malformed message must not raise — it just doesn't update state."""
    t = SimTransport(n_joints=2)
    t.publish(
        "/ctrl/x",
        {"joint_targets": "not_a_list", "control_mode": "joint_position", "horizon": 1},
    )
    assert t.state()["position"] == [0.0, 0.0]


def test_publish_with_non_list_step_is_noop_on_state() -> None:
    t = SimTransport(n_joints=2)
    t.publish(
        "/ctrl/x",
        {"joint_targets": ["not_a_step"], "control_mode": "joint_position", "horizon": 1},
    )
    assert t.state()["position"] == [0.0, 0.0]


# ── Introspection helpers ────────────────────────────────────────────────────


def test_call_count_starts_at_zero_and_increments() -> None:
    t = SimTransport(n_joints=1)
    assert t.call_count == 0
    t.publish("/a", {"joint_targets": [[0.5]], "control_mode": "joint_position", "horizon": 1})
    assert t.call_count == 1
    t.publish("/b", {"joint_targets": [[0.6]], "control_mode": "joint_position", "horizon": 1})
    assert t.call_count == 2


def test_last_call_is_none_before_any_publish() -> None:
    t = SimTransport(n_joints=1)
    assert t.last_call is None


def test_last_call_returns_most_recent_topic_and_msg() -> None:
    t = SimTransport(n_joints=1)
    t.publish("/first", {"joint_targets": [[0.1]], "horizon": 1})
    t.publish("/second", {"joint_targets": [[0.2]], "horizon": 1})
    last = t.last_call
    assert last is not None
    topic, msg = last
    assert topic == "/second"
    assert msg["joint_targets"] == [[0.2]]


def test_calls_returns_chronological_history() -> None:
    t = SimTransport(n_joints=1)
    t.publish("/a", {"joint_targets": [[0.0]], "horizon": 1})
    t.publish("/b", {"joint_targets": [[0.1]], "horizon": 1})
    t.publish("/c", {"joint_targets": [[0.2]], "horizon": 1})
    topics = [topic for topic, _ in t.calls]
    assert topics == ["/a", "/b", "/c"]


def test_calls_returns_independent_list() -> None:
    """Mutating the returned list must not corrupt internal history."""
    t = SimTransport(n_joints=1)
    t.publish("/a", {"joint_targets": [[0.0]], "horizon": 1})
    history = t.calls
    history.clear()
    assert t.call_count == 1


# ── Closed-loop trip: publish → state → publish ─────────────────────────────


def test_state_round_trip_through_multiple_publishes() -> None:
    t = SimTransport(n_joints=3)
    sequence = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]
    for target in sequence:
        t.publish(
            "/ctrl",
            {"joint_targets": [target], "control_mode": "joint_position", "horizon": 1},
        )
        assert t.state()["position"] == target
    assert t.call_count == len(sequence)


# ── Sanity: ill-shaped trajectory step does not raise ────────────────────────


@pytest.mark.parametrize(
    "msg",
    [
        {},  # missing key
        {"joint_targets": None},  # None
        {"joint_targets": [None]},  # step is None
        {"joint_targets": 42},  # int
    ],
)
def test_publish_does_not_raise_on_malformed_messages(msg: dict[str, object]) -> None:
    t = SimTransport(n_joints=2)
    t.publish("/x", msg)
    assert t.state()["position"] == [0.0, 0.0]
