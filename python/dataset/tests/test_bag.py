"""End-to-end tests for :class:`openral_dataset.Rosbag2Sink` (ADR-0019 PR3).

Per CLAUDE.md §1.11 — uses a real :class:`mcap.writer.Writer` against a
``tmp_path`` and re-reads with a real :func:`mcap.reader.make_reader`. No
mocks. The bag format is identical to what ``rosbag2`` with the mcap
backend writes; tests that need ``rosbag2_py`` (full ROS 2 integration)
live in PR3's HIL gate and skip cleanly without rclpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_dataset import RolloutRecorder, Rosbag2Sink
from openral_dataset.bag import (
    PHASE_END,
    PHASE_START,
    TOPIC_EPISODE,
    TOPIC_TICK,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_bag(bag_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Return [(topic, decoded_json_message), ...] in iteration order."""
    from mcap.reader import make_reader

    out: list[tuple[str, dict[str, object]]] = []
    with bag_path.open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            decoded = json.loads(message.data.decode("utf-8"))
            out.append((channel.topic, decoded))
    return out


def _zero_frame(robot: RobotDescription) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Build (state, images, action) of the shapes the SO-100 expects."""
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images = {
        "camera1": np.zeros((16, 16, 3), dtype=np.uint8),
        "camera2": np.zeros((16, 16, 3), dtype=np.uint8),
    }
    return state, images, action


def test_bag_tick_carries_active_span_trace_id(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """ISSUE-109: the /openral/tick record carries the producing tick's OTel ids.

    The Rosbag2Sink defers the actual mcap write to a worker thread, so
    the (trace_id, span_id) must ride on the ``DatasetFrame`` captured in
    the ``rskill.tick`` span — not be re-read off-thread (where the
    context is gone). Asserts the round-tripped tick carries the
    non-empty 32-hex / 16-hex ids.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")

    bag_path = tmp_path / "traced.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    rec.episode_start()
    with tracer.start_as_current_span("rskill.tick"):
        ctx = trace.get_current_span().get_span_context()
        exp_trace = f"{ctx.trace_id:032x}"
        exp_span = f"{ctx.span_id:016x}"
        rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    ticks = [msg for topic, msg in _read_bag(bag_path) if topic == TOPIC_TICK]
    assert len(ticks) == 1
    assert ticks[0]["trace_id"] == exp_trace
    assert ticks[0]["span_id"] == exp_span


# ── Construction-time validation (no mcap I/O) ───────────────────────────────


def test_bag_rejects_preexisting_file(tmp_path: Path) -> None:
    """Rosbag2Sink refuses to overwrite an existing file."""
    existing = tmp_path / "existing.mcap"
    existing.write_bytes(b"")
    with pytest.raises(ROSConfigError, match=r"already exists"):
        Rosbag2Sink(bag_path=existing)


def test_bag_rejects_missing_parent_dir(tmp_path: Path) -> None:
    """Rosbag2Sink refuses to write into a non-existent parent directory."""
    bad_path = tmp_path / "doesnt_exist" / "x.mcap"
    with pytest.raises(ROSConfigError, match=r"parent directory"):
        Rosbag2Sink(bag_path=bad_path)


def test_bag_rejects_unknown_compression(tmp_path: Path) -> None:
    """Unknown mcap compression strings are caught at first open_episode call."""
    bag_path = tmp_path / "x.mcap"
    sink = Rosbag2Sink(bag_path=bag_path, compression="bogus")
    # Compression is checked when the writer actually opens — fire an
    # episode_start that forces the open_writer path to execute.
    from openral_dataset.recorder import EpisodeHeader

    header = EpisodeHeader(episode_idx=0, task_string="t", fps=30.0, robot_name="x", stamp_ns=1)
    with pytest.raises(ROSConfigError, match=r"unknown mcap compression"):
        sink.open_episode(header)


# ── End-to-end round-trips ───────────────────────────────────────────────────


def test_bag_round_trip_one_episode(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Write one 2-frame episode and round-trip every message back via mcap reader."""
    bag_path = tmp_path / "single.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="pick the cube", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    rec.episode_start()
    rec.record_frame(observation_state=state, images=images, action=action)
    rec.record_frame(observation_state=state, images=images, action=action, reward=0.7)
    rec.episode_end(success=True)
    rec.finalize()

    # Sink counters reflect what was actually written.
    assert sink.n_ticks_written == 2
    assert sink.n_episode_markers_written == 2  # PHASE_START + PHASE_END
    assert sink.n_dropped == 0
    assert bag_path.is_file()

    messages = _read_bag(bag_path)
    # Order: episode_start → tick → tick → episode_end. mcap iter_messages
    # iterates by log_time so all four should be in order.
    assert [topic for topic, _ in messages] == [
        TOPIC_EPISODE,
        TOPIC_TICK,
        TOPIC_TICK,
        TOPIC_EPISODE,
    ]
    start_msg = messages[0][1]
    end_msg = messages[3][1]
    assert start_msg["phase"] == PHASE_START
    assert start_msg["task_string"] == "pick the cube"
    assert end_msg["phase"] == PHASE_END
    assert end_msg["success"] is True
    # Per-tick rewards survive the round-trip.
    assert messages[1][1]["reward"] == 0.0
    assert messages[2][1]["reward"] == pytest.approx(0.7)


def test_bag_round_trip_multiple_episodes(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Two episodes with different success outcomes — counters and markers align."""
    bag_path = tmp_path / "multi.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    for success in (True, False):
        rec.episode_start()
        rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success)
    rec.finalize()

    messages = _read_bag(bag_path)
    episode_markers = [m for t, m in messages if t == TOPIC_EPISODE]
    # 2 episodes x 2 markers (start + end) = 4 episode markers.
    assert len(episode_markers) == 4
    # End-phase markers should carry the success flag we passed in.
    end_markers = [m for m in episode_markers if m["phase"] == PHASE_END]
    assert [m["success"] for m in end_markers] == [True, False]
    # Episode indices on END markers should monotonically increase from 0.
    assert [m["episode_idx"] for m in end_markers] == [0, 1]


def test_bag_finalize_is_idempotent(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Calling finalize() twice (or before any frames) is safe."""
    bag_path = tmp_path / "idempotent.mcap"
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    rec.episode_start()
    rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()
    # Calling finalize() on the sink a second time is a no-op.
    sink.finalize()
    sink.finalize()
    assert sink.n_ticks_written == 1


def test_bag_uncompressed_round_trip(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """compression=None produces an uncompressed bag that still round-trips."""
    bag_path = tmp_path / "raw.mcap"
    sink = Rosbag2Sink(bag_path=bag_path, compression=None)
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    rec.episode_start()
    rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    messages = _read_bag(bag_path)
    assert len(messages) == 3  # start + 1 tick + end


def test_bag_construction_without_mcap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROSConfigError is raised at construction when mcap is unimportable."""
    import sys

    monkeypatch.setitem(sys.modules, "mcap", None)
    with pytest.raises(ROSConfigError, match=r"mcap>=1\.2"):
        Rosbag2Sink(bag_path=tmp_path / "x.mcap")
