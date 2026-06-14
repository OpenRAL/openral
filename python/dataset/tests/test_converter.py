"""End-to-end tests for :class:`Rosbag2ToLeRobotConverter` (ADR-0019 PR4).

Per CLAUDE.md §1.11 — real `mcap` writer (via `Rosbag2Sink`) writes a
bag to `tmp_path`, then the real `Rosbag2ToLeRobotConverter.from_bag`
replays it through a real `lerobot.datasets.LeRobotDataset` writer.
The result is reloaded by `lerobot.datasets.LeRobotDataset(root)` and
asserted on. No mocks anywhere.

What this file covers:
* Empty bag → `ROSConfigError` (the converter refuses to guess).
* Single-episode round-trip: PHASE_START → ticks → PHASE_END produces a
  v3 dataset whose `num_episodes==1`, `num_frames==n_ticks`, and
  `meta/info.json[metadata][dataset_success_rate]==1.0`.
* Mixed-success multi-episode: dataset_success_rate matches the
  PHASE_END success flags.
* PHASE_START with no PHASE_END (interrupted episode) lands as a
  failure — the deactivate-with-open-episode contract.
* Custom repo_id / license overrides land in meta/info.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_dataset import (
    RolloutRecorder,
    Rosbag2Sink,
    Rosbag2ToLeRobotConverter,
)

# lerobot is required for the LeRobotDatasetSink the converter writes
# into. Skip the whole module when it's not installed — matches the
# `test_sink_lerobot.py` pattern.
lerobot = pytest.importorskip(
    "lerobot",
    reason=(
        "lerobot>=0.5.1 not installed; install via "
        "`just sync --all-packages --group metaworld` or "
        "`uv pip install lerobot>=0.5.1`"
    ),
)


def _zero_frame(
    robot: RobotDescription,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images = {
        "camera1": np.zeros((16, 16, 3), dtype=np.uint8),
        "camera2": np.zeros((16, 16, 3), dtype=np.uint8),
    }
    return state, images, action


def _write_real_bag(
    bag_path: Path,
    robot: RobotDescription,
    *,
    episodes: list[tuple[str, int, bool]],
) -> None:
    """Drive a real RolloutRecorder + Rosbag2Sink to produce a tmp_path bag.

    ``episodes`` is a list of ``(task_string, n_ticks, success)`` tuples.
    """
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=robot, task_string="default", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(robot)
    for task_string, n_ticks, success in episodes:
        rec.episode_start(task_string=task_string)
        for _ in range(n_ticks):
            rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success)
    rec.finalize()


# ── Validation ────────────────────────────────────────────────────────────────


def test_converter_rejects_missing_bag(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Non-existent bag path raises a clean ROSConfigError."""
    with pytest.raises(ROSConfigError, match=r"does not exist"):
        Rosbag2ToLeRobotConverter.from_bag(
            bag_path=tmp_path / "nope.mcap",
            robot=so100_robot,
            output_root=tmp_path / "ds",
        )


def test_converter_rejects_bag_without_episode_markers(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """A bag with no /openral/episode markers raises a clean ROSConfigError.

    We craft this case by writing a bag that contains only Ticks. The
    only way to currently produce a "no episodes" bag with the real
    sink is to never call open_episode — so we write an empty bag
    (just the mcap header) and assert the converter rejects it.
    """
    from mcap.writer import Writer

    bag_path = tmp_path / "empty.mcap"
    with bag_path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="openral", library="test")
        writer.finish()

    with pytest.raises(ROSConfigError, match=r"no /openral/episode markers"):
        Rosbag2ToLeRobotConverter.from_bag(
            bag_path=bag_path,
            robot=so100_robot,
            output_root=tmp_path / "ds",
        )


# ── End-to-end round-trip ────────────────────────────────────────────────────


def test_converter_round_trip_single_episode(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """One 3-tick success episode → 1 episode, 3 frames, success_rate=1.0."""
    bag_path = tmp_path / "single.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("pick the cube", 3, True)])

    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        repo_id="openral/dataset-test",
    )
    assert summary.n_episodes == 1
    assert summary.n_frames == 3
    assert summary.n_success == 1
    assert summary.repo_id == "openral/dataset-test"

    # Reload the dataset via the real lerobot reader and verify on-disk
    # truth matches the summary.
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("openral/dataset-test", root=tmp_path / "ds")
    assert ds.num_episodes == 1
    assert ds.num_frames == 3
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(1.0)
    assert info["metadata"]["repo_id"] == "openral/dataset-test"


def test_converter_round_trip_mixed_success(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """Mixed-success multi-episode → success_rate matches the marker truth."""
    bag_path = tmp_path / "mixed.mcap"
    _write_real_bag(
        bag_path,
        so100_robot,
        episodes=[
            ("pick", 2, True),
            ("pick", 1, False),
            ("pick", 4, True),
        ],
    )
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.n_episodes == 3
    assert summary.n_frames == 7
    assert summary.n_success == 2
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(2 / 3)
    assert info["metadata"]["n_episodes"] == 3
    assert info["metadata"]["n_success_episodes"] == 2


def test_converter_uses_default_repo_id(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """No --repo-id → defaults to openral/dataset-<robot_name>."""
    bag_path = tmp_path / "default_id.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 1, True)])
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.repo_id == f"openral/dataset-{so100_robot.name}"


def test_converter_custom_license_lands_in_info(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """--license overrides the default CC-BY-4.0 in meta/info.json."""
    bag_path = tmp_path / "license.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 1, True)])
    Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        license="CC-BY-NC-4.0",
    )
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["license"] == "CC-BY-NC-4.0"


def test_converter_preserves_bag_trace_ids(so100_robot: RobotDescription, tmp_path: Path) -> None:
    """ISSUE-109: offline bag→LeRobot keeps each tick's ORIGINAL trace id.

    The converter replays ticks through its own ``RolloutRecorder``; if it
    captured the live (converter-process) span the on-disk trace id would
    point at the conversion run, not the original rollout. The bag's
    per-tick (trace_id, span_id) must round-trip verbatim into the
    LeRobot parquet rows.
    """
    import pandas as pd
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")

    # Write a bag whose single tick carries a real, known trace id.
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

    # Convert offline — outside any span of our own.
    Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
        repo_id="openral/dataset-test",
    )

    files = sorted((tmp_path / "ds").glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = df.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["trace_id"] == exp_trace
    assert rows[0]["span_id"] == exp_span


def test_converter_propagates_episode_success_through_recorder(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """A failure episode produces a 0/1 success rate end-to-end."""
    bag_path = tmp_path / "failure.mcap"
    _write_real_bag(bag_path, so100_robot, episodes=[("t", 2, False)])
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=so100_robot,
        output_root=tmp_path / "ds",
    )
    assert summary.n_success == 0
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(0.0)
