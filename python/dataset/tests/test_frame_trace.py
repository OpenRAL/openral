"""Tests for ``openral_dataset.read_frame_trace`` (ISSUE-109 pivot).

The replay ``--frame <repo>/<ep>/<frame>`` pivot resolves a written
LeRobotDataset frame back to the OTel ids of the tick that produced it.
The read goes straight at the parquet columns so it never touches the
video backend (torchcodec/ffmpeg) — verifiable on any host.

Per CLAUDE.md §1.11 — real ``LeRobotDataset`` writer + real SO-100
RobotDescription; ``pytest.skip`` when lerobot is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_dataset import LeRobotDatasetSink, RolloutRecorder, read_frame_trace

lerobot = pytest.importorskip(
    "lerobot",
    reason="lerobot>=0.5.1 not installed; install via `uv pip install lerobot>=0.5.1`",
)


def _zero_frame(robot: RobotDescription) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images = {
        "camera1": np.zeros((256, 256, 3), dtype=np.uint8),
        "camera2": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    return state, images, action


def _write_traced_dataset(root: Path, robot: RobotDescription) -> list[tuple[str, str]]:
    """Write a 1-episode, 2-frame dataset; return per-frame (trace_id, span_id)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    sink = LeRobotDatasetSink(root=root, robot=robot, fps=30.0, repo_id="openral/dataset-test")
    rec = RolloutRecorder(robot=robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(robot)

    ids: list[tuple[str, str]] = []
    rec.episode_start()
    for _ in range(2):
        with tracer.start_as_current_span("rskill.tick"):
            ctx = trace.get_current_span().get_span_context()
            ids.append((f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"))
            rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()
    return ids


def test_read_frame_trace_returns_producing_tick_ids(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """The pivot returns the exact (trace_id, span_id) of the requested frame."""
    root = tmp_path / "ds"
    ids = _write_traced_dataset(root, so100_robot)

    assert read_frame_trace(root=root, episode_idx=0, frame_idx=0) == ids[0]
    assert read_frame_trace(root=root, episode_idx=0, frame_idx=1) == ids[1]


def test_read_frame_trace_missing_frame_raises(
    so100_robot: RobotDescription, tmp_path: Path
) -> None:
    """Asking for a frame that was never written is a clean ROSConfigError."""
    root = tmp_path / "ds"
    _write_traced_dataset(root, so100_robot)

    with pytest.raises(ROSConfigError, match=r"no frame"):
        read_frame_trace(root=root, episode_idx=0, frame_idx=99)


def test_read_frame_trace_missing_dataset_raises(tmp_path: Path) -> None:
    """A root with no parquet data raises a clean ROSConfigError."""
    with pytest.raises(ROSConfigError, match=r"no LeRobot dataset"):
        read_frame_trace(root=tmp_path / "nope", episode_idx=0, frame_idx=0)
