"""Unit tests for ``openral dataset from-bag`` (ADR-0019 PR4).

Per CLAUDE.md §1.11 — real `Rosbag2Sink` writes a real `.mcap` bag,
then the CLI invokes the real `Rosbag2ToLeRobotConverter` to produce a
real `lerobot.datasets.LeRobotDataset` v3 root.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_cli.main import app
from openral_core import RobotDescription
from openral_dataset import RolloutRecorder, Rosbag2Sink
from typer.testing import CliRunner

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
    # SVT-AV1 (libsvtav1 ≥ v3) requires minimum 64×64; 96×96 gives headroom.
    images = {
        "camera1": np.zeros((96, 96, 3), dtype=np.uint8),
        "camera2": np.zeros((96, 96, 3), dtype=np.uint8),
    }
    return state, images, action


def _write_bag(
    bag_path: Path,
    robot: RobotDescription,
    *,
    n_episodes: int,
    n_ticks_each: int,
    success_pattern: tuple[bool, ...],
) -> None:
    """Drive a real RolloutRecorder + Rosbag2Sink to produce a tmp_path bag."""
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(robot=robot, task_string="pick", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(robot)
    for ep_idx in range(n_episodes):
        rec.episode_start()
        for _ in range(n_ticks_each):
            rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success_pattern[ep_idx % len(success_pattern)])
    rec.finalize()


@pytest.fixture(scope="session")
def real_robot_yaml() -> Path:
    """Path to the real SO-100 robot.yaml; the converter loads via from_yaml.

    Walks up from the test file to the repo root so the test is
    portable across checkout locations.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "robots" / "so100_follower" / "robot.yaml"
        if candidate.is_file():
            return candidate
    raise RuntimeError("could not locate robots/so100_follower/robot.yaml")


@pytest.fixture(scope="session")
def so100_robot(real_robot_yaml: Path) -> RobotDescription:
    """Real SO-100 RobotDescription loaded from the same yaml the CLI sees."""
    return RobotDescription.from_yaml(str(real_robot_yaml))


def test_from_bag_round_trip_via_cli(
    so100_robot: RobotDescription, real_robot_yaml: Path, tmp_path: Path
) -> None:
    """End-to-end: `openral dataset from-bag` produces a reload-able v3 dataset."""
    bag_path = tmp_path / "test.mcap"
    _write_bag(
        bag_path,
        so100_robot,
        n_episodes=2,
        n_ticks_each=2,
        success_pattern=(True, False),
    )
    ds_root = tmp_path / "ds"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "from-bag",
            str(bag_path),
            "--robot",
            str(real_robot_yaml),
            "--output",
            str(ds_root),
            "--repo-id",
            "openral/dataset-from-bag-test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "converted" in result.output
    assert "openral/dataset-from-bag-test" in result.output

    # Reload via real lerobot reader.
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("openral/dataset-from-bag-test", root=ds_root)
    assert ds.num_episodes == 2
    assert ds.num_frames == 4

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(0.5)
    assert info["metadata"]["repo_id"] == "openral/dataset-from-bag-test"


def test_from_bag_rejects_empty_bag(real_robot_yaml: Path, tmp_path: Path) -> None:
    """A bag with no episode markers fails with a non-zero exit code."""
    from mcap.writer import Writer

    bag_path = tmp_path / "empty.mcap"
    with bag_path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="openral", library="test")
        writer.finish()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "from-bag",
            str(bag_path),
            "--robot",
            str(real_robot_yaml),
            "--output",
            str(tmp_path / "ds"),
        ],
    )
    assert result.exit_code == 1
    # Rich wraps long error messages; collapse whitespace before matching.
    output_collapsed = " ".join(result.output.split())
    assert "no /openral/episode markers" in output_collapsed


def test_from_bag_rejects_missing_bag(real_robot_yaml: Path, tmp_path: Path) -> None:
    """A non-existent bag path fails at Typer's `exists=True` check (exit 2)."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "from-bag",
            str(tmp_path / "nope.mcap"),
            "--robot",
            str(real_robot_yaml),
            "--output",
            str(tmp_path / "ds"),
        ],
    )
    # Typer returns 2 for argument validation errors.
    assert result.exit_code == 2
