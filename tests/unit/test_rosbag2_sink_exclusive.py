"""``Rosbag2Sink`` never truncates a file or symlink placed at its path after construction."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_dataset import EpisodeHeader, Rosbag2Sink

pytest.importorskip("mcap")


def _header() -> EpisodeHeader:
    return EpisodeHeader(
        episode_idx=0, task_string="pick", fps=10.0, robot_name="so101_follower", stamp_ns=0
    )


def test_symlink_planted_before_first_episode_is_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    bag = tmp_path / "run.mcap"
    sink = Rosbag2Sink(bag_path=bag)  # the path is free here
    bag.symlink_to(victim)  # ... and taken before the first open_episode()
    with pytest.raises(ROSConfigError, match="refusing to overwrite"):
        sink.open_episode(_header())
    assert victim.read_text(encoding="utf-8") == "keep me"


def test_fresh_path_still_records(tmp_path: Path) -> None:
    bag = tmp_path / "run.mcap"
    sink = Rosbag2Sink(bag_path=bag)
    sink.open_episode(_header())
    sink.finalize()
    assert bag.stat().st_size > 0
