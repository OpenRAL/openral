"""openral_dataset — rosbag2 ↔ LeRobotDataset v3 bridge (ADR-0019).

Public API:
    RolloutRecorder       — in-memory per-rollout accumulator with multi-sink fan-out.
    DatasetSink           — Protocol every sink (online / offline) implements.
    LeRobotDatasetSink    — writes via ``lerobot.datasets.LeRobotDataset`` (v3.0).
    EpisodeHeader / DatasetFrame / EpisodeSummary — sink message dataclasses.
    features_from_robot   — pure RobotDescription → LeRobot v3 features dict mapping.

Example:
    >>> from openral_core import RobotDescription
    >>> from openral_dataset import LeRobotDatasetSink, RolloutRecorder
    >>> robot = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")
    >>> sink = LeRobotDatasetSink(root="/tmp/ds", robot=robot, fps=30.0)  # doctest: +SKIP
    >>> with RolloutRecorder(robot=robot, task_string="pick", fps=30.0, sinks=[sink]) as rec:
    ...     rec.episode_start()
    ...     # rec.record_frame(...)
    ...     rec.episode_end(success=True)  # doctest: +SKIP
"""

from __future__ import annotations

from openral_dataset.bag import Rosbag2Sink
from openral_dataset.converter import DatasetSummary, Rosbag2ToLeRobotConverter
from openral_dataset.frame_trace import read_frame_trace
from openral_dataset.recorder import (
    DatasetFrame,
    DatasetSink,
    EpisodeHeader,
    EpisodeSummary,
    RolloutRecorder,
)
from openral_dataset.schema_map import FeatureSpec, features_from_robot
from openral_dataset.sinks import LeRobotDatasetSink

__all__ = [
    "DatasetFrame",
    "DatasetSink",
    "DatasetSummary",
    "EpisodeHeader",
    "EpisodeSummary",
    "FeatureSpec",
    "LeRobotDatasetSink",
    "RolloutRecorder",
    "Rosbag2Sink",
    "Rosbag2ToLeRobotConverter",
    "features_from_robot",
    "read_frame_trace",
]
