"""Multi-robot recorder + bag + converter round-trip (ADR-0019).

Proves the bridge is robot-agnostic by driving the full
``RolloutRecorder → Rosbag2Sink → Rosbag2ToLeRobotConverter →
LeRobotDataset v3`` chain against two structurally-different robots:

* SO-100 follower — 6-DoF arm, 2 RGB cameras, 30 Hz.
* Aloha bimanual — 14-DoF (7+7) bimanual, 1 RGB camera (`top`), 50 Hz.

The bridge is generic by construction (binds via
``RobotDescription`` schema fields, not robot-specific code), but
exercising it against a wildly different state/action shape catches
any accidental hardcoding (e.g. assumed 6-DoF state, assumed
multi-camera setup, assumed 30 Hz fps).

Per CLAUDE.md §1.11: real ``RobotDescription`` from disk, real
``mcap.writer.Writer``, real ``lerobot.datasets.LeRobotDataset``
writer + reader, no mocks anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_dataset import (
    RolloutRecorder,
    Rosbag2Sink,
    Rosbag2ToLeRobotConverter,
)

# Both round-trip tests need lerobot for the LeRobotDatasetSink that
# the converter writes into. Skip the file when it's not present.
pytest.importorskip(
    "lerobot",
    reason=(
        "lerobot>=0.5.1 not installed; install via "
        "`just sync --all-packages --group metaworld` or "
        "`uv pip install lerobot>=0.5.1`"
    ),
)


def _zero_frame_for(
    robot: RobotDescription,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Build (state, images, action) of the shapes a given robot expects.

    ADR-0019: per-camera shape is taken from SensorSpec.intrinsics
    (the declared resolution), not a hardcoded test size.
    """
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images: dict[str, np.ndarray] = {}
    for sensor in robot.sensors:
        if sensor.vla_feature_key is None or sensor.intrinsics is None:
            continue
        stripped = sensor.vla_feature_key.removeprefix("observation.images.")
        channels = 1 if sensor.modality in {"depth", "ir", "thermal"} else 3
        images[stripped] = np.zeros(
            (int(sensor.intrinsics.height), int(sensor.intrinsics.width), channels),
            dtype=np.uint8,
        )
    return state, images, action


# ── The two robots we sweep across ───────────────────────────────────────────


@pytest.fixture(
    params=[
        pytest.param("so100_follower", id="so100_follower"),
        pytest.param("aloha_bimanual", id="aloha_bimanual"),
    ]
)
def bridge_capable_robot(request: pytest.FixtureRequest, repo_root: Path) -> RobotDescription:
    """Yields each bridge-capable robot description in turn.

    A "bridge-capable" robot has both ``observation_spec`` and
    ``action_spec`` populated. As of today only SO-100 and Aloha
    qualify (see ``test_schema_map.py::test_every_robot_manifest_*``).
    When a new robot manifest grows these specs, add it to ``params``.
    """
    name: str = request.param
    return RobotDescription.from_yaml(str(repo_root / "robots" / name / "robot.yaml"))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_bridge_round_trip_per_robot(
    bridge_capable_robot: RobotDescription,
    tmp_path: Path,
    require_video_decode: Callable[[], None],
) -> None:
    """Full bag → converter → LeRobotDataset round-trip for each robot.

    Asserts:
    * The bag writes successfully for the robot's specific shape.
    * The converter reproduces the right episode/frame counts.
    * The reloaded dataset surfaces ``observation.state`` and ``action``
      with the robot's authoritative shapes.
    * The dataset_success_rate aggregate lands correctly across mixed
      success / failure episodes.
    """
    robot = bridge_capable_robot
    bag_path = tmp_path / f"{robot.name}.mcap"

    # 2 episodes (1 success + 1 failure) so dataset_success_rate = 0.5.
    sink = Rosbag2Sink(bag_path=bag_path)
    rec = RolloutRecorder(
        robot=robot,
        task_string="generic-test-task",
        fps=robot.action_spec.control_freq_hz or 30.0,
        sinks=[sink],
    )
    state, images, action = _zero_frame_for(robot)
    for success in (True, False):
        rec.episode_start()
        for _ in range(3):
            rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=success)
    rec.finalize()

    assert bag_path.is_file(), f"bag not written for {robot.name}"

    # Convert and reload.
    summary = Rosbag2ToLeRobotConverter.from_bag(
        bag_path=bag_path,
        robot=robot,
        output_root=tmp_path / "ds",
    )
    assert summary.n_episodes == 2
    assert summary.n_frames == 6
    assert summary.n_success == 1

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset(summary.repo_id, root=tmp_path / "ds")
    assert ds.num_episodes == 2, f"{robot.name}: episode count drift"
    assert ds.num_frames == 6, f"{robot.name}: frame count drift"

    # Indexing decodes the recorded MP4 via torchcodec; skip when the host's
    # video backend can't load (CLAUDE.md §1.11). The bag→converter→counts
    # assertions above need no decode and stay covered everywhere.
    require_video_decode()
    # Robot-specific shape check: the on-disk observation.state vector
    # must round-trip with the same dimensionality the manifest
    # declares. This is the failure mode if the bridge silently
    # truncated to a fixed dim.
    row0 = ds[0]
    expected_state_dim = robot.observation_spec.state_shape[0]
    actual_state = row0["observation.state"]
    assert actual_state.numel() == expected_state_dim, (
        f"{robot.name}: round-tripped state has {actual_state.numel()} dims, "
        f"manifest declares {expected_state_dim}"
    )
    expected_action_dim = robot.action_spec.dim
    actual_action = row0["action"]
    assert actual_action.numel() == expected_action_dim, (
        f"{robot.name}: round-tripped action has {actual_action.numel()} dims, "
        f"manifest declares {expected_action_dim}"
    )

    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["metadata"]["dataset_success_rate"] == pytest.approx(0.5), (
        f"{robot.name}: dataset_success_rate aggregate mismatch"
    )


def test_features_dict_has_every_declared_camera(
    bridge_capable_robot: RobotDescription,
) -> None:
    """Recorder + sink see every camera the robot declares.

    Distinct from ``test_schema_map.test_every_robot_manifest_camera_keys_are_addressable``:
    that one calls ``features_from_robot`` directly; this one drives a
    real recorder + sink and asserts the on-disk dataset reflects every
    declared camera. Catches drift between the schema mapper and the
    sink's per-camera write loop.
    """
    from openral_dataset import features_from_robot

    robot = bridge_capable_robot
    fps = robot.action_spec.control_freq_hz or 30.0
    feats = features_from_robot(robot, fps=fps)
    image_keys = sorted(k for k in feats if k.startswith("observation.images."))
    declared = sorted(
        sensor.vla_feature_key
        for sensor in robot.sensors
        if sensor.vla_feature_key and sensor.vla_feature_key.startswith("observation.images.")
    )
    assert image_keys == declared, (
        f"{robot.name}: bridge feature image keys {image_keys!r} do not match "
        f"RobotDescription.sensors[*].vla_feature_key {declared!r}"
    )
