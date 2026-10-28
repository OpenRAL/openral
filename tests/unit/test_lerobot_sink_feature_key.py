"""``LeRobotDatasetSink`` refuses camera keys that walk out of the dataset root (CWE-22)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError

pytest.importorskip("lerobot")

from openral_dataset import LeRobotDatasetSink

REPO = Path(__file__).resolve().parents[2]


def _robot_with_key(key: str) -> RobotDescription:
    robot = RobotDescription.from_yaml(str(REPO / "robots/so101_follower/robot.yaml"))
    cams = [i for i, s in enumerate(robot.sensors) if s.vla_feature_key and s.intrinsics]
    assert cams, "fixture robot declares no camera feature"
    sensors = list(robot.sensors)
    sensors[cams[0]] = sensors[cams[0]].model_copy(update={"vla_feature_key": key})
    return robot.model_copy(update={"sensors": sensors})


@pytest.mark.parametrize("key", ["observation.images.x/../../../escape", "observation.images.a\\b"])
def test_separator_in_camera_key_is_refused(tmp_path: Path, key: str) -> None:
    with pytest.raises(ROSConfigError, match="path separator"):
        LeRobotDatasetSink(root=tmp_path / "ds", robot=_robot_with_key(key), fps=30.0)
