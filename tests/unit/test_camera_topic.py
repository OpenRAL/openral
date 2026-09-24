"""``openral_core.camera_topic``: the one builder of the camera topic layout (ADR-0108)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    CAMERA_TOPIC_PREFIX,
    CameraTopicKind,
    RobotDescription,
    ROSConfigError,
    camera_topic,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("kind", "suffix"),
    [
        (CameraTopicKind.IMAGE, "image"),
        (CameraTopicKind.CAMERA_INFO, "camera_info"),
        (CameraTopicKind.DEPTH_IMAGE, "depth/image"),
        (CameraTopicKind.DEPTH_CAMERA_INFO, "depth/camera_info"),
        (CameraTopicKind.POINTS, "points"),
    ],
)
def test_every_kind_lands_under_the_prefix(kind: CameraTopicKind, suffix: str) -> None:
    assert camera_topic("front_depth", kind) == f"/openral/cameras/front_depth/{suffix}"


def test_default_kind_is_the_image_stream() -> None:
    assert camera_topic("wrist") == "/openral/cameras/wrist/image"
    assert CAMERA_TOPIC_PREFIX == "/openral/cameras"


def test_prefix_override_for_a_namespaced_graph() -> None:
    assert camera_topic("top", CameraTopicKind.POINTS, prefix="/robot_b/cameras") == (
        "/robot_b/cameras/top/points"
    )


def test_real_manifest_sensor_names_build_their_topics() -> None:
    """Every RGB sensor the SO-101 manifest declares gets a well-formed topic."""
    desc = RobotDescription.from_yaml(REPO / "robots" / "so101_follower" / "robot.yaml")
    rgb = [s.name for s in desc.sensors if s.modality == "rgb"]
    assert rgb, "so101_follower declares RGB cameras"
    assert [camera_topic(n) for n in rgb] == [f"/openral/cameras/{n}/image" for n in rgb]


@pytest.mark.parametrize("name", ["", "cam/left"])
def test_rejects_a_name_that_is_not_one_path_segment(name: str) -> None:
    with pytest.raises(ROSConfigError, match="single path segment"):
        camera_topic(name)
