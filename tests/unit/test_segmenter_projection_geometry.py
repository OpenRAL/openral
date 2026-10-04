"""The segmenter's image geometry: which K and which optical frame a prompt is projected in.

Pinned against the real OpenArm manifest, whose ``head_zed.frame_id`` is the ZED
*body* frame (``zed_camera_link``) and whose intrinsics are nominal (fx 960),
while the driver on the Thor cell stamps its images
``zed_left_camera_frame_optical`` and publishes fx 1498.18 (2026-10-02). The node
must project in the image header's frame through the live K, and fall back to the
manifest only out loud. The last test stands up the real lifecycle node (no
model: configure builds none) to pin the ``camera_infos`` parameter handling.
"""

from __future__ import annotations

import pytest
from openral_core import RobotDescription, ROSConfigError, SensorSpec
from openral_perception_ros.camera_topics import parse_camera_entries
from openral_perception_ros.segmenter_node import projection_geometry, sensor_spec_by_name

_ROBOT_YAML = "robots/openarm/robot.yaml"
_ZED_OPTICAL = "zed_left_camera_frame_optical"
# Measured on the Thor cell, /zed/zed_node/rgb/color/rect/camera_info, 1920x1080.
_ZED_THOR_K = [1498.18, 0.0, 936.11, 0.0, 1498.18, 541.81, 0.0, 0.0, 1.0]


def _head_zed() -> SensorSpec:
    spec = sensor_spec_by_name(RobotDescription.from_yaml(_ROBOT_YAML), "head_zed")
    assert spec is not None
    return spec


def _zed_camera_info() -> object:
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    info = sensor_msgs.CameraInfo()
    info.header.frame_id = _ZED_OPTICAL
    info.width, info.height = 1920, 1080
    info.k = _ZED_THOR_K
    return info


def test_parse_camera_entries_skips_blanks_and_half_entries() -> None:
    assert parse_camera_entries([""]) == {}
    assert parse_camera_entries(
        ["head_zed=/zed/zed_node/rgb/color/rect/camera_info", "=x", "y="]
    ) == {"head_zed": "/zed/zed_node/rgb/color/rect/camera_info"}


def test_the_image_header_frame_and_live_k_win_over_the_manifest() -> None:
    spec = _head_zed()
    k, frame, notes = projection_geometry(
        spec,
        width=1920,
        height=1080,
        header_frame_id=_ZED_OPTICAL,
        camera_info=_zed_camera_info(),
    )
    assert frame == _ZED_OPTICAL
    assert frame != spec.frame_id  # the manifest names the body frame
    assert (k.fx, k.fy, k.cx, k.cy) == (1498.18, 1498.18, 936.11, 541.81)
    assert notes == ()


def test_live_k_is_rescaled_to_the_cached_frame() -> None:
    k, _, _ = projection_geometry(
        _head_zed(),
        width=960,
        height=540,
        header_frame_id=_ZED_OPTICAL,
        camera_info=_zed_camera_info(),
    )
    assert (k.width, k.height) == (960, 540)
    assert k.fx == pytest.approx(749.09)
    assert k.cy == pytest.approx(270.905)


def test_each_fallback_is_named() -> None:
    spec = _head_zed()
    k, frame, notes = projection_geometry(spec, width=1920, height=1080, header_frame_id="  ")
    assert frame == spec.frame_id
    assert spec.intrinsics is not None
    assert k.fx == spec.intrinsics.fx
    assert len(notes) == 2
    assert "nominal intrinsics" in notes[0]
    assert "header frame_id is empty" in notes[1]


def test_an_uncalibrated_camera_info_is_refused_not_used() -> None:
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    info = sensor_msgs.CameraInfo()
    info.width, info.height = 1920, 1080
    with pytest.raises(ROSConfigError, match="uncalibrated"):
        projection_geometry(
            _head_zed(), width=1920, height=1080, header_frame_id=_ZED_OPTICAL, camera_info=info
        )


def test_no_k_source_at_all_is_a_config_error() -> None:
    spec = _head_zed().model_copy(update={"intrinsics": None})
    with pytest.raises(ROSConfigError, match="no camera_infos entry"):
        projection_geometry(spec, width=1920, height=1080, header_frame_id=_ZED_OPTICAL)


def test_node_configure_wires_camera_infos_and_rejects_unknown_ids() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_perception_ros.segmenter_node import make_segmenter_node
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.parameter import Parameter

    def _node(name: str, infos: list[str]) -> object:
        node = make_segmenter_node(name)
        node.set_parameters(
            [
                Parameter("robot_yaml", Parameter.Type.STRING, _ROBOT_YAML),
                Parameter(
                    "manifest_path",
                    Parameter.Type.STRING,
                    "rskills/rskill-sam2_1-any-grasped_object_mask-bf16/rskill.yaml",
                ),
                Parameter(
                    "cameras",
                    Parameter.Type.STRING_ARRAY,
                    ["head_zed=/zed/zed_node/rgb/color/rect/image"],
                ),
                Parameter("camera_infos", Parameter.Type.STRING_ARRAY, infos),
            ]
        )
        return node

    rclpy.init()
    try:
        good = _node("segmenter_geometry_ok", ["head_zed=/zed/zed_node/rgb/color/rect/camera_info"])
        assert good.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert good._info_topics == {"head_zed": "/zed/zed_node/rgb/color/rect/camera_info"}
        good.trigger_cleanup()
        good.destroy_node()

        bad = _node("segmenter_geometry_bad", ["wrist_left=/cam/camera_info"])
        # rclpy's lifecycle turns a raising on_configure into ERROR, not SUCCESS.
        assert bad.trigger_configure() != TransitionCallbackReturn.SUCCESS
        bad.destroy_node()
    finally:
        rclpy.try_shutdown()
