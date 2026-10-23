"""LocateInView frame identity contract: exact tracker images, bounded in memory."""

from __future__ import annotations

from typing import Any

import pytest
from openral_perception_ros.ros_image_detector_node import (
    _LOCATE_FRAME_HISTORY_SIZE,
    image_source_stamp_ns,
    locate_frame_failure_reason,
    new_locate_frame_history,
    select_locate_frame,
)

_node = pytest.importorskip(
    "openral_world_state_ros.object_tracker_node",
    reason="the tracker node's package (packages/world_state) is not on the path",
)
_locate_source_stamp_failure_reason = _node._locate_source_stamp_failure_reason
_response_camera_failure_reason = _node._response_camera_failure_reason
_segment_response_failure_reason = _node._segment_response_failure_reason


def test_image_source_stamp_uses_header_time() -> None:
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    image = sensor_msgs.Image()
    image.header.stamp.sec = 4
    image.header.stamp.nanosec = 17

    assert image_source_stamp_ns(image) == 4_000_000_017


def test_locate_frame_selection_is_exact_or_explicitly_latest() -> None:
    history = [(10, b"first", 1, 1), (20, b"second", 1, 1)]

    assert select_locate_frame(history, 10) == history[0]
    assert select_locate_frame(history, 0) == history[1]
    assert select_locate_frame(history, 15) is None
    assert select_locate_frame([], 0) is None
    assert select_locate_frame(history, -1) is None
    assert locate_frame_failure_reason(history, 15) == "requested_frame_unavailable"
    assert locate_frame_failure_reason([], 10) == "no_frame"
    assert locate_frame_failure_reason(history, -1) == "invalid_source_stamp"


def test_locate_frame_history_evicts_old_images_at_fixed_bound() -> None:
    history = new_locate_frame_history()
    for stamp_ns in range(1, _LOCATE_FRAME_HISTORY_SIZE + 2):
        history.append((stamp_ns, b"frame", 1, 1))

    assert len(history) == _LOCATE_FRAME_HISTORY_SIZE
    assert select_locate_frame(history, 1) is None
    assert select_locate_frame(history, 2) is not None


@pytest.mark.parametrize(
    ("requested_ns", "actual_ns", "expected"),
    [
        (10, 10, ""),
        (10, 0, "locate_source_stamp_missing"),
        (10, 11, "locate_source_stamp_mismatch"),
        (0, 10, "locate_source_stamp_missing"),
    ],
)
def test_tracker_requires_actual_source_stamp_match(
    requested_ns: int, actual_ns: int, expected: str
) -> None:
    assert _locate_source_stamp_failure_reason(requested_ns, actual_ns) == expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    [("wrist", ""), ("", "locate_camera_missing"), ("top", "locate_camera_mismatch")],
)
def test_locate_response_requires_requested_camera(actual: str, expected: str) -> None:
    assert _response_camera_failure_reason("locate", "wrist", actual) == expected


def _segment_response(
    *, camera: str = "wrist", stamps_ns: tuple[int, ...] = (10,), frame_id: str = "wrist_optical"
) -> Any:
    segment_srv = pytest.importorskip("openral_msgs.srv")
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    response = segment_srv.SegmentInView.Response()
    response.ok = True
    response.camera = camera
    for stamp_ns in stamps_ns:
        image = sensor_msgs.Image()
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        image.header.frame_id = frame_id
        image.encoding = "mono8"
        response.masks.append(image)
    return response


@pytest.mark.parametrize(
    ("camera", "stamps_ns", "frame_id", "expected"),
    [
        ("wrist", (10,), "wrist_optical", ""),
        ("", (10,), "wrist_optical", "segment_camera_missing"),
        ("top", (10,), "wrist_optical", "segment_camera_mismatch"),
        ("wrist", (0,), "wrist_optical", "segment_mask_stamp_missing"),
        ("wrist", (11,), "wrist_optical", "segment_mask_stamp_mismatch"),
        ("wrist", (10,), "", "segment_mask_frame_missing"),
        ("wrist", (10,), "top_optical", "segment_mask_frame_mismatch"),
        # Every mask is checked; a correct first candidate does not excuse a stale second.
        ("wrist", (10, 11), "wrist_optical", "segment_mask_stamp_mismatch"),
    ],
)
def test_segment_response_requires_camera_and_every_mask_identity(
    camera: str, stamps_ns: tuple[int, ...], frame_id: str, expected: str
) -> None:
    response = _segment_response(camera=camera, stamps_ns=stamps_ns, frame_id=frame_id)

    assert _segment_response_failure_reason("wrist", 10, "wrist_optical", response) == expected


def test_segment_response_preserves_producer_failure_reason() -> None:
    response = _segment_response(camera="other", stamps_ns=(11,))
    response.failure_reason = "ROSPerceptionStale: exact frame unavailable"

    assert (
        _segment_response_failure_reason("wrist", 10, "wrist_optical", response)
        == "ROSPerceptionStale: exact frame unavailable"
    )
