"""HAL-side vision attachment client: fallback decisions and the mask/depth wire.

The bridge's ROS half needs a live graph and is covered by
``tests/integration/test_segment_in_view_service.py``. What is covered here is
everything the barrier's *safety* argument rests on and that can be decided
without one:

* :func:`~openral_hal.vision_attachment_bridge.resolve_segment_outcome` — the
  full truth table of "what does this round trip mean", including that **no**
  input combination produces "skip the attachment".
* the ``mono8`` mask wire, encoded by the perception node and decoded by the
  HAL, round-tripped through a **real** SAM 2.1 mask fixture rather than a
  hand-drawn square (CLAUDE.md §1.11).
* :func:`~openral_hal.depth_cloud.depth_grid_from_image`, the depth decoder that
  turns a driver's ``32FC1`` / ``16UC1`` frame into the metric raster the
  producer gates on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core.exceptions import ROSConfigError
from openral_hal.vision_attachment_bridge import (
    DEFAULT_SEGMENT_SERVICE,
    VisionAttachmentConfig,
    decode_mono8_mask,
    resolve_segment_outcome,
)
from PIL import Image

_ERASER_MASK = Path("tests/unit/fixtures/sam2_wrist_eraser_mask.png")


def _real_mask() -> np.ndarray:
    """The committed SAM 2.1 mask of the eraser in the SO-101's jaws."""
    return np.asarray(Image.open(_ERASER_MASK).convert("L")) > 0


def test_a_clean_reply_uses_its_masks() -> None:
    """ok=True with candidates is the only path that reaches the gates."""
    outcome = resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=3)
    assert outcome.use_masks is True
    assert outcome.reason == ""


def test_a_deadline_miss_falls_back_and_says_so() -> None:
    """The bounded wait expiring is a named, typed fallback — never a longer wait."""
    outcome = resolve_segment_outcome(timed_out=True, ok=False, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSDeadlineMissed:")


def test_a_deadline_miss_wins_over_a_late_ok() -> None:
    """Once the deadline fired, a reply that would have been fine is still late."""
    outcome = resolve_segment_outcome(timed_out=True, ok=True, failure_reason="", mask_count=3)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSDeadlineMissed:")


def test_a_typed_server_failure_is_carried_verbatim() -> None:
    """The node's own typed reason reaches the trace unrewritten."""
    outcome = resolve_segment_outcome(
        timed_out=False,
        ok=False,
        failure_reason="ROSPerceptionStale: no frame cached for camera 'wrist'",
        mask_count=0,
    )
    assert outcome.use_masks is False
    assert outcome.reason == "ROSPerceptionStale: no frame cached for camera 'wrist'"


def test_an_untyped_server_failure_still_gets_a_reason() -> None:
    """ok=False with an empty reason is still named, never silent."""
    outcome = resolve_segment_outcome(timed_out=False, ok=False, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSPerceptionStale:")


def test_ok_with_no_candidates_falls_back() -> None:
    """A success flag with an empty mask array cannot be gated, so it degrades."""
    outcome = resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSPerceptionStale:")


@pytest.mark.parametrize("timed_out", [True, False])
@pytest.mark.parametrize("ok", [True, False])
@pytest.mark.parametrize("mask_count", [0, 1, 3])
def test_every_outcome_is_named_or_clean(timed_out: bool, ok: bool, mask_count: int) -> None:
    """No input combination yields an unexplained fallback.

    The producer turns ``use_masks=False`` into the conservative GRIPPER_FORCE
    box, so an unnamed one would be an attachment nobody could explain from the
    trace (CLAUDE.md §1.4).
    """
    outcome = resolve_segment_outcome(
        timed_out=timed_out, ok=ok, failure_reason="", mask_count=mask_count
    )
    assert outcome.use_masks is (bool(outcome.reason) is False)
    assert outcome.use_masks or outcome.reason, "a fallback must carry its reason"


def test_mono8_round_trip_preserves_a_real_sam2_mask() -> None:
    """The perception encoder and the HAL decoder agree, pixel for pixel."""
    from openral_perception_ros.segmenter_node import mono8_bytes_from_mask

    mask = _real_mask()
    height, width = mask.shape
    payload = mono8_bytes_from_mask(mask)
    assert len(payload) == height * width
    decoded = decode_mono8_mask(payload, height=height, width=width)
    assert decoded.dtype == np.bool_
    assert np.array_equal(decoded, mask)
    assert int(decoded.sum()) == int(mask.sum()) > 0


def test_mono8_decode_accepts_any_nonzero_pixel() -> None:
    """A mask that arrived as 254 is still a mask, not an empty one."""
    decoded = decode_mono8_mask(bytes([0, 1, 254, 255]), height=2, width=2)
    assert decoded.tolist() == [[False, True], [True, True]]


def test_mono8_decode_rejects_a_size_mismatch() -> None:
    """A truncated payload is a typed error, not a reshaped guess."""
    with pytest.raises(ROSConfigError, match="mono8"):
        decode_mono8_mask(bytes([255, 0, 0]), height=2, width=2)


def test_default_config_points_at_the_perception_node_service() -> None:
    """The two halves agree on the service name without a literal in each."""
    from openral_perception_ros.segmenter_node import (
        DEFAULT_SEGMENT_SERVICE as PERCEPTION_DEFAULT,
    )

    assert VisionAttachmentConfig().service_name == DEFAULT_SEGMENT_SERVICE
    assert DEFAULT_SEGMENT_SERVICE == PERCEPTION_DEFAULT


def test_default_deadline_is_bounded_and_barrier_sized() -> None:
    """The wait is finite and stays the order of the ~100 ms barrier it rides in."""
    deadline = VisionAttachmentConfig().deadline_s
    assert 0.0 < deadline <= 1.0


def test_depth_decode_32fc1_is_metres() -> None:
    """A 32FC1 driver frame decodes straight through, zeros preserved."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    grid = np.array([[0.25, 0.0], [1.5, 2.0]], dtype="<f4")
    msg = RosImage()
    msg.height, msg.width = 2, 2
    msg.encoding = "32FC1"
    msg.step = 8
    msg.data = grid.tobytes()
    decoded = depth_grid_from_image(msg)
    assert decoded.dtype == np.float64
    assert decoded.tolist() == [[0.25, 0.0], [1.5, 2.0]]


def test_depth_decode_16uc1_is_millimetres() -> None:
    """A RealSense/OAK-D 16UC1 frame is rescaled to metres."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    grid = np.array([[250, 0], [1500, 2000]], dtype="<u2")
    msg = RosImage()
    msg.height, msg.width = 2, 2
    msg.encoding = "16UC1"
    msg.step = 4
    msg.data = grid.tobytes()
    decoded = depth_grid_from_image(msg)
    assert np.allclose(decoded, [[0.25, 0.0], [1.5, 2.0]])


def test_depth_decode_rejects_an_unsupported_encoding() -> None:
    """An RGB frame on the depth topic is a typed error, not garbage geometry."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    msg = RosImage()
    msg.height, msg.width = 1, 1
    msg.encoding = "rgb8"
    msg.step = 3
    msg.data = bytes(3)
    with pytest.raises(ROSConfigError, match="unsupported depth encoding"):
        depth_grid_from_image(msg)


def test_depth_decode_rejects_a_truncated_payload() -> None:
    """A short buffer never becomes a silently smaller depth frame."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    msg = RosImage()
    msg.height, msg.width = 4, 4
    msg.encoding = "32FC1"
    msg.step = 16
    msg.data = np.zeros(4, dtype="<f4").tobytes()
    with pytest.raises(ROSConfigError, match="payload has 4 samples"):
        depth_grid_from_image(msg)
