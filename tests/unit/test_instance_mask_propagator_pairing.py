"""The instance-mask propagator lifts a mask only with the robot mask of its own depth frame.

Review of #521 (CodeRabbit): the nearest robot mask to the RGB stamp could be any age, so the
arm's pixels where it had moved since were lifted as the object. The self-filter stamps each
mask with its depth frame's stamp (``robot_self_filter.hpp`` ``publish_mask``).
"""

from __future__ import annotations

import numpy as np
import pytest

_node = pytest.importorskip(
    "openral_world_state_ros.instance_mask_propagator_node",
    reason="the propagator's package (packages/world_state) is not on the path",
)


def test_the_robot_mask_is_the_depth_frames_own_or_the_frame_waits() -> None:
    old, own = np.zeros((2, 2), bool), np.ones((2, 2), bool)
    masks = {1_000_000_000: old, 1_200_000_000: own}
    mask, usable = _node.robot_mask_for(masks, 1_200_000_000)
    assert usable and mask is own
    # The depth frame's own mask not here yet: the 200 ms-old one is never used instead.
    mask, usable = _node.robot_mask_for(masks, 1_400_000_000)
    assert not usable and mask is None


def test_without_any_robot_mask_the_frame_is_used_unmasked() -> None:
    """No robot-mask publisher in this configuration: propagation must not starve."""
    assert _node.robot_mask_for({}, 1_000_000_000) == (None, True)
