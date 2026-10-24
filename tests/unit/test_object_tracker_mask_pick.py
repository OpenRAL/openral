"""Which of SAM's nested candidates the #418 tracker fits: the one the detector boxed.

Rendered into the real SO-101 ``top`` camera (``_tracker_scenes``).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.unit._tracker_scenes import RIGS, render, sphere

node = pytest.importorskip(
    "openral_world_state_ros.object_tracker_node",
    reason="the tracker node's package (packages/world_state) is not on the path",
)


def _dilate(mask: NDArray[np.bool_], px: int) -> NDArray[np.bool_]:
    m = mask.copy()
    for _ in range(px):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def test_the_detector_box_picks_the_object_over_its_spill_and_its_subpart() -> None:
    rig = RIGS["so101_top"]()
    x, y = rig.centre_xy
    _, labels = render([sphere((x, y, rig.support_z + 0.024), 0.024)], rig)
    obj = labels == 1
    rows, cols = np.nonzero(obj)
    box = (
        float(cols.min()) - 2,
        float(rows.min()) - 2,
        float(cols.max()) + 3,
        float(rows.max()) + 3,
    )
    spill = _dilate(obj, 4)  # SAM's "whole": the orange and the counter patch around it
    sub = obj & (np.arange(obj.shape[0])[:, None] < np.median(rows))  # an inner cut
    picked = node.pick_masks([sub, obj, spill], box)
    assert np.array_equal(picked[0], obj)
    # Without a detector box: largest first, the map decides whether to fall back.
    assert np.array_equal(node.pick_masks([sub, obj, spill], None)[0], spill)


def test_a_mask_that_trails_a_depth_period_still_pairs_with_its_frame() -> None:
    """Spark restock (2026-10-24): the head camera's mask trails its depth by more than one
    depth period. Waiting on the newest frame alone was reset by every new one (no frame ever
    fed); the newest frame whose mask has arrived must win over newer, mask-less ones."""
    pick = node.pick_depth_frame
    received = [0.0, 0.7, 1.4]  # one depth frame per 0.7 s wall
    # Frame 0's mask just arrived; frames 1 and 2 still wait for theirs.
    assert pick(received, lambda i: i == 0, 1.45, 1.0) == 0
    # Two have masks: the newer one is fed (the caller drops the older with it).
    assert pick(received, lambda i: i in (0, 1), 1.45, 1.0) == 1
    # None yet, the oldest has not waited out: keep waiting rather than age a frame.
    assert pick(received, lambda i: False, 0.9, 1.0) is None
    # None after the wait: the oldest is fed for ageing only.
    assert pick(received, lambda i: False, 1.0, 1.0) == 0
    assert pick([], lambda i: True, 5.0, 1.0) is None
