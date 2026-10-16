"""Which of SAM's nested candidates the #418 tracker fits: the one the detector boxed.

Rendered into the real SO-101 ``top`` camera (``_tracker_scenes``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit._tracker_scenes import RIGS, render, sphere

node = pytest.importorskip(
    "openral_world_state_ros.object_tracker_node",
    reason="the tracker node's package (packages/world_state) is not on the path",
)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
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
