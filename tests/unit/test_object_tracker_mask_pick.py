"""Which of SAM's nested candidates the #418 tracker fits: the object, not its support.

Rendered into the real SO-101 ``top`` and OpenArm ``head_zed`` cameras (``_tracker_scenes``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit._tracker_scenes import RIGS, box, cylinder, render, sphere

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


@pytest.mark.parametrize(
    ("rig_name", "kind"),
    [("so101_top", "orange"), ("so101_top", "plate"), ("openarm_head", "carton")],
)
def test_the_object_beats_its_spill_onto_the_support_and_its_subpart(
    rig_name: str, kind: str
) -> None:
    rig = RIGS[rig_name]()
    x, y = rig.centre_xy
    z0 = rig.support_z
    solids = {
        "orange": [sphere((x, y, z0 + 0.024), 0.024)],
        "plate": [cylinder((x, y), 0.1, z0, z0 + 0.018)],  # a low, wide body
        "carton": [box((x, y, z0 + 0.04), (0.04, 0.025, 0.04), 0.3)],
    }[kind]
    depth, labels = render(solids, rig)
    height = node.height_image(depth, rig.k, rig.t)
    obj = labels == 1
    spill = _dilate(obj, 8)  # SAM's "whole": the object and the support patch around it
    rows = np.nonzero(obj)[0]
    sub = obj & (np.arange(obj.shape[0])[:, None] < np.median(rows))  # an inner cut
    # An oblique box's outline runs along its contact line with the support for a stretch:
    # most of it, not all of it, is a depth edge.
    assert node.outline_edge_fraction(obj, height) >= 0.6
    assert node.outline_edge_fraction(spill, height) < 0.5
    picked = node.pick_masks([sub, obj, spill], None, height)
    assert picked and np.array_equal(picked[0], obj)
    assert not node.pick_masks([spill], None, height)  # no fit beats a fit of the support
