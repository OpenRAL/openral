"""Which of SAM's nested candidates the #418 tracker fits: the object, not its support.

Rendered into the real SO-101 ``top`` and OpenArm ``head_zed`` cameras (``_tracker_scenes``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit._tracker_scenes import RIGS, box, render, sphere

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


@pytest.mark.parametrize("rig_name", ["so101_top", "openarm_head"])
def test_the_object_beats_its_spill_onto_the_support_and_its_subpart(rig_name: str) -> None:
    rig = RIGS[rig_name]()
    x, y = rig.centre_xy
    solids = [sphere((x, y, rig.support_z + 0.024), 0.024)]
    if rig_name == "openarm_head":
        solids = [box((x, y, rig.support_z + 0.04), (0.04, 0.025, 0.04), 0.3)]
    depth, labels = render(solids, rig)
    obj = labels == 1
    spill = _dilate(obj, 8)  # SAM's "whole": the object and the support patch around it
    rows = np.nonzero(obj)[0]
    sub = obj & (np.arange(obj.shape[0])[:, None] < np.median(rows))  # an inner cut
    # An oblique box's outline runs along its contact line with the support for a stretch:
    # most of it, not all of it, is a depth edge.
    assert node.outline_edge_fraction(obj, depth) >= 0.6
    assert node.outline_edge_fraction(spill, depth) < 0.5
    picked = node.pick_mask([sub, obj, spill], None, depth)
    assert picked is not None and np.array_equal(picked, obj)
    assert node.pick_mask([spill], None, depth) is None  # no fit beats a fit of the support
