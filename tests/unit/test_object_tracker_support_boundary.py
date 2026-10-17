"""Support detection must not depend on which side of a cell boundary the support lands (#418).

PickOrange r6po_dbg7 (after the HAL-derived depth clouds): the counter top measured at z 0.0302,
on the 15 mm map's 0.030 layer boundary, filled one layer instead of two, and orange_3 under the
jaw was refused ``no_support`` all trial. Real camera rig (the PickOrange top camera), the orange
on its support, the map's layer boundary placed at the support +- epsilon.
"""

from __future__ import annotations

import pytest
from openral_core import ObjectQuery, ObjectRole
from openral_world_state.object_tracker import ObjectTracker

from tests.unit._tracker_scenes import RES, so101_top, sphere, step

_RIG = so101_top()
_R = 0.024  # the PickOrange orange
_EPS = 2e-4


@pytest.mark.parametrize("shift", [-_EPS, 0.0, _EPS, 0.5 * RES, RES - _EPS])
def test_an_orange_grounds_whichever_side_of_a_cell_edge_its_support_lands(shift: float) -> None:
    x, y = _RIG.centre_xy
    orange = sphere((x, y, _RIG.support_z + _R), _R)
    tr = ObjectTracker(_RIG.frame)
    tr.open_query("g", "q0", ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),
                  "obj:orange_1@0", stamp_ns=0)  # fmt: skip
    out, _ = step(tr, _RIG, [orange], stamp_ns=0, wall_s=0.0, shift_z=shift)
    assert [o.label for o in out.tracked.objects] == ["orange"], dict(tr.refusal_counts)
    (t,) = out.tracked.objects
    # The support is measured, not guessed: within a cell of the true top.
    assert abs(t.support_z - _RIG.support_z) <= RES + 1e-9


@pytest.mark.parametrize("below", [1e-6, 1e-4, _EPS])
@pytest.mark.parametrize("above", [1e-4, _EPS, 2e-3])
def test_a_seen_support_point_just_under_its_mapped_layer_still_finds_the_support(
    below: float, above: float
) -> None:
    """The r6po_dbg7 case. The counter top is at z 0.0302 and the 15 mm map's layer boundary at
    0.0300: the map puts the counter in the layer that starts at the boundary. SAM's mask spills
    onto the counter, and one seen point lies a hair under the boundary (0.0300). The A2 filter
    kept only cells starting below the lowest seen point, so it dropped the counter's whole layer
    and refused ``no_support`` for the rest of the trial. Rig: the support plane ``above`` the
    layer boundary, the mask taking in the support, one seen support point ``below`` it."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    from tests.unit._tracker_scenes import grid_from_depth, render

    x, y = _RIG.centre_xy
    depth, labels = render([sphere((x, y, _RIG.support_z + _R), _R)], _RIG)
    grid = grid_from_depth(depth, _RIG, shift_z=-above)  # boundary `above` under the support
    boundary = _RIG.support_z - above
    spill = binary_dilation(labels == 1, iterations=2) & (depth > 0)
    rows, cols = np.nonzero(spill)
    z = depth[rows, cols]
    k, t = _RIG.k, _RIG.t
    pts = np.stack(((cols + 0.5 - k.cx) / k.fx * z, (rows + 0.5 - k.cy) / k.fy * z, z), axis=1)
    surface = pts @ t[:3, :3].T + t[:3, 3]
    edge = surface[int(np.argmin(surface[:, 2]))].copy()
    edge[2] = boundary - below  # seen, but under the layer the map holds the support in
    surface = np.vstack([surface, edge])
    top = ObjectTracker(_RIG.frame)._support(grid, surface)
    assert top is not None, "no_support"
    assert abs(top - _RIG.support_z) <= RES + 1e-9


@pytest.mark.parametrize("shift", [-_EPS, 0.0, _EPS, 0.5 * RES, RES - _EPS])
def test_a_mask_that_takes_in_the_support_still_measures_it(shift: float) -> None:
    """The r6po_dbg7 case: SAM's mask spills a pixel onto the counter, so the lowest seen point
    is the support's own surface, at z 0.0300 on the layer boundary. The support's cells start at
    that boundary, not below it, and the A2 filter dropped them all: ``no_support``."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    from tests.unit._tracker_scenes import grid_from_depth, render

    x, y = _RIG.centre_xy
    depth, labels = render([sphere((x, y, _RIG.support_z + _R), _R)], _RIG)
    grid = grid_from_depth(depth, _RIG, shift_z=shift)
    spill = binary_dilation(labels == 1, iterations=2) & (depth > 0)
    rows, cols = np.nonzero(spill)
    z = depth[rows, cols]
    k, t = _RIG.k, _RIG.t
    pts = np.stack(((cols + 0.5 - k.cx) / k.fx * z, (rows + 0.5 - k.cy) / k.fy * z, z), axis=1)
    surface = pts @ t[:3, :3].T + t[:3, 3]
    assert surface[:, 2].min() == pytest.approx(_RIG.support_z, abs=1e-6), "the mask holds support"
    top = ObjectTracker(_RIG.frame)._support(grid, surface)
    assert top is not None
    assert abs(top - _RIG.support_z) <= RES + 1e-9
