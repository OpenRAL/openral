"""The attachment perception barrier waits for a grid that carries the payload's clearing (#418).

r7po_on1 (PickOrange, 2026-10-17, graph wall clock): the tracker bound trk:4 at 526.474; the
kernel first saw the attachment at 528.489 and the HAL armed its barrier ("waiting for 1
transparent depth frame"); the depth frame was counted at 528.591; the kernel checked the first
chunk against the bound payload at 528.615 and STOPPED on the held orange's own cells; the
barrier released at 528.731 and the bridge's identity clear landed at 528.782. The barrier's
voxel wait only required a grid whose source stamp post-dated the depth frame: the grid at
528.6 qualified although the bridge had not yet consumed the attachment revision, so it still
held the orange. The wait now also requires the grid to say, in its ``attachment_revision``,
that it was built with this attachment's clearing applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from openral_hal.sim_sensor_bridge import grid_settles_attachment

_REV = 1792246471229091316  # the revision the barrier armed for (r7po_on1)
_DEPTH_NS = 9_283_000_000  # the transparent depth frame the barrier counted


@dataclass(frozen=True)
class Grid:
    t_wall: float
    source_ns: int
    attachment_revision: int


# The grids the bridge published around the stop, in order (r7po_on1).
_TIMELINE = [
    Grid(528.50, _DEPTH_NS - 100_000_000, _REV - 1),  # before the depth frame: stale octree
    Grid(528.60, _DEPTH_NS + 10_000_000, _REV - 1),  # after the frame, bridge has not the revision
    Grid(528.70, _DEPTH_NS + 10_000_000, 0),  # bridge could not certify a clear (stale state)
    Grid(528.78, _DEPTH_NS + 10_000_000, _REV),  # identity clear (12 cells) applied
    Grid(528.88, _DEPTH_NS + 40_000_000, _REV),
]


def _settled(g: Grid, required: int = _REV) -> bool:
    return grid_settles_attachment(
        source_stamp_ns=g.source_ns,
        transparent_depth_ns=_DEPTH_NS,
        grid_attachment_revision=g.attachment_revision,
        required_revision=required,
    )


def test_the_barrier_does_not_release_before_the_post_clear_grid() -> None:
    released_at = next(g.t_wall for g in _TIMELINE if _settled(g))
    assert released_at == 528.78, "the first grid with the clear applied, not the first fresh one"
    # The old rule (source stamp after the depth frame) released at 528.60, before the clear.
    old = next(g.t_wall for g in _TIMELINE if g.source_ns > _DEPTH_NS)
    assert old == 528.60 < released_at


@pytest.mark.parametrize(
    ("source_ns", "rev", "want"),
    [
        (_DEPTH_NS + 1, _REV, True),  # fresh and cleared
        (_DEPTH_NS, _REV, False),  # not later than the transparent frame
        (_DEPTH_NS - 1, _REV, False),
        (_DEPTH_NS + 1, _REV - 1, False),  # an older revision's clearing
        (_DEPTH_NS + 1, 0, False),  # the bridge certified nothing
        (_DEPTH_NS + 1, _REV + 1, True),  # a newer set's clearing includes this one's
    ],
)
def test_a_grid_settles_the_barrier_only_when_fresh_and_cleared(
    source_ns: int, rev: int, want: bool
) -> None:
    assert (
        grid_settles_attachment(
            source_stamp_ns=source_ns,
            transparent_depth_ns=_DEPTH_NS,
            grid_attachment_revision=rev,
            required_revision=_REV,
        )
        is want
    )
