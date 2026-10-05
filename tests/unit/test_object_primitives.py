"""Object primitives from the voxel map (``docs/reference/object-primitives-design.md``).

The scene is the committed Isaac warehouse deploy scene, numerically:
``scenes/deploy/isaac_openarm_warehouse.yaml`` spawns three YCB props on the pallet deck
(top z 0.21 in ``world``) in front of an OpenArm whose base frame sits 0.698 m above the
URDF root (``robots/openarm/robot.yaml`` ``base_to_root_xyz_rpy``), at the 15 mm cells
``deploy_e2e`` gives a sim octomap. Each prop is modelled as the shell the head camera
leaves in the map — its top layer and the wall facing the robot — over a 7 cm footprint
(the scene names positions, not sizes). No mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import DeployScene, GraspDeclaration, PlaceRegion, Pose6D, RobotDescription
from openral_hal._grasp_target import (
    VoxelLattice,
    _in_region,
    cell_closed_region,
    support_top_from_voxels,
)
from openral_hal._object_primitives import (
    PrimitiveFit,
    PrimitiveTracker,
    nearest_primitive,
    point_box_gap_m,
    primitives_from_voxels,
    raised_anchor_xy,
)

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes" / "deploy" / "isaac_openarm_warehouse.yaml"
_ROBOT = _REPO / "robots" / "openarm" / "robot.yaml"
_RES = 0.015  # deploy_e2e.launch.py _octomap_resolution: sim ships 15 mm cells
_DECK_WORLD_Z = 0.21  # the scene's comment: Shelf_1 deck top
_FRAME = "openarm_base"
_FOOT = 0.07  # modelled prop footprint (the scene names positions only)
_HEIGHT = 0.12
_CAPS = {
    "max_half_extent_m": GraspDeclaration.MAX_HALF_EXTENT_M,
    "max_volume_m3": GraspDeclaration.MAX_VOLUME_M3,
}


def _scene() -> tuple[list[tuple[float, float]], float]:
    """The props' xy and the deck top, in ``openarm_base``, from the committed fixtures."""
    scene = DeployScene.from_yaml(str(_SCENE))
    base = scene.base_pose.xyz
    urdf = RobotDescription.from_yaml(str(_ROBOT)).assets.urdf
    assert urdf is not None and urdf.base_to_root_xyz_rpy is not None
    root_z = urdf.base_to_root_xyz_rpy[2]
    raw = yaml.safe_load(_SCENE.read_text())["scene"]["backend_options"]["objects"]
    xy = [(o["xyz"][0] - base[0], o["xyz"][1] - base[1]) for o in raw]
    return xy, _DECK_WORLD_Z - base[2] + root_z


_PROPS_XY, _DECK_Z = _scene()
# Lattice: x 0.2-0.8, |y| <= 0.5, two deck layers then 0.27 m of air above the deck top.
_ORIGIN = (0.2, -0.5, _DECK_Z - 2 * _RES)
_SIZE = (40, 67, 20)


def _lattice(
    boxes: list[tuple[float, float, float, float, float]], *, solid: bool = False
) -> VoxelLattice:
    """Deck + one shell (or solid) per ``(x, y, width_x, depth_y, height)`` box."""
    sx, sy, sz = _SIZE
    occ = np.zeros(sx * sy * sz, dtype=np.uint8)

    def fill(i: int, j: int, k: int) -> None:
        if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
            occ[i + sx * (j + sy * k)] = 1

    for i in range(sx):
        for j in range(sy):
            fill(i, j, 0)
            fill(i, j, 1)
    for x, y, w, d, h in boxes:
        i0, i1 = round((x - w / 2 - _ORIGIN[0]) / _RES), round((x + w / 2 - _ORIGIN[0]) / _RES)
        j0, j1 = round((y - d / 2 - _ORIGIN[1]) / _RES), round((y + d / 2 - _ORIGIN[1]) / _RES)
        k_top = 2 + round(h / _RES) - 1
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(2, k_top + 1):
                    if solid or k == k_top or i == i0:  # top face, or the wall facing -x
                        fill(i, j, k)
    return VoxelLattice(_FRAME, _ORIGIN, (0.0, 0.0, 0.0, 1.0), _RES, _SIZE, occ)


def _props(xy: list[tuple[float, float]] | None = None) -> VoxelLattice:
    return _lattice([(x, y, _FOOT, _FOOT, _HEIGHT) for x, y in (xy or _PROPS_XY)])


def _column(x: float, y: float, *, half_xy: float) -> PlaceRegion:
    """A search column over ``(x, y)``: from the deck down 0.15 m to 0.3 m above it."""
    return PlaceRegion(
        frame_id=_FRAME,
        pose=Pose6D(xyz=(x, y, _DECK_Z + 0.075), quat_xyzw=(0, 0, 0, 1), frame_id=_FRAME),
        half_extents=(half_xy, half_xy, 0.225),
    )


def _support(grid: VoxelLattice, column: PlaceRegion) -> float:
    """The leg's support measurement (``GraspTargetLeg._measure_support``) over ``column``."""
    centers = grid.occupied_centers()
    hx, hy, hz = column.half_extents
    grow = 0.05 + _RES  # the ring is counted on the column grown by the probe margin + a cell
    around = column.model_copy(update={"half_extents": (hx + grow, hy + grow, hz)})
    anchor = raised_anchor_xy(grid, column, column.pose.xyz[:2])
    assert anchor is not None
    z = support_top_from_voxels(
        grid,
        centers[_in_region(centers, column)],
        near_xy=anchor,
        min_cells=8,
        probe_margin_m=0.05,
        surface_centers=centers[_in_region(centers, around)],
    )
    assert z is not None
    return z


def _at(regions: list[PlaceRegion], y: float) -> PlaceRegion:
    """The one region whose centre lies within a cell of ``y`` (the lattice quantises)."""
    near = [r for r in regions if abs(r.pose.xyz[1] - y) <= _RES]
    assert len(near) == 1, (y, [r.pose.xyz for r in regions])
    return near[0]


def _fits(grid: VoxelLattice, column: PlaceRegion) -> tuple[list[PrimitiveFit], list[str]]:
    return primitives_from_voxels(
        grid, column, support_z=_support(grid, column), min_cells=8, stamp_ns=7, **_CAPS
    )


def test_the_scene_fixtures_place_three_props_on_the_deck_in_reach() -> None:
    assert len(_PROPS_XY) == 3
    assert all(0.45 <= x <= 0.55 for x, _ in _PROPS_XY)
    assert pytest.approx(0.21 - 0.698) == _DECK_Z


def test_each_prop_on_the_deck_is_one_primitive_within_the_kernels_contract() -> None:
    grid = _props()
    column = _column(0.5, 0.0, half_xy=0.45)
    support_z = _support(grid, column)
    assert support_z == pytest.approx(_DECK_Z)
    fits, skipped = _fits(grid, column)
    assert skipped == [] and len(fits) == 3
    for _, y in _PROPS_XY:
        _at([f.region for f in fits], y)
    for fit in fits:
        region = fit.region
        assert region.stamp_ns == 7 and region.frame_id == _FRAME
        # The tight box: over cell centres (half a cell short of the footprint each side),
        # its lower face one voxel above the deck, never into it (HZ-0115-6).
        assert abs(region.half_extents[0] - _FOOT / 2) <= _RES
        assert abs(region.half_extents[1] - _FOOT / 2) <= _RES
        assert region.pose.xyz[2] - region.half_extents[2] == pytest.approx(support_z + _RES)
        assert region.pose.xyz[2] + region.half_extents[2] == pytest.approx(
            _DECK_Z + _HEIGHT, abs=_RES
        )
        # What the kernel gets holds every cell of the component and passes the caps.
        closed, clamped = cell_closed_region(region, grid, max_half_extent_m=0.2)
        assert not clamped
        centers = grid.occupied_centers()
        above = centers[centers[:, 2] > support_z + _RES]
        held = above[_in_region(above, closed)]
        assert len(held) >= fit.cell_count
        GraspDeclaration(
            target_id="approach:openarm_left_finger_pair:1",
            contact_links=("openarm_left_finger_pair",),
            timeout_s=60.0,
            stamp_ns=1,
            region=closed,
        )


def test_the_prop_nearest_the_hand_is_chosen_and_a_tie_is_refused() -> None:
    grid = _props()
    fits, _ = _fits(grid, _column(0.5, 0.0, half_xy=0.45))
    regions = [f.region for f in fits]
    middle = regions.index(_at(regions, _PROPS_XY[1][1]))
    plus = regions.index(_at(regions, _PROPS_XY[2][1]))
    hover = _DECK_Z + _HEIGHT + 0.10
    index, why = nearest_primitive(regions, [(0.5, 0.03, hover)], ambiguity_m=_RES)
    assert (index, why) == (middle, "")
    index, why = nearest_primitive(regions, [(0.5, 0.20, hover)], ambiguity_m=_RES)
    assert (index, why) == (plus, "")
    between = (_PROPS_XY[1][1] + _PROPS_XY[2][1]) / 2
    index, why = nearest_primitive(regions, [(0.5, between, hover)], ambiguity_m=_RES)
    assert index is None and why.startswith("ambiguous")
    # A named search box (no hand) is unambiguous only with one primitive in it.
    assert nearest_primitive(regions[:1], [], ambiguity_m=_RES) == (0, "")
    assert nearest_primitive(regions, [], ambiguity_m=_RES)[0] is None
    assert nearest_primitive([], [(0.5, 0.0, hover)], ambiguity_m=_RES)[0] is None
    assert point_box_gap_m((0.5, _PROPS_XY[1][1], hover), regions[middle]) == pytest.approx(
        0.10, abs=_RES
    )


def test_a_touching_pair_is_one_primitive_and_a_merged_block_past_the_caps_is_none() -> None:
    """Two bodies within one voxel are one component (HZ-0115-30): the map alone cannot
    separate them. Under the caps the pair is one primitive — the documented residual;
    past them there is no primitive at all (fail safe: no region, the kernel stops)."""
    x, y = _PROPS_XY[1]
    pair = _lattice(
        [(x, y - _FOOT / 2, _FOOT, _FOOT, _HEIGHT), (x, y + _FOOT / 2, _FOOT, _FOOT, _HEIGHT)]
    )
    fits, skipped = _fits(pair, _column(x, y, half_xy=0.2))
    assert len(fits) == 1 and skipped == []
    assert max(fits[0].region.half_extents[:2]) == pytest.approx(_FOOT - _RES / 2, abs=_RES)
    crates = _lattice([(x, y - 0.13, 0.26, 0.26, 0.20), (x, y + 0.13, 0.26, 0.26, 0.20)])
    fits, skipped = _fits(crates, _column(x, y, half_xy=0.35))
    assert fits == []
    assert len(skipped) == 1 and "over the caps" in skipped[0]


def test_a_hand_beside_the_target_anchors_on_the_target_not_the_deck() -> None:
    """A lateral approach: the search column is centred on bare deck 10 cm from the prop.
    Anchored on the deck the support scan finds nothing under it; anchored on the nearest
    raised top surface it measures the deck and fits the prop."""
    grid = _props()
    x, y = _PROPS_XY[1]
    column = _column(x, y + 0.10, half_xy=0.2)
    centers = grid.occupied_centers()
    inside = centers[_in_region(centers, column)]
    assert (
        support_top_from_voxels(
            grid, inside, near_xy=column.pose.xyz[:2], min_cells=8, probe_margin_m=0.05
        )
        is None
    )
    anchor = raised_anchor_xy(grid, column, column.pose.xyz[:2])
    assert anchor is not None and abs(anchor[1] - (y + _FOOT / 2)) <= _RES
    fits, _ = _fits(grid, column)  # the column also reaches the +y prop
    assert len(fits) == 2 and _at([f.region for f in fits], y) is not None
    assert raised_anchor_xy(_lattice([]), column, column.pose.xyz[:2]) is None


def test_a_component_touching_the_column_edge_is_skipped() -> None:
    grid = _props()
    x, y = _PROPS_XY[1]
    fits, skipped = _fits(grid, _column(x + 0.03, y, half_xy=0.06))
    assert fits == [] and len(skipped) == 1 and "touch the column's edge" in skipped[0]
    fits, _ = _fits(grid, _column(x, y, half_xy=0.10))
    assert len(fits) == 1


def test_a_body_stacked_on_another_is_one_primitive_with_it() -> None:
    """A riser under the target clusters with it and the box reaches the support: the map
    cannot tell the two apart (the camera's ``not_on_support`` can). Pinned as the
    residual the design note keeps a camera confirmation for."""
    x, y = _PROPS_XY[1]
    stacked = _lattice([(x, y, _FOOT, _FOOT, 0.05), (x, y, _FOOT, _FOOT, 0.17)], solid=True)
    fits, skipped = _fits(stacked, _column(x, y, half_xy=0.2))
    assert len(fits) == 1 and skipped == []
    assert fits[0].region.pose.xyz[2] + fits[0].region.half_extents[2] == pytest.approx(
        _DECK_Z + 0.17, abs=_RES
    )


def test_too_few_cells_and_a_tilted_lattice_are_refused() -> None:
    grid = _props()
    column = _column(0.5, 0.0, half_xy=0.45)
    fits, skipped = primitives_from_voxels(
        grid, column, support_z=_support(grid, column), min_cells=60, **_CAPS
    )
    assert fits == [] and len(skipped) == 3 and all("< min_cells" in s for s in skipped)
    with pytest.raises(Exception, match="frame"):
        primitives_from_voxels(
            grid, column.model_copy(update={"frame_id": "map"}), support_z=0.0, min_cells=8, **_CAPS
        )


def test_the_tracker_keeps_identities_and_drops_what_the_map_cleared() -> None:
    column = _column(0.5, 0.0, half_xy=0.45)
    tracker = PrimitiveTracker(max_misses=2)

    def frame(grid: VoxelLattice, stamp: int) -> dict[int, PlaceRegion]:
        fits, _ = _fits(grid, column)
        return {
            p.primitive_id: p.region
            for p in tracker.update(fits, stamp_ns=stamp, tol_m=_RES, scanned=column)
        }

    first = frame(_props(), 1)
    assert sorted(first) == [0, 1, 2]
    assert frame(_props(), 2).keys() == first.keys(), "a still scene keeps every identity"
    # The middle prop nudged by one cell: the same identity, its box refreshed.
    x, y = _PROPS_XY[1]
    nudged = [_PROPS_XY[0], (x + _RES, y), _PROPS_XY[2]]
    third = frame(_props(nudged), 3)
    assert third.keys() == first.keys()
    moved = next(pid for pid, r in first.items() if r == _at(list(first.values()), y))
    assert third[moved].pose.xyz[0] == pytest.approx(first[moved].pose.xyz[0] + _RES)
    assert third[moved].stamp_ns == 7  # the fit's stamp, from the grid it was taken from
    # The middle prop taken away: one miss keeps it, the second drops it (the map cleared
    # its cells); the other two are untouched.
    gone = [_PROPS_XY[0], _PROPS_XY[2]]
    assert sorted(frame(_props(gone), 4)) == sorted(set(first) - {moved})
    assert sorted(p.primitive_id for p in tracker.primitives) == sorted(first)
    frame(_props(gone), 5)
    assert sorted(p.primitive_id for p in tracker.primitives) == sorted(set(first) - {moved})
    # Put back 12 cm away: a new identity (its centre lies in no remembered box).
    far = [_PROPS_XY[0], (x, y + 0.12), _PROPS_XY[2]]
    assert max(frame(_props(far), 6)) == 3
    # A primitive outside the scanned column is not a miss: scanning elsewhere keeps it.
    narrow = _column(*_PROPS_XY[0], half_xy=0.1)
    fits, _ = _fits(_props(far), narrow)
    kept = tracker.update(fits, stamp_ns=7, tol_m=_RES, scanned=narrow)
    assert len(kept) == 1 and len(tracker.primitives) == 3
    with pytest.raises(ValueError, match="max_misses"):
        PrimitiveTracker(max_misses=0)
