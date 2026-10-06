"""Camera-first object instances, map-confirmed (``docs/reference/object-primitives-design.md``).

The scene is the committed Isaac warehouse deploy scene, numerically:
``scenes/deploy/isaac_openarm_warehouse.yaml`` spawns three YCB props on the pallet deck
(top z 0.21 in ``world``) in front of an OpenArm whose base frame sits 0.698 m above the
URDF root (``robots/openarm/robot.yaml`` ``base_to_root_xyz_rpy``), at the 15 mm cells
``deploy_e2e`` gives a sim octomap. Each prop is modelled in the map as the shell the head
camera leaves — its top layer and the wall facing the robot — over a 7 cm footprint (the
scene names positions, not sizes), and ray-cast into the manifest's ``head_zed`` for the
per-instance masks a segmenter returns (``_head_camera_render``). No mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import DeployScene, GraspDeclaration, PlaceRegion, Pose6D, RobotDescription
from openral_hal._grasp_target import (
    VoxelLattice,
    _components,
    _ijk,
    cell_closed_region,
    region_covers_occupied,
    target_region_from_masks,
)
from openral_hal._object_primitives import (
    PrimitiveFit,
    PrimitiveTracker,
    instance_prompt,
    map_confirms,
    nearest_primitive,
    point_box_gap_m,
)

from tests.unit._head_camera_render import Box, head_camera, render

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes" / "deploy" / "isaac_openarm_warehouse.yaml"
_ROBOT = _REPO / "robots" / "openarm" / "robot.yaml"
_RES = 0.015  # deploy_e2e.launch.py _octomap_resolution: sim ships 15 mm cells
_DECK_WORLD_Z = 0.21  # the scene's comment: Shelf_1 deck top
_FRAME = "openarm_base"
_FOOT = 0.07  # modelled prop footprint (the scene names positions only)
_HEIGHT = 0.12


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


def _foot(x: float, y: float, w: float = _FOOT, d: float = _FOOT, h: float = _HEIGHT) -> Box:
    """The solid box the camera sees for a ``_lattice`` entry."""
    return ((x - w / 2, y - d / 2, _DECK_Z), (x + w / 2, y + d / 2, _DECK_Z + h))


_DECK: Box = ((0.2, -0.5, _DECK_Z - 2 * _RES), (0.8, 0.5, _DECK_Z))


def _camera_fits(boxes: list[Box], *, stamp_ns: int = 7) -> list[PlaceRegion]:
    """One fit per box from its own mask (what a prompt on it returns), in box order."""
    t, k = head_camera()
    depth, labels = render([*boxes, _DECK], t, k)
    regions = []
    for n in range(1, len(boxes) + 1):
        fit = target_region_from_masks(
            [labels == n],
            depth,
            k,
            t,
            support_z=_DECK_Z,
            resolution=_RES,
            frame_id=_FRAME,
            evidence_ref=f"segment_in_view:test@{stamp_ns}",
            stamp_ns=stamp_ns,
        )
        assert fit.region is not None, (n, fit.refusal)
        regions.append(fit.region)
    return regions


def _confirmed(grid: VoxelLattice, region: PlaceRegion) -> tuple[str, str]:
    return map_confirms(grid, region, support_z=_DECK_Z, min_cover=0.5)


def _fit(region: PlaceRegion, grid: VoxelLattice) -> PrimitiveFit:
    return PrimitiveFit(region, region_covers_occupied(grid, region)[0], _DECK_Z)


def test_the_scene_fixtures_place_three_props_on_the_deck_in_reach() -> None:
    assert len(_PROPS_XY) == 3
    assert all(0.45 <= x <= 0.55 for x, _ in _PROPS_XY)
    assert pytest.approx(0.21 - 0.698) == _DECK_Z


def test_each_prop_is_one_camera_instance_the_map_confirms_within_the_kernels_contract() -> None:
    grid = _props()
    regions = _camera_fits([_foot(x, y) for x, y in _PROPS_XY])
    for region, (x, y) in zip(regions, _PROPS_XY, strict=True):
        assert _confirmed(grid, region) == ("", "")
        assert region.pose.xyz[:2] == pytest.approx((x, y), abs=_RES)
        # The camera box, its lower face one voxel above the deck, never into it (HZ-0115-6).
        assert region.pose.xyz[2] - region.half_extents[2] == pytest.approx(_DECK_Z + _RES)
        closed, clamped = cell_closed_region(region, grid, max_half_extent_m=0.2)
        assert not clamped
        GraspDeclaration(
            target_id="approach:openarm_left_finger_pair:1",
            contact_links=("openarm_left_finger_pair",),
            timeout_s=60.0,
            stamp_ns=1,
            region=closed,
        )


def test_the_instance_nearest_the_hand_is_chosen_and_a_tie_is_refused() -> None:
    regions = _camera_fits([_foot(x, y) for x, y in _PROPS_XY])
    hover = _DECK_Z + _HEIGHT + 0.10
    index, why = nearest_primitive(
        regions, [(0.5, _PROPS_XY[1][1] + 0.03, hover)], ambiguity_m=_RES
    )
    assert (index, why) == (1, "")
    index, why = nearest_primitive(regions, [(0.5, _PROPS_XY[2][1], hover)], ambiguity_m=_RES)
    assert (index, why) == (2, "")
    # Midway between the two boxes' facing sides (each fit is its own camera view's).
    between = (
        regions[1].pose.xyz[1]
        + regions[1].half_extents[1]
        + regions[2].pose.xyz[1]
        - regions[2].half_extents[1]
    ) / 2
    index, why = nearest_primitive(regions, [(_PROPS_XY[1][0], between, hover)], ambiguity_m=_RES)
    assert index is None and why.startswith("ambiguous")
    # A named search box (no hand) is unambiguous only with one instance in it.
    assert nearest_primitive(regions[:1], [], ambiguity_m=_RES) == (0, "")
    assert nearest_primitive(regions, [], ambiguity_m=_RES)[0] is None
    assert nearest_primitive([], [(0.5, 0.0, hover)], ambiguity_m=_RES)[0] is None
    top = regions[1].pose.xyz[2] + regions[1].half_extents[2]
    assert point_box_gap_m((*_PROPS_XY[1], top + 0.10), regions[1]) == pytest.approx(0.10)


def test_a_packed_pair_the_map_merges_is_two_instances_from_two_masks() -> None:
    """Two cartons packed face to face: one 26-connected blob in the map (HZ-0115-30),
    which the map alone could only exempt whole. The prompts land one on each carton (the
    second only once the first's instance covers its top), each mask fits its own carton,
    the map confirms both (each box holds one body: its own carton and the touching sliver
    of the other), and the tracker keeps two identities."""
    x, y = _PROPS_XY[1]
    pair = [(x, y - _FOOT / 2), (x, y + _FOOT / 2)]
    grid = _props(pair)
    centers = grid.occupied_centers()
    above = centers[centers[:, 2] > _DECK_Z + _RES]
    assert len(_components(_ijk(grid, above))) == 1, "the map holds the pair as one body"

    column = _column(x, y, half_xy=0.2)
    hand = (x, y - 0.25)
    first = instance_prompt(grid, column, [], near_xy=hand)
    assert first is not None and first[0][1] < y - _RES, "the first prompt is on the near carton"
    near, far = _camera_fits([_foot(*xy) for xy in pair])
    second = instance_prompt(grid, column, [near], near_xy=hand)
    assert second is not None and second[0][1] > y + _RES, "then on the other one"
    assert instance_prompt(grid, column, [near, far], near_xy=hand) is None

    for region, (_, cy) in zip((near, far), pair, strict=True):
        assert _confirmed(grid, region) == ("", "")
        assert region.pose.xyz[1] == pytest.approx(cy, abs=_RES)
        assert region.half_extents[1] < _FOOT, "one carton, not the pair"
    tracker = PrimitiveTracker()
    ids = [
        p.primitive_id
        for region in (near, far)
        for p in tracker.update([_fit(region, grid)], stamp_ns=7, tol_m=_RES, scanned=region)
    ]
    assert ids == [0, 1] and len(tracker.primitives) == 2


def test_the_map_refuses_a_box_it_does_not_hold_or_that_holds_two_bodies() -> None:
    x, y = _PROPS_XY[1]
    (region,) = _camera_fits([_foot(x, y)])
    gone = _props([_PROPS_XY[0], _PROPS_XY[2]])  # the prop taken away: its cells cleared
    kind, why = _confirmed(gone, region)
    assert kind == "map_disagrees" and "occupied cells" in why
    # Two props 3 cm apart and a box over both (a mask that took in the pair): two
    # bodies in one box is never one instance.
    apart = _props([(x, y - 0.05), (x, y + 0.05)])
    both = region.model_copy(update={"half_extents": (0.05, 0.10, region.half_extents[2])})
    assert _confirmed(apart, both)[0] == "map_split"
    with pytest.raises(Exception, match="frame"):
        _confirmed(apart, both.model_copy(update={"frame_id": "map"}))


def test_the_tracker_keeps_identities_across_fits_and_drops_stale_and_cleared() -> None:
    grid = _props()
    tracker = PrimitiveTracker(max_misses=2, max_tracks=3)

    def fold(regions: list[PlaceRegion], stamp: int) -> list[int]:
        return [
            p.primitive_id
            for r in regions
            for p in tracker.update([_fit(r, grid)], stamp_ns=stamp, tol_m=_RES, scanned=r)
        ]

    boxes = [_foot(x, y) for x, y in _PROPS_XY]
    assert fold(_camera_fits(boxes, stamp_ns=1), 1) == [0, 1, 2]
    # The next captures: the same objects keep their identities, and their boxes and
    # stamps are the new fit's (``last_seen_ns`` is the instance's camera confirmation).
    x, y = _PROPS_XY[1]
    nudged = [boxes[0], _foot(x + _RES, y), boxes[2]]
    assert fold(_camera_fits(nudged, stamp_ns=2), 2) == [0, 1, 2]
    middle = next(p for p in tracker.primitives if p.primitive_id == 1)
    assert middle.last_seen_ns == 2 and middle.first_seen_ns == 1
    assert middle.region.pose.xyz[0] == pytest.approx(x + _RES, abs=_RES / 2)
    # Seen elsewhere 12 cm away: a new identity; past ``max_tracks`` the one seen
    # longest ago goes (bounded work).
    far = _camera_fits([_foot(x, y + 0.12)], stamp_ns=3)
    assert fold(far, 3) == [3]
    assert sorted(p.primitive_id for p in tracker.primitives) == [1, 2, 3]
    # Staleness and map contradiction are the leg's ``prune`` predicates: not re-fitted
    # since stamp 2, or no longer held by the map (its cells cleared).
    cleared = _props([_PROPS_XY[0], _PROPS_XY[2]])
    dropped = tracker.prune(lambda p: region_covers_occupied(cleared, p.region)[1])
    assert sorted(p.primitive_id for p in dropped) == [1, 3]
    assert [p.primitive_id for p in tracker.prune(lambda p: p.last_seen_ns > 2)] == [2]
    assert tracker.primitives == []
    with pytest.raises(ValueError, match="max_misses"):
        PrimitiveTracker(max_misses=0)
