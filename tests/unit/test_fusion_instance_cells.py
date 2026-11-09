"""EXPERIMENT exp/vision-voxel-fusion: a track's box fits the voxels vision labels its own."""

import math

import numpy as np
from openral_core import PlaceRegion, Pose6D, TrackStatus
from openral_world_state.object_tracker.tracker import _RECENT_S, _VISION_NEED_FRACTION

from tests.unit._tracker_scenes import cylinder, grid_from_depth, render
from tests.unit.test_object_tracker_cover import _CAN, _RIG, _confirmed_tracker


def _labelled(dx: float, radius: float = 0.035):
    """The grid with the can moved ``dx``, and its occupied cells above the table (the labels)."""
    x, y = _RIG.centre_xy
    depth, _ = render([cylinder((x + dx, y), radius, _RIG.support_z, _RIG.support_z + 0.1)], _RIG)
    grid = grid_from_depth(depth, _RIG)
    c = grid.occupied_centers()
    near = (np.hypot(c[:, 0] - (x + dx), c[:, 1] - y) <= radius + grid.resolution) & (
        c[:, 2] > _RIG.support_z + grid.resolution
    )
    local = (c[near] - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
    return grid, frozenset(tuple(v) for v in np.floor(local).astype(np.int64).tolist())


def test_the_box_follows_the_cells_vision_labels_its_own():
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    grid, cells = _labelled(0.06)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    moved = tr.store.get(t.object_id)
    assert moved.status is TrackStatus.CONFIRMED
    assert abs(moved.box.pose.xyz[0] - (t.box.pose.xyz[0] + 0.06)) < 0.02


def test_cells_far_larger_than_the_object_are_refused():
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    grid, cells = _labelled(0.06, radius=0.12)  # a 24 cm "can": something else
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 0
    assert tr.store.get(t.object_id).box.pose.xyz == t.box.pose.xyz


def test_an_unmoved_object_keeps_its_fitted_box_and_is_only_re_dated():
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    grid, cells = _labelled(0.01)  # within two cells: the same object, edge noise
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 0
    kept = tr.store.get(t.object_id)
    assert kept.box.pose.xyz == t.box.pose.xyz and kept.last_confirmed_stamp_ns == 10**9


def test_a_released_target_the_mask_still_follows_is_minted_again():
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    tr._map_cells_max[t.object_id] = tr._map_cells.get(t.object_id, 20)
    tr.store.replace(t.model_copy(update={"status": TrackStatus.RELEASED}))
    tr._let_go[t.object_id] = (t, 10, 0)  # what the released pass records
    grid, cells = _labelled(0.02)  # it slipped 2 cm: not new occupancy, but vision knows it
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    live = [o for o in tr.store.all() if o.status is TrackStatus.CONFIRMED]
    assert len(live) == 1 and live[0].object_id != t.object_id
    assert abs(live[0].box.pose.xyz[0] - (t.box.pose.xyz[0] + 0.02)) < 0.02


# -- regressions from the Spark pro#51 fusion runs (f1-f10), one per real failure ------------


def _released(tr, t, ref: int, since_ns: int = 0):
    """``t`` let go: what the released pass records (``_note_release``)."""
    tr.store.replace(t.model_copy(update={"status": TrackStatus.RELEASED}))
    tr._let_go[t.object_id] = (t, ref, since_ns)


def _second(tr, t, dx: float, scale: float = 1.0):
    """A second confirmed track of ``t``'s label and role, ``dx`` along x, ``scale`` x as wide."""
    h = t.box.half_extents
    box = t.box.model_copy(
        update={
            "pose": t.box.pose.model_copy(
                update={"xyz": (t.box.pose.xyz[0] + dx, *t.box.pose.xyz[1:])}
            ),
            "half_extents": (scale * h[0], scale * h[1], h[2]),
        }
    )
    fields = {
        k: getattr(t, k) for k in type(t).model_fields if k not in ("object_id", "epoch", "status")
    }
    fields["box"] = box
    o = tr.store.mint(**fields)
    tr.store.replace(o.model_copy(update={"status": TrackStatus.CONFIRMED}))
    return o


def test_a_lost_target_the_mask_still_follows_is_confirmed_where_it_is():
    """f5 (yw2 ep4): the map lost an orange the jaw pushed; its mask still followed it."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    tr.store.replace(t.model_copy(update={"status": TrackStatus.LOST}))
    grid, cells = _labelled(0.06)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    back = tr.store.get(t.object_id)
    assert back.status is TrackStatus.CONFIRMED
    assert abs(back.box.pose.xyz[0] - (t.box.pose.xyz[0] + 0.06)) < 0.02


def test_a_dropped_object_showing_a_third_of_its_cells_is_found_and_a_fifth_is_not():
    """f7 (ywb1 ep4): a dropped banana showed 18-23 of its 59 cells, under the map's 50 %."""
    grid, cells = _labelled(0.04)
    n = len(cells)
    for ref, found in ((math.floor(n / 0.35), 1), (math.ceil(n / 0.2), 0)):
        tr, _ = _confirmed_tracker(_CAN)
        t = tr.store.all()[0]
        _released(tr, t, ref)
        assert (n >= math.ceil(_VISION_NEED_FRACTION * ref)) is bool(found)
        assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == found, ref


def test_a_released_target_is_found_once_and_only_within_the_window():
    """f3/f4: one re-find per release (a second would mint a twin), never past ``recent_s``."""
    grid, cells = _labelled(0.02)
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    _released(tr, t, 10)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    assert tr._let_go[t.object_id] is None, "used: the twin is refused by the registry itself"
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 2 * 10**9, ()) == 0
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    _released(tr, t, 10)
    late = int((_RECENT_S + 1.0) * 1e9)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, late, ()) == 0
    assert tr._let_go[t.object_id] is None


def test_cells_inside_another_target_never_move_a_track_onto_it():
    """f1 (yf1 ep8): a track's mask bled onto the neighbour orange; the box must not follow."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    _second(tr, t, 0.06)
    grid, cells = _labelled(0.06)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 0
    assert tr.store.get(t.object_id).box.pose.xyz == t.box.pose.xyz


def test_an_implausible_track_never_blocks_a_real_one():
    """f6 (yw1 ep8): a 32 cm "orange" over two real ones held their cells as its own."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    big = _second(tr, t, 0.06, scale=3.0)
    assert tr._implausible_box(big.label, big.box, big.object_id)
    grid, cells = _labelled(0.06)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1


def test_the_hidden_side_grows_toward_the_hand_to_the_objects_size():
    """f10 (yw3 ep7, ywb2 ep3): vision sees only what the hand leaves visible; a box of those
    cells stopped the gripper on the orange's hidden side 2.3 cm outside it."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    size = tr._clear_size[t.object_id]
    grid, cells = _labelled(0.06)
    x_mid = np.median([c[0] for c in cells])
    near = frozenset(c for c in cells if c[0] <= x_mid)  # the far half is behind the hand
    hand_x = t.box.pose.xyz[0] + 0.06 + 0.06
    hand = PlaceRegion(
        frame_id=t.box.frame_id,
        half_extents=(0.02, 0.02, 0.02),
        pose=Pose6D(xyz=(hand_x, *t.box.pose.xyz[1:]), quat_xyzw=(0, 0, 0, 1),
                    frame_id=t.box.frame_id),
    )  # fmt: skip
    tr._map_cells[t.object_id] = len(near)
    assert tr.apply_instance_cells({t.object_id: near}, grid, 10**9, (hand,)) == 1
    box = tr.store.get(t.object_id).box
    lo_x = box.pose.xyz[0] - box.half_extents[0]
    hi_x = box.pose.xyz[0] + box.half_extents[0]
    assert hi_x - lo_x >= size - 1e-9, "never narrower than the object"
    seen_hi = grid.origin[0] + (max(c[0] for c in near) + 1) * grid.resolution
    assert hi_x > seen_hi + 0.5 * grid.resolution, "grown toward the hand, where it is hidden"


def test_a_slipped_target_is_found_on_the_table_though_its_released_box_is_in_the_hand():
    """A/B ab1b_po seeds 2/3: a released track keeps its in-hand box and support (7.5 cm up),
    and every re-find of the orange lying on the table kept 0 of its labelled cells."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    up = 0.06
    in_hand = t.model_copy(update={
        "support_z": t.support_z + up,
        "box": t.box.model_copy(update={"pose": t.box.pose.model_copy(
            update={"xyz": (t.box.pose.xyz[0], t.box.pose.xyz[1], t.box.pose.xyz[2] + up)})}),
    })  # fmt: skip
    _released(tr, in_hand, 10)
    grid, cells = _labelled(0.02)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    [found] = [o for o in tr.store.all() if o.status is TrackStatus.CONFIRMED]
    bottom = found.box.pose.xyz[2] - found.box.half_extents[2]
    assert bottom < t.support_z + 2 * grid.resolution, "it stands on the table, not in the air"


def test_a_refind_on_only_the_upper_cells_keeps_the_release_support():
    """Review of afff4f9be: a mask labelling only cells well above the release support lifted
    support_z above the kept floor_z and ``store.mint`` raised, killing the tracker node."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    _released(tr, t, 10)
    grid, cells = _labelled(0.02)
    k0 = min(c[2] for c in cells)
    top = frozenset(c for c in cells if c[2] >= k0 + 3)
    assert tr.apply_instance_cells({t.object_id: top}, grid, 10**9, ()) == 1
    [found] = [o for o in tr.store.all() if o.status is TrackStatus.CONFIRMED]
    assert found.support_z <= t.support_z and found.floor_z >= found.support_z


def test_a_vision_move_drops_the_hulls_posed_for_the_old_box():
    """Review of afff4f9be: a moved box kept its old tight hulls; the bind tested the pinch
    against where the object had been."""
    tr, _ = _confirmed_tracker(_CAN)
    t = tr.store.all()[0]
    assert t.tight, "the clear fit left hulls to drop"
    grid, cells = _labelled(0.06)
    assert tr.apply_instance_cells({t.object_id: cells}, grid, 10**9, ()) == 1
    assert tr.store.get(t.object_id).tight == ()
