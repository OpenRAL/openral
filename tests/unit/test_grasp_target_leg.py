"""The pre-grasp target producer leg's state machine (real pick-and-place design §2.2).

``GraspTargetTracker`` decides what the attachment envelope says about the grasp
declaration: dispatch's fields verbatim, plus a measured region only while it is
backed — retracted at once on contradicting evidence, frozen under a TTL on a lost
view, gone with the declaration, kept (but no longer re-measured) after the
declared gripper attaches. Real ``GraspDeclaration`` from the committed OpenArm
cell scene, real OpenArm manifest (CLAUDE.md §1.11). The ROS wiring is covered by
``tests/integration/test_grasp_target_leg_live.py``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import (
    AttachedCollisionObject,
    DeployScene,
    GraspDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_target import VoxelLattice, occupied_centers_in_box
from openral_hal._grasp_target_leg import (
    APPROACH_TARGET_PREFIX,
    GraspTargetLeg,
    GraspTargetTracker,
    _gate_refit,
    _Refusal,
    _refused_fit,
    approach_box,
    search_column,
)
from openral_hal.vision_attachment_bridge import (
    VisionAttachmentBridge,
    VisionAttachmentConfig,
    region_attachment,
)

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "tests" / "unit" / "fixtures" / "scenes" / "openarm_direct_dispatch_grasp.yaml"
_ROBOT = _REPO / "robots" / "openarm" / "robot.yaml"
_S = 1_000_000_000
_FREEZE_S = 2.0


def _CONFIRMED(_region: PlaceRegion) -> bool:  # noqa: N802 — reads as a constant at call sites
    """``on_attach``'s confirm for a payload that is the held region (the region path)."""
    return True


def _box(*, x: float = 0.45, z: float = 0.10, half: float = 0.15, stamp_ns: int = 0) -> PlaceRegion:
    return PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(x, 0.0, z), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"),
        half_extents=(half, half, 0.10 if half > 0.1 else half),
        evidence_ref="segment_in_view:test@0",
        stamp_ns=stamp_ns,
    )


def _dispatched(stamp_ns: int = 10 * _S) -> GraspDeclaration:
    """The scene's declaration as the runner arms it, plus a search box."""
    scene = DeployScene.from_yaml(str(_SCENE)).grasp_declaration
    assert scene is not None
    return scene.model_copy(
        update={
            "rskill_id": "openral/pi05-openarm-restock",
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "stamp_ns": stamp_ns,
            "search_box": _box(),
        }
    )


def _tracker() -> tuple[GraspTargetTracker, list[str]]:
    lines: list[str] = []
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=lines.append)
    tracker.on_declaration(_dispatched())
    return tracker, lines


def _measured(stamp_ns: int) -> PlaceRegion:
    return _box(half=0.04, z=0.09, stamp_ns=stamp_ns)


def test_no_declaration_no_envelope() -> None:
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append)
    assert tracker.envelope(now_ns=_S) is None
    assert not tracker.wants_measurement(now_ns=_S)


def test_the_envelope_keeps_dispatch_fields_verbatim_and_adds_the_region() -> None:
    tracker, _ = _tracker()
    dispatched = _dispatched()
    before = tracker.envelope(now_ns=11 * _S)
    assert before == dispatched  # no region until one is measured
    tracker.accept(_measured(11 * _S))
    after = tracker.envelope(now_ns=11 * _S)
    assert after is not None
    assert after.region == _measured(11 * _S)
    assert after.model_copy(update={"region": None}) == dispatched


def test_contradicting_evidence_retracts_at_once_and_logs_once() -> None:
    tracker, lines = _tracker()
    tracker.accept(_measured(11 * _S))
    for _ in range(5):
        tracker.refuse("ambiguous", "cluster sizes [12, 11]", retract=True, now_ns=11 * _S)
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope is not None and envelope.region is None
    assert sum("retracted — ambiguous" in line for line in lines) == 1


def test_a_lost_view_freezes_the_region_until_its_ttl() -> None:
    tracker, lines = _tracker()
    tracker.accept(_measured(11 * _S))
    tracker.refuse("seed_off_image", "occluded", retract=False, now_ns=12 * _S)
    held = tracker.envelope(now_ns=12 * _S)
    assert held is not None and held.region is not None
    tracker.refuse("seed_off_image", "occluded", retract=False, now_ns=13 * _S - 1)
    assert tracker.region is not None, "frozen region dropped before its TTL"
    gone = tracker.envelope(now_ns=13 * _S + 1)  # stamp + 2 s, no refusal needed
    assert gone is not None and gone.region is None
    assert any("freeze_ttl" in line for line in lines)


def _held_block_lattice() -> VoxelLattice:
    """20 mm cells filling exactly ``_measured``'s box (x 0.41-0.49, |y| <= 0.04, z 0.05-0.13)."""
    return VoxelLattice(
        "openarm_base",
        (0.41, -0.04, 0.05),
        (0.0, 0.0, 0.0, 1.0),
        0.02,
        (4, 4, 4),
        np.ones(64, dtype=np.uint8),
    )


def _refit(x: float, half: tuple[float, float, float], z: float = 0.09) -> PlaceRegion:
    return _measured(12 * _S).model_copy(
        update={
            "pose": Pose6D(xyz=(x, 0.0, z), quat_xyzw=(0, 0, 0, 1), frame_id="openarm_base"),
            "half_extents": half,
        }
    )


#: tf origins of the declared finger link: just above the held box (its top is z=0.13),
#: and 30 cm above it — reaching for something else, or not moving at all.
_HAND_NEAR = (0.47, 0.02, 0.18)
_HAND_FAR = (0.45, 0.0, 0.45)


def _gate(
    region: PlaceRegion,
    held: PlaceRegion,
    *,
    hands: tuple[tuple[float, float, float], ...] = (_HAND_NEAR,),
    lattice: VoxelLattice | None = None,
) -> _Refusal:
    with pytest.raises(_Refusal) as caught:
        _gate_refit(lattice or _held_block_lattice(), region, held, min_cover=0.5, hands=hands)
    return caught.value


def test_a_refit_shrunk_inside_the_held_region_with_the_hand_at_it_freezes() -> None:
    """The hand covers part of the target: the fit shrinks and shifts 25 mm (past the
    one-voxel tracking gate) but stays inside the held box, the map still holds the
    target and the declared finger link is right at it — a lost view, not a move."""
    tracker, lines = _tracker()
    held = _measured(11 * _S)
    tracker.accept(held)
    refusal = _gate(_refit(0.425, (0.015, 0.04, 0.03)), held)
    assert (refusal.kind, refusal.retract) == ("occluded_refit", False)
    tracker.refuse(refusal.kind, refusal.detail, retract=refusal.retract, now_ns=12 * _S)
    assert tracker.region == held, "the partial fit must not replace the held region"
    assert any("view lost — occluded_refit" in line for line in lines)
    gone = tracker.envelope(now_ns=13 * _S + 1)  # the freeze TTL still bounds it
    assert gone is not None and gone.region is None


def test_a_refit_shrunk_inside_the_held_region_with_no_hand_near_retracts() -> None:
    """Nothing of the robot's is at the target (a person's hand, the target tipped):
    the same shrink is unexplained and retracts at once, as does an unlocated hand."""
    tracker, lines = _tracker()
    held = _measured(11 * _S)
    tracker.accept(held)
    shrunk = _refit(0.425, (0.015, 0.04, 0.03))
    far = _gate(shrunk, held, hands=(_HAND_FAR,))
    assert (far.kind, far.retract) == ("unoccluded_refit", True)
    tracker.refuse(far.kind, far.detail, retract=far.retract, now_ns=12 * _S)
    assert tracker.region is None
    assert any("retracted — unoccluded_refit" in line for line in lines)
    assert _gate(shrunk, held, hands=()).kind == "unoccluded_refit"


def test_a_refit_the_map_does_not_cover_retracts_even_with_the_hand_at_it() -> None:
    held = _measured(11 * _S)
    empty = VoxelLattice(
        "openarm_base",
        (0.41, -0.04, 0.05),
        (0.0, 0.0, 0.0, 1.0),
        0.02,
        (4, 4, 4),
        np.zeros(64, dtype=np.uint8),
    )
    gone = _gate(_refit(0.425, (0.015, 0.04, 0.03)), held, lattice=empty)
    assert (gone.kind, gone.retract) == ("map_disagrees", True)


def test_a_refit_reaching_outside_the_held_region_retracts() -> None:
    tracker, lines = _tracker()
    held = _measured(11 * _S)
    tracker.accept(held)
    moved = _gate(_refit(0.50, (0.04, 0.04, 0.04)), held)  # 50 mm past the held face
    assert (moved.kind, moved.retract) == ("target_moved", True)
    tracker.refuse(moved.kind, moved.detail, retract=moved.retract, now_ns=12 * _S)
    assert tracker.region is None
    assert any("retracted — target_moved" in line for line in lines)
    elsewhere = _gate(_refit(0.70, (0.04, 0.04, 0.04)), held)  # nothing in the map there
    assert (elsewhere.kind, elsewhere.retract) == ("map_disagrees", True)


def test_a_hand_hovering_over_the_target_within_the_approach_distance_occludes() -> None:
    """The hand point is the jaw's TCP (the finger hinge), a finger length above the
    fingertips: waiting over the target it sits 12 cm over the held top while it shadows
    part of the box from the head camera (Isaac i28: the shrunken re-fit retracted as
    ``unoccluded_refit``). Over the held footprint within the approach distance that
    armed the target, it is the robot's own hand — not a person's."""
    held = _measured(11 * _S)
    shrunk = _refit(0.425, (0.015, 0.04, 0.03))
    hover = (0.45, 0.0, 0.25)

    def gate(rise: float, hand: tuple[float, float, float] = hover) -> _Refusal:
        with pytest.raises(_Refusal) as caught:
            _gate_refit(
                _held_block_lattice(), shrunk, held, min_cover=0.5, hands=(hand,), hand_rise_m=rise
            )
        return caught.value

    assert (gate(0.20).kind, gate(0.20).retract) == ("occluded_refit", False)
    assert gate(0.0).kind == "unoccluded_refit"
    assert gate(0.20, (0.60, 0.0, 0.25)).kind == "unoccluded_refit"  # beside it, not over it
    # A re-fit reaching outside the held region stays a move, hand or not.
    assert _gate(_refit(0.50, (0.04, 0.04, 0.04)), held).kind == "target_moved"


def test_a_refused_refit_with_the_hand_over_the_held_target_is_a_lost_view() -> None:
    """With the fingers removed from the fit, the hand still OCCLUDES the target's lower
    part from the head camera as it closes: what is left of the box no longer reaches the
    support and the re-fit is refused (``not_on_support``, Isaac i25). With a region held
    and the hand over it, that is a lost view (held region kept under its TTL); with
    none held yet, or the hand elsewhere, it is the contradiction it names."""
    held = _measured(11 * _S)

    def refused(prev: PlaceRegion | None, hand: tuple[float, float, float]) -> _Refusal:
        return _refused_fit(
            "not_on_support", "points=900", prev, (hand,), reach_m=0.07, hand_rise_m=0.20
        )

    at = refused(held, (0.45, 0.0, 0.25))
    assert (at.kind, at.retract) == ("hand_at_target", False)
    assert "not_on_support" in at.detail
    assert refused(held, _HAND_FAR).kind == "not_on_support"
    assert refused(None, (0.45, 0.0, 0.25)).kind == "not_on_support"
    few = _refused_fit("too_few_points", "points=3", None, (), reach_m=0.07, hand_rise_m=0.2)
    assert (few.kind, few.retract) == ("too_few_points", False)


def test_a_refit_within_the_tracking_gate_replaces_the_held_region() -> None:
    held = _measured(11 * _S)
    nudged = _refit(0.46, (0.04, 0.04, 0.04))
    assert _gate_refit(_held_block_lattice(), nudged, held, min_cover=0.5) is nudged


def test_a_lost_view_with_nothing_held_is_a_plain_refusal() -> None:
    tracker, lines = _tracker()
    tracker.refuse("no_grid", "no grid yet", retract=False, now_ns=11 * _S)
    assert tracker.region is None
    assert any("refused — no_grid" in line for line in lines)


def test_dispatch_retraction_clears_everything() -> None:
    tracker, lines = _tracker()
    tracker.accept(_measured(11 * _S))
    tracker.on_declaration(_dispatched().model_copy(update={"active": False}))
    assert tracker.envelope(now_ns=11 * _S) is None
    assert not tracker.wants_measurement(now_ns=11 * _S)
    assert any("dispatch retracted" in line for line in lines)


def test_expiry_clears_the_declaration() -> None:
    tracker, _ = _tracker()
    tracker.accept(_measured(11 * _S))
    timeout_ns = int(_dispatched().timeout_s * _S)
    assert tracker.envelope(now_ns=10 * _S + timeout_ns + 1) is None
    assert tracker.declaration is None


def test_a_new_declaration_starts_without_a_region() -> None:
    tracker, _ = _tracker()
    tracker.accept(_measured(11 * _S))
    tracker.on_declaration(_dispatched())  # the latched copy again: kept
    assert tracker.region is not None
    tracker.on_declaration(_dispatched(stamp_ns=20 * _S))
    assert tracker.region is None


def test_attach_on_a_declared_jaw_link_stops_measuring_and_keeps_the_region() -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    jaws = {j.name: j.child_link for j in robot.joints if j.role == "gripper"}
    tracker, _ = _tracker()
    tracker.accept(_measured(11 * _S))
    tracker.on_attach("openarm_left_link7", confirm=_CONFIRMED)  # not a declared contact link
    assert tracker.wants_measurement(now_ns=11 * _S)
    tracker.on_attach(jaws["left_gripper"], confirm=_CONFIRMED)
    assert not tracker.wants_measurement(now_ns=11 * _S)
    late = tracker.envelope(now_ns=11 * _S + 10 * int(_FREEZE_S * _S))
    assert late is not None and late.region is not None, "handover must keep the region"


def test_a_region_over_the_declaration_caps_is_refused() -> None:
    tracker, lines = _tracker()
    tracker.accept(_box(half=0.25, stamp_ns=11 * _S))
    assert tracker.region is None
    assert any("declaration_bounds" in line for line in lines)


def test_no_search_box_means_no_measurement() -> None:
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append)
    tracker.on_declaration(_dispatched().model_copy(update={"search_box": None}))
    assert not tracker.wants_measurement(now_ns=11 * _S)


def test_the_support_column_reaches_below_the_box_and_must_be_level() -> None:
    column = search_column(_box(z=0.10), below_m=0.15)
    assert column.pose.xyz[2] - column.half_extents[2] == pytest.approx(-0.15)
    assert column.pose.xyz[2] + column.half_extents[2] == pytest.approx(0.20)
    tilted = _box().model_copy(
        update={
            "pose": Pose6D(
                xyz=(0.45, 0.0, 0.1), quat_xyzw=(0.0, 0.2588, 0.0, 0.9659), frame_id="openarm_base"
            )
        }
    )
    with pytest.raises(ROSConfigError, match="gravity-aligned"):
        search_column(tilted, below_m=0.15)


def test_the_freeze_ttl_derives_from_the_deploys_grid_age_unless_set() -> None:
    """The kernel's voxel deadline (``grid_max_age_s``) sets the freeze: twice it."""
    robot = RobotDescription.from_yaml(str(_ROBOT))

    def leg(**kwargs: float) -> GraspTargetLeg:
        bridge = VisionAttachmentBridge(
            None,
            robot,
            config=VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True, **kwargs),
        )
        assert bridge._grasp_target is not None
        return bridge._grasp_target

    assert leg()._freeze_s == pytest.approx(2.0)  # the real cell's 1.0 s fallback
    assert leg(grid_max_age_s=0.4)._freeze_s == pytest.approx(0.8)
    assert leg(grid_max_age_s=0.4, grasp_target_freeze_s=0.5)._freeze_s == pytest.approx(0.5)
    with pytest.raises(ROSConfigError, match="grid_max_age_s"):
        leg(grid_max_age_s=0.0)
    # The kernel ages a grasp region out at grasp_region_max_age_s = 2 x its voxel
    # deadline (deploy passes 2 x world_voxel_deadline_s; grid_max_age_s is that
    # deadline): a longer freeze would publish a region the kernel already refuses.
    assert leg(grid_max_age_s=0.4, grasp_target_freeze_s=0.8)._freeze_s == pytest.approx(0.8)
    with pytest.raises(ROSConfigError, match="grasp_target_freeze_s"):
        leg(grid_max_age_s=0.4, grasp_target_freeze_s=0.81)
    with pytest.raises(ROSConfigError, match="grasp_target_freeze_s"):
        leg(grid_max_age_s=0.4, grasp_target_freeze_s=1.6)  # the old 4x ceiling


def test_the_bridge_owns_the_leg_only_when_enabled_and_validates_its_rate() -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    off = VisionAttachmentBridge(None, robot, config=VisionAttachmentConfig(camera="head_zed"))
    assert off._grasp_target is None
    on = VisionAttachmentBridge(
        None, robot, config=VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
    )
    assert on._grasp_target is not None
    with pytest.raises(ROSConfigError, match="2-5 Hz"):
        VisionAttachmentBridge(
            None,
            robot,
            config=VisionAttachmentConfig(
                camera="head_zed", grasp_target_enabled=True, grasp_target_rate_hz=10.0
            ),
        )


def test_the_support_tunables_are_constructor_args_with_documented_defaults() -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
    bridge = VisionAttachmentBridge(None, robot, config=config)
    default = GraspTargetLeg(None, bridge, config)
    assert (default._search_below_m, default._probe_margin_m) == (0.15, 0.05)
    assert default._occluder_margin_m == 0.05
    wide = GraspTargetLeg(
        None, bridge, config, support_search_below_m=0.30, support_probe_margin_m=0.08
    )
    assert (wide._search_below_m, wide._probe_margin_m) == (0.30, 0.08)
    with pytest.raises(ROSConfigError, match="support_probe_margin_m"):
        GraspTargetLeg(None, bridge, config, support_probe_margin_m=0.0)
    with pytest.raises(ROSConfigError, match="support_search_below_m"):
        GraspTargetLeg(None, bridge, config, support_search_below_m=-0.1)
    with pytest.raises(ROSConfigError, match="occluder_margin_m"):
        GraspTargetLeg(None, bridge, config, occluder_margin_m=-0.01)
    # Upper bounds that keep the gates meaningful.
    assert GraspTargetLeg(None, bridge, config, occluder_margin_m=0.10)._occluder_margin_m == 0.10
    with pytest.raises(ROSConfigError, match="occluder_margin_m"):
        GraspTargetLeg(None, bridge, config, occluder_margin_m=0.11)
    assert GraspTargetLeg(None, bridge, config, support_search_below_m=0.5)._search_below_m == 0.5
    with pytest.raises(ROSConfigError, match="support_search_below_m"):
        GraspTargetLeg(None, bridge, config, support_search_below_m=0.6)


# ── The target must stand on the measured support ────────────────────────────────

_CELLS = (24, 24, 20)
_ITEM_I = _ITEM_J = range(8, 13)  # a 10x10 cm item, k=11-15, on a board at k=9-10


def _cells(i: range, j: range, k: range) -> set[tuple[int, int, int]]:
    return {(a, b, c) for a in i for b in j for c in k}


def _shelf_lattice(board_i: range, board_j: range) -> VoxelLattice:
    """A 20 mm base-aligned octomap: bench (k=5), a board (k=9-10), the item's visible
    shell; the bench under the board and in the item's +x shadow is hidden."""
    item = _cells(_ITEM_I, _ITEM_J, range(15, 16))
    for k in range(11, 16):
        item |= _cells(range(8, 9), _ITEM_J, range(k, k + 1))
        item |= _cells(_ITEM_I, range(8, 9), range(k, k + 1))
        item |= _cells(_ITEM_I, range(12, 13), range(k, k + 1))
    board = _cells(board_i, board_j, range(9, 11))
    bench = _cells(range(24), range(24), range(5, 6))
    bench -= _cells(board_i, board_j, range(5, 6)) | _cells(range(13, 18), _ITEM_J, range(5, 6))
    occ = np.zeros(int(np.prod(_CELLS)), dtype=np.uint8)
    for a, b, c in item | board | bench:
        occ[a + _CELLS[0] * (b + _CELLS[1] * c)] = 1
    return VoxelLattice("openarm_base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, _CELLS, occ)


def _item_search_box() -> PlaceRegion:
    return PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(0.21, 0.21, 0.28), quat_xyzw=(0, 0, 0, 1), frame_id="openarm_base"),
        half_extents=(0.12, 0.12, 0.08),
    )


def _leg() -> GraspTargetLeg:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
    return GraspTargetLeg(None, VisionAttachmentBridge(None, robot, config=config), config)


def test_a_target_on_a_visible_shelf_board_seeds_on_the_board() -> None:
    grid = _shelf_lattice(range(7, 24), range(24))
    point, support_z = _leg()._seed(grid, _item_search_box())
    assert support_z == pytest.approx(0.22)  # the board's top face
    assert point[2] == pytest.approx(0.32)


def test_a_shelf_board_edge_over_a_bench_is_not_the_targets_support() -> None:
    """The board is barely wider than the item and its lip hidden: no ring there, so the
    first surface found is the bench 10 cm below — the target does not stand on it."""
    grid = _shelf_lattice(range(7, 14), range(7, 14))
    with pytest.raises(_Refusal) as caught:
        _leg()._seed(grid, _item_search_box())
    assert (caught.value.kind, caught.value.retract) == ("not_on_support", True)
    assert "support z=0.120" in caught.value.detail


def test_the_bridge_hands_its_support_and_occluder_tunables_to_the_target_leg() -> None:
    """The HAL params land in VisionAttachmentConfig; the bridge must pass them on."""
    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(
        camera="head_zed",
        grasp_target_enabled=True,
        grasp_target_support_search_below_m=0.30,
        grasp_target_support_probe_margin_m=0.08,
        grasp_target_occluder_margin_m=0.02,
    )
    leg = VisionAttachmentBridge(None, robot, config=config)._grasp_target
    assert leg is not None
    assert (leg._search_below_m, leg._probe_margin_m, leg._occluder_margin_m) == (
        0.30,
        0.08,
        0.02,
    )


def test_a_probe_margin_under_two_grid_cells_is_a_typed_lost_view() -> None:
    """The margin is a constructor arg, the resolution arrives with the grid: a coarser
    grid than the margin can ring is named, never silently widened."""
    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
    bridge = VisionAttachmentBridge(None, robot, config=config)
    leg = GraspTargetLeg(None, bridge, config, support_probe_margin_m=0.03)
    grid = _shelf_lattice(range(7, 24), range(24))
    with pytest.raises(_Refusal) as caught:
        leg._seed(grid, _item_search_box())
    assert (caught.value.kind, caught.value.retract) == ("probe_margin_under_two_cells", False)


def test_a_search_box_padded_tightly_around_the_target_still_finds_the_table() -> None:
    """An 8 cm item (x, y in 0.17-0.25 m) whose octomap shell spans the five 20 mm cells
    0.16-0.26 m, in a detection box padded only 35 mm around the item: the support ring
    (two to three cells out from the shell) lies wholly outside the box's footprint, so a
    ring counted on the column alone found no support under an ordinary target."""
    item = range(8, 13)
    cells = _cells(range(24), range(24), range(5, 6))
    cells -= _cells(range(8, 16), item, range(5, 6))  # under the item + its +x shadow
    cells |= _cells(item, item, range(10, 11))  # the visible shell: top, front, both sides
    for k in range(6, 11):
        cells |= _cells(range(8, 9), item, range(k, k + 1))
        cells |= _cells(item, range(8, 9), range(k, k + 1)) | _cells(
            item, range(12, 13), range(k, k + 1)
        )
    occ = np.zeros(int(np.prod(_CELLS)), dtype=np.uint8)
    for a, b, c in cells:
        occ[a + _CELLS[0] * (b + _CELLS[1] * c)] = 1
    grid = VoxelLattice("openarm_base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, _CELLS, occ)
    pad = 0.035
    box = PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(0.21, 0.21, 0.17), quat_xyzw=(0, 0, 0, 1), frame_id="openarm_base"),
        half_extents=(0.04 + pad, 0.04 + pad, 0.06),
    )
    column = occupied_centers_in_box(grid, search_column(box, below_m=0.15))
    assert not np.any((column[:, 2] < 0.12) & (np.abs(column[:, :2] - 0.21).max(axis=1) > 0.07)), (
        "the scene must hold the trap: no table cell of the ring inside the column"
    )
    point, support_z = _leg()._seed(grid, box)
    assert support_z == pytest.approx(0.12)  # the table's top face
    assert point[2] == pytest.approx(0.22)


def test_the_hand_is_located_through_the_attach_link_not_the_manifest_only_jaw_link() -> None:
    """The OpenArm's declared contact links (``openarm_<side>_finger_pair``) are manifest
    links, not tf frames: looked up on tf they located no hand, so every hand occlusion
    retracted. The hand is the bridge's TCP — the attach link through ``tf_frames``
    (``openarm_left_link7`` -> ``openarm_left_ee_base_link``) plus the gripper joint's
    ``origin_xyz`` — and a shrink with it at the held box is a lost view."""
    from geometry_msgs.msg import TransformStamped

    with _live_leg("test_grasp_target_jaw_point") as live:
        bridge, robot = live.bridge, live.robot
        ee = TransformStamped()
        ee.header.frame_id, ee.child_frame_id = "openarm_base", "openarm_left_ee_base_link"
        ee.transform.translation.x, ee.transform.translation.y = 0.47, 0.02
        ee.transform.translation.z = 0.25
        ee.transform.rotation.w = 1.0  # the published tree: no finger_pair frame in it
        live.buffer.set_transform_static(ee, "test")

        assert bridge._lookup("openarm_base", "openarm_left_finger_pair") is None
        (origin,) = (
            j.origin_xyz for j in robot.joints if j.child_link == "openarm_left_finger_pair"
        )
        hand = bridge.jaw_point("openarm_left_finger_pair", "openarm_base")
        assert hand == pytest.approx(np.add((0.47, 0.02, 0.25), origin))
        assert bridge.jaw_point("openarm_right_finger_pair", "openarm_base") is None  # no tf
        assert bridge.jaw_point("openarm_left_link7", "openarm_base") is None  # not a jaw

        hands = live.leg._hands(_dispatched(), "openarm_base")
        assert hands == [hand]
        held = _measured(11 * _S)
        refusal = _gate(_refit(0.425, (0.015, 0.04, 0.03)), held, hands=tuple(hands))
        assert (refusal.kind, refusal.retract) == ("occluded_refit", False)


# ── Approach-armed target (no named target: the policy picks, the hand's approach arms) ──

_LEFT = ("openarm_left_finger_pair",)
_RIGHT = ("openarm_right_finger_pair",)


def _goal_scope(stamp_ns: int = 10 * _S) -> GraspDeclaration:
    """The runner's goal-scope declaration: no target named, no search box, every hand."""
    return GraspDeclaration(
        target_id="approach",
        contact_links=_LEFT + _RIGHT,
        rskill_id="openral/pi05-openarm-restock",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=60.0,
        stamp_ns=stamp_ns,
    )


def _approach_tracker() -> tuple[GraspTargetTracker, list[str]]:
    lines: list[str] = []
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=lines.append)
    tracker.on_declaration(_goal_scope())
    return tracker, lines


def _near(*hands: tuple[str, ...]) -> list[tuple[tuple[str, ...], PlaceRegion]]:
    return [
        (hand, approach_box([_HAND_NEAR], approach_m=0.1, frame_id="openarm_base"))
        for hand in hands
    ]


_CELL = 0.02


def _approach(
    tracker: GraspTargetTracker,
    near: list[tuple[tuple[str, ...], PlaceRegion]],
    *,
    now_ns: int = 11 * _S,
) -> None:
    tracker.on_approach(near, now_ns=now_ns, move_m=_CELL)


def test_the_approach_box_spans_the_hands_tcps_grown_and_capped() -> None:
    box = approach_box([(0.4, 0.0, 0.2), (0.4, 0.1, 0.2)], approach_m=0.1, frame_id="b")
    assert box.pose.xyz == pytest.approx((0.4, 0.05, 0.2))
    assert box.half_extents == pytest.approx((0.1, 0.15, 0.1))
    assert box.pose.quat_xyzw == (0.0, 0.0, 0.0, 1.0), "gravity-aligned: a support column"
    wide = approach_box([(0.0, 0.0, 0.0), (0.0, 0.5, 0.0)], approach_m=0.2, frame_id="b")
    assert max(wide.half_extents) == GraspDeclaration.MAX_HALF_EXTENT_M
    with pytest.raises(ROSConfigError):
        approach_box([], approach_m=0.1, frame_id="b")


def test_a_goal_scope_declaration_waits_for_an_approach_and_measures_nothing() -> None:
    tracker, _ = _approach_tracker()
    assert tracker.wants_approach(now_ns=11 * _S)
    assert not tracker.wants_measurement(now_ns=11 * _S), "no box until a hand approaches"
    _approach(tracker, [])
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope == _goal_scope(), "region-less, so the kernel exempts nothing"


def test_one_approaching_hand_arms_a_one_hand_declaration_with_the_goals_attribution() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    target = tracker.target
    assert target is not None
    assert target.target_id.startswith(f"{APPROACH_TARGET_PREFIX}openarm_left_finger_pair:")
    assert target.contact_links == _LEFT, "the exemption is for the approaching hand only"
    assert (target.rskill_id, target.trace_id, target.stamp_ns, target.timeout_s) == (
        "openral/pi05-openarm-restock",
        "4bf92f3577b34da6a3ce929d0e0e4736",
        10 * _S,
        60.0,
    )
    assert target.search_box == _near(_LEFT)[0][1]
    assert tracker.wants_measurement(now_ns=11 * _S)
    tracker.accept(_measured(11 * _S))
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope is not None and envelope.region == _measured(11 * _S)
    assert envelope.contact_links == _LEFT
    assert sum("armed from the approach" in line for line in lines) == 1


def test_two_hands_approaching_at_once_arm_neither() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT, _RIGHT))
    assert tracker.target == _goal_scope()
    assert not tracker.wants_measurement(now_ns=11 * _S)
    assert any("2 hands approaching at once" in line for line in lines)


def test_the_hand_leaving_the_approach_distance_retracts_at_once() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    _approach(tracker, _near(_RIGHT))  # the left hand moved off; the other hand is not it
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope == _goal_scope(), "back to the region-less goal declaration"
    assert any("retracted — approach_ended" in line for line in lines)
    # It re-arms only from a fresh approach, from scratch (no region carried over).
    _approach(tracker, _near(_RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _RIGHT and tracker.region is None


def test_the_held_hand_keeps_its_arming_while_the_other_hand_also_approaches() -> None:
    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    _approach(tracker, _near(_LEFT, _RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _LEFT
    assert tracker.region is not None


def test_the_approach_target_dies_with_the_goal() -> None:
    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_declaration(_goal_scope().model_copy(update={"active": False}))
    assert tracker.envelope(now_ns=11 * _S) is None
    assert tracker.target is None and tracker.region is None
    # A new goal starts with no approach held.
    tracker.on_declaration(_goal_scope(stamp_ns=20 * _S))
    assert tracker.target == _goal_scope(stamp_ns=20 * _S)


def test_the_approach_target_dies_with_the_goal_timeout() -> None:
    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    assert tracker.envelope(now_ns=71 * _S) is None  # stamp 10 s + timeout 60 s


def test_a_named_search_box_wins_and_no_approach_runs() -> None:
    tracker, _ = _tracker()
    assert not tracker.wants_approach(now_ns=11 * _S)
    _approach(tracker, _near(_LEFT))
    assert tracker.target == _dispatched()


def test_a_named_hand_without_a_box_narrows_the_approach_to_that_hand() -> None:
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append)
    tracker.on_declaration(_goal_scope().model_copy(update={"contact_links": _RIGHT}))
    _approach(tracker, _near(_LEFT))
    assert not tracker.wants_measurement(now_ns=11 * _S), "the undeclared hand never arms"
    _approach(tracker, _near(_LEFT, _RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _RIGHT


def test_after_the_approach_hand_attaches_the_region_is_kept_for_the_handover() -> None:
    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_attach("openarm_right_finger_pair", confirm=_CONFIRMED)  # other hand
    assert tracker.wants_measurement(now_ns=11 * _S)
    tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
    _approach(tracker, [])  # the closed hand reads anything now; the handover rules
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope is not None and envelope.region is not None
    assert envelope.contact_links == _LEFT


def test_the_approach_distance_is_validated_and_off_by_default() -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
    bridge = VisionAttachmentBridge(None, robot, config=config)
    assert bridge._grasp_target is not None and bridge._grasp_target._approach_m is None
    on = VisionAttachmentBridge(
        None,
        robot,
        config=VisionAttachmentConfig(
            camera="head_zed", grasp_target_enabled=True, grasp_target_approach_m=0.10
        ),
    )
    assert on._grasp_target is not None and on._grasp_target._approach_m == 0.10
    assert on._grasp_target._hands_of_robot == (_LEFT, _RIGHT)
    for bad in (0.0, -0.1, GraspDeclaration.MAX_HALF_EXTENT_M + 0.01):
        with pytest.raises(ROSConfigError, match="approach_m"):
            GraspTargetLeg(None, bridge, config, approach_m=bad)


def test_the_leg_arms_the_hand_whose_tcp_is_near_occupied_cells_and_not_the_far_one() -> None:
    """Real bridge, real tf2 buffer, real lattice: the left TCP 5 cm above a block arms the
    left hand; the right TCP 30 cm off arms nothing; the left moving away retracts."""
    import time

    with _live_leg("test_grasp_target_approach") as live:
        live.place("left", (0.45, 0.0, 0.18))  # the block's top face is z = 0.13
        live.place("right", (0.45, -0.30, 0.18))
        leg = live.leg
        now_ns = leg._now_ns()
        leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
        leg.tracker.on_declaration(_goal_scope())

        leg._detect_approach(0.10, now_ns)
        target = leg.tracker.target
        assert target is not None and target.contact_links == _LEFT
        assert target.search_box is not None
        assert target.search_box.pose.xyz == pytest.approx((0.45, 0.0, 0.18))

        live.place("left", (0.45, 0.0, 0.40))  # lifted 27 cm clear of the block
        leg._detect_approach(0.10, now_ns)
        assert leg.tracker.target == _goal_scope(), "the approach ended with the hand away"


# ── grid freshness: the world data's age, not the republish's ───────────────────


def _offline_leg(**kwargs: float) -> GraspTargetLeg:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    bridge = VisionAttachmentBridge(
        None,
        robot,
        config=VisionAttachmentConfig(
            camera="head_zed", grasp_target_enabled=True, grasp_target_approach_m=0.10, **kwargs
        ),
    )
    assert bridge._grasp_target is not None
    return bridge._grasp_target


def test_a_freshly_republished_grid_of_stale_world_data_is_a_lost_view() -> None:
    """A stalled octomap: the bridge republishes (fresh receipt), ``source_stamp`` frozen."""
    import time

    leg = _offline_leg()  # freeze 2.0 s
    now_ns = 100 * _S
    leg._bridge._grid = (_held_block_lattice(), now_ns - 1 * _S, time.monotonic())
    grid, source_ns = leg._fresh_grid(now_ns)
    assert source_ns == now_ns - 1 * _S and grid.resolution == pytest.approx(0.02)
    leg._bridge._grid = (_held_block_lattice(), now_ns - 3 * _S, time.monotonic())
    with pytest.raises(_Refusal) as stale:
        leg._fresh_grid(now_ns)
    assert stale.value.kind == "grid_source_stale" and not stale.value.retract
    leg._bridge._grid = (_held_block_lattice(), 0, time.monotonic())
    with pytest.raises(_Refusal) as unknown:
        leg._fresh_grid(now_ns)
    assert unknown.value.kind == "grid_source_unknown", "unset = unknown age = stale (kernel)"


# ── handover is per hand ─────────────────────────────────────────────────────────


def test_an_attach_on_an_unarmed_hand_does_not_block_the_other_hand() -> None:
    """The goal-scope declaration names every hand; a false-positive ATTACH on the right
    hand (nothing armed for it) neither hands over nor stops the left hand arming."""
    tracker, lines = _approach_tracker()
    tracker.on_attach("openarm_right_finger_pair", confirm=_CONFIRMED)
    assert tracker.handed_over is None
    assert tracker.wants_approach(now_ns=11 * _S)
    assert any("no armed target for its hand" in line for line in lines)
    _approach(tracker, _near(_LEFT))
    target = tracker.target
    assert target is not None and target.contact_links == _LEFT
    # ... and an ATTACH on the right while the left is armed does not hand the left over.
    tracker.accept(_measured(11 * _S))
    tracker.on_attach("openarm_right_finger_pair", confirm=_CONFIRMED)
    assert tracker.handed_over is None and tracker.wants_measurement(now_ns=11 * _S)


def _pick(tracker: GraspTargetTracker, box: PlaceRegion, *, now_ns: int) -> GraspDeclaration:
    """Arm the left hand at ``box``, measure, and hand it over: one pick's ATTACH."""
    _approach(tracker, [(_LEFT, box)], now_ns=now_ns)
    tracker.accept(_measured(now_ns))
    picked = tracker.target
    assert picked is not None and picked.contact_links == _LEFT
    tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
    assert tracker.handed_over == _LEFT
    return picked


def test_a_completed_pick_re_arms_the_hand_under_a_fresh_identity() -> None:
    """Multi-pick per goal: while handed over nothing arms; the DETACH drops the region at
    once but keeps the handover; once the hand holds nothing the pick is complete and the
    hand re-arms — under a fresh ``target_id``, the goal's ``stamp_ns``, behind the backoff."""
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append, first_pick=7)
    tracker.on_declaration(_goal_scope())
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    first = _pick(tracker, here, now_ns=11 * _S)
    assert first.target_id == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:7"
    second_place = approach_box([(0.30, 0.20, 0.10)], approach_m=0.1, frame_id="openarm_base")
    for hand in (_LEFT, _RIGHT):  # handed over: nothing arms, either hand
        _approach(tracker, [(hand, second_place)], now_ns=12 * _S)
        assert tracker.target is first
    record = _released_record()
    tracker.on_release(_LEFT, record)
    assert tracker.region is None, "the region is dropped at the DETACH"
    assert tracker.handed_over == _LEFT, "kept until the hand holds nothing"
    envelope = tracker.envelope(now_ns=12 * _S)
    assert envelope is not None and envelope.region is None
    tracker.accept(_measured(12 * _S))
    assert tracker.region is None, "nothing re-latches a region while handed over"
    assert not tracker.on_pick_complete(_RIGHT, now_ns=12 * _S), "not the handed-over hand"
    assert tracker.on_pick_complete(_LEFT, now_ns=12 * _S)
    assert tracker.handed_over is None and tracker.target == _goal_scope()
    assert tracker.spent(_LEFT) == (True, record), "the leg guards the released payload"
    # Behind the backoff: the same place does not re-arm at once.
    _approach(tracker, [(_LEFT, here)], now_ns=12 * _S + 1)
    assert tracker.target == _goal_scope()
    second = _pick(tracker, second_place, now_ns=13 * _S)
    assert second.target_id == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:8"
    assert second.stamp_ns == first.stamp_ns, "the goal's stamp: its timeout backstop"
    # A new goal forgets the spent payload, never the counter.
    tracker.on_release(_LEFT, record)
    tracker.on_pick_complete(_LEFT, now_ns=14 * _S)
    tracker.on_declaration(_goal_scope(stamp_ns=20 * _S))
    assert tracker.spent(_LEFT) == (False, None)
    third = _pick(tracker, here, now_ns=21 * _S)
    assert third.target_id == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:9"


def test_a_re_activated_hals_tracker_never_reuses_an_identity() -> None:
    """The counter is seeded from the wall clock: a tracker built mid-goal by a re-activated
    HAL mints identities past every one the previous tracker minted."""
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    minted: list[int] = []
    for _ in range(2):
        tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append)
        tracker.on_declaration(_goal_scope())
        for k in range(3):
            picked = _pick(tracker, here, now_ns=(11 + 10 * k) * _S)
            minted.append(int(picked.target_id.rsplit(":", 1)[1]))
            tracker.on_release(_LEFT, None)
            assert tracker.on_pick_complete(_LEFT, now_ns=(12 + 10 * k) * _S)
    assert minted == sorted(set(minted)), f"an identity was reused: {minted}"


def test_a_pick_with_no_release_record_keeps_the_hand_guarded() -> None:
    """No tf2 at the DETACH: no record. The hand is guarded for the rest of the goal, and a
    pick completing with no DETACH seen at all is guarded the same way."""
    tracker, _ = _approach_tracker()
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    _pick(tracker, here, now_ns=11 * _S)
    assert tracker.on_pick_complete(_LEFT, now_ns=12 * _S)
    assert tracker.spent(_LEFT) == (True, None)
    tracker.clear_spent(_LEFT, _released_record())
    assert tracker.spent(_LEFT) == (True, None), "only the guarded record ends the guard"


def test_the_end_of_the_released_payload_guard_is_logged_once() -> None:
    """The guard ending is the hand's way back to arming: visible, once, and only for
    the record actually guarded."""
    tracker, lines = _approach_tracker()
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    _pick(tracker, here, now_ns=11 * _S)
    record = _released_record()
    tracker.on_release(_LEFT, record)
    assert tracker.on_pick_complete(_LEFT, now_ns=12 * _S)
    tracker.clear_spent(_LEFT, _released_record())
    assert not any("cleared the payload" in line for line in lines), "not the guarded record"
    tracker.clear_spent(_LEFT, record)
    tracker.clear_spent(_LEFT, record)
    assert tracker.spent(_LEFT) == (False, None)
    assert sum("cleared the payload it released" in line for line in lines) == 1, lines


def test_an_approach_computed_before_a_pick_completed_is_stale() -> None:
    """The leg's tick reads the generation first; a pick completing on the HAL's thread
    before its ``on_approach`` lands makes that verdict stale (it could not see the guard)."""
    tracker, _ = _approach_tracker()
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    _pick(tracker, here, now_ns=11 * _S)
    tracker.on_release(_LEFT, _released_record())
    generation = tracker.generation
    tracker.on_pick_complete(_LEFT, now_ns=12 * _S)
    elsewhere = approach_box([(0.30, 0.20, 0.10)], approach_m=0.1, frame_id="openarm_base")
    tracker.on_approach([(_LEFT, elsewhere)], now_ns=20 * _S, move_m=_CELL, generation=generation)
    assert tracker.target == _goal_scope()
    tracker.on_approach(
        [(_LEFT, elsewhere)], now_ns=20 * _S, move_m=_CELL, generation=tracker.generation
    )
    assert tracker.target is not None and tracker.target.contact_links == _LEFT


def test_a_confirm_that_raises_hands_nothing_over() -> None:
    """``confirm`` is tf2/geometry; an exception in it is never "on target": evaluated before
    any state changes, it ends the arming as ``attach_off_target`` with no region."""
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))

    def broken(_region: PlaceRegion) -> bool:
        raise RuntimeError("tf2 lookup exploded")

    tracker.on_attach("openarm_left_finger_pair", confirm=broken)
    assert tracker.handed_over == _LEFT, "re-measurement stops: the hand holds something"
    assert tracker.region is None, "the region must not be handed over unconfirmed"
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope is not None and envelope.region is None
    assert any("attach_off_target" in line and "RuntimeError" in line for line in lines)


def test_a_named_declaration_stays_handed_over_after_its_pick() -> None:
    """A named target was picked; a new target needs a new declaration from dispatch."""
    tracker, _ = _tracker()
    tracker.accept(_measured(11 * _S))
    tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
    handed = tracker.handed_over
    assert handed is not None
    tracker.on_release(handed, None)
    assert not tracker.on_pick_complete(handed, now_ns=12 * _S)
    assert tracker.handed_over == handed and tracker.region == _measured(11 * _S)
    assert not tracker.wants_measurement(now_ns=12 * _S)


# ── approach: the support surface is not a target; no re-arm flood ──────────────


def test_the_support_layer_under_the_hand_is_not_an_approach() -> None:
    """A hand 4 cm over an empty table: the box holds a full table layer (>= min_cells)
    but nothing above it."""
    from openral_hal._grasp_target_leg import approach_cell_count

    size = (10, 10, 6)
    occ = np.zeros(size[::-1], dtype=np.uint8)
    occ[0, :, :] = 1  # the table top, k = 0
    table = VoxelLattice(
        "openarm_base", (0.35, -0.1, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, size, occ.ravel()
    )
    box = approach_box([(0.45, 0.0, 0.05)], approach_m=0.08, frame_id="openarm_base")
    assert len(occupied_centers_in_box(table, box)) >= 8, "the raw count would arm"
    assert approach_cell_count(table, box) == 0
    occ[1:4, 4:6, 4:6] = 1  # a 4 x 4 x 6 cm object on it
    on_it = VoxelLattice(
        "openarm_base", (0.35, -0.1, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, size, occ.ravel()
    )
    assert approach_cell_count(on_it, box) == 12, "all three of its layers count"


def test_a_refused_arming_does_not_re_arm_in_place_until_the_hand_moves_or_backs_off() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=11 * _S)
    assert tracker.target == _goal_scope()
    for step in range(1, 10):  # every tick inside the backoff, the hand still
        _approach(tracker, _near(_LEFT), now_ns=11 * _S + step * _S // 10)
        assert tracker.target == _goal_scope()
    assert sum("armed from the approach" in line for line in lines) == 1, "re-arm flood"
    # Moving by under a voxel is still the same place.
    jitter = approach_box([(0.48, 0.02, 0.18)], approach_m=0.1, frame_id="openarm_base")
    _approach(tracker, [(_LEFT, jitter)], now_ns=12 * _S)
    assert tracker.target == _goal_scope()
    # Moving by more than a voxel re-arms at once.
    moved = approach_box([(0.50, 0.02, 0.18)], approach_m=0.1, frame_id="openarm_base")
    _approach(tracker, [(_LEFT, moved)], now_ns=12 * _S)
    target = tracker.target
    assert target is not None and target.search_box == moved
    # Refused again, the hand still: after the backoff (one freeze window) it re-arms.
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=12 * _S)
    _approach(tracker, [(_LEFT, moved)], now_ns=12 * _S + int(_FREEZE_S * _S))
    assert tracker.target == _goal_scope()
    _approach(tracker, [(_LEFT, moved)], now_ns=12 * _S + int(_FREEZE_S * _S) + 1)
    target = tracker.target
    assert target is not None and target.contact_links == _LEFT


def test_a_backed_off_hand_still_counts_toward_ambiguity() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=11 * _S)
    _approach(tracker, _near(_LEFT, _RIGHT), now_ns=11 * _S + 1)
    assert tracker.target == _goal_scope(), "the right hand is not the only one approaching"
    assert any("2 hands approaching at once" in line for line in lines)


# ── handover races: segmentation pending, a measurement in flight ──────────────────


def test_the_armed_hand_keeps_its_arming_while_its_grasp_is_being_resolved() -> None:
    """An ATTACH that goes to segmentation: the hand's leg is ``pending`` across a tick.
    It is no longer "approaching" (it closed), but its arming must survive until the
    ATTACH resolves — else the region it measured is retracted and nothing hands over."""
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    armed = tracker.target
    tracker.on_approach([], now_ns=11 * _S + 1, move_m=_CELL, holding=[_LEFT])
    assert tracker.target is armed and tracker.region == _measured(11 * _S)
    assert not any("approach_ended" in line for line in lines)
    tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)  # ``_finish``
    assert tracker.handed_over == _LEFT
    assert not any("attach_unarmed" in line or "no armed target" in line for line in lines)


def test_a_grasp_resolved_to_nothing_ends_the_arming_once_the_hand_leaves() -> None:
    tracker, lines = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.on_approach([], now_ns=11 * _S + 1, move_m=_CELL, holding=[_LEFT])
    assert tracker.target is not None and tracker.target.contact_links == _LEFT
    tracker.on_approach([], now_ns=11 * _S + 2, move_m=_CELL)  # no attachment; hand away
    assert tracker.target == _goal_scope()
    assert any("approach_ended" in line for line in lines)


def test_a_handed_over_target_takes_no_late_accept_or_refuse() -> None:
    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    generation = tracker.generation
    tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
    assert tracker.generation != generation, "a request asked before the grasp is stale"
    tracker.refuse("map_disagrees", "late reply", retract=True, now_ns=12 * _S)
    tracker.accept(_measured(12 * _S).model_copy(update={"half_extents": (0.05, 0.05, 0.05)}))
    assert tracker.handed_over == _LEFT and tracker.region == _measured(11 * _S)


def _payload(gripper: Any, tracker: GraspTargetTracker) -> AttachedCollisionObject:
    """The real record a region ATTACH latches (``region_attachment``), the manifest's links.

    Posed with an identity ``attach link <- base`` transform: only its presence matters to
    the callers (the hand is holding), never where it is.
    """
    target, region = tracker.target, tracker.region
    assert target is not None and region is not None
    declaration = target.model_copy(update={"region": region})
    return region_attachment(
        declaration,
        attach_link=gripper.producer.attach_link,
        touch_links=gripper.producer.touch_links,
        t_link_from_region=np.eye(4),
        stamp_ns=region.stamp_ns,
    )


def _released_record() -> AttachedCollisionObject:
    """A real frozen release record: the left gripper's region payload of ``_measured``,
    released and frozen in the base frame (``freeze_released_attachment``)."""
    from openral_hal.vision_attachment_bridge import freeze_released_attachment

    robot = RobotDescription.from_yaml(str(_ROBOT))
    bridge = VisionAttachmentBridge(None, robot, config=VisionAttachmentConfig(camera="head_zed"))
    gripper = next(g for g in bridge._legs if g.jaw_link == _LEFT[0])
    declaration = _goal_scope().model_copy(
        update={
            "target_id": f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:1",
            "contact_links": _LEFT,
            "region": _measured(11 * _S),
        }
    )
    held = region_attachment(
        declaration,
        attach_link=gripper.producer.attach_link,
        touch_links=gripper.producer.touch_links,
        t_link_from_region=np.eye(4),
        stamp_ns=11 * _S,
    )
    return freeze_released_attachment(held, base_link=robot.base_frame, t_base_from_link=np.eye(4))


class _LiveLeg:
    """A real bridge on a real rclpy node and tf2 buffer, the hands placed by TCP."""

    def __init__(self, node: Any, tf2_ros: Any, **config: Any) -> None:
        self.node = node
        self.robot = RobotDescription.from_yaml(str(_ROBOT))
        self.bridge = VisionAttachmentBridge(
            node,
            self.robot,
            config=VisionAttachmentConfig(
                camera="head_zed",
                grasp_target_enabled=True,
                grasp_target_approach_m=0.10,
                tf_frames={
                    "openarm_left_link7": "openarm_left_ee_base_link",
                    "openarm_right_link7": "openarm_right_ee_base_link",
                },
                **config,
            ),
        )
        self.buffer = tf2_ros.Buffer()
        self.bridge._tf_buffer = self.buffer
        leg = self.bridge._grasp_target
        assert leg is not None
        self.leg = leg
        self.origin = {j.child_link: j.origin_xyz for j in self.robot.joints if j.role == "gripper"}

    def place(self, side: str, tcp: tuple[float, float, float]) -> None:
        from geometry_msgs.msg import TransformStamped

        ee = TransformStamped()
        ee.header.frame_id = "openarm_base"
        ee.child_frame_id = f"openarm_{side}_ee_base_link"
        offset = self.origin[f"openarm_{side}_finger_pair"]
        x, y, z = (t - o for t, o in zip(tcp, offset, strict=True))
        ee.transform.translation.x, ee.transform.translation.y = x, y
        ee.transform.translation.z = z
        ee.transform.rotation.w = 1.0
        self.buffer.set_transform_static(ee, "test")

    def gripper(self, jaw_link: str) -> Any:
        return next(g for g in self.bridge._legs if g.jaw_link == jaw_link)


@contextmanager
def _live_leg(name: str, **config: Any) -> Iterator[_LiveLeg]:
    """A ``_LiveLeg`` whose rclpy context and node are torn down whatever fails, setup too."""
    tf2_ros = pytest.importorskip("tf2_ros")
    rclpy = pytest.importorskip("rclpy")
    rclpy.init()
    try:
        node = rclpy.create_node(name)
        try:
            yield _LiveLeg(node, tf2_ros, **config)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_a_pending_segmentation_spanning_a_leg_tick_still_hands_over() -> None:
    """The leg's own tick (``_detect_approach``) while the left leg is ``pending``: the
    closed hand is skipped as an approach, yet its arming is kept for the ATTACH."""
    import time

    with _live_leg("test_grasp_target_pending_handover") as live:
        leg, now_ns = live.leg, live.leg._now_ns()
        live.place("left", (0.45, 0.0, 0.18))
        live.place("right", (0.45, -0.30, 0.18))
        leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        leg._detect_approach(0.10, now_ns)
        armed = leg.tracker.target
        assert armed is not None and armed.contact_links == _LEFT
        left = live.gripper("openarm_left_finger_pair")
        left.pending = True  # ``_begin_segmentation``: ATTACH -> SegmentInView in flight
        live.place("left", (0.45, 0.0, 0.40))  # even with the TCP read far away
        leg._detect_approach(0.10, now_ns + 1)
        assert leg.tracker.target is armed, "the pending hand's arming was retracted"
        left.pending = False
        leg.tracker.accept(_measured(now_ns))
        left.attachment = _payload(left, leg.tracker)  # ``_finish`` latched the payload
        leg.tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
        assert leg.tracker.handed_over == _LEFT


def test_a_measurement_in_flight_at_handover_is_discarded() -> None:
    """Both late outcomes of a grasp-leg request asked before the ATTACH: its refusal
    (reply or deadline) must not retract the handed-over arming nor replace its region."""
    rclpy_task = pytest.importorskip("rclpy.task")

    with _live_leg("test_grasp_target_inflight_handover") as live:
        leg = live.leg
        now_ns = leg._now_ns()
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        region = _measured(now_ns)
        leg.tracker.accept(region)
        target = leg.tracker.target
        assert target is not None

        def in_flight() -> tuple[Any, Any]:
            future = rclpy_task.Future()
            snapshot = (
                np.zeros((2, 2)),
                now_ns,
                None,
                np.eye(4),
                0.05,
                target,
                leg.tracker.generation,
            )
            leg._inflight = (future, snapshot[6])
            return future, snapshot

        # A reply: resolved with no response would be a lost-view refusal.
        future, snapshot = in_flight()
        leg.tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
        future.set_result(None)
        leg._on_reply(future, snapshot)
        assert leg._inflight is None
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region == region
        assert leg.tracker.target is target

        # A deadline: asked under the current generation (a new goal), then a handover.
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns + 1))
        assert leg.tracker.handed_over is None
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns + 3 * _S)
        leg.tracker.accept(_measured(now_ns + 3 * _S))
        target = leg.tracker.target
        in_flight()
        leg.tracker.on_attach("openarm_left_finger_pair", confirm=_CONFIRMED)
        leg._on_deadline()
        assert leg._inflight is None and leg.tracker.handed_over == _LEFT
        assert leg.tracker.region == _measured(now_ns + 3 * _S)


def _grip(bridge: VisionAttachmentBridge, q: float, *, t0_ns: int) -> int:
    """The left jaw commanded closed and read at ``q`` for 0.8 s of 50 Hz HAL ticks — the
    bridge's real trigger path (``observe_command`` / ``observe_joint_state``). ``q`` = 0.2
    stalls on an object (ATTACH); 0.0 closes onto nothing once loaded (DETACH)."""
    from openral_core import Action, ControlMode, JointState

    close = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_names=["left_gripper"],
        joint_targets=[[0.0]],
    )
    t = t0_ns
    for _ in range(40):
        t += 20_000_000
        bridge.observe_command(close)
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[q, 0.0],
                effort=[0.0, 0.0],
                stamp_ns=t,
            )
        )
    return t


def test_two_picks_in_one_goal_through_the_bridges_detach() -> None:
    """The bridge end to end, event-driven: pick 1 takes the measured region; its DETACH
    drops the region at once (handover kept while the release window is open); the window
    closing completes the pick; the hand by the payload it just released does not arm;
    lifted clear and back, it arms pick 2 under a fresh identity and the goal's stamp, and
    pick 2's region is a new payload."""
    import time

    with _live_leg("test_grasp_target_two_picks", release_timeout_s=0.05) as live:
        leg, bridge = live.leg, live.bridge
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")

        def detect(tcp: tuple[float, float, float], at_ns: int) -> None:
            live.place("left", tcp)
            live.place("right", (0.45, -0.30, 0.40))
            leg._bridge._grid = (_held_block_lattice(), at_ns, time.monotonic())
            leg._detect_approach(0.10, at_ns)

        goal = _goal_scope(stamp_ns=now_ns)
        leg.tracker.on_declaration(goal)
        detect((0.45, 0.0, 0.18), now_ns)
        first = leg.tracker.target
        assert first is not None and first.contact_links == _LEFT
        assert first.target_id.startswith(f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:")
        live.place("left", (0.45, 0.0, 0.10))  # inside ``_measured``'s box
        leg.tracker.accept(_measured(now_ns))

        # ── Pick 1: ATTACH on the region, then DETACH ───────────────────────────────────
        t = _grip(bridge, 0.2, t0_ns=1)
        assert left.attachment is not None and left.attachment.object_id == first.target_id
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region == _measured(now_ns)
        _grip(bridge, 0.0, t0_ns=t)
        assert left.attachment is None and left.release is not None, "release window open"
        assert leg.tracker.region is None, "the region is dropped at the DETACH"
        assert leg.tracker.handed_over == _LEFT, "kept while the window is open"
        detect((0.45, 0.0, 0.18), now_ns + 1 * _S)
        assert leg.tracker.target is first, "nothing arms while the hand releases"

        # ── The window times out with the hand still by the payload: pick complete ──────
        time.sleep(0.1)
        bridge._poll_releases()
        assert left.release is None and leg.tracker.handed_over is None
        assert leg.tracker.spent(_LEFT)[0], "the released payload is guarded"
        for step in range(2, 5):
            detect((0.45, 0.0, 0.18), now_ns + step * 3 * _S)  # well past any backoff
            assert leg.tracker.target == goal, "the hand armed on its just-released payload"

        # ── Lifted clear, then back: pick 2 under a fresh identity ──────────────────────
        detect((0.45, 0.0, 0.45), now_ns + 15 * _S)
        assert leg.tracker.spent(_LEFT) == (False, None), "clear of the payload: guard over"
        detect((0.45, 0.0, 0.18), now_ns + 18 * _S)
        second = leg.tracker.target
        assert second is not None and second.contact_links == _LEFT
        n_first = int(first.target_id.rsplit(":", 1)[1])
        assert second.target_id == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:{n_first + 1}"
        assert second.stamp_ns == goal.stamp_ns, "the goal's stamp: its timeout backstop"
        live.place("left", (0.45, 0.0, 0.10))
        leg.tracker.accept(_measured(now_ns + 18 * _S))
        taken = bridge._region_payload(left, stamp_ns=now_ns + 18 * _S)
        assert taken is not None and taken[0].object_id == second.target_id, (
            "pick 2's region is a new payload"
        )


def _blocks_on_the_bridge_lock(bridge: VisionAttachmentBridge, call: Any) -> bool:
    """Whether ``call``, run on another thread, waits while this thread holds the bridge
    lock — and then completes cleanly once it is released."""
    import threading

    done = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as exc:  # surfaced below
            errors.append(exc)
        finally:
            done.set()

    with bridge._lock:
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        waited = not done.wait(0.2)
    assert done.wait(5.0), "never completed after the lock was released"
    assert not errors, errors
    return waited


def test_every_bridge_and_leg_entry_point_is_serialized() -> None:
    """The lead decision after three rounds of proprio-thread vs executor races: ONE bridge
    lock, taken at every entry point that reads or mutates leg, tracker or attachment state.
    Each one, run on another thread while the lock is held, must wait for it — so no
    interleaving (a heartbeat inside a half-applied change, a release poll between a
    re-ATTACH's trigger update and its close, an executor drop between ``_region_payload``
    and ``on_attach``) can happen."""
    from openral_core import Action, ControlMode, JointState
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg

    rclpy_task = pytest.importorskip("rclpy.task")
    with _live_leg("test_bridge_entry_points_serialized", place_target_enabled=True) as live:
        bridge, leg = live.bridge, live.leg
        place = bridge._place_target
        assert place is not None
        _attachment_publisher(live)
        left = live.gripper("openarm_left_finger_pair")
        state = JointState(
            name=["left_gripper", "right_gripper"],
            position=[0.0, 0.0],
            effort=[0.0, 0.0],
            stamp_ns=1,
        )
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_names=["left_gripper"],
            joint_targets=[[0.0]],
        )
        grasp_msg = GraspDeclarationMsg()
        _goal_scope(stamp_ns=leg._now_ns()).fill_idl(grasp_msg)
        place_msg = PlaceDeclarationMsg()
        stray = rclpy_task.Future()  # never the one in flight: dropped after the lock
        calls = {
            "observe_joint_state": lambda: bridge.observe_joint_state(state),
            "observe_command": lambda: bridge.observe_command(action),
            "clear_command": bridge.clear_command,
            "attachment_action_ack_ready": bridge.attachment_action_ack_ready,
            "heartbeat": bridge._heartbeat,
            "poll_releases": bridge._poll_releases,
            "publish_attachment": bridge._publish_attachment,
            "segmentation_reply": lambda: bridge._on_reply(
                left,
                stray,
                generation=left.generation,
                stamp_ns=1,
                t_link_from_cam=np.eye(4),
                tcp_in_link=(0.0, 0.0, 0.0),
            ),
            "segmentation_deadline": lambda: bridge._on_deadline(
                left,
                generation=left.generation,
                stamp_ns=1,
                t_link_from_cam=np.eye(4),
                tcp_in_link=(0.0, 0.0, 0.0),
            ),
            "grasp_declaration": lambda: leg._on_declaration(grasp_msg),
            "grasp_tick": leg._tick,
            "grasp_reply": lambda: leg._on_reply(stray, (None,) * 7),  # type: ignore[arg-type]
            "grasp_deadline": leg._on_deadline,
            "place_declaration": lambda: place._on_declaration(place_msg),
            "place_tick": place._tick,
            "place_witness": place.on_joint_state,
            "teardown": bridge.teardown,
        }
        unserialized = [
            name for name, call in calls.items() if not _blocks_on_the_bridge_lock(bridge, call)
        ]
        assert not unserialized, f"entry points not under the bridge lock: {unserialized}"


class _OrderedLock:
    """A re-entrant lock that records any bridge-lock acquisition made by a thread holding
    the tracker lock but not the bridge lock — the one forbidden order."""

    _held = __import__("threading").local()

    def __init__(self, name: str, violations: list[str]) -> None:
        import threading

        self._lock, self.name, self._violations = threading.RLock(), name, violations
        self.nested: list[tuple[str, ...]] = []

    def __enter__(self) -> None:
        stack: list[str] = getattr(_OrderedLock._held, "stack", [])
        _OrderedLock._held.stack = stack
        if self.name == "bridge" and "tracker" in stack and "bridge" not in stack:
            self._violations.append(f"bridge lock taken under the tracker lock: {stack}")
        self.nested.append(tuple(stack))
        self._lock.acquire()
        stack.append(self.name)

    def __exit__(self, *_exc: object) -> None:
        _OrderedLock._held.stack.pop()
        self._lock.release()


def test_the_lock_order_is_bridge_then_tracker_never_the_reverse() -> None:
    """Bridge lock -> tracker lock, never the reverse (nothing run under the tracker lock —
    its log sink, ``on_attach``'s ``confirm`` — takes the bridge lock), across a whole pick:
    approach, region ATTACH with its confirm, DETACH, release poll, heartbeat, ticks."""
    import time

    violations: list[str] = []
    probe_bridge, probe_tracker = (
        _OrderedLock("bridge", violations),
        _OrderedLock("tracker", violations),
    )
    with probe_tracker, probe_bridge:  # the recorder itself catches the forbidden order
        pass
    assert len(violations) == 1
    violations.clear()

    with _live_leg("test_bridge_lock_order", release_timeout_s=0.05) as live:
        leg, bridge = live.leg, live.bridge
        bridge_lock = _OrderedLock("bridge", violations)
        tracker_lock = _OrderedLock("tracker", violations)
        bridge._lock = bridge_lock  # type: ignore[assignment]
        leg.tracker._lock = tracker_lock  # type: ignore[assignment]
        _attachment_publisher(live)
        now_ns = leg._now_ns()
        live.place("left", (0.45, 0.0, 0.18))
        live.place("right", (0.45, -0.30, 0.40))
        leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        leg._detect_approach(0.10, now_ns)
        live.place("left", (0.45, 0.0, 0.10))
        leg.tracker.accept(_measured(now_ns))
        t = _grip(bridge, 0.2, t0_ns=1)  # region ATTACH: confirm runs under both locks
        assert leg.tracker.handed_over == _LEFT
        _grip(bridge, 0.0, t0_ns=t)  # DETACH
        time.sleep(0.1)
        bridge._heartbeat()  # release poll -> window closes -> pick completes
        leg._tick()
        assert leg.tracker.handed_over is None
        assert not violations, violations
        assert ("bridge",) in tracker_lock.nested, "never exercised the nested order"


def test_a_detach_that_finds_nothing_latched_still_completes_the_pick() -> None:
    """Review finding: pick completion was re-checked only on a DETACH that dropped an
    attachment (``on_detach``) and on a window close. A handed-over hand whose regrasp was
    superseded before its reply (nothing latched, maybe still holding the barrier) then
    DETACHes: nothing else would ever complete its pick, so the hand stayed handed over —
    no further pick — for the rest of the goal. Every DETACH and every barrier release
    re-checks."""
    for pending in (False, True):
        with _live_leg(f"test_grasp_target_empty_detach_{int(pending)}") as live:
            leg, bridge = live.leg, live.bridge
            now_ns = leg._now_ns()
            left = live.gripper("openarm_left_finger_pair")
            live.place("left", (0.45, 0.0, 0.18))
            live.place("right", (0.45, -0.30, 0.40))
            leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
            leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
            leg._detect_approach(0.10, now_ns)
            live.place("left", (0.45, 0.0, 0.10))
            leg.tracker.accept(_measured(now_ns))
            t = _grip(bridge, 0.2, t0_ns=1)  # region ATTACH: handed over
            first = leg.tracker.target
            assert leg.tracker.handed_over == _LEFT and first is not None
            # A regrasp whose reply a newer event superseded: nothing latched.
            left.attachment = None
            left.pending = pending
            _grip(bridge, 0.0, t0_ns=t)  # DETACH: nothing to release, no window
            assert left.attachment is None and left.release is None and not left.pending
            assert leg.tracker.handed_over is None, f"pending={pending}: stayed handed over"
            assert leg.tracker.spent(_LEFT)[0], "no release record: guarded for the goal"


def _attachment_publisher(live: _LiveLeg) -> None:
    """The bridge's real ``/openral/attachment_state`` publisher (``setup`` creates it)."""
    from openral_msgs.msg import AttachmentState

    live.bridge._attachment_pub = live.node.create_publisher(
        AttachmentState, "/openral/attachment_state", 10
    )


def test_an_attachment_change_refreshes_the_other_hands_pre_handover_arming() -> None:
    """HZ-0115-3, producer side. The kernel retires an armed grasp declaration whenever the
    attachment set empties at a new revision — before its handover too, whichever hand let
    go — and never re-arms that identity. The bridge owns the set, so at every publish that
    loses or replaces an object (or is empty) the other hand's pre-handover arming drops its
    region at once and takes a fresh identity, BEFORE the envelope of that very snapshot is
    filled; and a region measured at or before the change is never accepted again."""
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import AttachmentState

    with _live_leg("test_grasp_target_attachment_change") as live:
        leg, bridge = live.leg, live.bridge
        _attachment_publisher(live)
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        right = live.gripper("openarm_right_finger_pair")
        live.place("left", (0.45, 0.0, 0.18))
        live.place("right", (0.45, -0.30, 0.40))
        leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        leg._detect_approach(0.10, now_ns)
        armed = leg.tracker.target
        assert armed is not None and armed.contact_links == _LEFT
        before = _measured(leg._now_ns())
        leg.tracker.accept(before)
        assert leg.tracker.region == before
        # A real addition — the right hand ATTACHes something it grasped unarmed — while the
        # left hand is armed: not a change, the left arming is NOT refreshed.
        right.attachment = _released_record().model_copy(
            update={"object_id": "cell:other", "attach_link": right.producer.attach_link}
        )
        bridge._publish_attachment()
        assert leg.tracker.target is armed and leg.tracker.region == before

        # The right hand lets go (its DETACH; no tf2 here, so no window): the set empties.
        generation = leg.tracker.generation
        right.attachment = None
        # The envelope the detach snapshot itself carries: what the bridge publishes.
        filled: list[Any] = []
        sub = live.node.create_subscription(
            AttachmentState, "/openral/attachment_state", filled.append, 10
        )
        bridge._publish_attachment()
        rclpy = pytest.importorskip("rclpy")
        deadline = time.monotonic() + 5.0
        while not filled and time.monotonic() < deadline:
            rclpy.spin_once(live.node, timeout_sec=0.05)
        live.node.destroy_subscription(sub)
        fresh = leg.tracker.target
        assert fresh is not None and fresh.contact_links == _LEFT
        n = int(armed.target_id.rsplit(":", 1)[1])
        assert fresh.target_id == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:{n + 1}"
        assert fresh.stamp_ns == armed.stamp_ns, "the goal's stamp"
        assert leg.tracker.region is None, "the pre-detach region was kept"
        assert leg.tracker.generation != generation, "a measurement in flight survived"
        (snapshot,) = filled
        assert snapshot.grasp_declaration.target_id == fresh.target_id, (
            "the detach snapshot carried the retired identity"
        )
        assert not snapshot.grasp_declaration.region_valid
        # The region measured before the detach is never re-used ...
        leg.tracker.accept(before)
        assert leg.tracker.region is None, "a pre-detach measurement was accepted"
        # ... a measurement after it is.
        after = _measured(leg._now_ns() + 1)
        leg.tracker.accept(after)
        assert leg.tracker.region == after and leg.tracker.target is fresh
        assert left.attachment is None


def test_an_attachment_change_leaves_a_handed_over_pick_to_its_own_path() -> None:
    tracker, _ = _approach_tracker()
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    picked = _pick(tracker, here, now_ns=11 * _S)
    tracker.on_attachment_changed(now_ns=12 * _S)
    assert tracker.target is picked and tracker.region == _measured(11 * _S)
    assert tracker.handed_over == _LEFT
    # With nothing armed, only the change time is kept: a later arming needs a newer region.
    tracker.on_release(_LEFT, None)
    assert tracker.on_pick_complete(_LEFT, now_ns=12 * _S)
    elsewhere = approach_box([(0.30, 0.20, 0.10)], approach_m=0.1, frame_id="openarm_base")
    _approach(tracker, [(_LEFT, elsewhere)], now_ns=13 * _S)
    tracker.accept(_measured(12 * _S))
    assert tracker.region is None
    tracker.accept(_measured(12 * _S + 1))
    assert tracker.region == _measured(12 * _S + 1)


def test_a_mid_grasp_hand_keeps_its_identity_but_never_its_pre_change_region() -> None:
    """Review findings: a change the kernel does not retire on (the set stays non-empty) used
    to refresh the arming of a hand that is mid-grasp — its segmentation in flight — so the
    ATTACH found no identity and the bimanual pick failed when the other hand regrasped or
    let go. The mid-grasp hand keeps its identity; but it must NOT keep the region measured
    before the change (the next review: handing that over is less conservative than the
    pre-fix drop) — its handover latches with no region (fail closed). A carried payload or a
    release window is not mid-grasp. An emptying change refreshes regardless."""
    from openral_hal.vision_attachment_bridge import ReleaseWindow

    with _live_leg("test_grasp_target_mid_grasp_refresh") as live:
        leg, bridge = live.leg, live.bridge
        _attachment_publisher(live)
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        right = live.gripper("openarm_right_finger_pair")
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        other = _released_record().model_copy(
            update={"object_id": "cell:other", "attach_link": right.producer.attach_link}
        )
        right.attachment = other
        bridge._publish_attachment()  # an addition
        armed = leg.tracker.target
        region = _measured(leg._now_ns())
        leg.tracker.accept(region)
        assert armed is not None and leg.tracker.region == region
        left.pending = True  # the left ATTACH's SegmentInView is in flight
        # The right hand's DETACH: its payload is replaced by its frozen release record
        # (a change, the set non-empty).
        right.attachment = None
        right.release = ReleaseWindow(
            record=other.model_copy(update={"stamp_ns": other.stamp_ns + 1}),
            opened_s=time.monotonic(),
            hand_link=right.producer.attach_link,
            hand_boxes=(),
            jaws=(),
        )
        bridge._publish_attachment()
        assert leg.tracker.target is armed, "a mid-grasp identity was refreshed"
        assert leg.tracker.region is None, "a region measured before the change was kept"
        # The left segmentation resolves: handed over, with no region (fail closed).
        left.pending = False
        segmented = other.model_copy(
            update={"object_id": "cell:left", "attach_link": left.producer.attach_link}
        )
        left.attachment = segmented  # ``_finish`` latched the segmented payload
        leg.on_attach(left.jaw_link, segmented, region=None)
        assert leg.tracker.handed_over == _LEFT
        envelope = leg.tracker.envelope(now_ns=leg._now_ns())
        assert envelope is not None and envelope.target_id == armed.target_id
        assert envelope.region is None, "the handover carried a pre-change region"


def test_a_carried_payload_or_release_window_is_not_mid_grasp() -> None:
    """Only an ATTACH being resolved keeps an identity across a change: a hand that already
    carries a payload (or is in its release window) is refreshed like any other."""
    with _live_leg("test_grasp_target_not_mid_grasp") as live:
        leg, bridge = live.leg, live.bridge
        _attachment_publisher(live)
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        right = live.gripper("openarm_right_finger_pair")
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        other = _released_record().model_copy(
            update={"object_id": "cell:other", "attach_link": right.producer.attach_link}
        )
        right.attachment = other
        bridge._publish_attachment()
        armed = leg.tracker.target
        assert armed is not None
        assert not leg._resolving(_LEFT)
        left.pending = True
        assert leg._resolving(_LEFT)
        left.attachment = other  # carrying (a REGRASP segmenting): not an ATTACH resolving
        assert not leg._resolving(_LEFT)
        left.attachment = None
        left.pending = False
        right.attachment = other.model_copy(update={"stamp_ns": other.stamp_ns + 2})
        bridge._publish_attachment()
        refreshed = leg.tracker.target
        assert refreshed is not None and refreshed.target_id != armed.target_id
        # Mid-grasp, and the right hand lets go: the set empties — refreshed anyway.
        leg.tracker.accept(_measured(leg._now_ns()))
        left.pending = True
        right.attachment = None
        bridge._publish_attachment()
        fresh = leg.tracker.target
        assert fresh is not None and fresh.target_id != refreshed.target_id
        assert leg.tracker.region is None, "the empty edge retires it in the kernel too"


def test_an_attachment_change_after_a_retraction_still_advances_the_identity() -> None:
    """Review finding: the counter advanced at a change only while an arming was held. An
    arming retracted since the kernel last saw it is still the kernel's armed identity, and
    the change's detach edge retires it there — so a re-arm under that identity would be
    refused for the goal. Any identity armed since the last advance advances it; nothing
    armed since, nothing to retire, no advance."""
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append, first_pick=7)
    tracker.on_declaration(_goal_scope())
    later = 11 * _S + int(_FREEZE_S * _S) + 1  # past the refusal backoff

    def identity() -> str:
        target = tracker.target
        assert target is not None and target.contact_links == _LEFT
        return target.target_id

    _approach(tracker, _near(_LEFT))
    assert identity() == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:7"
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=11 * _S)
    assert tracker.target == _goal_scope()
    tracker.on_attachment_changed(now_ns=12 * _S)  # the kernel may still hold :7
    _approach(tracker, _near(_LEFT), now_ns=later)
    assert identity() == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:8"
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=later)
    tracker.on_attachment_changed(now_ns=later + 1)  # :8 was armed: advance
    tracker.on_attachment_changed(now_ns=later + 2)  # nothing armed since: no advance
    _approach(tracker, _near(_LEFT), now_ns=2 * later)
    assert identity() == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:9"


def test_a_tick_that_cannot_sample_the_hands_restarts_the_away_window() -> None:
    """Review finding: the away window (liveness after a kernel fault retirement) was only
    sampled in ``on_approach``; a tick with no fresh grid, a request in flight, or a stale
    verdict produced no sample and left the window running — unknown counted as away. Each
    must restart it: away on every sample of a window otherwise earns a fresh identity."""
    with _live_leg("test_grasp_target_unknown_is_not_away") as live:
        leg = live.leg
        tracker = leg.tracker
        tracker.on_declaration(_goal_scope(stamp_ns=leg._now_ns()))
        here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
        there = approach_box([(0.30, 0.20, 0.10)], approach_m=0.1, frame_id="openarm_base")
        far = approach_box([(0.45, 0.40, 0.40)], approach_m=0.1, frame_id="openarm_base")
        freeze_ns = int(_FREEZE_S * _S)

        def sample(now_ns: int, *, near: PlaceRegion | None, at: PlaceRegion) -> None:
            tracker.on_approach(
                [] if near is None else [(_LEFT, near)],
                now_ns=now_ns,
                move_m=_CELL,
                located={_LEFT: at},
            )

        def identity() -> str:
            target = tracker.target
            assert target is not None and target.contact_links == _LEFT
            return target.target_id

        def stale_grid() -> None:
            leg._bridge._grid = (_held_block_lattice(), leg._now_ns(), time.monotonic() - 60.0)
            leg._tick()

        def in_flight() -> None:
            leg._inflight = (None, tracker.generation)
            leg._tick()
            leg._inflight = None

        unknowns: dict[str, Any] = {
            "no grid": lambda: (setattr(leg._bridge, "_grid", None), leg._tick()),
            "stale grid": stale_grid,
            "request in flight": in_flight,
            "stale verdict": lambda: tracker.on_approach(
                [], now_ns=0, move_m=_CELL, generation=tracker.generation - 1, located={}
            ),
            "control (a real sample)": lambda: None,
        }
        sample(10 * _S, near=here, at=here)
        first = identity()
        for round_, (name, unknown) in enumerate(unknowns.items(), start=1):
            t0 = round_ * 20 * _S
            sample(t0, near=here, at=here)  # the arming follows the hand back
            sample(t0 + 1, near=None, at=far)  # leaves: approach_ended
            sample(t0 + freeze_ns // 2, near=None, at=far)  # away since here
            unknown()
            sample(t0 + freeze_ns // 2 + freeze_ns, near=None, at=far)  # a whole window
            sample(t0 + 2 * freeze_ns, near=there, at=there)
            if name.startswith("control"):
                assert identity() != first, "the control round never earned a fresh one"
            else:
                assert identity() == first, f"{name}: an unsampled tick counted as away"


def test_the_released_payload_guard_is_the_oriented_box_not_its_bounding_sphere() -> None:
    """HZ-0115-11 with an elongated payload (a 30 cm bar lying along x): its bounding sphere
    (radius ~15 cm) kept the hand guarded 15 cm to the side of it. The guard is the
    separating-axis lower bound between the payload's oriented box and the column the next
    measurement would search (``search_column``: the approach box reaching
    ``support_search_below_m`` = 15 cm below it, where ``_seed`` looks): one voxel of
    clearance ends it, within one voxel it holds.

    Review finding: the gap was tested against the approach box alone, so a hand 25 cm over
    the bar (approach box bottom 15 cm over it) was "clear" although the column it would
    search next reaches down to the bar and would seed on it."""
    from openral_core import BoxShape

    with _live_leg("test_grasp_target_elongated_guard") as live:
        leg = live.leg
        assert leg._search_below_m == pytest.approx(0.15)
        grid = _held_block_lattice()
        record = _released_record()
        bar = record.model_copy(
            update={
                "primitives": [
                    record.primitives[0].model_copy(
                        update={"shape": BoxShape(half_extents_m=(0.15, 0.01, 0.01))}
                    )
                ],
                "pose_in_link": record.pose_in_link.model_copy(update={"xyz": (0.45, 0.0, 0.0)}),
            }
        )
        leg.tracker.on_declaration(_goal_scope())
        here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
        _pick(leg.tracker, here, now_ns=11 * _S)
        leg.tracker.on_release(_LEFT, bar)
        assert leg.tracker.on_pick_complete(_LEFT, now_ns=12 * _S)

        def box(x: float, y: float, z: float) -> PlaceRegion:
            return approach_box([(x, y, z)], approach_m=0.1, frame_id="openarm_base")

        # 1 cm over the bar's top face: guarded.
        assert leg._near_spent(_LEFT, box(0.45, 0.0, 0.12), grid)
        # The hand 25 cm over the bar: its approach box clears it by 14 cm, but the search
        # column reaches 15 cm lower, onto the bar. Still guarded.
        assert leg._near_spent(_LEFT, box(0.45, 0.0, 0.25), grid)
        assert leg.tracker.spent(_LEFT)[0]
        # 15 cm to the side at the same height: the sphere said "near" (0.15 <= 0.1 + 0.02
        # + 0.15); the oriented box clears by 4 cm.
        assert not leg._near_spent(_LEFT, box(0.45, 0.15, 0.18), grid)
        assert leg.tracker.spent(_LEFT) == (False, None), "clear of the bar: guard over"
        # Straight above, the column bottom 4 cm over the bar's top also clears.
        leg.tracker.on_declaration(_goal_scope(stamp_ns=20 * _S))
        _pick(leg.tracker, here, now_ns=21 * _S)
        leg.tracker.on_release(_LEFT, bar)
        assert leg.tracker.on_pick_complete(_LEFT, now_ns=22 * _S)
        assert leg._near_spent(_LEFT, box(0.45, 0.0, 0.25), grid)
        assert not leg._near_spent(_LEFT, box(0.45, 0.0, 0.30), grid)


def test_a_guarded_hand_does_not_block_the_other_hands_approach() -> None:
    """Review finding: a hand guarded for the rest of the goal (no release record: a tf2 gap
    at its DETACH) was still counted as a competing approach, so whenever it lingered by
    occupied cells the other hand never armed (``approach_ambiguous``). A guarded hand is
    not a candidate; one hand per declaration is the kernel's own rule (HZ-0115-12)."""
    with _live_leg("test_grasp_target_guarded_not_ambiguous") as live:
        leg, now_ns = live.leg, live.leg._now_ns()
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
        _pick(leg.tracker, here, now_ns=now_ns)
        assert leg.tracker.on_pick_complete(_LEFT, now_ns=now_ns)  # no DETACH record
        assert leg.tracker.spent(_LEFT) == (True, None)
        # Both hands over the same block; the left is guarded for the goal.
        live.place("left", (0.47, 0.02, 0.18))
        live.place("right", (0.43, -0.02, 0.18))
        at = now_ns + 5 * _S  # past the left hand's backoff
        leg._bridge._grid = (_held_block_lattice(), at, time.monotonic())
        leg._detect_approach(0.10, at)
        armed = leg.tracker.target
        assert armed is not None and armed.contact_links == _RIGHT, "the guarded hand blocked it"
        assert leg.tracker.spent(_LEFT) == (True, None), "still guarded: no record to clear"


def test_a_hand_that_left_and_stayed_away_re_arms_under_a_fresh_identity() -> None:
    """Liveness after a kernel fault retirement (grid re-frame, rejected attachment model):
    the producer cannot see it, and a re-arm keeps the pick's identity, so the kernel would
    refuse the pick for the rest of the goal. Only a hand that left the approach distance
    and was then seen away — located, its approach box off its last target's column — on
    every sample for a continuous freeze window re-arms under a fresh identity. Wiggling, a
    refusal in place, coming straight back, a tf2 gap (unlocated) or a ``min_cells`` dip with
    the hand still over the target is not away, and any of them restarts the window."""
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append, first_pick=7)
    tracker.on_declaration(_goal_scope())
    here = approach_box([(0.45, 0.0, 0.18)], approach_m=0.1, frame_id="openarm_base")
    there = approach_box([(0.30, 0.20, 0.10)], approach_m=0.1, frame_id="openarm_base")
    far = approach_box([(0.45, 0.40, 0.40)], approach_m=0.1, frame_id="openarm_base")
    freeze_ns = int(_FREEZE_S * _S)

    def tick(
        now_ns: int, *, near: PlaceRegion | None = None, at: PlaceRegion | None = None
    ) -> None:
        """One leg tick: ``near`` = approaching there; ``at`` = located there (else unlocated)."""
        box = near if near is not None else at
        tracker.on_approach(
            [] if near is None else [(_LEFT, near)],
            now_ns=now_ns,
            move_m=_CELL,
            located={} if box is None else {_LEFT: box},
        )

    def identity() -> str:
        target = tracker.target
        assert target is not None and target.contact_links == _LEFT
        return target.target_id

    def leave(now_ns: int) -> None:
        tick(now_ns, at=far)
        assert tracker.target == _goal_scope(), "approach_ended"

    armed_at = [here]

    def back(now_ns: int) -> None:
        """Approach again — elsewhere (more than a voxel off), so no backoff holds it."""
        box = there if armed_at[-1] is here else here
        tick(now_ns, near=box)
        armed_at.append(box)

    first = f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:7"
    tick(11 * _S, near=here)
    assert identity() == first
    # A refusal in place, re-armed after the backoff without leaving: the same identity.
    tracker.refuse("no_support", "nothing under it", retract=True, now_ns=11 * _S)
    tick(11 * _S + freeze_ns + 1, near=here)
    assert identity() == first
    # Leaving and coming straight back: the same identity.
    leave(20 * _S)
    back(20 * _S + 1)
    assert identity() == first
    # Away, but back inside the window: the same identity.
    leave(30 * _S)
    tick(30 * _S + freeze_ns // 2, at=far)
    back(30 * _S + freeze_ns // 2 + 1)
    assert identity() == first, "back inside the window: not yet away long enough"
    # Unlocated (a tf2 gap) for longer than the window: never seen away.
    leave(40 * _S)
    for step in range(1, 4):
        tick(40 * _S + step * freeze_ns)
    back(40 * _S + 4 * freeze_ns)
    assert identity() == first, "a tf2 gap counted as away"
    # A min_cells dip: located, its box still over the target's column, not in ``near``.
    target_box = armed_at[-1]
    tick(50 * _S, at=far)  # approach_ended
    for step in range(1, 4):
        tick(50 * _S + step * freeze_ns, at=target_box)
    back(50 * _S + 4 * freeze_ns)
    assert identity() == first, "a min_cells dip over the target counted as away"
    # Away, a tf2 gap mid-window restarts it: not yet a fresh identity.
    leave(60 * _S)
    tick(60 * _S + freeze_ns // 2)  # unlocated
    tick(60 * _S + freeze_ns // 2 + 1, at=far)
    tick(60 * _S + freeze_ns + 1, at=far)
    back(60 * _S + freeze_ns + 2)
    assert identity() == first, "the window was not restarted by the unlocated sample"
    # Away on every sample for a whole window: a fresh identity.
    leave(70 * _S)
    tick(70 * _S + freeze_ns // 2, at=far)  # the first sample after leaving: since then
    tick(70 * _S + freeze_ns, at=far)
    tick(70 * _S + freeze_ns // 2 + freeze_ns, at=far)  # a whole window, continuously
    back(70 * _S + freeze_ns // 2 + freeze_ns + 1)
    assert identity() == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:8"
    # A new goal forgets who was away.
    leave(80 * _S)
    tick(80 * _S + freeze_ns + 1, at=far)
    tracker.on_declaration(_goal_scope(stamp_ns=90 * _S))
    tick(91 * _S, near=here)
    assert identity() == f"{APPROACH_TARGET_PREFIX}{_LEFT[0]}:8"


# ── ATTACH on what the arming measured, or not (HZ-0115-8/-11) ─────────────────────


def _segmented_attach(live: _LiveLeg, gripper: Any, stamp_ns: int) -> None:
    """The bridge's segmentation outcome for one ATTACH: ``_finish`` with no mask, i.e. the
    real producer's conservative jaw-span box at the leg's TCP, then ``on_attach``."""
    gripper.pending = True
    live.bridge._finish(
        gripper,
        generation=gripper.generation,
        stamp_ns=stamp_ns,
        masks=[],
        scores=[],
        reason="test: SegmentInView unavailable",
        tcp_in_link=gripper.tcp_in_link,
    )


def test_jaws_closing_on_a_neighbour_hand_nothing_over() -> None:
    """The region was refused for this ATTACH (the jaw is not at it), the grasp segmented,
    and the payload lies beside the measured target: the arming ends, its region is not
    handed over for a payload it was not measured for (``attach_off_target``)."""
    with _live_leg("test_grasp_target_off_target") as live:
        leg, bridge = live.leg, live.bridge
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        leg.tracker.accept(_measured(now_ns))  # X: x 0.41-0.49, |y| <= 0.04
        live.place("left", (0.45, 0.16, 0.09))  # closing on Y, 12 cm beside X
        assert bridge._region_payload(left, stamp_ns=now_ns) is None, "jaw is not at X"
        _segmented_attach(live, left, now_ns)
        assert left.attachment is not None and not left.pending
        assert leg.tracker.handed_over == _LEFT, "re-measurement stops: the hand holds Y"
        assert leg.tracker.region is None, "X's region must not be handed over for Y"
        envelope = leg.tracker.envelope(now_ns=now_ns)
        assert envelope is not None and envelope.region is None
        # The hand empty with no DETACH event: still handed over (the pick completes on the
        # bridge's DETACH, never by polling the legs).
        left.attachment = None
        assert leg.tracker.handed_over == _LEFT and not leg.tracker.wants_approach(now_ns=now_ns)


def test_jaws_closing_on_the_target_via_segmentation_hand_it_over() -> None:
    """The region first accepted mid-close (none was held at the ATTACH, so it segmented):
    the jaw is at it and the segmented payload lies in it — handed over."""
    with _live_leg("test_grasp_target_on_target_segmented") as live:
        leg, bridge = live.leg, live.bridge
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        live.place("left", (0.45, 0.0, 0.09))  # closing on X
        assert bridge._region_payload(left, stamp_ns=now_ns) is None, "no region yet"
        leg.tracker.accept(_measured(now_ns))  # accepted while the segmentation runs
        _segmented_attach(live, left, now_ns)
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region == _measured(now_ns)

        # The same region accepted mid-close, but the jaw closed on Y: not handed over.
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns + 1))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns + 1)
        live.place("left", (0.45, 0.16, 0.09))
        leg.tracker.accept(_measured(now_ns))
        _segmented_attach(live, left, now_ns + 1)
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region is None


def test_a_region_payload_hands_over_only_the_region_it_was_built_from() -> None:
    """``_region_payload`` read region R; a re-fit R' accepted on another thread before the
    ATTACH reached the tracker is not what the payload is: nothing handed over."""
    with _live_leg("test_grasp_target_region_moved_under_attach") as live:
        leg, bridge = live.leg, live.bridge
        now_ns = leg._now_ns()
        left = live.gripper("openarm_left_finger_pair")
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns)
        live.place("left", (0.45, 0.0, 0.09))
        leg.tracker.accept(_measured(now_ns))
        taken = bridge._region_payload(left, stamp_ns=now_ns)
        assert taken is not None
        leg.tracker.accept(_measured(now_ns + 1))  # the interleaved re-fit
        leg.on_attach(left.jaw_link, taken[0], region=taken[1])
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region is None

        # The region it was built from DROPPED since (an attachment change): no region to
        # compare is not "absent" — the payload is a stale measurement, attach_off_target.
        leg.tracker.on_declaration(_goal_scope(stamp_ns=now_ns + 2))
        _approach(leg.tracker, _near(_LEFT), now_ns=now_ns + 2)
        leg.tracker.accept(_measured(now_ns + 2))
        taken = bridge._region_payload(left, stamp_ns=now_ns + 2)
        assert taken is not None
        leg.tracker.on_attachment_changed(now_ns=now_ns + 3)
        assert leg.tracker.region is None
        leg.on_attach(left.jaw_link, taken[0], region=taken[1])
        assert leg.tracker.handed_over == _LEFT and leg.tracker.region is None
        assert leg.tracker._status == "attach_off_target", "handed over as 'absent'"


# ── the tracker is serialized: the proprio thread vs the executor ──────────────────


def test_an_attach_on_another_thread_runs_atomically_with_accept() -> None:
    """Deterministic interleaving: while ``on_attach`` (proprio thread) is between reading
    the arming and handing it over — inside its ``confirm`` — an ``accept`` (executor) must
    wait; it then sees the handover and is ignored, so the handed-over region is the one
    ``confirm`` vouched for."""
    import threading

    tracker, _ = _approach_tracker()
    _approach(tracker, _near(_LEFT))
    tracker.accept(_measured(11 * _S))
    entered, accept_started, accept_done = (threading.Event() for _ in range(3))
    seen: list[bool] = []

    def confirm(region: PlaceRegion) -> bool:
        entered.set()
        assert accept_started.wait(timeout=5.0)
        seen.append(accept_done.wait(timeout=0.2))  # must time out: accept is blocked
        return region == _measured(11 * _S)

    def rival() -> None:
        accept_started.set()
        tracker.accept(_measured(12 * _S))
        accept_done.set()

    attach = threading.Thread(
        target=tracker.on_attach, args=(_LEFT[0],), kwargs={"confirm": confirm}
    )
    worker = threading.Thread(target=rival)
    attach.start()
    assert entered.wait(timeout=5.0)
    worker.start()
    attach.join(timeout=10.0)
    worker.join(timeout=10.0)
    assert accept_done.is_set()
    assert seen == [False], "accept ran while on_attach was mid-handover"
    assert tracker.handed_over == _LEFT and tracker.region == _measured(11 * _S)


def test_two_threads_hammering_the_tracker_never_replace_a_handed_over_region() -> None:
    """The proprio thread declares/arms/attaches while the executor accepts re-fits and runs
    approach ticks; whenever the hand is handed over its region is the one it took."""
    import threading

    lines: list[str] = []
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=lines.append)
    tracker.on_declaration(_goal_scope())
    stop = threading.Event()
    broken: list[str] = []

    def executor() -> None:
        k = 0
        while not stop.is_set():
            k += 1
            tracker.accept(_measured(11 * _S + k))
            tracker.on_approach(_near(_LEFT), now_ns=11 * _S, move_m=_CELL, holding=[_LEFT])

    def proprio() -> None:
        for n in range(1, 400):
            now = 11 * _S + n * 3 * _S  # past each pick's backoff
            tracker.on_approach(_near(_LEFT), now_ns=now, move_m=_CELL)
            took: list[PlaceRegion] = []

            def confirm(region: PlaceRegion, took: list[PlaceRegion] = took) -> bool:
                took.append(region)
                return True

            tracker.on_attach(_LEFT[0], confirm=confirm)
            if tracker.handed_over is not None and took and tracker.region != took[0]:
                broken.append(f"pick {n}: {tracker.region} != {took[0]}")
            tracker.on_declaration(_goal_scope(stamp_ns=now))  # the next goal

    worker = threading.Thread(target=executor)
    worker.start()
    try:
        proprio()
    finally:
        stop.set()
        worker.join(timeout=10.0)
    assert not broken, broken[:3]


def test_a_tick_commits_its_request_only_under_the_lock_it_snapshotted() -> None:
    """Review finding: ``_tick`` ran ``_request`` (the in-flight record, the deadline timer,
    tf2) outside the bridge lock, so a teardown on another thread could leave a deadline
    timer on a torn-down leg, and an ATTACH meanwhile could be raced. Now only the pure
    column scan (``_seed``) runs outside; the request is committed under the lock, and not
    at all when the leg was torn down or the target changed while the column was scanned."""
    with _live_leg("test_grasp_target_tick_commit") as live:
        leg = live.leg
        now_ns = leg._now_ns()
        leg.tracker.on_declaration(_dispatched(stamp_ns=now_ns))  # a named search box
        leg._bridge._grid = (_held_block_lattice(), now_ns, time.monotonic())
        requested: list[int] = []
        leg._request = lambda *args: requested.append(args[2])  # type: ignore[method-assign]  # reason: records the commit

        def scan_then(during: Any) -> Any:
            def seed(*_: Any) -> tuple[tuple[float, float, float], float]:
                during()  # another thread, while the column is scanned outside the lock
                return (0.45, 0.0, 0.09), 0.05

            return seed

        leg._seed = scan_then(lambda: None)  # type: ignore[method-assign]  # reason: forces the interleaving
        leg._tick()
        assert requested == [leg.tracker.generation], "the control tick sent nothing"
        requested.clear()
        # The target changed while the column was scanned: the scan is stale, no request.
        leg._seed = scan_then(  # type: ignore[method-assign]  # reason: forces the interleaving
            lambda: leg.tracker.on_declaration(_dispatched(stamp_ns=now_ns + 1))
        )
        leg._tick()
        assert requested == [], "a request was sent for a target that changed mid-scan"
        # Torn down while the column was scanned: nothing is sent, no timer is left.
        leg._seed = scan_then(leg.teardown)  # type: ignore[method-assign]  # reason: forces the interleaving
        leg._tick()
        assert requested == [] and leg._inflight is None and leg._deadline_timer is None
        leg._on_deadline()  # a deadline already queued at teardown: a no-op
        assert leg._deadline_timer is None


def test_both_legs_share_one_voxel_subscription_and_one_lazy_decode() -> None:
    """Review finding: the grasp and place legs each subscribed ``/openral/world_voxels`` and
    eagerly decoded and scanned every grid, armed or not. The bridge owns ONE subscription;
    each grid is decoded once, shared by both legs, and its whole-grid scan runs only on first
    use (cached on the lattice)."""
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import OccupancyVoxels

    with _live_leg("test_grasp_target_one_voxel_sub", place_target_enabled=True) as live:
        bridge, leg = live.bridge, live.leg
        place = bridge._place_target
        assert place is not None
        bridge.setup()
        try:
            topics = [sub.topic_name for sub in live.node.subscriptions]
            assert topics.count("/openral/world_voxels") == 1, topics
            msg = OccupancyVoxels()
            msg.header.frame_id = "openarm_base"
            msg.source_stamp.sec = 12
            msg.origin.x, msg.origin.y, msg.origin.z = 0.41, -0.04, 0.05
            msg.orientation.w = 1.0
            msg.resolution = 0.02
            msg.size_x = msg.size_y = msg.size_z = 4
            msg.occupancy = [1] * 64
            bridge._on_voxels(msg)
            assert bridge._grid is not None
            assert leg._grid is bridge._grid and place._grid is bridge._grid
            lattice, source_ns, _ = bridge._grid
            assert source_ns == 12 * _S
            assert "_occupied_centers" not in lattice.__dict__, "scanned with nothing armed"
            first = lattice.occupied_centers()
            assert lattice.occupied_centers() is first, "the scan is not cached per grid"
            msg.size_x = 5  # 64 cells for a 5x4x4 grid: malformed, dropped
            bridge._on_voxels(msg)
            assert bridge._grid is None and leg._grid is None and place._grid is None
        finally:
            bridge.teardown()
        assert "/openral/world_voxels" not in [s.topic_name for s in live.node.subscriptions]


# ── self-filtered fit (the fingers in the target's mask) ─────────────────────

_SELF_FILTERED = "/openral/world_cloud/self_filtered"
#: The target (``_measured``'s 8 cm block, standing on z=0.03) and a finger closing in
#: over it — both inside the SAM mask, only the target in the self-filter's output.
_TARGET_BOX = (np.array([0.41, -0.04, 0.03]), np.array([0.49, 0.04, 0.13]))
_FINGER_BOX = (np.array([0.44, -0.015, 0.14]), np.array([0.46, 0.015, 0.20]))


def _oblique_capture() -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]:
    """Ray-cast the target and finger from a head camera 45 degrees above the target.

    Returns ``(depth, mask, t_base_from_cam, intrinsics, kept)``: the z-depth raster, the
    mask over both boxes (SAM sees the fingers too), and the base-frame points of the
    target's pixels only — what the robot self-filter lets through of this capture.
    """
    from openral_core import IntrinsicsPinhole

    k = IntrinsicsPinhole(width=160, height=120, fx=200.0, fy=200.0, cx=80.0, cy=60.0)
    origin = np.array([0.0, 0.0, 0.50])
    z = np.array([0.45, 0.0, 0.08]) - origin
    z /= np.linalg.norm(z)
    x = np.cross(z, (0.0, 0.0, 1.0))
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, np.cross(z, x), z, origin
    rows, cols = np.mgrid[0 : k.height, 0 : k.width]
    rays = np.stack(((cols + 0.5 - k.cx) / k.fx, (rows + 0.5 - k.cy) / k.fy, np.ones(rows.shape)))
    rays = rays.reshape(3, -1).T @ t[:3, :3].T  # parameter along these is the optical z
    depth = np.full(rays.shape[0], np.inf)
    hit = np.zeros(rays.shape[0], dtype=np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        for label, (lo, hi) in ((1, _TARGET_BOX), (2, _FINGER_BOX)):
            t1, t2 = (lo - origin) / rays, (hi - origin) / rays
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            closer = (near <= far) & (near > 0) & (near < depth)
            depth[closer], hit[closer] = near[closer], label
    target = hit == 1
    kept = origin + rays[target] * depth[target][:, None]
    depth[~np.isfinite(depth)] = 0.0
    shape = (k.height, k.width)
    return depth.reshape(shape), (hit > 0).reshape(shape), t, k, kept


def test_the_fingers_in_the_mask_leave_the_fit_only_through_the_self_filtered_cloud(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Isaac i24-i28: the closing fingers entered the target's SAM mask, the 3 Hz re-fit
    grew up to the hand and the held region was retracted mid-grasp. With the self-filter's
    cloud of the same capture the fit is the target alone; without it the capture is not
    fitted at all — a lost view, so nothing hand-grown is ever accepted (Isaac i38 armed a
    32 cm hand column from an unfiltered fit) — and it is said out loud (CLAUDE.md §1.4)."""
    pytest.importorskip("openral_msgs")
    rclpy_task = pytest.importorskip("rclpy.task")
    from openral_hal.depth_cloud import camera_info_from_intrinsics
    from openral_msgs.srv import SegmentInView
    from sensor_msgs.msg import Image as ImageMsg

    depth, mask, t_base_from_cam, k, kept = _oblique_capture()
    with _live_leg(
        "test_grasp_target_self_filtered", self_filtered_cloud_topic=_SELF_FILTERED
    ) as live:
        bridge, leg = live.bridge, live.leg
        now_ns = leg._now_ns()
        bridge._camera_info = camera_info_from_intrinsics(
            width=k.width, height=k.height, fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy, frame_id="cam"
        )
        grid = (_held_block_lattice(), now_ns, time.monotonic())

        def measure(previous: PlaceRegion | None) -> PlaceRegion:
            response = SegmentInView.Response(ok=True)
            image = ImageMsg(height=k.height, width=k.width, encoding="mono8", step=k.width)
            image.header.stamp.sec, image.header.stamp.nanosec = divmod(now_ns, _S)
            image.data = (mask.astype(np.uint8) * 255).tobytes()
            response.masks = [image]
            future = rclpy_task.Future()
            future.set_result(response)
            snapshot = (depth, now_ns, k, t_base_from_cam, 0.03, _dispatched(), 0)
            return leg._measure(future, snapshot, now_ns, grid, [], previous)

        bridge._kept_clouds.append((now_ns, "openarm_base", kept))
        filtered = measure(None)
        top = filtered.pose.xyz[2] + filtered.half_extents[2]
        pad = 3**0.5 * 0.02 / 2 + 0.01  # target_region_from_mask's padding
        assert top == pytest.approx(_TARGET_BOX[1][2] + pad, abs=0.005), "the finger is in the fit"
        assert leg.unfiltered_fits == 0

        bridge._kept_clouds.clear()  # the self-filter dropped this capture
        for previous in (None, filtered):  # nothing held yet / a region held
            with pytest.raises(_Refusal) as refused:
                measure(previous)
            # A lost view, never a fit with the finger in it (Isaac i38 armed one).
            assert (refused.value.kind, refused.value.retract) == ("unfiltered", False)
        assert leg.unfiltered_fits == 2, "an unfiltered capture went unreported"

        bridge._kept_clouds.append((now_ns, "openarm_base", kept))
        assert measure(filtered) == filtered
        assert leg.unfiltered_fits == 0, "the filter's return was not noticed"
    err = capfd.readouterr().err
    assert err.count("grasp target fit unfiltered") == 1, "logged per fit, not per transition"
    assert f"no self-filtered cloud on {_SELF_FILTERED!r} at stamp {now_ns}" in err
    assert "self-filtered again after 2 unfiltered fit(s)" in err


def test_the_kernel_gets_the_cell_closed_region_while_the_held_fit_stays_tight() -> None:
    """The published declaration carries the fit closed over the grid's cells (2 cm here:
    +1 cm sideways and up, the bottom kept); the tracker, its envelope (the region payload's
    source) and so every producer gate keep the tight fit. A grid in another frame publishes
    the tight fit (the kernel refuses that frame itself)."""
    with _live_leg("test_grasp_target_cell_closed") as live:
        attachment_state = pytest.importorskip("openral_msgs.msg").AttachmentState
        leg = live.leg
        now_ns = leg._now_ns()
        lattice = _held_block_lattice()
        leg._bridge._grid = (lattice, now_ns, time.monotonic())
        leg.tracker.on_declaration(_dispatched(stamp_ns=now_ns))
        tight = _measured(now_ns + 1)
        leg.tracker.accept(tight)
        assert leg.tracker.region == tight

        msg = attachment_state()
        leg.fill(msg, now_ns=now_ns + 2)
        assert msg.grasp_declaration_valid and msg.grasp_declaration.region_valid
        published = PlaceRegion.from_idl(msg.grasp_declaration.region)
        assert published.half_extents == pytest.approx((0.05, 0.05, 0.045))
        assert published.pose.xyz == pytest.approx((0.45, 0.0, 0.095))
        assert published.pose.xyz[2] - published.half_extents[2] == pytest.approx(0.05)
        assert published.stamp_ns == tight.stamp_ns
        assert published.evidence_ref == tight.evidence_ref
        assert leg.tracker.region == tight, "the held fit was replaced"
        envelope = leg.tracker.envelope(now_ns=now_ns + 2)
        assert envelope is not None and envelope.region == tight

        other = VoxelLattice(
            "world", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, (1, 1, 1), np.zeros(1, np.uint8)
        )
        leg._bridge._grid = (other, now_ns, time.monotonic())
        leg.fill(msg, now_ns=now_ns + 3)
        assert PlaceRegion.from_idl(msg.grasp_declaration.region).half_extents == pytest.approx(
            tight.half_extents
        )
