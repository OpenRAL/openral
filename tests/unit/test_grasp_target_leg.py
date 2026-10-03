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

from pathlib import Path

import numpy as np
import pytest
from openral_core import DeployScene, GraspDeclaration, PlaceRegion, Pose6D, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_target import VoxelLattice, occupied_centers_in_box
from openral_hal._grasp_target_leg import (
    APPROACH_TARGET_PREFIX,
    GraspTargetLeg,
    GraspTargetTracker,
    _gate_refit,
    _Refusal,
    approach_box,
    search_column,
)
from openral_hal.vision_attachment_bridge import VisionAttachmentBridge, VisionAttachmentConfig

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "tests" / "unit" / "fixtures" / "scenes" / "openarm_direct_dispatch_grasp.yaml"
_ROBOT = _REPO / "robots" / "openarm" / "robot.yaml"
_S = 1_000_000_000
_FREEZE_S = 2.0


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
    tracker.on_attach("openarm_left_link7")  # not a declared contact link
    assert tracker.wants_measurement(now_ns=11 * _S)
    tracker.on_attach(jaws["left_gripper"])
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
    # A freeze past four kernel voxel deadlines holds a region the map has long moved past.
    assert leg(grid_max_age_s=0.4, grasp_target_freeze_s=1.6)._freeze_s == pytest.approx(1.6)
    with pytest.raises(ROSConfigError, match="grasp_target_freeze_s"):
        leg(grid_max_age_s=0.4, grasp_target_freeze_s=1.7)


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
    tf2_ros = pytest.importorskip("tf2_ros")
    rclpy = pytest.importorskip("rclpy")
    from geometry_msgs.msg import TransformStamped

    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(
        camera="head_zed",
        grasp_target_enabled=True,
        tf_frames={
            "openarm_left_link7": "openarm_left_ee_base_link",
            "openarm_right_link7": "openarm_right_ee_base_link",
        },
    )
    rclpy.init()
    node = rclpy.create_node("test_grasp_target_jaw_point")
    try:
        bridge = VisionAttachmentBridge(node, robot, config=config)
        buffer = tf2_ros.Buffer()
        ee = TransformStamped()
        ee.header.frame_id, ee.child_frame_id = "openarm_base", "openarm_left_ee_base_link"
        ee.transform.translation.x, ee.transform.translation.y = 0.47, 0.02
        ee.transform.translation.z = 0.25
        ee.transform.rotation.w = 1.0  # the published tree: no finger_pair frame in it
        buffer.set_transform_static(ee, "test")
        bridge._tf_buffer = buffer

        assert bridge._lookup("openarm_base", "openarm_left_finger_pair") is None
        (origin,) = (
            j.origin_xyz for j in robot.joints if j.child_link == "openarm_left_finger_pair"
        )
        hand = bridge.jaw_point("openarm_left_finger_pair", "openarm_base")
        assert hand == pytest.approx(np.add((0.47, 0.02, 0.25), origin))
        assert bridge.jaw_point("openarm_right_finger_pair", "openarm_base") is None  # no tf
        assert bridge.jaw_point("openarm_left_link7", "openarm_base") is None  # not a jaw

        leg = bridge._grasp_target
        assert leg is not None
        hands = leg._hands(_dispatched(), "openarm_base")
        assert hands == [hand]
        held = _measured(11 * _S)
        refusal = _gate(_refit(0.425, (0.015, 0.04, 0.03)), held, hands=tuple(hands))
        assert (refusal.kind, refusal.retract) == ("occluded_refit", False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


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
    tracker.on_approach([])
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope == _goal_scope(), "region-less, so the kernel exempts nothing"


def test_one_approaching_hand_arms_a_one_hand_declaration_with_the_goals_attribution() -> None:
    tracker, lines = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    target = tracker.target
    assert target is not None
    assert target.target_id == f"{APPROACH_TARGET_PREFIX}openarm_left_finger_pair"
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
    tracker.on_approach(_near(_LEFT, _RIGHT))
    assert tracker.target == _goal_scope()
    assert not tracker.wants_measurement(now_ns=11 * _S)
    assert any("2 hands approaching at once" in line for line in lines)


def test_the_hand_leaving_the_approach_distance_retracts_at_once() -> None:
    tracker, lines = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_approach(_near(_RIGHT))  # the left hand moved off; the other hand is not it
    envelope = tracker.envelope(now_ns=11 * _S)
    assert envelope == _goal_scope(), "back to the region-less goal declaration"
    assert any("retracted — approach_ended" in line for line in lines)
    # It re-arms only from a fresh approach, from scratch (no region carried over).
    tracker.on_approach(_near(_RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _RIGHT and tracker.region is None


def test_the_held_hand_keeps_its_arming_while_the_other_hand_also_approaches() -> None:
    tracker, _ = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_approach(_near(_LEFT, _RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _LEFT
    assert tracker.region is not None


def test_the_approach_target_dies_with_the_goal() -> None:
    tracker, _ = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_declaration(_goal_scope().model_copy(update={"active": False}))
    assert tracker.envelope(now_ns=11 * _S) is None
    assert tracker.target is None and tracker.region is None
    # A new goal starts with no approach held.
    tracker.on_declaration(_goal_scope(stamp_ns=20 * _S))
    assert tracker.target == _goal_scope(stamp_ns=20 * _S)


def test_the_approach_target_dies_with_the_goal_timeout() -> None:
    tracker, _ = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    tracker.accept(_measured(11 * _S))
    assert tracker.envelope(now_ns=71 * _S) is None  # stamp 10 s + timeout 60 s


def test_a_named_search_box_wins_and_no_approach_runs() -> None:
    tracker, _ = _tracker()
    assert not tracker.wants_approach(now_ns=11 * _S)
    tracker.on_approach(_near(_LEFT))
    assert tracker.target == _dispatched()


def test_a_named_hand_without_a_box_narrows_the_approach_to_that_hand() -> None:
    tracker = GraspTargetTracker(freeze_s=_FREEZE_S, log=[].append)
    tracker.on_declaration(_goal_scope().model_copy(update={"contact_links": _RIGHT}))
    tracker.on_approach(_near(_LEFT))
    assert not tracker.wants_measurement(now_ns=11 * _S), "the undeclared hand never arms"
    tracker.on_approach(_near(_LEFT, _RIGHT))
    target = tracker.target
    assert target is not None and target.contact_links == _RIGHT


def test_after_the_approach_hand_attaches_the_region_is_kept_for_the_handover() -> None:
    tracker, _ = _approach_tracker()
    tracker.on_approach(_near(_LEFT))
    tracker.accept(_measured(11 * _S))
    tracker.on_attach("openarm_right_finger_pair")  # the other hand: not this target
    assert tracker.wants_measurement(now_ns=11 * _S)
    tracker.on_attach("openarm_left_finger_pair")
    tracker.on_approach([])  # the closed hand reads anything now; the handover rules
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
    tf2_ros = pytest.importorskip("tf2_ros")
    rclpy = pytest.importorskip("rclpy")
    import time

    from geometry_msgs.msg import TransformStamped

    robot = RobotDescription.from_yaml(str(_ROBOT))
    config = VisionAttachmentConfig(
        camera="head_zed",
        grasp_target_enabled=True,
        grasp_target_approach_m=0.10,
        tf_frames={
            "openarm_left_link7": "openarm_left_ee_base_link",
            "openarm_right_link7": "openarm_right_ee_base_link",
        },
    )
    origin = {j.child_link: j.origin_xyz for j in robot.joints if j.role == "gripper"}
    rclpy.init()
    node = rclpy.create_node("test_grasp_target_approach")
    try:
        bridge = VisionAttachmentBridge(node, robot, config=config)
        buffer = tf2_ros.Buffer()

        def place(side: str, tcp: tuple[float, float, float]) -> None:
            ee = TransformStamped()
            ee.header.frame_id = "openarm_base"
            ee.child_frame_id = f"openarm_{side}_ee_base_link"
            offset = origin[f"openarm_{side}_finger_pair"]
            x, y, z = (t - o for t, o in zip(tcp, offset, strict=True))
            ee.transform.translation.x, ee.transform.translation.y = x, y
            ee.transform.translation.z = z
            ee.transform.rotation.w = 1.0
            buffer.set_transform_static(ee, "test")

        place("left", (0.45, 0.0, 0.18))  # the block's top face is z = 0.13
        place("right", (0.45, -0.30, 0.18))
        bridge._tf_buffer = buffer
        leg = bridge._grasp_target
        assert leg is not None
        leg._grid = (_held_block_lattice(), time.monotonic())
        leg.tracker.on_declaration(_goal_scope())

        leg._detect_approach(0.10)
        target = leg.tracker.target
        assert target is not None and target.contact_links == _LEFT
        assert target.search_box is not None
        assert target.search_box.pose.xyz == pytest.approx((0.45, 0.0, 0.18))

        place("left", (0.45, 0.0, 0.40))  # lifted 27 cm clear of the block
        leg._detect_approach(0.10)
        assert leg.tracker.target == _goal_scope(), "the approach ended with the hand away"
    finally:
        node.destroy_node()
        rclpy.shutdown()
