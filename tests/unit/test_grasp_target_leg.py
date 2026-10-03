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
from openral_hal._grasp_target import VoxelLattice
from openral_hal._grasp_target_leg import (
    GraspTargetLeg,
    GraspTargetTracker,
    _gate_refit,
    _Refusal,
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


def _gate(region: PlaceRegion, held: PlaceRegion) -> _Refusal:
    with pytest.raises(_Refusal) as caught:
        _gate_refit(_held_block_lattice(), region, held, min_cover=0.5)
    return caught.value


def test_a_refit_shrunk_inside_the_held_region_is_occlusion_and_freezes() -> None:
    """The hand covers part of the target: the fit shrinks and shifts 25 mm (past the
    one-voxel tracking gate) but stays inside the held box — a lost view, not a move."""
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
    wide = GraspTargetLeg(
        None, bridge, config, support_search_below_m=0.30, support_probe_margin_m=0.08
    )
    assert (wide._search_below_m, wide._probe_margin_m) == (0.30, 0.08)
    with pytest.raises(ROSConfigError, match="support_probe_margin_m"):
        GraspTargetLeg(None, bridge, config, support_probe_margin_m=0.0)
    with pytest.raises(ROSConfigError, match="support_search_below_m"):
        GraspTargetLeg(None, bridge, config, support_search_below_m=-0.1)
