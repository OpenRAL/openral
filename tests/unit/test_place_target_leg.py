"""Real place producer leg: the support measured under the carried payload, no fixture.

Real pick-and-place design §2.3. Pure core of ``_place_target_leg`` (no ROS): the real
OpenArm manifest, a real ``VoxelLattice`` at the real cell's 20 mm built the way the
octomap bridge publishes one (single-layer surfaces: a table, a two-board shelf), and the
vision producer's own fallback payload (the 10 cm jaw-span box on ``link7``). Nothing
about the cell is surveyed and nobody names a place target: the only inputs are the map,
the held payload and the runner's goal-scope declaration (``target_id = surface``, no
region — the goal is the only thing that licenses a place allowance), optionally with a
reasoner-grounded hint box.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    JointState,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_target import VoxelLattice
from openral_hal._grasp_trigger import PositionStallConfig, PositionStallTrigger
from openral_hal._place_target_leg import (
    PlacePatch,
    PlaceRefusal,
    PlaceTargetLeg,
    PlaceTargetTracker,
    _posed_primitives,
    _witness_candidates,
    measure_under_payload,
    place_tick,
    verify_patch,
)
from openral_hal._vision_attachment_evidence import (
    VisionAttachmentEvidenceProducer,
    VisionGateConfig,
    jaw_span_primitive,
)
from openral_hal.vision_attachment_bridge import (
    ReleaseWindow,
    VisionAttachmentBridge,
    VisionAttachmentConfig,
    _GripperLeg,
)

_ROOT = Path(__file__).resolve().parents[2]
_OPENARM = _ROOT / "robots" / "openarm" / "robot.yaml"
_BASE = "openarm_base"
_RES = 0.02
_EXT = 0.015  # openral_core.depth_extrinsic.MAX_PLANAR_ERR_M, the leg's default
_DEPTH = VisionAttachmentConfig().place_target_search_depth_m
# The deploy passes the kernel's world_voxel_deadline_s; the test default is 1.0 s.
_GRID_MAX_AGE_S = VisionAttachmentConfig().grid_max_age_s
_FREEZE_S = 2.0 * _GRID_MAX_AGE_S
_TABLE_TOP = 0.31
_STAMP = 5_000_000_000
_MS = 1_000_000
_Posed = list[tuple[AttachedCollisionPrimitive, NDArray[np.float64]]]


def _lattice(occupied: Callable[[NDArray[np.float64]], NDArray[np.bool_]]) -> VoxelLattice:
    """A base-aligned 20 mm lattice; layer k spans z in [0.01 + 0.02 k, 0.03 + 0.02 k]."""
    origin = np.array([0.20, -0.46, 0.01])
    size = (26, 46, 30)
    k, j, i = np.meshgrid(*(np.arange(n) for n in size[::-1]), indexing="ij")
    centres = (np.stack((i, j, k), axis=-1).reshape(-1, 3) + 0.5) * _RES + origin
    return VoxelLattice(
        frame_id=_BASE,
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        resolution=_RES,
        size=size,
        occupancy=occupied(centres).astype(np.uint8),
    )


def _board(
    c: NDArray[np.float64],
    *,
    top: float,
    x: tuple[float, float] = (0.30, 0.60),
    y: tuple[float, float] = (-0.20, 0.20),
) -> NDArray[np.bool_]:
    """A one-cell-thick horizontal surface whose top face is at ``top``."""
    return (
        (c[:, 0] > x[0])
        & (c[:, 0] < x[1])
        & (c[:, 1] > y[0])
        & (c[:, 1] < y[1])
        & (np.abs(c[:, 2] - (top - _RES / 2)) < 1e-6)
    )


def _table(c: NDArray[np.float64]) -> NDArray[np.bool_]:
    return _board(c, top=_TABLE_TOP)


def _two_boards(c: NDArray[np.float64]) -> NDArray[np.bool_]:
    """A shelf: a lower board (top 0.15 m) 12 cm under an upper board (top 0.29 m)."""
    return _board(c, top=0.15) | _board(c, top=0.29)


def _blob(c: NDArray[np.float64], centre: tuple[float, float, float]) -> NDArray[np.bool_]:
    return np.all(np.abs(c - np.asarray(centre)) <= 0.03, axis=1)


@cache
def _openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_OPENARM))


def _payload(
    *, bottom_z: float, xy: tuple[float, float] = (0.45, 0.0)
) -> tuple[AttachedCollisionObject, NDArray[np.float64]]:
    """The vision producer's fallback box on the left hand, posed so its bottom is at z."""
    description = _openarm()
    producer = VisionAttachmentEvidenceProducer(description, gripper_joint="left_gripper")
    joint = next(j for j in description.joints if j.name == "left_gripper")
    span = VisionGateConfig().jaw_span_m
    object_id = "grasped_payload:left_gripper"
    obj = AttachedCollisionObject(
        object_id=object_id,
        attach_link=producer.attach_link,
        touch_links=list(producer.touch_links),
        primitives=[jaw_span_primitive(object_id=object_id, jaw_span_m=span)],
        pose_in_link=Pose6D(
            xyz=joint.origin_xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=producer.attach_link
        ),
        confidence=0.3,
        evidence_kind=AttachmentEvidenceKind.GRIPPER_FORCE,
        stamp_ns=1_500_000_000,
    )
    t_base_link = np.eye(4)
    t_base_link[:3, 3] = np.array([xy[0], xy[1], bottom_z + span]) - np.asarray(joint.origin_xyz)
    return obj, t_base_link


def _posed(bottom_z: float, xy: tuple[float, float] = (0.45, 0.0)) -> _Posed:
    obj, t = _payload(bottom_z=bottom_z, xy=xy)
    return _posed_primitives(obj, t)


def _under(grid: VoxelLattice, bottom_z: float, xy: tuple[float, float] = (0.45, 0.0)) -> object:
    return measure_under_payload(
        grid, _posed(bottom_z, xy), extrinsic_error_m=_EXT, search_depth_m=_DEPTH
    )


# ── measurement ──────────────────────────────────────────────────────────────


def test_the_table_under_the_payload_yields_a_payload_sized_slab() -> None:
    patch = _under(_lattice(_table), bottom_z=_TABLE_TOP + 0.10)
    assert isinstance(patch, PlacePatch), patch
    assert patch.plane_z == pytest.approx(_TABLE_TOP)
    # 10 cm payload + 2 x (one voxel + 15 mm extrinsic) = 17 cm -> 9-10 cells (lattice
    # phase); free height 10 cm + 35 mm -> 7 cells: from the payload, not a fixed 0.10 m.
    assert patch.ni in (9, 10) and patch.nj in (9, 10)
    assert patch.free_layers == 7
    region = patch.region
    assert region.frame_id == _BASE and region.geometry == ()
    assert region.half_extents[2] == pytest.approx(0.01)
    assert min(region.half_extents[:2]) * 2 >= 0.17 - 1e-9  # payload + pad, every side
    # The slab is the support layer itself, centred under the payload.
    assert region.pose.xyz[2] == pytest.approx(_TABLE_TOP - _RES / 2)
    assert region.pose.xyz[0] == pytest.approx(0.45, abs=_RES)
    assert region.pose.xyz[1] == pytest.approx(0.0, abs=_RES)
    assert "map_measured_support:plane_z=0.310" in region.evidence_ref


def test_over_a_shelf_it_finds_the_top_board() -> None:
    patch = _under(_lattice(_two_boards), bottom_z=0.40)
    assert isinstance(patch, PlacePatch), patch
    assert patch.plane_z == pytest.approx(0.29)


def test_inside_a_shelf_gap_lower_than_the_payload_is_no_headroom() -> None:
    """Between the boards: the lower one is 12 cm under the upper; the payload needs 13.5."""
    result = _under(_lattice(_two_boards), bottom_z=0.17)
    assert isinstance(result, tuple) and result[0] is PlaceRefusal.NO_HEADROOM, result


def test_half_over_the_board_edge_is_partial_support() -> None:
    result = _under(_lattice(_table), bottom_z=_TABLE_TOP + 0.10, xy=(0.45, 0.20))
    assert isinstance(result, tuple) and result[0] is PlaceRefusal.PARTIAL_SUPPORT, result


def test_clutter_under_the_payload_is_partial_support() -> None:
    def cluttered(c: NDArray[np.float64]) -> NDArray[np.bool_]:
        return _table(c) | _blob(c, (0.45, 0.02, _TABLE_TOP + 0.03))

    result = _under(_lattice(cluttered), bottom_z=_TABLE_TOP + 0.12)
    assert isinstance(result, tuple) and result[0] is PlaceRefusal.PARTIAL_SUPPORT, result


def test_nothing_within_the_search_depth_is_no_surface() -> None:
    result = _under(_lattice(_table), bottom_z=_TABLE_TOP + _DEPTH + 0.05)
    assert isinstance(result, tuple) and result[0] is PlaceRefusal.NO_SURFACE, result


# ── re-verification ──────────────────────────────────────────────────────────


def _patch() -> PlacePatch:
    patch = _under(_lattice(_table), bottom_z=_TABLE_TOP + 0.10)
    assert isinstance(patch, PlacePatch)
    return patch


def test_occlusion_is_a_lost_view_and_new_clutter_contradicts() -> None:
    patch = _patch()
    assert verify_patch(_lattice(_table), patch) is None
    x, y, _ = patch.region.pose.xyz

    def occluded(c: NDArray[np.float64]) -> NDArray[np.bool_]:  # payload over the board
        return _table(c) & ~_blob(c, (x, y, _TABLE_TOP - 0.01))

    verdict = verify_patch(_lattice(occluded), patch)
    assert verdict is not None and verdict[0] is PlaceRefusal.SUPPORT_OCCLUDED

    def arrived(c: NDArray[np.float64]) -> NDArray[np.bool_]:
        return _table(c) | _blob(c, (x, y, _TABLE_TOP + 0.05))

    verdict = verify_patch(_lattice(arrived), patch)
    assert verdict is not None and verdict[0] is PlaceRefusal.FREE_VOLUME_OCCUPIED


def test_the_set_down_payload_is_a_lost_view_not_a_contradiction() -> None:
    """Its own cells would contradict; excluded (as the octomap bridge clears them), the
    support under it is out of view too, so the latched patch rides its freeze TTL."""
    patch = _patch()
    x, y, _ = patch.region.pose.xyz
    resting = _lattice(lambda c: _table(c) | _blob(c, (x, y, _TABLE_TOP + 0.05)))
    verdict = verify_patch(resting, patch)
    assert verdict is not None and verdict[0] is PlaceRefusal.FREE_VOLUME_OCCUPIED
    verdict = verify_patch(resting, patch, _posed(_TABLE_TOP + 0.002, (x, y)))
    assert verdict is not None and verdict[0] is PlaceRefusal.SUPPORT_OCCLUDED


# ── the tick loop: the goal scope, and no place target named ─────────────────


def _goal(stamp_ns: int = _STAMP, timeout_s: float = 60.0) -> PlaceDeclaration:
    """The runner's goal-scope declaration (``place_approach_enabled``): names nothing."""
    return PlaceDeclaration(
        target_id="surface", rskill_id="openral/itest-place", timeout_s=timeout_s, stamp_ns=stamp_ns
    )


def _tracker(lines: list[str], *, goal: bool = True) -> PlaceTargetTracker:
    tracker = PlaceTargetTracker(freeze_s=_FREEZE_S, extrinsic_error_m=_EXT, log=lines.append)
    if goal:
        tracker.on_declaration(_goal())
    return tracker


def _tick(
    tracker: PlaceTargetTracker,
    grid: VoxelLattice,
    *,
    bottom_z: float | None,
    xy: tuple[float, float] = (0.45, 0.0),
    now_ns: int,
    released: bool = False,
    grid_stamp_ns: int | None = None,
) -> None:
    """One tick with the payload held at ``bottom_z`` (or released there / absent).

    ``grid_stamp_ns`` is the grid's ``source_stamp`` (defaults to ``now_ns``: fresh data).
    """
    obj, t = _payload(bottom_z=bottom_z if bottom_z is not None else 0.5, xy=xy)
    posed = _posed_primitives(obj, t)
    key = (obj.object_id, obj.stamp_ns)
    place_tick(
        tracker,
        grid,
        now_ns if grid_stamp_ns is None else grid_stamp_ns,
        carried=None if released or bottom_z is None else (obj, posed),
        published={} if bottom_z is None else {key: posed},
        extrinsic_error_m=_EXT,
        search_depth_m=_DEPTH,
        now_ns=now_ns,
    )


def test_a_surface_under_the_held_payload_rides_the_goal_declaration() -> None:
    lines: list[str] = []
    tracker = _tracker(lines)
    envelope = tracker.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is None, "goal declared, nothing measured"
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    envelope = tracker.envelope(now_ns=_STAMP + 10 * _MS)
    assert envelope is not None and envelope.region is not None
    assert envelope.target_id == "surface", "the goal's own target, never a producer one"
    assert envelope.stamp_ns == _STAMP and envelope.timeout_s == 60.0, "the goal's backstop"
    assert envelope.object_id == "grasped_payload:left_gripper", "scoped to that payload"
    assert envelope.region.stamp_ns == _STAMP
    assert envelope.is_live(now_ns=_STAMP + 10 * _MS)
    assert any(line.startswith("place_target_latched") for line in lines)
    assert any("not a proven support" in line for line in lines)


def test_occlusion_freezes_the_region_and_its_age_never_passes_the_kernel_bound() -> None:
    """The kernel drops a place region older than 2 x its voxel deadline; so must we."""
    lines: list[str] = []
    tracker = _tracker(lines)
    now = _STAMP
    for step in range(5):  # descending onto the table while it is in view
        _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10 - 0.015 * step, now_ns=now)
        now += 500 * _MS
    region = tracker.patch.region if tracker.patch else None
    assert region is not None
    last_verified = region.stamp_ns
    x, y, _ = region.pose.xyz

    def occluded(c: NDArray[np.float64]) -> NDArray[np.bool_]:
        return _table(c) & ~_blob(c, (x, y, _TABLE_TOP - 0.01))

    bound_ns = int(2 * _GRID_MAX_AGE_S * 1e9)
    held_until = last_verified
    while now < last_verified + 2 * bound_ns:
        _tick(tracker, _lattice(occluded), bottom_z=_TABLE_TOP + 0.004, xy=(x, y), now_ns=now)
        envelope = tracker.envelope(now_ns=now)
        if envelope is not None and envelope.region is not None:
            assert now - envelope.region.stamp_ns <= bound_ns
            held_until = now
        now += 100 * _MS
    assert held_until - last_verified >= bound_ns - 100 * _MS, "held through the freeze"
    assert any("view lost" in line for line in lines)
    assert any("reason=freeze_ttl" in line for line in lines)


def test_the_payload_moving_off_the_patch_re_measures_or_retracts() -> None:
    lines: list[str] = []
    tracker = _tracker(lines)
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    assert tracker.patch is not None
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, xy=(0.45, 0.30), now_ns=_STAMP)
    envelope = tracker.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is None, "off the table: retracted"
    assert lines[-1].startswith("place_target_retracted reason=no_surface")


def test_the_region_carries_through_the_release_window_then_dies_with_the_payload() -> None:
    tracker = _tracker([])
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    x, y, _ = tracker.patch.region.pose.xyz if tracker.patch else (0, 0, 0)
    resting = _lattice(lambda c: _table(c) | _blob(c, (x, y, _TABLE_TOP + 0.05)))
    _tick(tracker, resting, bottom_z=_TABLE_TOP + 0.002, xy=(x, y), now_ns=_STAMP, released=True)
    envelope = tracker.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is not None, "frozen record keeps it"
    _tick(tracker, resting, bottom_z=None, now_ns=_STAMP)  # the window closed
    envelope = tracker.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is None


def test_a_dispatch_declaration_is_an_optional_hint() -> None:
    def declaration(box_y: float) -> PlaceDeclaration:
        return PlaceDeclaration(
            target_id="surface:shelf",
            rskill_id="openral/itest-place",
            timeout_s=60.0,
            stamp_ns=_STAMP,
            search_box=PlaceRegion(
                frame_id=_BASE,
                pose=Pose6D(xyz=(0.45, box_y, 0.3), quat_xyzw=(0, 0, 0, 1), frame_id=_BASE),
                half_extents=(0.2, 0.2, 0.1),
            ),
        )

    named = _tracker([], goal=False)
    named.on_declaration(declaration(0.0))
    _tick(named, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    envelope = named.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is not None
    assert envelope.target_id == "surface:shelf", "dispatch's own fields when it names one"

    elsewhere = _tracker([], goal=False)
    elsewhere.on_declaration(declaration(-0.4))
    _tick(elsewhere, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    envelope = elsewhere.envelope(now_ns=_STAMP)
    assert envelope is not None and envelope.region is None
    assert elsewhere.patch is None


def test_the_freeze_may_not_exceed_the_kernels_region_age_bound() -> None:
    bridge = cast("VisionAttachmentBridge", object())
    too_long = VisionAttachmentConfig(
        place_target_enabled=True, grid_max_age_s=1.0, place_target_freeze_s=2.5
    )
    with pytest.raises(ROSConfigError, match="2 x grid_max_age_s"):
        PlaceTargetLeg(object(), bridge, too_long)
    PlaceTargetLeg(object(), bridge, VisionAttachmentConfig(place_target_enabled=True))


# ── witness ──────────────────────────────────────────────────────────────────


def _armed(lines: list[str]) -> PlaceTargetTracker:
    tracker = _tracker(lines)
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    assert tracker.patch is not None
    return tracker


def _on_patch(
    tracker: PlaceTargetTracker, dz: float = 0.004, xy: tuple[float, float] | None = None
) -> tuple[AttachedCollisionObject, NDArray[np.float64]]:
    patch = tracker.patch
    assert patch is not None
    x, y, _ = patch.region.pose.xyz
    return _payload(bottom_z=patch.plane_z + dz, xy=xy or (x, y))


def test_the_witness_attests_once_on_a_loaded_payload_at_the_measured_plane() -> None:
    lines: list[str] = []
    tracker = _armed(lines)
    obj, t_base_link = _on_patch(tracker)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=_STAMP + 100 * _MS)
    (decorated,) = tracker.decorate([obj])
    witness = decorated.support_contact
    assert witness is not None
    assert witness.support_id == "surface"
    assert witness.evidence_kind is AttachmentEvidenceKind.MAP_SUPPORT_PROXIMITY
    assert "not sensed contact" in (witness.evidence_ref or "")
    assert "not a proven support" in (witness.evidence_ref or "")
    assert witness.max_penetration_m == pytest.approx(min(_EXT, 0.01))
    assert 0.0 < witness.patch_radius_m <= 0.5
    # The plane in the OBJECT frame: 4 mm + the box half-height below the object origin.
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0))
    assert witness.contact_point_in_object[2] == pytest.approx(-(0.05 + 0.004))
    assert decorated.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE, "object kind kept"
    assert any("not sensed contact" in line for line in lines)
    # Once per declaration: the same witness keeps riding; no re-attest.
    assert not tracker.attest([(obj, t_base_link, True)], now_ns=_STAMP + 200 * _MS)
    assert tracker.decorate([obj])[0].support_contact == witness


@pytest.mark.parametrize(
    ("dz", "xy", "loaded"),
    [
        (0.03, None, True),  # above max(1 voxel, 15 mm extrinsic) = 20 mm
        (0.004, None, False),  # the trigger no longer reads loaded
        (0.004, (0.45, 0.30), True),  # centre off the measured patch
    ],
)
def test_no_witness_off_the_plane_unloaded_or_off_the_patch(
    dz: float, xy: tuple[float, float] | None, loaded: bool
) -> None:
    tracker = _armed([])
    obj, t_base_link = _on_patch(tracker, dz, xy)
    assert not tracker.attest([(obj, t_base_link, loaded)], now_ns=_STAMP + 100 * _MS)
    assert tracker.decorate([obj])[0].support_contact is None


def test_a_retracted_region_drops_the_witness() -> None:
    lines: list[str] = []
    tracker = _armed(lines)
    obj, t_base_link = _on_patch(tracker)
    now = _STAMP + 100 * _MS
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now)
    tracker.refuse(PlaceRefusal.FREE_VOLUME_OCCUPIED, "x", now_ns=now)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now + 100 * _MS), "drop publishes"
    assert tracker.decorate([obj])[0].support_contact is None
    assert any("witness dropped" in line for line in lines)
    envelope = tracker.envelope(now_ns=now)
    assert envelope is not None and envelope.region is None


def test_a_frozen_release_record_keeps_the_witness_until_the_window_closes() -> None:
    """Design note §2.3 "Release": the set-down payload keeps its map-support witness."""
    lines: list[str] = []
    tracker = _armed(lines)
    obj, t_base_link = _on_patch(tracker)
    description = _openarm()
    joint = next(j for j in description.joints if j.name == "left_gripper")
    trigger = PositionStallTrigger(
        description,
        joint_name="left_gripper",
        config=PositionStallConfig(consecutive_s=0.0, settle_s=0.03),
    )
    trigger.command(0.0)  # closed on the payload: the jaw stalls 0.2 rad short
    for stamp in range(2):
        trigger.update(
            JointState(name=["left_gripper"], position=[0.2], stamp_ns=stamp * 33_333_333)
        )
    assert trigger.attached
    leg = _GripperLeg(
        joint_name="left_gripper",
        trigger=trigger,
        producer=VisionAttachmentEvidenceProducer(description, gripper_joint="left_gripper"),
        object_id=obj.object_id,
        tcp_frame="",
        tcp_in_link=joint.origin_xyz,
        attachment=obj,
    )
    now = _STAMP + 100 * _MS
    assert tracker.attest(_witness_candidates([leg], lambda _: t_base_link), now_ns=now)
    witness = tracker.decorate([obj])[0].support_contact
    assert witness is not None
    trigger.command(0.7)  # released: the jaw opens past its hold
    trigger.update(JointState(name=["left_gripper"], position=[0.5], stamp_ns=2 * 33_333_333))
    assert not trigger.attached
    leg.release = ReleaseWindow.open(
        description, obj, base_link=_BASE, t_base_from_link=t_base_link, now_s=0.0
    )
    leg.attachment = None
    record = leg.release.record
    assert (record.object_id, record.stamp_ns) == (obj.object_id, obj.stamp_ns), "key preserved"
    assert not tracker.attest(
        _witness_candidates([leg], lambda _: t_base_link), now_ns=now + 100 * _MS
    ), "the witness carries through the window"
    assert tracker.decorate([record])[0].support_contact == witness, "and rides the record"
    leg.release = None
    assert tracker.attest(_witness_candidates([leg], lambda _: t_base_link), now_ns=now + 200 * _MS)
    assert tracker.decorate([record])[0].support_contact is None


def test_the_freeze_ttl_takes_the_witness_with_the_region() -> None:
    """No part of the place allowance outlives the region age the kernel accepts."""
    lines: list[str] = []
    tracker = _armed(lines)
    obj, t_base_link = _on_patch(tracker)
    now = _STAMP + 100 * _MS
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now)
    late = _STAMP + int(_FREEZE_S * 1e9) + 100 * _MS
    tracker.refuse(PlaceRefusal.SUPPORT_OCCLUDED, "occluded", now_ns=late)
    envelope = tracker.envelope(now_ns=late)
    assert tracker.patch is None and envelope is not None and envelope.region is None
    assert tracker.attest([(obj, t_base_link, True)], now_ns=late), "the drop re-publishes"
    assert tracker.decorate([obj])[0].support_contact is None


# ── goal scope: retraction, expiry, re-declaration, stalled data ─────────────


def test_no_goal_no_measurement_and_no_allowance() -> None:
    lines: list[str] = []
    tracker = _tracker(lines, goal=False)
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    assert tracker.patch is None and tracker.envelope(now_ns=_STAMP) is None
    assert lines[-1].startswith("place_target_refused reason=no_declaration")


def test_a_retraction_during_the_hold_ends_the_allowance_until_a_new_declaration() -> None:
    """Goal end / cancel / E-stop: nothing about the hold may re-arm by itself."""
    lines: list[str] = []
    tracker = _armed(lines)
    obj, t_base_link = _on_patch(tracker)
    now = _STAMP + 100 * _MS
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now)
    tracker.on_declaration(_goal().model_copy(update={"active": False}))
    assert tracker.patch is None and tracker.envelope(now_ns=now) is None
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now), "the drop re-publishes"
    assert tracker.decorate([obj])[0].support_contact is None
    assert any("place_target_retracted reason=no_declaration" in line for line in lines)
    # The same payload, still held over the same table, for well past the witness period.
    for step in range(1, 6):
        _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=now + step * _MS)
        assert not tracker.attest([(obj, t_base_link, True)], now_ns=now + step * _MS)
        assert tracker.envelope(now_ns=now + step * _MS) is None
        assert tracker.decorate([obj])[0].support_contact is None
    # A new goal re-measures and re-arms, under its own stamp.
    tracker.on_declaration(_goal(stamp_ns=now + 10 * _MS))
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=now + 20 * _MS)
    envelope = tracker.envelope(now_ns=now + 20 * _MS)
    assert envelope is not None and envelope.region is not None
    assert envelope.stamp_ns == now + 10 * _MS
    obj, t_base_link = _on_patch(tracker)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=now + 30 * _MS)
    assert tracker.decorate([obj])[0].support_contact is not None


def test_an_expired_goal_declaration_ends_the_allowance_with_a_logged_transition() -> None:
    lines: list[str] = []
    tracker = _tracker(lines, goal=False)
    tracker.on_declaration(_goal(timeout_s=1.0))
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP)
    obj, t_base_link = _on_patch(tracker)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=_STAMP + 100 * _MS)
    late = _STAMP + 1100 * _MS
    assert tracker.envelope(now_ns=late) is None and tracker.patch is None
    assert any("expired" in line and "reason=no_declaration" in line for line in lines)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=late), "the witness drop publishes"
    assert tracker.decorate([obj])[0].support_contact is None


def test_a_new_declaration_rechecks_the_latched_patch_against_its_hint() -> None:
    lines: list[str] = []
    tracker = _armed(lines)
    assert tracker.patch is not None
    elsewhere = PlaceDeclaration(
        target_id="surface:shelf",
        timeout_s=60.0,
        stamp_ns=_STAMP + 1,
        search_box=PlaceRegion(
            frame_id=_BASE,
            pose=Pose6D(xyz=(0.45, -0.4, 0.3), quat_xyzw=(0, 0, 0, 1), frame_id=_BASE),
            half_extents=(0.2, 0.2, 0.1),
        ),
    )
    tracker.on_declaration(elsewhere)
    assert tracker.patch is None, "the latch from the previous declaration does not survive"
    envelope = tracker.envelope(now_ns=_STAMP + 2)
    assert envelope is not None and envelope.region is None
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=_STAMP + 2)
    assert tracker.patch is None, "re-measured under the new hint: outside it"
    assert lines[-1].startswith("place_target_refused reason=outside_hint")


def test_a_stalled_map_ages_the_region_out_on_its_data_stamp() -> None:
    """An octree republished unchanged keeps a fresh header but an old ``source_stamp``:
    re-verifying on it never makes the region younger than the data."""
    tracker = _tracker([])
    data_ns = _STAMP
    now = _STAMP
    _tick(tracker, _lattice(_table), bottom_z=_TABLE_TOP + 0.10, now_ns=now, grid_stamp_ns=data_ns)
    bound_ns = int(_FREEZE_S * 1e9)
    while now <= data_ns + bound_ns + 200 * _MS:
        _tick(
            tracker,
            _lattice(_table),
            bottom_z=_TABLE_TOP + 0.10,
            now_ns=now,
            grid_stamp_ns=data_ns,
        )
        envelope = tracker.envelope(now_ns=now)
        assert envelope is not None
        if envelope.region is not None:
            assert envelope.region.stamp_ns == data_ns
            assert now - data_ns <= bound_ns
        now += 100 * _MS
    envelope = tracker.envelope(now_ns=now)
    assert envelope is not None and envelope.region is None
