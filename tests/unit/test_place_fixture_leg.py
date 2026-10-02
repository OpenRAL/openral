"""Real place producer leg: unit fixture verified against the map, proximity witness.

Real pick-and-place design §2.3. Pure core of ``_place_fixture_leg`` (no ROS): the real
OpenArm manifest, the test unit ``tests/unit/fixtures/robot_units/openarm_shelf_cell.yaml``
loaded through the real ``load_robot_unit`` (its ``cell:shelf_top`` is a 40 x 80 cm slab,
top face at z = 0.31 m in ``openarm_base``), a real ``VoxelLattice`` at the real cell's
20 mm, and the vision producer's own fallback payload (the jaw-span box on ``link7``).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachmentEvidenceKind,
    JointState,
    PlaceDeclaration,
    Pose6D,
    RobotDescription,
    UnitFixture,
    load_robot_unit,
)
from openral_hal._grasp_target import VoxelLattice
from openral_hal._grasp_trigger import GraspTriggerConfig, GripperEffortTrigger
from openral_hal._place_fixture_leg import (
    PlaceFixtureTracker,
    _posed_primitives,
    _witness_candidates,
    fixture_region,
    verify_fixture,
)
from openral_hal._vision_attachment_evidence import (
    VisionAttachmentEvidenceProducer,
    VisionGateConfig,
    jaw_span_primitive,
)
from openral_hal.vision_attachment_bridge import ReleaseWindow, _GripperLeg

_ROOT = Path(__file__).resolve().parents[2]
_OPENARM = _ROOT / "robots" / "openarm" / "robot.yaml"
_SHELF_UNIT = _ROOT / "tests" / "unit" / "fixtures" / "robot_units" / "openarm_shelf_cell.yaml"
_BASE = "openarm_base"
_RES = 0.02
_FACE_Z = 0.31
_GRID_STAMP_NS = 5_000_000_000


@pytest.fixture(scope="module")
def shelf(tmp_path_factory: pytest.TempPathFactory) -> UnitFixture:
    robot_dir = tmp_path_factory.mktemp("robots") / "openarm"
    (robot_dir / "units").mkdir(parents=True)
    shutil.copy(_OPENARM, robot_dir / "robot.yaml")
    shutil.copy(_SHELF_UNIT, robot_dir / "units" / "shelf_cell.yaml")
    return load_robot_unit(robot_dir / "robot.yaml", "shelf_cell").fixture("cell:shelf_top")


def _lattice(occupied: Callable[[NDArray[np.float64]], NDArray[np.bool_]]) -> VoxelLattice:
    """A base-aligned 20 mm lattice over the shelf; cell centres at z = 0.30 + k * 0.02."""
    origin = np.array([0.20, -0.46, 0.21])
    size = (26, 46, 20)
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


def _slab(c: NDArray[np.float64], *, top: float = _FACE_Z) -> NDArray[np.bool_]:
    """The shelf: one cell layer whose centres sit half a voxel under ``top``."""
    return (
        (np.abs(c[:, 0] - 0.45) <= 0.20)
        & (np.abs(c[:, 1]) <= 0.40)
        & (np.abs(c[:, 2] - (top - _RES / 2)) < 1e-6)
    )


def _blob(c: NDArray[np.float64], centre: tuple[float, float, float]) -> NDArray[np.bool_]:
    return np.all(np.abs(c - np.asarray(centre)) <= 0.03, axis=1)


def _verify(shelf: UnitFixture, grid: VoxelLattice, **kw: object) -> tuple[str, str] | None:
    return verify_fixture(grid, shelf, min_face_cover=0.5, free_height_m=0.10, **kw)  # type: ignore[arg-type]  # reason: payload kw only


# ── verification ─────────────────────────────────────────────────────────────


def test_an_occupied_face_with_an_empty_volume_above_it_verifies(shelf: UnitFixture) -> None:
    assert _verify(shelf, _lattice(_slab)) is None


def test_a_face_two_voxels_off_the_survey_is_face_missing(shelf: UnitFixture) -> None:
    verdict = _verify(shelf, _lattice(lambda c: _slab(c, top=_FACE_Z + 2 * _RES)))
    assert verdict is not None and verdict[0] == "face_missing", verdict


def test_a_blob_above_the_face_is_free_volume_occupied(shelf: UnitFixture) -> None:
    verdict = _verify(shelf, _lattice(lambda c: _slab(c) | _blob(c, (0.45, 0.10, 0.39))))
    assert verdict is not None and verdict[0] == "free_volume_occupied", verdict


def test_a_grid_in_another_frame_is_frame_mismatch(shelf: UnitFixture) -> None:
    grid = _lattice(_slab)
    other = VoxelLattice(
        "world", grid.origin, grid.orientation_xyzw, grid.resolution, grid.size, grid.occupancy
    )
    verdict = _verify(shelf, other)
    assert verdict is not None and verdict[0] == "frame_mismatch"


def test_the_carried_payloads_own_cells_are_not_counted(shelf: UnitFixture) -> None:
    """The payload resting on the face: its residue fails the check unless excluded."""
    obj, t_base_link = _payload(bottom_z=_FACE_Z + 0.002)
    centre = t_base_link[:3, 3] + np.asarray(obj.pose_in_link.xyz)
    grid = _lattice(lambda c: _slab(c) | _blob(c, (centre[0], centre[1], centre[2])))
    assert _verify(shelf, grid) is not None
    assert _verify(shelf, grid, payload=_posed_primitives(obj, t_base_link)) is None


# ── tracker: declaration, region, grid freshness ─────────────────────────────


def _declaration(stamp_ns: int = 1_000_000_000, target: str = "cell:shelf_top") -> PlaceDeclaration:
    return PlaceDeclaration(
        target_id=target,
        rskill_id="openral/itest-place",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=60.0,
        stamp_ns=stamp_ns,
    )


def _verified_tracker(shelf: UnitFixture, lines: list[str]) -> PlaceFixtureTracker:
    tracker = PlaceFixtureTracker(fixtures=[shelf], unit="shelf_cell", log=lines.append)
    tracker.on_declaration(_declaration())
    assert tracker.grid_fresh(0.1)
    tracker.verified(
        fixture_region(shelf, unit="shelf_cell", grid_stamp_ns=_GRID_STAMP_NS), resolution=_RES
    )
    return tracker


def test_a_verified_fixture_rides_the_envelope_as_the_region(shelf: UnitFixture) -> None:
    lines: list[str] = []
    tracker = _verified_tracker(shelf, lines)
    envelope = tracker.envelope(now_ns=2_000_000_000)
    assert envelope is not None and envelope.region is not None
    region = envelope.region
    assert region.frame_id == _BASE
    assert region.half_extents == shelf.half_extents and region.geometry == ()
    assert region.evidence_ref == (
        f"unit_fixture:shelf_cell/cell:shelf_top@2026-10-02;map_verified@{_GRID_STAMP_NS}"
    )
    assert region.stamp_ns == _GRID_STAMP_NS
    assert any(line.startswith("place_fixture_verified") for line in lines)


def test_a_stale_grid_retracts_the_region_with_its_reason(shelf: UnitFixture) -> None:
    lines: list[str] = []
    tracker = _verified_tracker(shelf, lines)
    assert not tracker.grid_fresh(1.5)
    envelope = tracker.envelope(now_ns=2_000_000_000)
    assert envelope is not None and envelope.region is None, "the declaration stays, region-less"
    assert lines[-1].startswith("place_fixture_unverified reason=grid_stale")
    tracker.grid_fresh(1.6)
    assert sum("grid_stale" in line for line in lines) == 1, "logged once per transition"


def test_an_unknown_fixture_passes_the_declaration_region_less(shelf: UnitFixture) -> None:
    lines: list[str] = []
    tracker = PlaceFixtureTracker(fixtures=[shelf], unit="shelf_cell", log=lines.append)
    tracker.on_declaration(_declaration(target="cell:front_table"))
    envelope = tracker.envelope(now_ns=2_000_000_000)
    assert envelope is not None and envelope.target_id == "cell:front_table"
    assert envelope.region is None and tracker.fixture is None
    assert "reason=unknown_fixture" in lines[-1]


# ── witness ──────────────────────────────────────────────────────────────────


def _payload(
    *, bottom_z: float, xy: tuple[float, float] = (0.45, 0.0)
) -> tuple[AttachedCollisionObject, NDArray[np.float64]]:
    """The vision producer's fallback box on the left hand, posed so its bottom is at z."""
    description = RobotDescription.from_yaml(str(_OPENARM))
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


def test_the_witness_attests_once_on_a_loaded_payload_at_the_plane(shelf: UnitFixture) -> None:
    lines: list[str] = []
    tracker = _verified_tracker(shelf, lines)
    obj, t_base_link = _payload(bottom_z=_FACE_Z + 0.004)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=3_000_000_000)
    (decorated,) = tracker.decorate([obj])
    witness = decorated.support_contact
    assert witness is not None
    assert witness.support_id == "cell:shelf_top"
    assert witness.evidence_kind is AttachmentEvidenceKind.DECLARED_FIXTURE
    assert "not sensed contact" in (witness.evidence_ref or "")
    assert witness.max_penetration_m == pytest.approx(min(shelf.survey_uncertainty_m, 0.01))
    assert 0.0 < witness.patch_radius_m <= 0.5
    # The plane in the OBJECT frame: 4 mm + the box half-height below the object origin.
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0))
    assert witness.contact_point_in_object[2] == pytest.approx(-(0.05 + 0.004))
    assert decorated.evidence_kind is AttachmentEvidenceKind.GRIPPER_FORCE, "object kind kept"
    assert any("not sensed contact" in line for line in lines)
    # Once per declaration: the same witness (same stamp) keeps riding; no re-attest.
    assert not tracker.attest([(obj, t_base_link, True)], now_ns=3_100_000_000)
    assert tracker.decorate([obj])[0].support_contact == witness


@pytest.mark.parametrize(
    ("bottom_z", "xy", "loaded"),
    [
        (_FACE_Z + 0.03, (0.45, 0.0), True),  # above max(1 voxel, survey) = 20 mm
        (_FACE_Z + 0.004, (0.45, 0.0), False),  # the trigger no longer reads loaded
        (_FACE_Z + 0.004, (0.70, 0.0), True),  # centre off the face
    ],
)
def test_no_witness_off_the_plane_unloaded_or_off_the_face(
    shelf: UnitFixture, bottom_z: float, xy: tuple[float, float], loaded: bool
) -> None:
    tracker = _verified_tracker(shelf, [])
    obj, t_base_link = _payload(bottom_z=bottom_z, xy=xy)
    assert not tracker.attest([(obj, t_base_link, loaded)], now_ns=3_000_000_000)
    assert tracker.decorate([obj])[0].support_contact is None


def test_no_witness_without_a_verified_region(shelf: UnitFixture) -> None:
    tracker = _verified_tracker(shelf, [])
    tracker.grid_fresh(2.0)
    obj, t_base_link = _payload(bottom_z=_FACE_Z + 0.004)
    assert not tracker.attest([(obj, t_base_link, True)], now_ns=3_000_000_000)


def test_retraction_drops_it_and_a_new_declaration_re_arms(shelf: UnitFixture) -> None:
    lines: list[str] = []
    tracker = _verified_tracker(shelf, lines)
    obj, t_base_link = _payload(bottom_z=_FACE_Z + 0.004)
    assert tracker.attest([(obj, t_base_link, True)], now_ns=3_000_000_000)
    first = tracker.decorate([obj])[0].support_contact
    tracker.on_declaration(_declaration().model_copy(update={"active": False}))
    assert tracker.attest([(obj, t_base_link, True)], now_ns=3_100_000_000), "drop re-publishes"
    assert tracker.decorate([obj])[0].support_contact is None
    assert any("witness dropped" in line for line in lines)
    tracker.on_declaration(_declaration(stamp_ns=3_200_000_000))
    tracker.verified(
        fixture_region(shelf, unit="shelf_cell", grid_stamp_ns=_GRID_STAMP_NS), resolution=_RES
    )
    assert tracker.attest([(obj, t_base_link, True)], now_ns=3_300_000_000)
    second = tracker.decorate([obj])[0].support_contact
    assert second is not None and first is not None and second.stamp_ns != first.stamp_ns


def test_a_frozen_release_record_keeps_the_witness_until_the_window_closes(
    shelf: UnitFixture,
) -> None:
    """Design note §2.3 "Release": the set-down payload keeps its shelf witness."""
    lines: list[str] = []
    tracker = _verified_tracker(shelf, lines)
    obj, t_base_link = _payload(bottom_z=_FACE_Z + 0.004)
    description = RobotDescription.from_yaml(str(_OPENARM))
    joint = next(j for j in description.joints if j.name == "left_gripper")
    assert joint.effort_limit is not None
    trigger = GripperEffortTrigger(
        description, joint_name="left_gripper", config=GraspTriggerConfig(consecutive_ticks=1)
    )
    trigger.update(
        JointState(name=["left_gripper"], position=[0.0], effort=[joint.effort_limit], stamp_ns=0)
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
    # Held and loaded on the face: the witness arms.
    assert tracker.attest(_witness_candidates([leg], lambda _: t_base_link), now_ns=3_000_000_000)
    witness = tracker.decorate([obj])[0].support_contact
    assert witness is not None
    # DETACH: the trigger unloads, the attachment goes, the window freezes it in base.
    trigger.update(JointState(name=["left_gripper"], position=[0.0], effort=[0.0], stamp_ns=1))
    assert not trigger.attached
    leg.release = ReleaseWindow.open(
        description, obj, base_link=_BASE, t_base_from_link=t_base_link, now_s=0.0
    )
    leg.attachment = None
    record = leg.release.record
    assert (record.object_id, record.stamp_ns) == (obj.object_id, obj.stamp_ns), "key preserved"
    assert record.attach_link == _BASE
    assert not tracker.attest(
        _witness_candidates([leg], lambda _: t_base_link), now_ns=3_100_000_000
    ), "the witness carries through the window"
    (decorated,) = tracker.decorate([record])
    assert decorated.support_contact == witness, "and rides the frozen record"
    # The window closes: no candidate, the witness drops and the set re-publishes.
    leg.release = None
    assert tracker.attest(_witness_candidates([leg], lambda _: t_base_link), now_ns=3_200_000_000)
    assert tracker.decorate([record])[0].support_contact is None
    assert any("witness dropped" in line for line in lines)
