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

import pytest
from openral_core import DeployScene, GraspDeclaration, PlaceRegion, Pose6D, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_target_leg import GraspTargetTracker, support_z_of
from openral_hal.vision_attachment_bridge import VisionAttachmentBridge, VisionAttachmentConfig

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
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


def test_support_plane_is_the_search_box_bottom_and_must_be_level() -> None:
    assert support_z_of(_box(z=0.10)) == pytest.approx(0.0)
    tilted = _box().model_copy(
        update={
            "pose": Pose6D(
                xyz=(0.45, 0.0, 0.1), quat_xyzw=(0.0, 0.2588, 0.0, 0.9659), frame_id="openarm_base"
            )
        }
    )
    with pytest.raises(ROSConfigError, match="gravity-aligned"):
        support_z_of(tilted)


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
