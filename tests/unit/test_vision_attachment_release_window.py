"""The frozen release window, on the real OpenArm manifest (design note §1.6, §2.3 "Release").

The position-stall DETACH fires when the jaws open, while the fingers still surround the
released object. Dropping the attachment then lets the octomap re-mark the object inside the
fingers' 20 mm world margin and the first retreat chunk stops on it. The bridge instead keeps
the payload as a checked attached record frozen in the base frame until the links it exempts
are clear, a timeout, or a new grasp. What is decided here without a ROS graph:

* ``freeze_released_attachment`` — the record moves onto the collision root (``openarm_base``,
  identity FK, no primitives), keeps its geometry and evidence, and exempts exactly what the
  held record exempted (the hand and its jaws).
* ``box_gap_lower_bound_m`` — exact on a face normal, never above the true distance.
* ``ReleaseWindow`` — the separation test on the manifest's real hand and finger geometry, the
  timeout, and "unknown is never clear".

The ATTACH-closes path and the published wire are covered live by
``tests/integration/test_vision_attachment_release_window_live.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import AttachedCollisionObject, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_hal._vision_attachment_evidence import VisionAttachmentEvidenceProducer
from openral_hal.vision_attachment_bridge import (
    ReleaseWindow,
    VisionAttachmentBridge,
    VisionAttachmentConfig,
    _release_base_link,
    _xyz_rpy_matrix,
    box_gap_lower_bound_m,
    freeze_released_attachment,
)

_OPENARM = Path("robots/openarm/robot.yaml")
_BASE = "openarm_base"
_HAND = "openarm_left_link7"
_FINGER = "openarm_left_finger_pair"
_CLEAR_M = VisionAttachmentConfig().release_clear_m
_TIMEOUT_S = VisionAttachmentConfig().release_timeout_s


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_OPENARM))


def _hand_at_zero(description: RobotDescription) -> np.ndarray:
    """``openarm_base <- openarm_left_link7`` at q = 0: the manifest chain's origins composed."""
    edges = {e.child_link: e for e in [*description.joints, *description.fixed_attachments]}
    t, link = np.eye(4), _HAND
    while link in edges:
        edge = edges[link]
        t = _xyz_rpy_matrix(edge.origin_xyz, edge.origin_rpy) @ t
        link = edge.parent_link
    assert link == _BASE
    return t


def _held(description: RobotDescription) -> AttachedCollisionObject:
    """What the left leg holds after an ATTACH the segmenter never answered: the jaw box."""
    joint = next(j for j in description.joints if j.name == "left_gripper")
    producer = VisionAttachmentEvidenceProducer(description, gripper_joint=joint.name)
    held, _ = producer.on_grasp(
        masks=[],
        depth_m=np.zeros((0, 0)),
        intrinsics=None,
        t_link_from_cam=np.eye(4),
        tcp_in_link=joint.origin_xyz,
        object_id="grasped_payload:left_gripper",
        stamp_ns=1_000,
    )
    assert held is not None
    return held


def _lifted(t: np.ndarray, dz: float) -> np.ndarray:
    moved = t.copy()
    moved[2, 3] += dz
    return moved


def test_the_frozen_record_sits_on_the_collision_root_and_exempts_what_the_held_one_did(
    description: RobotDescription,
) -> None:
    held = _held(description)
    t_hand = _hand_at_zero(description)
    frozen = freeze_released_attachment(held, base_link=_BASE, t_base_from_link=t_hand)

    assert frozen.attach_link == _BASE == frozen.pose_in_link.frame_id
    assert frozen.touch_links == [_HAND, _FINGER]
    assert {held.attach_link, *held.touch_links} == set(frozen.touch_links)
    expected = t_hand @ homogeneous_from_quat_xyz(
        held.pose_in_link.xyz, held.pose_in_link.quat_xyzw
    )
    got = homogeneous_from_quat_xyz(frozen.pose_in_link.xyz, frozen.pose_in_link.quat_xyzw)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert frozen.primitives == held.primitives
    assert (frozen.object_id, frozen.evidence_kind, frozen.confidence, frozen.stamp_ns) == (
        held.object_id,
        held.evidence_kind,
        held.confidence,
        held.stamp_ns,
    )
    assert frozen.evidence_ref == f"{held.evidence_ref}|frozen_release"


def test_the_base_frame_is_proven_to_be_the_root(description: RobotDescription) -> None:
    assert _release_base_link(description) == _BASE
    with pytest.raises(ROSConfigError, match="not the root"):
        _release_base_link(description.model_copy(update={"base_frame": _HAND}))


@pytest.mark.parametrize("field", ["release_clear_m", "release_timeout_s"])
def test_non_positive_release_bounds_are_refused(description: RobotDescription, field: str) -> None:
    with pytest.raises(ROSConfigError, match="must be positive"):
        VisionAttachmentBridge(
            None, description, config=VisionAttachmentConfig(camera="head_zed", **{field: 0.0})
        )


def test_box_gap_is_exact_on_a_face_and_never_above_the_true_distance() -> None:
    unit = (np.zeros(3), np.eye(3), np.full(3, 0.05))
    assert box_gap_lower_bound_m(
        unit, (np.array([0.0, 0.13, 0.0]), np.eye(3), np.full(3, 0.05))
    ) == (pytest.approx(0.03))
    rng = np.random.default_rng(7)

    def rotation() -> np.ndarray:
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        return q * np.sign(np.diag(r))

    def surface(box: tuple[np.ndarray, np.ndarray, np.ndarray], n: int) -> np.ndarray:
        centre, rot, half = box
        local = rng.uniform(-1.0, 1.0, size=(n, 3))
        face = rng.integers(0, 3, size=n)
        local[np.arange(n), face] = np.sign(local[np.arange(n), face])
        return centre + (local * half) @ rot.T

    for _ in range(50):
        a = (rng.normal(scale=0.1, size=3), rotation(), rng.uniform(0.01, 0.06, size=3))
        b = (rng.normal(scale=0.1, size=3), rotation(), rng.uniform(0.01, 0.06, size=3))
        gap = box_gap_lower_bound_m(a, b)
        sampled = np.min(
            np.linalg.norm(surface(a, 400)[:, None, :] - surface(b, 400)[None, :, :], axis=-1)
        )
        # A sampled pair's distance is >= the true distance >= the bound.
        assert gap <= sampled + 1e-12


def test_the_window_closes_on_separation_only_once_hand_and_jaws_are_clear(
    description: RobotDescription,
) -> None:
    t_hand = _hand_at_zero(description)
    window = ReleaseWindow.open(
        description, _held(description), base_link=_BASE, t_base_from_link=t_hand, now_s=10.0
    )
    jaws = {"left_gripper": 0.0}

    def reason(dz: float) -> tuple[float | None, str]:
        clearance = window.clearance_m(_lifted(t_hand, dz), jaws)
        return clearance, window.close_reason(
            now_s=10.5, clearance_m=clearance, clear_m=_CLEAR_M, timeout_s=_TIMEOUT_S
        )

    at_release, verdict = reason(0.0)
    assert at_release is not None and at_release <= 0.0, "the jaws surround the payload"
    assert verdict == ""
    # 10 cm up the frozen payload still overlaps the finger box: the window holds.
    assert reason(0.10)[1] == ""
    # Pinned on the manifest's hand + finger boxes against the 10 cm jaw box: at 20 cm the
    # gap is positive but inside margin + one voxel — still held; at 25 cm it is clear.
    gap_20, verdict_20 = reason(0.20)
    assert gap_20 is not None and 0.0 < gap_20 <= _CLEAR_M and verdict_20 == ""
    gap_25, verdict_25 = reason(0.25)
    assert gap_25 is not None and gap_25 > _CLEAR_M and verdict_25 == "separation"
    # The payload never moved: the record's pose is the DETACH pose throughout.
    assert window.record.pose_in_link.xyz == pytest.approx(
        tuple((t_hand @ np.array([*_held(description).pose_in_link.xyz, 1.0]))[:3])
    )


def test_an_unknown_jaw_angle_is_never_clear_and_the_timeout_still_bounds_it(
    description: RobotDescription,
) -> None:
    t_hand = _hand_at_zero(description)
    window = ReleaseWindow.open(
        description, _held(description), base_link=_BASE, t_base_from_link=t_hand, now_s=10.0
    )
    far = _lifted(t_hand, 1.0)
    assert window.clearance_m(far, {}) is None
    assert window.clearance_m(far, {"left_gripper": 0.0}) > _CLEAR_M

    def verdict(now_s: float) -> str:
        return window.close_reason(
            now_s=now_s, clearance_m=None, clear_m=_CLEAR_M, timeout_s=_TIMEOUT_S
        )

    assert verdict(10.0 + _TIMEOUT_S - 0.01) == ""
    assert verdict(10.0 + _TIMEOUT_S) == "timeout"
