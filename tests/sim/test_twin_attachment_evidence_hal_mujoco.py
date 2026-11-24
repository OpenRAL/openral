# SPDX-License-Identifier: Apache-2.0
"""Sim grasp evidence on the bimanual and end-effector-gripper twins (#740).

``SimAttachmentEvidenceTracker`` arms one hand per gripper: aloha_agilex and r1pro
have two (one per arm), galaxea_a1 has one end-effector gripper with no manifest
joint. Each test runs the twin's real MJCF (``robots/<robot>/robot.yaml`` →
``assets.mjcf``) with a free box parked between each hand's open jaws on a small
mocap platform, closes and opens the grippers through their own actuators, and
reads what the tracker would publish on ``/openral/attachment_state``.

Gravity is off so a released box stays where the jaws left it until it is
nudged away (the tracker releases on contact loss plus relative motion, never on
contact loss alone).

What is proved:

* each hand attaches its own payload, on its own attach link and touch links, and
  releases it without disturbing the other hand's record;
* nothing is looser than the single-gripper case: every hand needs both of its own
  jaws, hands' jaws and touch links are disjoint, a payload held by one hand is
  never a candidate for the other, and a record that moves to another slot of the
  published set is published unchanged, its support witness included. This checks the
  tracker's snapshot only; how the safety kernel keeps a moved record's witness (it
  keys re-arming across slots) is
  `AWitnessKeepsItsStateWhenItsRecordMovesSlot` in the kernel's lifecycle suite.

No mocks (CLAUDE.md §1.11): when ``mujoco`` is missing or a twin's upstream files
cannot be fetched (``AssetFetchError``), that case skips with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any  # reason: mujoco ships no stubs, so MjModel/MjData handles are Any

import pytest

pytest.importorskip("mujoco")

import mujoco
import numpy as np
from openral_core import AttachedCollisionObject, GraspDeclaration, RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_core.exceptions import ROSConfigError
from openral_core.tracking_thresholds import load_tracking_thresholds
from openral_hal._sim_attachment_evidence import (
    SimAttachmentEvidenceTracker,
    _ContactCandidate,
    _Hand,
)
from openral_hal.mujoco_twin import _PlanarBaseTwin

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_SETTLE_STEPS = 400


@dataclass
class _Gripper:
    hand: int
    actuator: int
    open_ctrl: float
    closed_ctrl: float
    #: The mirror finger's actuator (gym-aloha), driven at ``-ctrl`` as the HAL does.
    mirror: int | None = None


@dataclass
class _Rig:
    desc: RobotDescription
    model: Any
    data: Any
    tracker: SimAttachmentEvidenceTracker
    grippers: list[_Gripper]
    #: ``(qpos address, dof address, held value)`` of the R1 Pro's planar base joints: the
    #: HAL's ``_PlanarBaseTwin`` pins them before every physics step, and so does this rig.
    pinned: list[tuple[int, int, float]]
    step_ns: int = 0

    def run(self, steps: int) -> list[list[AttachedCollisionObject]]:
        """Step the physics; every set the tracker would publish, in order."""
        published = []
        for _ in range(steps):
            for qpos_addr, dof, value in self.pinned:
                self.data.qpos[qpos_addr] = value
                self.data.qvel[dof] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.step_ns += 1_000_000
            out = self.tracker.update(self.data, stamp_ns=self.step_ns)
            if out is not None:
                published.append(out)
        return published

    def set_gripper(self, hand: int, *, closed: bool) -> None:
        """Drive the gripper of tracker hand ``hand`` (the hand whose payload is ``hand``)."""
        (g,) = (g for g in self.grippers if g.hand == hand)
        _drive(self.data, g.actuator, g.mirror, g.closed_ctrl if closed else g.open_ctrl)

    def nudge_away(self, payload: int) -> None:
        """Push a released box out of the open jaws, as a lifting arm would."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"payload{payload}_free")
        self.data.qvel[int(self.model.jnt_dofadr[joint]) + 2] = -0.3


def _description(robot: str) -> tuple[RobotDescription, Path]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    assert desc.assets.mjcf is not None
    if str(desc.assets.mjcf).startswith("gym_aloha:"):
        pytest.importorskip("gym_aloha", reason="aloha_bimanual's twin ships with gym-aloha")
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"{robot} twin files not fetchable on this host: {exc}")
    assert path is not None
    return desc, path


def _compile(
    path: Path,
    boxes: list[tuple[Any, Any, list[float]]],
    *,
    support: bool = True,
    unnamed: frozenset[int] = frozenset(),
) -> tuple[Any, Any]:
    """The twin plus one free box per hand, each on a mocap pad.

    ``support``: the pad sits under the box in world -z, flush, so the attach-time
    support witness is real. Otherwise it is a backstop behind the box along the
    hand's approach axis (the box's local -z), which keeps closing jaws from squeezing
    the box into the palm; it attests nothing.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    spec.option.gravity = [0.0, 0.0, 0.0]
    for i, (pos, quat, half) in enumerate(boxes):
        # An unnamed payload is one the tracker refuses to attach (it raises).
        body = spec.worldbody.add_body(
            name="" if i in unnamed else f"payload{i}", pos=pos, quat=quat
        )
        body.add_freejoint(name=f"payload{i}_free")
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            mass=0.05,
            friction=[2.0, 0.05, 0.01],
            # Group 4: gym-aloha's compiler takes inertia only from geom groups 4-5
            # (`inertiagrouprange`); every other twin's default range includes it.
            group=4,
        )
        if not support:
            backstop = spec.worldbody.add_body(name=f"platform{i}", pos=pos, quat=quat, mocap=True)
            backstop.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half[0] * 0.6, half[1], 0.004],
                pos=[0.0, 0.0, -half[2] - 0.004],
            )
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=np.float64))
        depth = float(np.abs(rot.reshape(3, 3)[2]) @ np.asarray(half))
        platform = spec.worldbody.add_body(
            name=f"platform{i}", pos=np.asarray(pos) - [0.0, 0.0, depth + 0.004], mocap=True
        )
        platform.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, 0.005, 0.004])
    model = spec.compile()
    return model, mujoco.MjData(model)


def _gripper_actuators(
    model: Any, desc: RobotDescription, hands: list[_Hand]
) -> list[tuple[int, int]]:
    """``(hand index, actuator)`` per ``sim.grippers`` entry."""
    assert desc.sim is not None
    out = []
    for gripper in desc.sim.grippers:
        if gripper.actuator_index is not None:
            actuator = gripper.actuator_index
        else:
            sim_name = next(j.sim_joint_name for j in desc.joints if j.name == gripper.joint)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            actuator = next(a for a in range(model.nu) if _actuated_joint(model, a) == joint)
        body = int(model.jnt_bodyid[_actuated_joint(model, actuator)])
        hand = next(i for i, h in enumerate(hands) if body in h.body_groups)
        out.append((hand, actuator))
    return out


def _actuated_joint(model: Any, actuator: int) -> int:
    """The joint an actuator drives: its own, or a tendon's first (the Panda's split fingers)."""
    target = int(model.actuator_trnid[actuator, 0])
    if model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_TENDON:
        return int(model.wrap_objid[int(model.tendon_adr[target])])
    return target


def _drive(data: Any, actuator: int, mirror: int | None, value: float) -> None:
    """Command a gripper actuator, and its mirror finger at ``-value`` (``MujocoArmHAL``)."""
    data.ctrl[actuator] = value
    if mirror is not None:
        data.ctrl[mirror] = -value


def _mirror_actuators(
    desc: RobotDescription, actuators: list[tuple[int, int]]
) -> dict[int, int | None]:
    """Each gripper actuator's ``sim.grippers[].mirror_actuator_index``."""
    assert desc.sim is not None
    return {
        actuator: gripper.mirror_actuator_index
        for gripper, (_, actuator) in zip(desc.sim.grippers, actuators, strict=True)
    }


def _ctrl_ranges(
    model: Any, desc: RobotDescription, actuators: list[tuple[int, int]]
) -> dict[int, tuple[float, float]]:
    """Each gripper actuator's control range: the MJCF's, or the manifest's when it has none.

    The SO-101 twin's jaw actuator declares no ``ctrlrange``; the HAL drives it over the
    manifest's ``sim.grippers[].ctrl_range``.
    """
    assert desc.sim is not None
    out = {}
    for gripper, (_, actuator) in zip(desc.sim.grippers, actuators, strict=True):
        low, high = (float(v) for v in model.actuator_ctrlrange[actuator])
        lo, hi = gripper.ctrl_range
        out[actuator] = (low, high) if high > low else (float(lo), float(hi))
    return out


def _jaw_roots(model: Any, data: Any, hand: _Hand) -> tuple[Any, Any]:
    """World positions of the hand's two jaw roots (the finger bodies its joints move).

    A sliding jaw's origin; a hinged jaw's (the SO-ARM's, the OpenArm's) centre of mass,
    since it turns about its origin and only its mass centre opens and closes.
    """

    def at(root: int) -> Any:
        hinged = int(model.body_jntnum[root]) > 0 and model.jnt_type[
            int(model.body_jntadr[root])
        ] == int(mujoco.mjtJoint.mjJNT_HINGE)
        return (data.xipos if hinged else data.xpos)[root].copy()

    first, second = (at(root) for root in sorted(set(hand.body_groups.values())))
    return first, second


def _settle(model: Any, data: Any) -> None:
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)


def _payload_body(model: Any, index: int) -> int:
    """Payload ``index``'s body, found through its (always named) free joint."""
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"payload{index}_free")
    return int(model.jnt_bodyid[joint])


def _payload_clear(model: Any, data: Any, n_boxes: int) -> bool:
    """No payload touches anything but its own platform."""
    payloads = {_payload_body(model, i) for i in range(n_boxes)}
    platforms = {model.body(f"platform{i}").id for i in range(n_boxes)}
    for contact in data.contact[: data.ncon]:
        pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if pair & payloads and not pair & platforms:
            return False
    return True


def _rig(robot: str, *, support: bool = True, unnamed: frozenset[int] = frozenset()) -> _Rig:
    """The twin with one box between each hand's open jaws, touching nothing but its platform.

    The box spans the line between the jaw roots, centred on their midpoint and slid
    out along the hand's approach direction (attach body → jaws) until it clears the
    open jaws; it is two thirds of the jaws' open-minus-shut spread long.
    """
    desc, path = _description(robot)
    model, data = _compile(path, [])
    # The HAL's start pose: gym-aloha's fingers sit outside their limits at qpos 0.
    if desc.sim is not None and desc.sim.keyframe_index is not None:
        mujoco.mj_resetDataKeyframe(model, data, desc.sim.keyframe_index)
    mujoco.mj_forward(model, data)
    hands = SimAttachmentEvidenceTracker(model, desc)._hands
    actuators = _gripper_actuators(model, desc, hands)
    ranges = _ctrl_ranges(model, desc, actuators)
    mirrors = _mirror_actuators(desc, actuators)
    # Every joint-driven actuator holds its joint where it is.
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    # Which end of each gripper's ctrl range opens it: the one with the wider jaws.
    spreads = {}
    for end in (0, 1):
        for _, actuator in actuators:
            _drive(data, actuator, mirrors[actuator], ranges[actuator][end])
        _settle(model, data)
        spreads[end] = [
            float(np.linalg.norm(np.subtract(*_jaw_roots(model, data, h)))) for h in hands
        ]
    # Per hand: the OpenArm's mirrored grippers open at opposite ends of their ranges.
    open_end = [0 if spreads[0][i] > spreads[1][i] else 1 for i in range(len(hands))]
    for owner, actuator in actuators:
        _drive(data, actuator, mirrors[actuator], ranges[actuator][open_end[owner]])
    _settle(model, data)
    qpos, ctrl = data.qpos.copy(), data.ctrl.copy()
    placements = []
    for index, hand in enumerate(hands):
        p1, p2 = _jaw_roots(model, data, hand)
        axis = (p2 - p1) / np.linalg.norm(p2 - p1)
        middle = (p1 + p2) / 2
        approach = middle - data.xpos[hand.attach_body_id]
        approach -= (approach @ axis) * axis
        approach /= np.linalg.norm(approach)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(
            quat, np.stack([axis, np.cross(approach, axis), approach], axis=1).flatten()
        )
        half = [abs(spreads[0][index] - spreads[1][index]) / 3, 0.015, 0.015]
        placements.append((middle, approach, quat, half))
    for slide in np.arange(0.0, 0.08, 0.005):
        boxes = [(m + slide * a, q, h) for m, a, q, h in placements]
        model, data = _compile(path, boxes, support=support, unnamed=unnamed)
        data.qpos[: len(qpos)] = qpos
        data.ctrl[: len(ctrl)] = ctrl
        mujoco.mj_forward(model, data)
        if _payload_clear(model, data, len(boxes)):
            break
    else:
        pytest.fail(f"{robot}: no box fits between the open jaws without touching them")
    tracker = SimAttachmentEvidenceTracker(model, desc, stable_ticks=1)
    grippers = [
        _Gripper(
            hand=hand,
            actuator=actuator,
            mirror=mirrors[actuator],
            open_ctrl=float(ranges[actuator][open_end[hand]]),
            closed_ctrl=float(ranges[actuator][1 - open_end[hand]]),
        )
        for hand, actuator in actuators
    ]
    pinned = []
    for name in _PlanarBaseTwin.BASE_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            addr = int(model.jnt_qposadr[joint])
            pinned.append((addr, int(model.jnt_dofadr[joint]), float(data.qpos[addr])))
    return _Rig(
        desc=desc, model=model, data=data, tracker=tracker, grippers=grippers, pinned=pinned
    )


@pytest.mark.parametrize(
    ("robot", "attach_links", "support", "witnessed"),
    [
        # aloha_agilex's box rests flush on its platform at attach: a support witness.
        ("aloha_agilex", ("fl_link6", "fr_link6"), True, True),
        # r1pro's narrow jaws need the backstop pad, which attests no support: none.
        ("r1pro", ("left_gripper_link", "right_gripper_link"), False, False),
        # anvil_openarm_v2's hinged fingers close on a box over a pad they lift it off (#752).
        (
            "anvil_openarm_v2",
            ("openarm_left_ee_base_link", "openarm_right_ee_base_link"),
            True,
            False,
        ),
        # aloha_bimanual: the gym-aloha ViperX twin, resolved by sim_joint_name (#752).
        ("aloha_bimanual", ("left_link6", "right_link6"), True, False),
    ],
)
def test_each_hand_attaches_and_releases_independently(
    robot: str, attach_links: tuple[str, str], support: bool, witnessed: bool
) -> None:
    rig = _rig(robot, support=support)
    assert rig.tracker.attach_links == attach_links
    assert rig.run(50) == [], "a box touching no jaw attached"

    rig.set_gripper(0, closed=True)
    sets = rig.run(400)
    assert [[o.attach_link for o in s] for s in sets] == [[attach_links[0]]]
    left = sets[0][0]
    assert left.object_id == "sim:payload0"
    assert set(left.touch_links) == set(rig.tracker._hands[0].touch_links)

    rig.set_gripper(1, closed=True)
    sets = rig.run(400)
    assert len(sets) == 1, sets
    held_left, right = sets[0]
    assert held_left == left, "the right hand's attach re-measured the left hand's record"
    assert (right.object_id, right.attach_link) == ("sim:payload1", attach_links[1])
    assert set(right.touch_links) == set(rig.tracker._hands[1].touch_links)
    # The kernel keys witness re-arming by (object, support, stamp) across slots, so a
    # record that moves slot keeps its witness; aloha_agilex proves it on a real one.
    assert (right.support_contact is not None) == witnessed

    # A hinged jaw (anvil_openarm_v2) can lose contact and release while it opens.
    rig.set_gripper(0, closed=False)
    sets = rig.run(100)
    rig.nudge_away(0)
    sets += rig.run(400)
    released = [s for s in sets if [o.object_id for o in s] == ["sim:payload1"]]
    assert released, f"left release never left the right payload alone: {sets}"
    # The right record moved from slot 1 to slot 0 unchanged, witness included: the
    # kernel recognises its key in the new slot (no re-arm, no loss).
    assert released[0][0] == right

    rig.set_gripper(1, closed=False)
    final = rig.run(100)
    rig.nudge_away(1)
    final += rig.run(400)
    assert final and final[-1] == []


def test_end_effector_gripper_attaches_and_releases() -> None:
    """galaxea_a1: the gripper is an end effector with no manifest joint (PR #746)."""
    rig = _rig("galaxea_a1")
    (hand,) = rig.tracker._hands
    assert rig.tracker.attach_links == ("arm_seg6",)
    assert sorted(hand.touch_links) == ["gripper1", "gripper2"]
    assert rig.run(50) == []

    rig.set_gripper(0, closed=True)
    sets = rig.run(400)
    assert [[(o.object_id, o.attach_link) for o in s] for s in sets] == [
        [("sim:payload0", "arm_seg6")]
    ]

    rig.set_gripper(0, closed=False)
    rig.run(100)
    rig.nudge_away(0)
    final = rig.run(400)
    assert final and final[-1] == []


@pytest.mark.parametrize(
    "robot",
    [
        "aloha_agilex",
        "r1pro",
        "galaxea_a1",
        "widowx",
        "franka_panda",
        "so100_follower",
        "so101_follower",
        "anvil_openarm_v2",
        "aloha_bimanual",
    ],
)
def test_no_hand_is_looser_than_a_single_gripper(robot: str) -> None:
    """Every hand needs both of its own jaws; hands share no jaw and no touch link."""
    desc, path = _description(robot)
    model, _ = _compile(path, [])
    tracker = SimAttachmentEvidenceTracker(model, desc)
    assert tracker._required_contact_groups == 2  # every twin here is a parallel gripper
    seen_bodies: set[int] = set()
    seen_links: set[str] = set()
    for hand in tracker._hands:
        assert len(set(hand.body_groups.values())) == 2, hand.attach_link
        assert not seen_bodies & set(hand.body_groups)
        assert not seen_links & set(hand.touch_links)
        seen_bodies |= set(hand.body_groups)
        seen_links |= set(hand.touch_links)


@pytest.mark.parametrize(
    ("robot", "attach_link", "support"),
    [
        # The hand is a fixed child of panda_link7 named `hand` in the twin (#752).
        ("franka_panda", "panda_hand", True),
        # One moving jaw against the fixed jaw on the hand body (`fixed_jaw_link`, #752).
        ("so100_follower", "gripper_base", True),
        ("so101_follower", "gripper", False),
    ],
)
def test_single_gripper_twin_attaches_and_releases(
    robot: str, attach_link: str, support: bool
) -> None:
    """The twins issue #752 left publishing "nothing carried" while they held a payload."""
    rig = _rig(robot, support=support)
    assert rig.tracker.attach_links == (attach_link,)
    assert rig.run(50) == [], "a box touching no jaw attached"

    rig.set_gripper(0, closed=True)
    sets = rig.run(400)
    assert [[(o.object_id, o.attach_link) for o in s] for s in sets] == [
        [("sim:payload0", attach_link)]
    ]
    assert set(sets[0][0].touch_links) == set(rig.tracker._hands[0].touch_links)

    rig.set_gripper(0, closed=False)
    released = rig.run(100)
    rig.nudge_away(0)
    released += rig.run(400)
    assert released and released[-1] == []


def test_a_fixed_hand_link_resolves_only_at_its_declared_pose() -> None:
    """The Panda's `panda_hand` is the twin's `hand` only while the manifest transform fits."""
    desc, path = _description("franka_panda")
    model, _ = _compile(path, [])
    (row,) = (a for a in desc.fixed_attachments if a.child_link == "panda_hand")
    shifted = row.model_copy(update={"origin_xyz": [0.0, 0.0, 0.112]})  # 5 mm off the flange
    others = [a for a in desc.fixed_attachments if a is not row]
    with pytest.raises(ROSConfigError, match="attach link 'panda_hand'"):
        SimAttachmentEvidenceTracker(
            model, desc.model_copy(update={"fixed_attachments": [*others, shifted]})
        )


def test_the_fixed_jaw_is_the_second_contact_branch() -> None:
    """Without `fixed_jaw_link` the SO-101 has one jaw; one elsewhere than the hand refuses."""
    desc, path = _description("so101_follower")
    model, _ = _compile(path, [])

    def with_fixed_jaw(link: str | None) -> RobotDescription:
        joints = [
            j.model_copy(update={"fixed_jaw_link": link}) if j.role == "gripper" else j
            for j in desc.joints
        ]
        return desc.model_copy(update={"joints": joints})

    with pytest.raises(ROSConfigError, match="needs 2 physical contact branches, found 1"):
        SimAttachmentEvidenceTracker(model, with_fixed_jaw(None))
    with pytest.raises(ROSConfigError, match="fixed_jaw_link 'wrist'"):
        SimAttachmentEvidenceTracker(model, with_fixed_jaw("wrist"))
    (hand,) = SimAttachmentEvidenceTracker(model, desc)._hands
    assert hand.attach_body_id in hand.body_groups
    # Only the hand body itself: the moving jaw under it stays its own branch.
    assert len(set(hand.body_groups.values())) == 2


def test_a_held_payload_is_never_a_candidate_for_the_other_hand() -> None:
    """Both hands' jaws on the left hand's payload attach it once, to the left hand."""
    rig = _rig("aloha_agilex")
    rig.set_gripper(0, closed=True)
    (held,) = rig.run(400)[-1]
    left, right = rig.tracker._hands
    root = left.attached_root
    assert root is not None and held.attach_link == left.attach_link
    both_jaws = set(right.body_groups.values())
    # A stability count the right hand built before the left attached is discarded.
    right.candidates[root] = _ContactCandidate(contact_bodies=both_jaws, stable_ticks=2)
    assert not rig.tracker._update_hand(
        right, {root: both_jaws}, {root}, rig.data, stamp_ns=rig.step_ns + 1
    )
    assert right.attachment is None and right.attached_root is None
    assert root not in right.candidates
    # ...and when the right hand is no longer even touching it.
    right.candidates[root] = _ContactCandidate(contact_bodies=both_jaws, stable_ticks=2)
    assert not rig.tracker._update_hand(right, {}, {root}, rig.data, stamp_ns=rig.step_ns + 2)
    assert root not in right.candidates


def test_an_unattachable_payload_in_one_hand_does_not_block_the_other() -> None:
    """The right hand pinching an unnamed payload is refused for that hand alone.

    Every update keeps publishing: the left hand's release goes out while the right
    jaws are still closed on the unattachable body, which never attaches.
    """
    rig = _rig("aloha_agilex", unnamed=frozenset({1}))
    rig.set_gripper(0, closed=True)
    assert [[o.object_id for o in s] for s in rig.run(400)] == [["sim:payload0"]]

    rig.set_gripper(1, closed=True)
    assert rig.run(200) == [], "the unnamed payload never attaches (and nothing raises)"
    assert rig.tracker._hands[1].refused, "the refusal is recorded"

    rig.set_gripper(0, closed=False)
    rig.run(100)
    rig.nudge_away(0)
    published = rig.run(400)
    assert published and published[0] == [], f"the left release was held back: {published}"
    assert rig.tracker._hands[1].attached_root is None

    # A refusal is per payload, not per hand: the right hand still grasps a valid one.
    # Open it, take the unnamed box out, and put the (named) left box where it stood.
    rig.set_gripper(1, closed=False)
    rig.run(100)
    named, unnamed = (
        int(rig.model.jnt_qposadr[rig.model.joint(f"payload{i}_free").id]) for i in (0, 1)
    )
    named_dof, unnamed_dof = (
        int(rig.model.jnt_dofadr[rig.model.joint(f"payload{i}_free").id]) for i in (0, 1)
    )
    slot = rig.data.qpos[unnamed : unnamed + 7].copy()
    rig.data.qpos[unnamed : unnamed + 3] += [0.0, 0.0, -1.0]  # parked well below the hands
    rig.data.qpos[named : named + 7] = slot
    rig.data.qvel[named_dof : named_dof + 6] = 0.0
    rig.data.qvel[unnamed_dof : unnamed_dof + 6] = 0.0
    mujoco.mj_forward(rig.model, rig.data)
    rig.set_gripper(1, closed=True)
    published = rig.run(400)
    assert [[(o.object_id, o.attach_link) for o in s] for s in published][:1] == [
        [("sim:payload0", "fr_link6")]
    ], f"the right hand could not grasp a valid payload after a refusal: {published}"


def test_construction_refuses_what_the_kernel_would_reject() -> None:
    """Links the kernel model lacks, and two hands over one finger, refuse up front."""
    desc, path = _description("aloha_agilex")
    model, _ = _compile(path, [])
    tracker = SimAttachmentEvidenceTracker(model, desc)
    names = {n for h in tracker._hands for n in (h.attach_link, *h.touch_links)}
    assert tracker.bind_kernel_links(names) == []
    assert tracker.bind_kernel_links(names - {"fr_link8"}) == ["fr_link8"]
    # An end-effector entry over the left fingers as well: two hands, one gripper.
    assert desc.sim is not None
    qpos = [int(model.jnt_qposadr[model.joint(n).id]) for n in ("fl_joint7", "fl_joint8")]
    twice = desc.sim.grippers[0].model_copy(
        update={"joint": None, "end_effector": "left_gripper", "qpos_addrs": qpos}
    )
    sim = desc.sim.model_copy(update={"grippers": [*desc.sim.grippers, twice]})
    with pytest.raises(ROSConfigError, match="share finger bodies"):
        SimAttachmentEvidenceTracker(model, desc.model_copy(update={"sim": sim}))


def test_end_effector_jaws_are_only_the_listed_fingers() -> None:
    """A hinged sibling on the A1 wrist is no jaw and no touch link."""
    desc, path = _description("galaxea_a1")
    spec = mujoco.MjSpec.from_file(str(path))
    wrist = spec.body("arm_seg6")
    guard = wrist.add_body(name="tool_guard", pos=[0.0, 0.03, 0.0])
    guard.add_joint(name="tool_guard_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0])
    guard.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, 0.005, 0.005])
    model = spec.compile()
    (hand,) = SimAttachmentEvidenceTracker(model, desc)._hands
    assert sorted(hand.touch_links) == ["gripper1", "gripper2"]
    assert model.body("tool_guard").id not in hand.body_groups


@pytest.mark.parametrize(("robot", "per_hand"), [("aloha_agilex", 8), ("galaxea_a1", 16)])
def test_every_hand_gets_a_share_of_the_kernel_primitive_cap(robot: str, per_hand: int) -> None:
    """N hands' records together never exceed the kernel's ``attached_max_primitives``."""
    desc, path = _description(robot)
    model, _ = _compile(path, [])
    tracker = SimAttachmentEvidenceTracker(model, desc)
    cap = load_tracking_thresholds().caps.attached_max_primitives
    assert tracker._max_primitives_per_hand == per_hand
    assert tracker._max_primitives_per_hand * len(tracker._hands) <= cap


def test_names_are_adopted_only_from_a_twin_whose_chain_moves_alike() -> None:
    """``adopt_kernel_names`` compares the arm across configurations, not only at qpos0.

    The Panda twin against itself adopts. Against a copy whose ``joint4`` axis is flipped,
    the two chains agree at qpos0 (every joint at zero) but move the hand differently, so
    the names are refused and nothing changes.
    """
    desc, path = _description("franka_panda")
    scene = SimAttachmentEvidenceTracker(mujoco.MjModel.from_xml_path(str(path)), desc)
    same = SimAttachmentEvidenceTracker(mujoco.MjModel.from_xml_path(str(path)), desc)
    spec = mujoco.MjSpec.from_file(str(path))
    spec.joint("joint4").axis = -np.asarray(spec.joint("joint4").axis)
    spec.joint("joint4").range = -np.asarray(spec.joint("joint4").range)[::-1]
    flipped = SimAttachmentEvidenceTracker(spec.compile(), desc)
    (before,) = [(h.attach_link, list(h.touch_links)) for h in scene._hands]

    assert not scene.adopt_kernel_names(flipped)
    assert [(h.attach_link, list(h.touch_links)) for h in scene._hands] == [before]
    # A commanded arm joint one model cannot pose is never left out of the comparison.
    joints = [
        j.model_copy(update={"sim_joint_name": "joint4_unmapped"})
        if j.sim_joint_name == "joint4"
        else j
        for j in desc.joints
    ]
    blind = SimAttachmentEvidenceTracker(
        mujoco.MjModel.from_xml_path(str(path)), desc.model_copy(update={"joints": joints})
    )
    assert not blind.adopt_kernel_names(same)
    assert not scene.adopt_kernel_names(blind)
    same._hands[0].attach_link = "kernel_hand"
    assert scene.adopt_kernel_names(same)
    assert scene._hands[0].attach_link == "kernel_hand"


def test_a_grasp_declaration_leaves_in_the_kernels_naming() -> None:
    """On an MJCF-lowered kernel (SO-100's links are its body names) the declaration the
    producer publishes names ``Moving_Jaw``, the kernel's link, not ``gripper_finger``: the
    kernel matches ``contact_links`` against its own collision links (#754 review)."""
    rig = _rig("so100_follower")
    kernel = [rig.model.body(b).name for b in range(1, rig.model.nbody)]
    assert rig.tracker.bind_kernel_links(kernel) == []
    rig.tracker.set_grasp_declaration(
        GraspDeclaration(
            target_id="payload0",
            contact_links=("gripper_finger",),
            timeout_s=60.0,
            stamp_ns=rig.step_ns,
        )
    )
    out = rig.tracker.grasp_declaration(rig.data, stamp_ns=rig.step_ns)
    assert out is not None and out.contact_links == ("Moving_Jaw",)
