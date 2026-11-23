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
  published set loses its support witness (the safety kernel keys witness
  re-arming by slot).

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
from openral_core import AttachedCollisionObject, RobotDescription
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
        self.data.ctrl[g.actuator] = g.closed_ctrl if closed else g.open_ctrl

    def run_tolerant(self, steps: int) -> tuple[list[list[AttachedCollisionObject]], int]:
        """``run``, but an update that raises ``ROSConfigError`` is counted, not fatal."""
        published, raised = [], 0
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self.step_ns += 1_000_000
            try:
                out = self.tracker.update(self.data, stamp_ns=self.step_ns)
            except ROSConfigError:
                raised += 1
                continue
            if out is not None:
                published.append(out)
        return published, raised

    def nudge_away(self, payload: int) -> None:
        """Push a released box out of the open jaws, as a lifting arm would."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"payload{payload}_free")
        self.data.qvel[int(self.model.jnt_dofadr[joint]) + 2] = -0.3


def _description(robot: str) -> tuple[RobotDescription, Path]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    assert desc.assets.mjcf is not None
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
            type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.05, friction=[2.0, 0.05, 0.01]
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
            actuator = next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == joint)
        body = int(model.jnt_bodyid[int(model.actuator_trnid[actuator, 0])])
        hand = next(i for i, h in enumerate(hands) if body in h.body_groups)
        out.append((hand, actuator))
    return out


def _jaw_roots(model: Any, data: Any, hand: _Hand) -> tuple[Any, Any]:
    """World origins of the hand's two jaw roots (the finger bodies its joints move)."""
    first, second = (data.xpos[root].copy() for root in sorted(set(hand.body_groups.values())))
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
    mujoco.mj_forward(model, data)
    hands = SimAttachmentEvidenceTracker(model, desc)._hands
    actuators = _gripper_actuators(model, desc, hands)
    # Every joint-driven actuator holds its joint where it is.
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    # Which end of each gripper's ctrl range opens it: the one with the wider jaws.
    spreads = {}
    for end in (0, 1):
        for _, actuator in actuators:
            data.ctrl[actuator] = model.actuator_ctrlrange[actuator][end]
        _settle(model, data)
        spreads[end] = [
            float(np.linalg.norm(np.subtract(*_jaw_roots(model, data, h)))) for h in hands
        ]
    open_end = 0 if sum(spreads[0]) > sum(spreads[1]) else 1
    for _, actuator in actuators:
        data.ctrl[actuator] = model.actuator_ctrlrange[actuator][open_end]
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
        half = [abs(spreads[open_end][index] - spreads[1 - open_end][index]) / 3, 0.015, 0.015]
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
            open_ctrl=float(model.actuator_ctrlrange[actuator][open_end]),
            closed_ctrl=float(model.actuator_ctrlrange[actuator][1 - open_end]),
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
    ("robot", "attach_links", "witnessed"),
    [
        # aloha_agilex's box rests flush on its platform at attach: a support witness.
        ("aloha_agilex", ("fl_link6", "fr_link6"), True),
        # r1pro's narrow jaws need the backstop pad, which attests no support: none.
        ("r1pro", ("left_gripper_link", "right_gripper_link"), False),
    ],
)
def test_each_hand_attaches_and_releases_independently(
    robot: str, attach_links: tuple[str, str], witnessed: bool
) -> None:
    rig = _rig(robot, support=witnessed)
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

    rig.set_gripper(0, closed=False)
    rig.run(100)
    rig.nudge_away(0)
    sets = rig.run(400)
    released = [s for s in sets if [o.object_id for o in s] == ["sim:payload1"]]
    assert released, f"left release never left the right payload alone: {sets}"
    # The right record moved from slot 1 to slot 0 unchanged, witness included: the
    # kernel recognises its key in the new slot (no re-arm, no loss).
    assert released[0][0] == right

    rig.set_gripper(1, closed=False)
    rig.run(100)
    rig.nudge_away(1)
    final = rig.run(400)
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


@pytest.mark.parametrize("robot", ["aloha_agilex", "r1pro", "galaxea_a1", "widowx"])
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


def test_a_transition_survives_a_later_hand_failing() -> None:
    """A release committed in an update that a later hand then failed still publishes.

    The right hand grips an unnamed payload, so every update raises after the left hand
    has been advanced. The left hand releases during those failing updates; once the
    right hand lets go, the next update must publish the release.
    """
    rig = _rig("aloha_agilex", unnamed=frozenset({1}))
    rig.set_gripper(0, closed=True)
    assert [[o.object_id for o in s] for s in rig.run(400)] == [["sim:payload0"]]

    rig.set_gripper(1, closed=True)
    published, raised = rig.run_tolerant(200)
    assert raised > 0 and published == [], "the right hand's unnamed payload raises"

    rig.set_gripper(0, closed=False)
    rig.run_tolerant(100)
    rig.nudge_away(0)
    published, raised = rig.run_tolerant(400)
    assert raised > 0 and published == [], "the left release happened in failing updates"

    assert rig.tracker._hands[0].attached_root is None  # committed, not yet published
    rig.set_gripper(1, closed=False)
    published, _ = rig.run_tolerant(400)
    assert published and published[0] == [], f"the left release was lost: {published}"


def test_construction_refuses_what_the_kernel_would_reject() -> None:
    """Links the kernel model lacks, and two hands over one finger, refuse up front."""
    desc, path = _description("aloha_agilex")
    model, _ = _compile(path, [])
    hands = SimAttachmentEvidenceTracker(model, desc)._hands
    names = {n for h in hands for n in (h.attach_link, *h.touch_links)}
    SimAttachmentEvidenceTracker(model, desc, kernel_link_names=names)  # all known: arms
    with pytest.raises(ROSConfigError, match="not in the safety kernel's collision model"):
        SimAttachmentEvidenceTracker(model, desc, kernel_link_names=names - {"fr_link8"})
    # An end-effector entry over the left fingers as well: two hands, one gripper.
    assert desc.sim is not None
    qpos = [int(model.jnt_qposadr[model.joint(n).id]) for n in ("fl_joint7", "fl_joint8")]
    twice = desc.sim.grippers[0].model_copy(
        update={"joint": None, "end_effector": "left_gripper", "qpos_addrs": qpos}
    )
    sim = desc.sim.model_copy(update={"grippers": [*desc.sim.grippers, twice]})
    with pytest.raises(ROSConfigError, match="share finger bodies"):
        SimAttachmentEvidenceTracker(model, desc.model_copy(update={"sim": sim}))


@pytest.mark.parametrize(("robot", "per_hand"), [("aloha_agilex", 8), ("galaxea_a1", 16)])
def test_every_hand_gets_a_share_of_the_kernel_primitive_cap(robot: str, per_hand: int) -> None:
    """N hands' records together never exceed the kernel's ``attached_max_primitives``."""
    desc, path = _description(robot)
    model, _ = _compile(path, [])
    tracker = SimAttachmentEvidenceTracker(model, desc)
    cap = load_tracking_thresholds().caps.attached_max_primitives
    assert tracker._max_primitives_per_hand == per_hand
    assert tracker._max_primitives_per_hand * len(tracker._hands) <= cap
