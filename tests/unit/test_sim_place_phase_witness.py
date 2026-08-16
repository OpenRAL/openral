"""Place-phase support-contact attestation, gated on a typed declaration.

Round-5 (`spark:~/openral-runs/2026-08-14-round5/baguette/seed1_run2`) is why
this file exists. After genuine separation from the pick support the baguette
was driven into its target cabinet and the safety kernel stopped at -1.78 mm
with the payload 66-67 mm inside the cabinet region — correctly, per the
pick witness's anti-scope (`NewContactOutsideThePatchStillStopsMidCarry`),
because nothing distinguished "the payload arrived at its declared
destination" from "the payload grazed a wall."

ADR-0097 supplies the distinction and nothing else: a typed place-phase
declaration from the dispatch layer. Every test here is a real compiled
``MjModel`` with analytically known geometry, and the whole behaviour matrix
is checked in both directions — a declaration without contact attests
nothing, contact without a declaration attests nothing, and only the two
together, on the declared target, produce a witness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from openral_core import (
    AttachmentEvidenceKind,
    EndEffectorSpec,
    JointSpec,
    JointType,
    PlaceDeclaration,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError
from openral_hal._sim_attachment_evidence import SimAttachmentEvidenceTracker

mujoco = pytest.importorskip("mujoco")

# The safety kernel's own caps on what an attestation may claim
# (``support_witness_max_patch_radius_m`` / ``support_witness_max_penetration_m``).
# A place witness is under the identical bounds — ADR-0097 adds no new licence,
# only a new precondition.
_KERNEL_MAX_PATCH_RADIUS_M = 0.5
_KERNEL_MAX_PENETRATION_M = 0.01

_CUP_RADIUS_M = 0.04
_PENETRATION_M = 0.0008

# Counter (the PICK support): top face z = 0.4208, so a cup of radius 0.04
# centred at z = 0.46 rests on it 0.8 mm deep.
_COUNTER_TOP_Z = 0.4208
_CARRY_Z = 0.46
# Cabinet (the PLACE target): its shelf's top face is z = 0.3008, and the shelf
# lives on a CHILD body of the declared cabinet — "a surface identified as part
# of it" in ADR-0097's words, which the resolution must cover.
_SHELF_TOP_Z = 0.3008
_CABINET_X = 0.6
_PLACE_Z = _SHELF_TOP_Z + _CUP_RADIUS_M - _PENETRATION_M
# Sideboard: an undeclared surface with the identical top face, so a test that
# places on it differs from the cabinet test in the DECLARATION alone.
_SIDEBOARD_X = -0.6

_MJCF = f"""
<mujoco model="place_phase_witness">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="counter" pos="0 0 {_COUNTER_TOP_Z - 0.02}">
      <geom name="counter_top" type="box" size="0.3 0.25 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="cabinet" pos="{_CABINET_X} 0 0">
      <geom name="cabinet_back" type="box" pos="0.18 0 0.45" size="0.02 0.25 0.45"
            contype="4" conaffinity="4"/>
      <body name="cabinet_shelf" pos="0 0 0">
        <geom name="cabinet_shelf_top" type="box" pos="0 0 {_SHELF_TOP_Z - 0.02}"
              size="0.16 0.25 0.02" contype="4" conaffinity="4"/>
      </body>
    </body>
    <body name="sideboard" pos="{_SIDEBOARD_X} 0 {_SHELF_TOP_Z - 0.02}">
      <geom name="sideboard_top" type="box" size="0.16 0.25 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="robot0_link7" pos="0 0 0.6">
      <joint name="robot0_joint7" type="slide" axis="0 0 1"/>
      <joint name="robot0_slide_x" type="slide" axis="1 0 0"/>
      <geom name="wrist" type="sphere" size="0.01" contype="1" conaffinity="1"/>
      <body name="gripper0_leftfinger" pos="-0.045 0 -0.14">
        <joint name="gripper0_left_joint" type="slide" axis="-1 0 0"/>
        <geom name="left_pad" type="box" size="0.005 0.02 0.02" contype="1" conaffinity="1"/>
      </body>
      <body name="gripper0_rightfinger" pos="0.045 0 -0.14">
        <joint name="gripper0_right_joint" type="slide" axis="1 0 0"/>
        <geom name="right_pad" type="box" size="0.005 0.02 0.02" contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="cup" pos="0 0 {_CARRY_Z}">
      <freejoint name="cup_joint"/>
      <geom name="cup_body" type="sphere" size="{_CUP_RADIUS_M}" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _description() -> RobotDescription:
    return RobotDescription(
        name="place_phase_rig",
        embodiment_kind="manipulator",
        joints=[
            JointSpec(
                name="joint7",
                joint_type=JointType.PRISMATIC,
                parent_link="base",
                child_link="panda_link7",
                sim_joint_name="robot0_joint7",
                role="arm",
            ),
            JointSpec(
                name="left_finger",
                joint_type=JointType.PRISMATIC,
                parent_link="panda_link7",
                child_link="left_finger",
                sim_joint_name="gripper0_left_joint",
                role="gripper",
            ),
            JointSpec(
                name="right_finger",
                joint_type=JointType.PRISMATIC,
                parent_link="panda_link7",
                child_link="right_finger",
                sim_joint_name="gripper0_right_joint",
                role="gripper",
            ),
        ],
        end_effectors=[
            EndEffectorSpec(
                name="parallel_gripper",
                kind="parallel_gripper",
                parent_link="panda_link7",
                actuated=True,
            )
        ],
        capabilities=RobotCapabilities(),
        safety=SafetyEnvelope(),
    )


def _qpos_address(model: Any, joint: str) -> int:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
    assert joint_id >= 0, joint
    return int(model.jnt_qposadr[joint_id])


def _move_assembly(model: Any, data: Any, *, x: float, z: float) -> None:
    """Drive the whole gripper+payload assembly kinematically to (x, z).

    ``mj_forward`` only — never integration — so the pose under test is exactly
    the pose the arithmetic in each test describes.
    """
    data.qpos[_qpos_address(model, "robot0_slide_x")] = x
    data.qpos[_qpos_address(model, "robot0_joint7")] = z - _CARRY_Z
    cup = _qpos_address(model, "cup_joint")
    data.qpos[cup : cup + 3] = np.asarray([x, 0.0, z], dtype=np.float64)
    mujoco.mj_forward(model, data)


class _Rig:
    """A compiled scene with the cup already grasped off the counter."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_MJCF)
        self.data = mujoco.MjData(self.model)
        self.tracker = SimAttachmentEvidenceTracker(self.model, _description(), stable_ticks=1)
        for joint in ("gripper0_left_joint", "gripper0_right_joint"):
            self.data.qpos[_qpos_address(self.model, joint)] = -0.008
        self.tick = 0
        attachment = self.step()
        assert attachment, "the tracker never confirmed the grasp"
        assert attachment[0].object_id == "sim:cup"
        # The pick witness: the cup is still resting on the counter it was
        # grasped from. This is Entry 012's behaviour, unchanged by ADR-0097.
        assert attachment[0].support_contact is not None
        assert attachment[0].support_contact.support_id == "sim:counter"

    def step(self, *, seconds: float = 0.0) -> list[Any] | None:
        """One post-step observation; ``seconds`` advances the producer clock."""
        self.tick += max(1, int(seconds * 1e9))
        mujoco.mj_forward(self.model, self.data)
        return self.tracker.update(self.data, stamp_ns=self.tick)

    def carry_to(self, *, x: float, z: float) -> None:
        _move_assembly(self.model, self.data, x=x, z=z)


def _declaration(
    *,
    target_id: str = "sim:cabinet",
    stamp_ns: int,
    timeout_s: float = 60.0,
    object_id: str = "",
    active: bool = True,
) -> PlaceDeclaration:
    return PlaceDeclaration(
        target_id=target_id,
        object_id=object_id,
        rskill_id="openral/pi05-robocasa",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=timeout_s,
        stamp_ns=stamp_ns,
        active=active,
    )


# -- The anti-scope, unchanged without a declaration -------------------------


def test_insertion_without_a_declaration_attests_nothing() -> None:
    """Round-5's actual behaviour, and it must stay exactly this.

    The cup is driven into the cabinet and comes to rest on its shelf. With no
    declaration in force there is no place witness, so the kernel keeps
    stopping on that contact — the pick witness's anti-scope, verbatim.
    """
    rig = _Rig()
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is None


def test_a_declaration_without_contact_attests_nothing() -> None:
    """Declaring intent licenses nothing; only measured contact does."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z + 0.20)  # clear above the shelf
    assert rig.step() is None


# -- Arming ------------------------------------------------------------------


def test_declaration_plus_contact_on_the_declared_target_attests() -> None:
    """Both halves present: a bounded witness for the cabinet's own shelf.

    The attested support is ``cabinet_shelf``, a CHILD of the declared
    ``cabinet`` — ADR-0097's "or a surface identified as part of it".
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    transition = rig.step()

    assert transition is not None
    (attachment,) = transition
    assert attachment.object_id == "sim:cup"
    witness = attachment.support_contact
    assert witness is not None
    assert witness.support_id == "sim:cabinet_shelf"
    assert witness.evidence_kind is AttachmentEvidenceKind.SIM_GEOM_DISTANCE
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert witness.max_penetration_m == pytest.approx(_PENETRATION_M, abs=1e-9)
    # Bounded by the same caps as the pick witness. ADR-0097 changes WHICH
    # contact may be exempted, never HOW MUCH.
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M
    assert witness.max_penetration_m <= _KERNEL_MAX_PENETRATION_M


def test_the_place_witness_is_keyed_apart_from_the_pick_witness() -> None:
    """A new ``(support_id, stamp_ns)`` is what re-arms the kernel's latch.

    The kernel re-arms only on a genuinely new attestation key. The place
    witness must therefore differ from the pick witness in both halves of that
    key, or a payload placed onto the surface it was picked from would inherit
    a latch the kernel had already killed.
    """
    rig = _Rig()
    pick = rig.tracker.update(rig.data, stamp_ns=rig.tick)
    assert pick is None  # nothing changed on a bare re-observation
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    transition = rig.step(seconds=1.0)
    assert transition is not None
    witness = transition[0].support_contact
    assert witness is not None
    assert witness.support_id != "sim:counter"
    assert witness.stamp_ns == rig.tick


# -- Wrong target, undeclared target -----------------------------------------


def test_contact_on_an_undeclared_surface_attests_nothing() -> None:
    """The sideboard's top face is identical to the shelf's; only the
    declaration differs, and that is the whole difference in outcome."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_SIDEBOARD_X, z=_PLACE_Z)
    assert rig.step() is None


def test_a_wrong_target_declaration_still_only_exempts_its_own_patch() -> None:
    """HZ-0097-2: a mis-declared target misdirects, it never widens.

    Declaring the sideboard and then placing on it does attest — that is the
    hazard, recorded and accepted — but the attestation is bounded by exactly
    the same caps as any other, and it describes only the payload's own
    footprint on that one surface.
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(target_id="sim:sideboard", stamp_ns=rig.tick))
    rig.carry_to(x=_SIDEBOARD_X, z=_PLACE_Z)
    transition = rig.step()
    assert transition is not None
    witness = transition[0].support_contact
    assert witness is not None
    assert witness.support_id == "sim:sideboard"
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M
    assert witness.max_penetration_m <= _KERNEL_MAX_PENETRATION_M


def test_non_load_bearing_contact_inside_the_declared_target_attests_nothing() -> None:
    """A cup pressed against the cabinet's back panel is not supported by it.

    The declaration names the cabinet and the contact is genuinely on it, but a
    vertical face carries no weight. ``_dominant_support`` discards it, so
    driving a payload into the back of its own declared target still stops the
    robot.
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    # Back panel inner face: cabinet x + 0.18 - 0.02 = 0.76.
    rig.carry_to(x=_CABINET_X + 0.16 - _PENETRATION_M, z=0.55)
    assert rig.step() is None


def test_an_unresolvable_target_is_refused_not_ignored() -> None:
    """A declaration naming nothing in the scene fails closed and loudly."""
    rig = _Rig()
    with pytest.raises(ROSConfigError, match="names no MuJoCo body"):
        rig.tracker.set_place_declaration(_declaration(target_id="sim:pantry", stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is None


def test_a_declaration_scoped_to_another_payload_does_not_arm() -> None:
    """``object_id`` narrows the declaration; a mismatch attests nothing."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick, object_id="sim:baguette"))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is None


# -- Hysteresis --------------------------------------------------------------


def test_one_declaration_licenses_exactly_one_attestation() -> None:
    """Separation kills the exemption and the producer never re-arms it.

    The kernel's latch dies the moment the payload lifts off its attested
    support, and only a genuinely new attestation revives it. If the producer
    re-attested every tick the payload stayed in contact, a lift-and-touch-again
    would be silently forgiven instead of being the fresh violation it is.
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is not None  # armed

    assert rig.step() is None  # still in contact: no re-attestation
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z + 0.20)  # lift clear
    assert rig.step() is None
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)  # touch down again
    assert rig.step() is None  # re-contact is a NEW violation, not a re-arm


def test_a_fresh_declaration_re_arms_after_a_genuine_lift() -> None:
    """Re-arming is possible, but only through a new declaration.

    This is the counterpart to the test above: the producer is not permanently
    poisoned, it simply refuses to re-arm on its own authority.
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is not None
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z + 0.20)
    assert rig.step() is None

    rig.tick += 1_000_000
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    transition = rig.step()
    assert transition is not None
    assert transition[0].support_contact is not None


# -- Lifecycle (HZ-0097-3) ---------------------------------------------------


def test_retraction_disarms_an_armed_place_witness() -> None:
    """Goal end / cancel / E-stop retract the declaration, and the exemption
    goes with it — re-published as an attachment carrying no witness at all,
    which is what makes the kernel drop the latch rather than keep it."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is not None

    rig.tracker.set_place_declaration(None)
    transition = rig.step()
    assert transition is not None
    assert transition[0].support_contact is None
    # And it settles: one disarm, not a republish every tick.
    assert rig.step() is None


def test_a_timed_out_declaration_disarms_without_any_retraction() -> None:
    """The crash backstop. No retraction message ever arrives — the dispatcher
    is gone — and the exemption still dies on the declaration's own window."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick, timeout_s=2.0))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is not None

    transition = rig.step(seconds=3.0)
    assert transition is not None
    assert transition[0].support_contact is None


def test_a_stale_declaration_cannot_arm_a_later_execution() -> None:
    """HZ-0097-3's core hazard: a declaration outliving the goal that made it.

    The declaration is installed, the goal ends without anyone retracting it,
    and only afterwards does the payload reach the cabinet. Nothing arms.
    """
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick, timeout_s=1.0))
    rig.step(seconds=5.0)
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is None


def test_releasing_the_payload_ends_the_place_phase() -> None:
    """A new grasp is a new place phase; a live declaration never carries the
    previous payload's attestation over to the next one."""
    rig = _Rig()
    rig.tracker.set_place_declaration(_declaration(stamp_ns=rig.tick))
    rig.carry_to(x=_CABINET_X, z=_PLACE_Z)
    assert rig.step() is not None

    # Open the jaws and withdraw the gripper, leaving the cup on the shelf:
    # the payload stops following the hand, which is what the tracker reads as
    # a release.
    for joint in ("gripper0_left_joint", "gripper0_right_joint"):
        rig.data.qpos[_qpos_address(rig.model, joint)] = 0.02
    rig.data.qpos[_qpos_address(rig.model, "robot0_joint7")] = 0.30
    mujoco.mj_forward(rig.model, rig.data)
    released = None
    for _ in range(8):
        released = rig.step()
        if released == []:
            break
    assert released == []
    # The place phase is over: a fresh grasp would have to declare its own.
    assert rig.tracker._place_attested_stamp_ns is None
