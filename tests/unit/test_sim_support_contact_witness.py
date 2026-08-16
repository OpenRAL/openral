"""Support-contact attestation from MuJoCo signed distance probes.

The 2026-08-14 acceptance round is the reason this file exists. A cup rested on
a RoboCasa island at 0.000 mm — a textbook support contact — and the producer
attested nothing, so the safety kernel had no exemption and E-stopped on the
real support contact at -11.11 mm. The contact list was empty: MuJoCo's
``contype``/``conaffinity`` bitmasks suppress whole geom pairs, and cup↔island
happened to be one of them while baguette↔counter happened not to be.

Every fixture here is a real compiled ``MjModel`` with analytically known
geometry, so the attested numbers are checked against arithmetic rather than
against whatever the producer happens to emit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from openral_core import (
    AttachmentEvidenceKind,
    EndEffectorSpec,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_hal._sim_attachment_evidence import (
    SimAttachmentEvidenceTracker,
    support_contact_witness,
)

mujoco = pytest.importorskip("mujoco")

# The safety kernel's own bounds on what an attestation may claim
# (``support_witness_max_patch_radius_m`` / ``support_witness_max_penetration_m``
# in cpp/openral_safety_kernel). The producer must respect them by construction.
_KERNEL_MAX_PATCH_RADIUS_M = 0.5
_KERNEL_MAX_PENETRATION_M = 0.01

# -- Analytically known scene geometry ---------------------------------------
# Island slab: half-extents (0.5, 0.3, 0.02) centred at z = 0.40, so its top
# face is the plane z = 0.42 and its outward normal there is exactly +z.
_ISLAND_TOP_Z = 0.42
# Cup: a sphere of radius 0.04 whose lowest point is the unique closest point to
# a horizontal face below it — directly under the centre, so the probe's contact
# point is pinned analytically in x and y as well as in z.
_CUP_RADIUS_M = 0.04
_CUP_XY = (0.10, 0.05)
_CUP_PENETRATION_M = 0.0008
# The penetration the 2026-08-14 baguette run attested, reproduced exactly.
_BAGUETTE_PENETRATION_M = 0.00118
_BAGUETTE_HALF_EXTENTS = (0.14, 0.03, 0.04)
# The producer's conservative AABB padding.
_AABB_PAD_M = 1e-4

# ``island_top`` and ``cup_body`` share no bits in either direction
# (1 & 4 == 0 and 4 & 1 == 0), so MuJoCo generates NO contact record for the
# pair however deeply they overlap. That is the island defect, reproduced.
# ``baguette_body`` is affine to the island (its conaffinity 4 meets the
# island's contype 4), so that pair DOES produce contact records — the
# baguette-class control that the old contact-list producer handled.
_SCENE_MJCF = """
<mujoco model="support_contact_witness">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="island" pos="0 0 0.4">
      <geom name="island_top" type="box" size="0.5 0.3 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="shelf" pos="0 0 0.9">
      <geom name="shelf_underside" type="box" size="0.5 0.3 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="loose_bin" pos="2 0 0.5">
      <freejoint name="loose_bin_joint"/>
      <geom name="bin_body" type="box" size="0.2 0.2 0.1" contype="4" conaffinity="4"/>
    </body>
    <body name="cup" pos="{cup_x} {cup_y} {cup_z}">
      <freejoint name="cup_joint"/>
      <geom name="cup_body" type="sphere" size="{cup_r}" contype="1" conaffinity="1"/>
    </body>
    <body name="baguette" pos="-0.2 0.05 {baguette_z}">
      <freejoint name="baguette_joint"/>
      <geom name="baguette_body" type="box" size="0.14 0.03 0.04" contype="1" conaffinity="4"/>
    </body>
  </worldbody>
</mujoco>
"""

_RESTING_CUP_Z = _ISLAND_TOP_Z + _CUP_RADIUS_M - _CUP_PENETRATION_M


def _scene(*, cup_x: float = _CUP_XY[0], cup_z: float = _RESTING_CUP_Z) -> tuple[Any, Any]:
    """Compile the scene and run forward kinematics only (never integration)."""
    mjcf = _SCENE_MJCF.format(
        cup_x=cup_x,
        cup_y=_CUP_XY[1],
        cup_z=cup_z,
        cup_r=_CUP_RADIUS_M,
        baguette_z=_ISLAND_TOP_Z + _BAGUETTE_HALF_EXTENTS[2] - _BAGUETTE_PENETRATION_M,
    )
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body(model: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _contact_records(model: Any, data: Any, payload: str) -> list[Any]:
    """The MuJoCo solver contacts naming this payload body."""
    payload_id = _body(model, payload)
    records = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        bodies = {
            int(model.geom_bodyid[int(contact.geom1)]),
            int(model.geom_bodyid[int(contact.geom2)]),
        }
        if payload_id in bodies:
            records.append(contact)
    return records


def test_island_class_payload_generates_no_mujoco_contact_records() -> None:
    """The fixture reproduces the defect's precondition, not just its symptom.

    If MuJoCo ever stopped suppressing this pair, the attestation test below
    would pass for the wrong reason.
    """
    model, data = _scene()
    assert _contact_records(model, data, "cup") == []
    # The control payload must keep producing records, or the equivalence test
    # further down is comparing the probe against nothing.
    assert _contact_records(model, data, "baguette") != []


def test_attests_support_the_contact_list_cannot_see() -> None:
    """RED before the fix: zero contact records meant zero attestations.

    Every number here is arithmetic on the fixture. The cup is a sphere of
    radius 0.04 centred 0.0008 m below a tangent rest on the plane z = 0.42, so
    the probe's closest points are (0.10, 0.05, 0.4192) on the cup and
    (0.10, 0.05, 0.42) on the island: signed distance -0.0008, midpoint
    (0.10, 0.05, 0.4196).
    """
    model, data = _scene()
    witness = support_contact_witness(
        model,
        data,
        root_body_id=_body(model, "cup"),
        robot_body_ids=frozenset(),
        stamp_ns=17,
    )

    assert witness is not None
    assert witness.support_id == "sim:island"
    assert witness.evidence_kind is AttachmentEvidenceKind.SIM_GEOM_DISTANCE
    assert witness.evidence_ref == "mujoco_geom_distance:island"
    assert witness.stamp_ns == 17

    # The island's top face is exactly +z and the cup carries no rotation, so
    # the object-frame normal is exactly +z too.
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)

    # The object frame is the cup's body frame, centred on the sphere. The
    # attested point is the midpoint of the penetration segment.
    expected_z = (_ISLAND_TOP_Z - _CUP_PENETRATION_M / 2.0) - _RESTING_CUP_Z
    assert expected_z == pytest.approx(-0.0396, abs=1e-9)
    assert witness.contact_point_in_object == pytest.approx((0.0, 0.0, expected_z), abs=1e-9)

    # The physical depth, never a quantised one.
    assert witness.max_penetration_m == pytest.approx(_CUP_PENETRATION_M, abs=1e-9)

    # The patch spans the payload's own footprint. MuJoCo's local AABB for a
    # sphere is a cube of half-extent r, padded by the producer's 1e-4, and the
    # attested point sits on the cup's own axis — so the lateral reach is the
    # padded half-extent's diagonal.
    padded = _CUP_RADIUS_M + _AABB_PAD_M
    assert witness.patch_radius_m == pytest.approx(padded * math.sqrt(2.0), rel=1e-9)

    # Bounded by construction against the kernel's caps.
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M
    assert witness.max_penetration_m <= _KERNEL_MAX_PENETRATION_M


def test_baguette_class_attestation_matches_the_solver_contacts() -> None:
    """Where contacts DO exist, the probe reproduces what they reported.

    This is the equivalence claim for the payloads the old contact-list basis
    handled correctly: the attested depth is MuJoCo's own deepest ``dist``, and
    the attested plane is the one its contact frames and positions describe.
    """
    model, data = _scene()
    records = _contact_records(model, data, "baguette")
    assert records

    witness = support_contact_witness(
        model,
        data,
        root_body_id=_body(model, "baguette"),
        robot_body_ids=frozenset(),
        stamp_ns=23,
    )
    assert witness is not None
    assert witness.support_id == "sim:island"
    assert witness.evidence_kind is AttachmentEvidenceKind.SIM_GEOM_DISTANCE

    solver_depth = max(-float(record.dist) for record in records)
    assert solver_depth == pytest.approx(_BAGUETTE_PENETRATION_M, abs=1e-9)
    assert witness.max_penetration_m == pytest.approx(solver_depth, abs=1e-9)

    # MuJoCo orients this pair's contact frames along +z; so does the probe.
    for record in records:
        assert np.asarray(record.frame)[:3] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)

    # The solver's contact points and the probe's agree on the support plane,
    # to the last digit: MuJoCo places a contact at the midpoint of the overlap
    # and so does the probe, from the same two surfaces.
    solver_plane_z = float(np.mean([float(record.pos[2]) for record in records]))
    baguette_origin_z = float(data.xpos[_body(model, "baguette")][2])
    probe_plane_z = witness.contact_point_in_object[2] + baguette_origin_z
    assert probe_plane_z == pytest.approx(solver_plane_z, abs=1e-9)
    assert witness.contact_point_in_object[2] == pytest.approx(
        -_BAGUETTE_HALF_EXTENTS[2] + _BAGUETTE_PENETRATION_M / 2.0, abs=1e-9
    )

    # The attested point is re-centred on the loaf's own footprint, so the patch
    # is the smallest disc covering it — the same number the contact-list
    # producer reached by averaging four symmetric solver contacts, without
    # depending on there having been four of them.
    assert witness.contact_point_in_object[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert witness.patch_radius_m == pytest.approx(
        math.hypot(
            _BAGUETTE_HALF_EXTENTS[0] + _AABB_PAD_M, _BAGUETTE_HALF_EXTENTS[1] + _AABB_PAD_M
        ),
        rel=1e-9,
    )
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M
    assert witness.max_penetration_m <= _KERNEL_MAX_PENETRATION_M


def test_no_support_within_the_probe_window_attests_nothing() -> None:
    """Fail closed: a payload in free flight gets no fabricated witness."""
    model, data = _scene(cup_z=_ISLAND_TOP_Z + 0.30)
    assert (
        support_contact_witness(
            model,
            data,
            root_body_id=_body(model, "cup"),
            robot_body_ids=frozenset(),
            stamp_ns=1,
        )
        is None
    )


def test_penetration_past_the_kernel_cap_attests_nothing() -> None:
    """A payload 12 mm inside the island is not resting on it; it is colliding.

    Clamping to the cap would launder a real penetration into an exemption, so
    the producer withholds the attestation entirely and lets the kernel stop.
    """
    model, data = _scene(cup_z=_ISLAND_TOP_Z + _CUP_RADIUS_M - 0.012)
    assert (
        support_contact_witness(
            model,
            data,
            root_body_id=_body(model, "cup"),
            robot_body_ids=frozenset(),
            stamp_ns=1,
        )
        is None
    )


def test_overhead_surface_is_not_support() -> None:
    """A shelf underside opposes no gravity, so it is contact but not support."""
    shelf_underside_z = 0.9 - 0.02
    model, data = _scene(cup_z=shelf_underside_z - _CUP_RADIUS_M + _CUP_PENETRATION_M)
    assert (
        support_contact_witness(
            model,
            data,
            root_body_id=_body(model, "cup"),
            robot_body_ids=frozenset(),
            stamp_ns=1,
        )
        is None
    )


def test_free_floating_neighbour_is_never_a_support() -> None:
    """Another loose object cannot license an exemption, however flush it is."""
    bin_top_z = 0.5 + 0.1
    model, data = _scene(cup_x=2.0, cup_z=bin_top_z + _CUP_RADIUS_M - _CUP_PENETRATION_M)
    # Precondition: the cup really is touching the bin, and only the bin.
    assert (
        support_contact_witness(
            model,
            data,
            root_body_id=_body(model, "cup"),
            robot_body_ids=frozenset(),
            stamp_ns=1,
        )
        is None
    )


def test_robot_geometry_is_never_a_support() -> None:
    """The gripper holding the payload is ``touch_links``' job, not a witness."""
    model, data = _scene()
    assert (
        support_contact_witness(
            model,
            data,
            root_body_id=_body(model, "cup"),
            robot_body_ids=frozenset({_body(model, "island")}),
            stamp_ns=1,
        )
        is None
    )


def test_the_support_under_the_load_wins_a_two_surface_contact() -> None:
    """Two coplanar supports at once: the seat is attested, not the ledge.

    A plank rests centred on a wide island while its tip also grazes a narrow
    three-geom ledge whose top face is the same plane. Both are load-bearing and
    equally horizontal, so the dominance rule falls through to load path — the
    island lies under the plank's centre of mass, the ledge catches its rim.

    The ledge deliberately outnumbers the island three geom pairs to one: a
    distance probe reports one closest point per pair, so any rule keyed on pair
    count or point spread would measure tessellation and pick the ledge. This
    test is the pin on that.
    """
    mjcf = f"""
<mujoco model="two_supports">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="island" pos="0 0 0.4">
      <geom name="island_top" type="box" size="0.12 0.12 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="ledge" pos="-0.19 0 0.4">
      <geom name="ledge_a" type="box" pos="0 -0.02 0" size="0.004 0.004 0.02"
            contype="4" conaffinity="4"/>
      <geom name="ledge_b" type="box" pos="0 0.00 0" size="0.004 0.004 0.02"
            contype="4" conaffinity="4"/>
      <geom name="ledge_c" type="box" pos="0 0.02 0" size="0.004 0.004 0.02"
            contype="4" conaffinity="4"/>
    </body>
    <body name="plank" pos="0 0 {_ISLAND_TOP_Z + 0.02 - _CUP_PENETRATION_M}">
      <freejoint name="plank_joint"/>
      <geom name="plank_body" type="box" size="0.20 0.05 0.02" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    witness = support_contact_witness(
        model,
        data,
        root_body_id=_body(model, "plank"),
        robot_body_ids=frozenset(),
        stamp_ns=5,
    )
    assert witness is not None
    assert witness.support_id == "sim:island"
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M


# -- The witness on the real attach transition -------------------------------

_GRIPPER_MJCF = """
<mujoco model="support_contact_witness_attach">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="island" pos="0 0 {island_z}">
      <geom name="island_top" type="box" size="0.5 0.3 0.02" contype="4" conaffinity="4"/>
    </body>
    <body name="robot0_link7" pos="0 0 0.6">
      <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
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
    <body name="cup" pos="0 0 {cup_z}">
      <freejoint name="cup_joint"/>
      <geom name="cup_body" type="sphere" size="{cup_r}" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _gripper_description() -> RobotDescription:
    return RobotDescription(
        name="support_witness_test_robot",
        embodiment_kind="manipulator",
        joints=[
            JointSpec(
                name="joint7",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
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


def _grasp(*, island_z: float) -> Any:
    """Close both jaws on the cup and return the confirmed attachment."""
    model = mujoco.MjModel.from_xml_string(
        _GRIPPER_MJCF.format(island_z=island_z, cup_z=_RESTING_CUP_Z, cup_r=_CUP_RADIUS_M)
    )
    data = mujoco.MjData(model)
    tracker = SimAttachmentEvidenceTracker(model, _gripper_description())
    # Both jaw joints slide outward on positive qpos, so closing is negative.
    for joint in ("gripper0_left_joint", "gripper0_right_joint"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[int(model.jnt_qposadr[joint_id])] = -0.008

    attachments = None
    for tick in range(8):
        mujoco.mj_forward(model, data)
        attachments = tracker.update(data, stamp_ns=tick)
        if attachments:
            break
    assert attachments, "the tracker never confirmed the grasp"
    (attachment,) = attachments
    assert attachment.object_id == "sim:cup"
    return attachment


def test_attach_transition_carries_the_witness_for_a_suppressed_pair() -> None:
    """The acceptance run's exact shape: grasp a cup resting on an island.

    cup↔island produces no contact record, so before the fix the attachment
    went out with ``support_contact=None`` and the kernel had nothing to apply.
    """
    witness = _grasp(island_z=0.4).support_contact
    assert witness is not None
    assert witness.support_id == "sim:island"
    assert witness.evidence_kind is AttachmentEvidenceKind.SIM_GEOM_DISTANCE
    assert witness.max_penetration_m == pytest.approx(_CUP_PENETRATION_M, abs=1e-9)
    assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert witness.max_penetration_m <= _KERNEL_MAX_PENETRATION_M
    assert witness.patch_radius_m <= _KERNEL_MAX_PATCH_RADIUS_M


def test_lifted_payload_attaches_without_a_witness() -> None:
    """Grasp in free space: the attachment ships, the attestation does not."""
    assert _grasp(island_z=0.05).support_contact is None
