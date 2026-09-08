"""Support-contact attestation from MuJoCo signed distance probes.

Regression for the 2026-08-14 acceptance round: a cup resting on a RoboCasa
island at 0.000 mm got no attestation (MuJoCo's ``contype``/``conaffinity``
bitmasks suppress the cup↔island pair, so the contact list was empty), and
the safety kernel E-stopped on the real support contact at -11.11 mm with no
exemption to apply. baguette↔counter is the control pair where contacts DO
generate.

Fixtures are real compiled ``MjModel``s with analytically known geometry, so
attested numbers are checked against arithmetic, not against producer output.
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
# (1 & 4 == 0, 4 & 1 == 0) so MuJoCo emits NO contact record for that pair
# however deeply they overlap — the island defect, reproduced. ``baguette_body``
# is affine to the island (conaffinity 4 meets contype 4), so that pair DOES
# produce contact records — the control.
_SCENE_MJCF = """
<mujoco model="support_contact_witness">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" contype="4" conaffinity="4"/>
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
    """Pins the defect's precondition: no contact records for cup↔island, some for baguette."""
    model, data = _scene()
    assert _contact_records(model, data, "cup") == []
    # baguette is the control payload; must keep producing records or the
    # equivalence test below compares the probe against nothing.
    assert _contact_records(model, data, "baguette") != []


def test_attests_support_the_contact_list_cannot_see() -> None:
    """Zero MuJoCo contact records still yields an attestation, checked against arithmetic.

    Cup: sphere r=0.04 centred 0.0008 m below tangent rest on plane z=0.42, so
    closest points are (0.10, 0.05, 0.4192) on cup / (0.10, 0.05, 0.42) on
    island: signed distance -0.0008, midpoint (0.10, 0.05, 0.4196).
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
    assert witness.evidence_ref == "openral_hal.convex_distance.convex_geom_distance:island"
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
    """Where solver contacts DO exist, the probe's depth and plane match them exactly."""
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
    """A payload 12 mm inside the island is colliding, not resting: no clamp-to-cap laundering."""
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


_TESSELLATED_MJCF = """
<mujoco model="tessellated_counter">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="counter" pos="0 0 0.4">
      <geom name="strip_under" type="box" size="0.10 0.3 0.02" pos="0 0 0"
            contype="4" conaffinity="4"/>
      <geom name="strip_beside" type="box" size="0.05 0.3 0.02" pos="0.30 0 0"
            contype="4" conaffinity="4"/>
    </body>
    <body name="tray" pos="0 0 0.4225">
      <freejoint name="tray_joint"/>
      <geom name="tray_body" type="box" size="0.45 0.2 0.0025" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_a_neighbouring_strip_cannot_tilt_the_attested_support_plane() -> None:
    """#190 regression: a tray flush across two coplanar strips must not tilt the attested plane.

    A support geom's analytic face normal is only meaningful at a point *on*
    that geom. Averaging hits from an off-geom neighbour strip (0.25 m outside
    it) pulled in a lateral normal and tilted the plane 45 degrees off
    vertical. Fix: cross-check every hit against the instrument's own contact
    direction and drop the off-geom ones.
    """
    model = mujoco.MjModel.from_xml_string(_TESSELLATED_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    witness = support_contact_witness(
        model,
        data,
        root_body_id=_body(model, "tray"),
        robot_body_ids=frozenset(),
        stamp_ns=1,
    )

    assert witness is not None, "the tray rests flush on the counter; that is real support"
    normal = np.asarray(witness.contact_normal_in_object, dtype=float)
    tilt_deg = math.degrees(math.acos(float(np.clip(normal @ np.array([0.0, 0.0, 1.0]), -1, 1))))
    assert tilt_deg < 1e-6, f"attested support plane is {tilt_deg:.1f} deg off vertical"


def test_a_plane_support_attests_nothing_because_it_cannot_be_certified() -> None:
    """A payload resting on a floor plane earns no exemption (fail-closed).

    The certified instrument has no bounded hull for a plane and refuses to
    measure it (#170); the witness path issues only from certified
    measurements (#190). Pin for a future plane-aware branch to flip.
    """
    from openral_hal.convex_distance import convex_geom_distance

    model, data = _scene(cup_z=_CUP_RADIUS_M - _CUP_PENETRATION_M)
    cup_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_body"))
    floor_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    measured = convex_geom_distance(model, data, cup_geom, floor_geom)
    assert not measured.certified and "PLANE" in measured.uncertified_reason

    witness = support_contact_witness(
        model,
        data,
        root_body_id=_body(model, "cup"),
        robot_body_ids=frozenset(),
        stamp_ns=1,
    )
    assert witness is None


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
    """Two coplanar supports at once: the load-bearing island wins, not the tessellated ledge.

    A plank centred on a wide island also grazes a narrow three-geom ledge at
    its rim, same plane. Both are equally horizontal, so dominance falls
    through to load path (centre of mass over island). The ledge deliberately
    outnumbers the island 3:1 in geom pairs, to pin that dominance is not
    decided by pair count / point spread.
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
    """Grasping a cup on the island: attachment carries a witness despite no contact record."""
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
