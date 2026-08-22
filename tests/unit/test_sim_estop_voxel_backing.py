# SPDX-License-Identifier: Apache-2.0
"""What backs a world-voxel stop — the map-side half of the stop record.

The 2026-08-22 validation round declared two of four stops FALSE POSITIVES and
could not have been right about either, because nothing in the stack could
answer the question a world-voxel stop actually turns on: *what is at the cell
the kernel stopped on?* The near-miss probes measure MuJoCo against MuJoCo and
never look at ``/openral/world_voxels`` at all, and their world side excludes
exactly the two classes a map/world disagreement hides in — every robot body,
and every non-collidable geom. So "zero pairs within 100 mm" was read as
"nothing was there" when it could only ever mean "nothing SOLID and NOT the
robot was within the window".

Two independent defects in that reading, both pinned here:

1. **The window was too narrow to contain the answer.** Kernel distances are
   OBB-to-voxel; probe distances are mesh-to-mesh. A box around a rounded link
   is sub-millimetre on its faces and 23-88 mm out at its corners (measured
   against ``panda_mj_description``), so the backing geometry of a legitimate
   stop routinely sits outside a 100 mm mesh-to-mesh window.
   :func:`collision_model_mesh_slop` makes that term a number and the snapshot
   widens the window to it.
2. **The robot's own body was invisible to the diagnostic.** The depth
   self-filter is supposed to keep the robot out of its own world map; when it
   fails there is no report, because the probe excludes robot bodies from its
   world side on purpose. :func:`voxel_backing_record` classifies a cell
   against ALL geometry, so self-occupancy becomes a verdict instead of
   silence.

The self-filter's coverage of the base and mount is pinned here too
(``test_self_filter_covers_base_and_mount_including_unprefixed``): the
2026-08-22 "base mapped as world occupancy" hypothesis was refuted by the
round's own artifacts, and ``manipulator_mount`` — a robosuite body sharing no
prefix with any joint — is the body that hypothesis turned on.

Real compiled MuJoCo models throughout, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import (
    collision_model_mesh_slop,
    voxel_backing_record,
)

mujoco = pytest.importorskip("mujoco")

# A robocasa-shaped mobile manipulator. The chassis, the pedestal and the arm
# each carry a real solid geom, and `manipulator_mount` deliberately shares no
# prefix with any joint name — robosuite names it exactly that way, and it is
# reachable by the self-filter only through the parent-descendant closure.
#
# `pantry_side_panel` is a REAL obstacle standing beside the parked base: it is
# what proves that removing the robot from world occupancy removes no
# protection. `region_marker` carries neither contype nor conaffinity — a
# RoboCasa placement region, which `mj_ray` strikes (and so does the depth
# synth) but which the near-miss probe deliberately never measures.
_MJCF = """
<mujoco model="estop_voxel_backing">
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="mobilebase0_base" pos="0 0 0.12">
      <joint name="mobilebase0_joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="mobilebase0_joint_mobile_side" type="slide" axis="0 1 0"/>
      <joint name="mobilebase0_joint_mobile_yaw" type="hinge" axis="0 0 1"/>
      <geom name="chassis_col" type="box" size="0.25 0.25 0.06"/>
      <body name="mobilebase0_support" pos="0 0 0.06">
        <geom name="support_col" type="cylinder" size="0.08 0.15"/>
        <body name="manipulator_mount" pos="0 0 0.15">
          <geom name="mount_plate" type="box" size="0.14 0.14 0.01"/>
          <body name="robot0_link0" pos="0 0 0.01">
            <geom name="link0_col" type="cylinder" size="0.06 0.05"/>
            <body name="robot0_link1" pos="0 0 0.05">
              <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
              <geom name="link1_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.05"/>
              <body name="robot0_link2" pos="0 0 0.1">
                <joint name="robot0_joint2" type="hinge" axis="0 1 0"/>
                <geom name="link2_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.05"/>
                <body name="gripper0_right_finger" pos="0 -0.06 0.1">
                  <joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0"/>
                  <geom name="finger_col" type="box" size="0.01 0.02 0.03"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="pantry_side_panel" pos="0 0.36 0.4">
      <geom name="pantry_panel" type="box" size="0.3 0.02 0.4"/>
    </body>
    <body name="cab_region" pos="-0.36 0 0.4">
      <geom name="region_marker" type="box" size="0.02 0.3 0.3"
            contype="0" conaffinity="0"/>
    </body>
    <body name="carried_cup" pos="0.30 0 0.40">
      <freejoint name="carried_cup_joint"/>
      <geom name="cup_body" type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""

_ROBOT_SIM_JOINTS = [
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "robot0_joint1",
    "robot0_joint2",
    "gripper0_right_finger_joint1",
]

# A base-frame grid on `mobilebase0_support`, the body `base_frame` denotes on
# a robosuite mobile base (ADR-0095). 50 mm cells over a 1.6 m box, the deploy
# lattice at half the deploy resolution.
_GRID_ORIGIN = (-0.8, -0.8, -0.8)
_GRID_RES = 0.05
_GRID_SIZE = (32, 32, 32)
_BASE_BODY = "mobilebase0_support"


def _model_data() -> tuple[Any, Any]:
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _index_at(base_xyz: tuple[float, float, float]) -> int:
    """The grid index whose cell contains ``base_xyz`` (base-frame metres)."""
    idx = [int((base_xyz[k] - _GRID_ORIGIN[k]) / _GRID_RES) for k in range(3)]
    assert all(0 <= idx[k] < _GRID_SIZE[k] for k in range(3)), f"{base_xyz} outside the grid"
    return idx[0] + _GRID_SIZE[0] * (idx[1] + _GRID_SIZE[1] * idx[2])


def _robot_bodies(model: Any) -> frozenset[int]:
    return robot_self_body_ids(model, _ROBOT_SIM_JOINTS)


def _backing(
    model: Any,
    data: Any,
    base_xyz: tuple[float, float, float],
    *,
    attached: frozenset[int] = frozenset(),
) -> dict[str, object]:
    return voxel_backing_record(
        model,
        data,
        voxel_index=_index_at(base_xyz),
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=attached,
        base_frame_body=_BASE_BODY,
    )


def test_self_filter_covers_base_and_mount_including_unprefixed() -> None:
    """The 2026-08-22 hypothesis, refuted: base and mount ARE self-filtered.

    The round suspected the chassis/mount were mapped into the octomap as world
    occupancy because they are not arm links. They are not: every one of them is
    in the depth self-filter's body set, which is the same set the stop record
    reports as ``probe_excluded_robot_bodies``. ``manipulator_mount`` is the
    hard case — robosuite gives it no joint-shared prefix, so it is reachable
    only through the parent-descendant closure, and it is exactly the body the
    hypothesis named.
    """
    model, _data = _model_data()
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in _robot_bodies(model)}

    assert {
        "mobilebase0_base",
        "mobilebase0_support",
        "manipulator_mount",
        "robot0_link0",
        "robot0_link1",
        "robot0_link2",
        "gripper0_right_finger",
    } <= names
    # ...and nothing of the world came with them.
    assert not names & {"pantry_side_panel", "cab_region", "carried_cup"}


def test_cell_on_the_robots_own_mount_is_self_occupancy_not_silence() -> None:
    """A cell inside the robot's own base/mount reports, instead of nothing.

    This is the failure mode the near-miss probe cannot see at all: its world
    side excludes every robot body, so a robot mapped into its own world map
    produces an empty pair list that reads as "nothing was there".
    """
    model, data = _model_data()
    # The mount plate spans +-0.14 x +-0.14 x +-0.01 about base-frame z=0.15;
    # this cell reaches its rim, where neither the pedestal (r=0.08) nor the
    # arm base (r=0.06) can explain the return. `manipulator_mount` shares no
    # prefix with any joint, so only the self-filter's descendant closure
    # covers it — the exact body the refuted hypothesis named.
    record = _backing(model, data, (0.115, 0.0, 0.145))

    assert record["verdict"] == "self_occupancy_suspect"
    assert record["classes"] == ["self_occupancy_suspect"]
    bodies = {str(b["body"]) for b in record["backing"]}  # type: ignore[index]
    assert bodies == {"manipulator_mount"}


def test_a_real_obstacle_beside_the_base_still_maps_as_world() -> None:
    """The conservativeness proof: excluding the robot excludes no obstacle.

    Keeping the robot out of world occupancy is only safe if a real obstacle
    parked right beside it is still world. ``pantry_side_panel`` stands 0.36 m
    off the base axis, closer than the chassis is wide, and is still classified
    ``solid_world`` — never confused with the robot next to it.
    """
    model, data = _model_data()
    record = _backing(model, data, (0.0, 0.36, 0.28))

    assert record["verdict"] == "solid_world"
    bodies = {str(b["body"]) for b in record["backing"]}  # type: ignore[index]
    assert bodies == {"pantry_side_panel"}


def test_real_geometry_outranks_the_robot_when_both_back_a_cell() -> None:
    """Precedence is adjudication order: real geometry EXPLAINS the stop.

    A cell straddling the arm and a real obstacle is a legitimate stop, not a
    map defect, so ``solid_world`` wins the verdict while the robot hit stays
    visible in ``classes``.
    """
    model, data = _model_data()
    # The finger sits at base-frame (0, -0.06, 0.31); park the panel's face so
    # one cell contains both. The panel is a world body; the finger is the robot.
    record = _backing(model, data, (0.0, 0.36, 0.28))
    assert record["verdict"] == "solid_world"

    mount = _backing(model, data, (0.115, 0.0, 0.145))
    assert mount["verdict"] == "self_occupancy_suspect"
    # Both verdicts come from the same unfiltered sweep, not from two policies.
    assert mount["method"] == record["method"]


def test_marker_geometry_is_named_not_silently_dropped() -> None:
    """A non-collidable region marker is reported as its own class.

    The near-miss probe excludes these deliberately (measuring against one
    manufactured the "payload 134 mm inside cab_1_left_group_reg_main" of
    rounds 5/6). But `mj_ray` strikes them and so does the depth synth, so they
    CAN become occupancy — and a cell backed only by a marker is a map defect
    the probe is structurally unable to report.
    """
    model, data = _model_data()
    record = _backing(model, data, (-0.36, 0.0, 0.28))

    assert record["verdict"] == "noncollidable_world"
    entry = next(b for b in record["backing"])  # type: ignore[call-overload]
    assert entry["body"] == "cab_region"
    assert entry["collidable"] is False


def test_attached_payload_is_distinguished_from_the_world() -> None:
    """A carried object still in world occupancy is a clearing defect, not an obstacle."""
    model, data = _model_data()
    cup = frozenset({int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carried_cup"))})
    record = _backing(model, data, (0.30, 0.0, 0.19), attached=cup)

    assert record["verdict"] == "attached_payload"


def test_empty_space_is_unbacked_and_says_how_hard_it_looked() -> None:
    """``unbacked`` must mean "looked and found nothing", never "did not look"."""
    model, data = _model_data()
    record = _backing(model, data, (0.55, 0.55, 0.55))

    assert record["verdict"] == "unbacked"
    assert record["classes"] == []
    assert record["backing"] == []
    # The coverage attestation the near-miss probe learned to carry.
    assert int(record["rays_cast"]) == 3 * 3 * 3  # type: ignore[call-overload]
    assert int(record["rays_hit"]) == 0  # type: ignore[call-overload]


def test_the_cell_is_located_through_the_base_frame_not_the_world() -> None:
    """The grid is base-relative, so its cells must move with the base.

    A world-frame reading of a base-frame grid is the ADR-0095 class of bug: it
    put the whole cloud 0.70 m out. The same index must name a different world
    point once the base drives away, and must still find the mount.
    """
    model, data = _model_data()
    parked = _backing(model, data, (0.115, 0.0, 0.145))

    data.qpos[0] = 1.25  # mobilebase0_joint_mobile_forward
    mujoco.mj_forward(model, data)
    driven = _backing(model, data, (0.115, 0.0, 0.145))

    assert driven["verdict"] == "self_occupancy_suspect"
    assert driven["base_xyz"] == parked["base_xyz"]
    assert driven["world_xyz"] != parked["world_xyz"]
    assert pytest.approx(1.25, abs=1e-6) == (
        driven["world_xyz"][0] - parked["world_xyz"][0]  # type: ignore[index]
    )


def test_an_index_outside_the_grid_is_refused_not_guessed() -> None:
    model, data = _model_data()
    record = voxel_backing_record(
        model,
        data,
        voxel_index=_GRID_SIZE[0] * _GRID_SIZE[1] * _GRID_SIZE[2],
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    assert record["verdict"] == "out_of_range"
    assert record["rays_cast"] == 0


def test_collision_model_slop_is_tight_on_faces_and_loose_at_corners() -> None:
    """The budget term the round did not have, against the REAL Panda meshes.

    ``panda_mobile``'s manifest documents its OBBs as enclosing the RoboCasa
    ``robot0_linkN_collision`` meshes, and a mesh verification confirmed they
    are tight — which withdrew the "OBBs are too big" hypothesis. Both halves
    are true of different questions: the faces are sub-millimetre, and the
    corners are tens of millimetres out, necessarily, because a box around a
    rounded link cannot be otherwise. That corner term is what makes a kernel
    OBB-to-voxel distance and a probe mesh-to-mesh distance comparable.
    """
    from openral_core import RobotDescription

    descriptions = pytest.importorskip("robot_descriptions.loaders.mujoco")
    try:
        panda = descriptions.load_robot_description("panda_mj_description")
    except Exception as exc:  # reason: description fetch is a network dependency
        pytest.skip(f"panda_mj_description unavailable: {exc}")

    # The real manifest's OBBs against the real upstream meshes. Only the joint
    # NAME binding differs: panda_mobile's `sim_joint_name`s are robosuite's
    # (`robot0_joint1`), and upstream ships them bare (`joint1`). Rebinding the
    # names keeps both the geometry under test and the manifest that declares
    # it real — nothing here is a stand-in for either.
    manifest = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml").model_dump()
    for joint in manifest["joints"]:
        sim_name = joint.get("sim_joint_name") or ""
        if sim_name.startswith("robot0_joint"):
            joint["sim_joint_name"] = sim_name.removeprefix("robot0_")
    slop = collision_model_mesh_slop(panda, RobotDescription.model_validate(manifest))

    link1 = slop["links"]["panda_link1"]  # type: ignore[index]
    # Faces: sub-millimetre. The OBBs really do hug the meshes.
    assert max(abs(v) for v in link1["face_slop_m"]) < 0.001  # type: ignore[index]
    # Corners: tens of millimetres, and that is the whole point.
    assert 0.030 < link1["corner_slop_m"] < 0.080  # type: ignore[index]
    # Every kernel-checked link resolved, so the budget is not silently partial.
    assert slop["unresolved_links"] == []
    assert float(slop["max_corner_slop_m"]) > 0.020  # type: ignore[arg-type]
