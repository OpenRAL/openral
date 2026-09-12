# SPDX-License-Identifier: Apache-2.0
"""What backs a world-voxel stop — the map-side half of the stop record.

The 2026-08-22 validation round called two of four stops false positives using probes that compare
MuJoCo to MuJoCo, never touch ``/openral/world_voxels``, and exclude robot bodies and non-collidable
geoms from the world side — so "no pairs within 100 mm" only ever meant "nothing solid and
non-robot was in the window". Two defects pinned here:

1. Window too narrow: kernel distances are OBB-to-voxel, probe distances mesh-to-mesh. A box around
   a rounded link is sub-mm on faces but 23-88 mm out at corners (measured against
   ``panda_mj_description``); ``collision_model_mesh_slop`` makes that a number and the snapshot
   widens the window by it.
2. Robot body invisible to the probe: if the depth self-filter fails to keep the robot out of the
   world map, the probe (which excludes robot bodies) reports nothing.
   ``voxel_backing_record`` classifies against ALL geometry, so self-occupancy becomes a
   verdict.

``test_self_filter_covers_base_and_mount_including_unprefixed`` pins self-filter coverage of
``manipulator_mount`` (unprefixed robosuite body) — refutes the 2026-08-22 "base mapped as world
occupancy" hypothesis.

Real compiled MuJoCo models throughout, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import (
    collision_model_mesh_slop,
    estop_ground_truth_snapshot,
    grid_current_at,
    occupied_cell_keys,
    preattach_verdict,
    voxel_backing_for_cell,
    voxel_backing_record,
)

mujoco = pytest.importorskip("mujoco")

# A robocasa-shaped mobile manipulator. `manipulator_mount` deliberately shares no prefix with
# any joint name (robosuite names it that way), reachable by the self-filter only through the
# parent-descendant closure. `pantry_side_panel` is a real obstacle beside the parked base, proving
# robot exclusion doesn't remove protection. `region_marker` has neither contype nor conaffinity —
# a RoboCasa placement region that `mj_ray`/depth synth strike but the near-miss probe never
# measures.
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
    <!-- RoboCasa-shaped fixture: collidable slab wearing a non-collidable visual shell, both in
         ONE 50 mm cell, shell nearer the probe's ray start. `counter_1_right` is this shape and
         what the 2026-09-06 battery kept stopping on. -->
    <body name="counter_1_right_group" pos="0 -0.317 0.40">
      <geom name="counter_top" type="box" size="0.10 0.02 0.10" pos="0 0.027 0"/>
      <geom name="counter_top_visual" type="box" size="0.10 0.005 0.10" pos="0 -0.028 0"
            contype="0" conaffinity="0"/>
    </body>
    <!-- RoboCasa counter as `counter.py` actually builds it: one full-span non-collidable
         `*_top_visual` and collidable chunks tiling the SAME volume (shared surface). Stepping
         past the shell's face lands inside the chunk with no further ray entry — the coincident
         case needs more than the walk-past fix. -->
    <body name="counter_2_right_group" pos="0.60 0 0.40">
      <geom name="counter2_top_visual" type="box" size="0.10 0.02 0.10"
            contype="0" conaffinity="0"/>
      <geom name="counter2_top_0" type="box" size="0.10 0.02 0.10"/>
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
    """Refutes the 2026-08-22 "chassis/mount mapped as world occupancy" hypothesis: both are in
    the depth self-filter's body set (``probe_excluded_robot_bodies``), including the hard case
    ``manipulator_mount``, reachable only through the parent-descendant closure.
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
    """A cell inside the robot's own base/mount reports self-occupancy instead of silence — the
    failure mode the near-miss probe can't see (its world side excludes all robot bodies).
    """
    model, data = _model_data()
    # Mount plate spans +-0.14 x +-0.14 x +-0.01 about base-frame z=0.15; this cell reaches its rim,
    # where neither the pedestal (r=0.08) nor the arm base (r=0.06) explains the return —
    # `manipulator_mount` only, covered via the self-filter's descendant closure.
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


def test_the_grids_own_rotation_places_the_cell() -> None:
    """``OccupancyVoxels`` is an oriented lattice, so its rotation locates cells.

    The grid is published on the OctoMap's lattice, whose axes turn relative to
    ``base_frame`` as the robot does. A probe that ignores that rotation
    interrogates a cube the kernel never stopped on — it would still return a
    confident verdict, about the wrong place, which is worse than no verdict.
    """
    model, data = _model_data()
    # +90 degrees about z: the grid's +x runs along the base frame's +y.
    quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    index = _index_at((0.115, 0.0, 0.145))
    aligned = voxel_backing_record(
        model,
        data,
        voxel_index=index,
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    rotated = voxel_backing_record(
        model,
        data,
        voxel_index=index,
        grid_origin=_GRID_ORIGIN,
        grid_orientation_xyzw=quat,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    assert rotated["base_xyz"] != aligned["base_xyz"], (
        "the same index on a rotated lattice is a different point"
    )
    # The cell offset from origin rotates: (dx, dy) -> (-dy, dx).
    ox, oy, _ = _GRID_ORIGIN
    ax, ay, az = aligned["base_xyz"]  # type: ignore[misc]  # reason: dict[str, object] value
    rx, ry, rz = rotated["base_xyz"]  # type: ignore[misc]  # reason: dict[str, object] value
    assert pytest.approx(-(ay - oy) + ox, abs=1e-6) == rx
    assert pytest.approx((ax - ox) + oy, abs=1e-6) == ry
    assert pytest.approx(az, abs=1e-9) == rz


def test_the_shared_cell_helper_carries_the_grids_rotation() -> None:
    """The regression: the late-arrival caller used to drop the orientation.

    ``voxel_backing_record`` has always honoured ``grid_orientation_xyzw`` (the
    test above). The bug was in a *caller*: two sites unpacked a cached
    ``/openral/world_voxels`` cell into its arguments, and only one passed the
    rotation. The other — ``SimSensorBridge._late_voxel_backing``, which is the
    path most stops take because the snapshot is never delayed for the kernel's
    evidence — silently took the identity default and probed a cube the kernel
    never stopped on, then reported a confident ``unbacked`` verdict about it.

    Measured on a live ``robocasa_baguette`` carry: three world-collision stops,
    every one decoding to a cell 2.9-3.1 m from the stopping link with 27 rays
    cast and 0 hits. ``voxel_backing_for_cell`` is now the single place a cell
    dict is unpacked, so the two paths cannot disagree again.
    """
    model, data = _model_data()
    quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    cell = {
        "index": _index_at((0.115, 0.0, 0.145)),
        "origin": _GRID_ORIGIN,
        "orientation": quat,
        "resolution": _GRID_RES,
        "size": _GRID_SIZE,
    }
    through_helper = voxel_backing_for_cell(
        model,
        data,
        cell,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=frozenset(),
        base_frame_body=_BASE_BODY,
    )
    rotated = voxel_backing_record(
        model,
        data,
        voxel_index=int(cell["index"]),  # type: ignore[arg-type]
        grid_origin=_GRID_ORIGIN,
        grid_orientation_xyzw=quat,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    assert through_helper is not None
    assert through_helper["base_xyz"] == rotated["base_xyz"], (
        "the helper placed the cell somewhere the grid's own rotation does not"
    )

    identity = voxel_backing_record(
        model,
        data,
        voxel_index=int(cell["index"]),  # type: ignore[arg-type]
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    assert through_helper["base_xyz"] != identity["base_xyz"], (
        "the rotation made no difference here, so this fixture cannot catch the bug"
    )


def test_the_shared_cell_helper_declines_a_cell_with_no_index() -> None:
    """A stop that named no voxel has no cube to probe, and says so with ``None``."""
    model, data = _model_data()
    assert (
        voxel_backing_for_cell(
            model,
            data,
            {"origin": _GRID_ORIGIN, "resolution": _GRID_RES, "size": _GRID_SIZE},
            robot_body_ids=_robot_bodies(model),
            attached_body_ids=frozenset(),
            base_frame_body=_BASE_BODY,
        )
        is None
    )


def test_an_unset_grid_orientation_is_refused_not_read_as_identity() -> None:
    """All-zeros is what an unset field carries, and it is not a rotation."""
    model, data = _model_data()
    record = voxel_backing_record(
        model,
        data,
        voxel_index=_index_at((0.115, 0.0, 0.145)),
        grid_origin=_GRID_ORIGIN,
        grid_orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    assert record["verdict"] == "out_of_range"
    assert record["rays_cast"] == 0


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


def _real_panda_and_manifest() -> tuple[Any, Any]:
    """The real upstream Panda MjModel paired with the real `panda_mobile` manifest.

    Only the joint NAME binding differs between them: `panda_mobile`'s
    ``sim_joint_name``s are robosuite's (``robot0_joint1``) and upstream ships
    them bare (``joint1``). Rebinding the names keeps both the geometry under
    test and the manifest that declares it real — nothing here is a stand-in
    for either.
    """
    from openral_core import RobotDescription

    descriptions = pytest.importorskip("robot_descriptions.loaders.mujoco")
    try:
        panda = descriptions.load_robot_description("panda_mj_description")
    except Exception as exc:  # reason: description fetch is a network dependency
        pytest.skip(f"panda_mj_description unavailable: {exc}")
    manifest = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml").model_dump()
    for joint in manifest["joints"]:
        sim_name = joint.get("sim_joint_name") or ""
        if sim_name.startswith("robot0_joint"):
            joint["sim_joint_name"] = sim_name.removeprefix("robot0_")
    return panda, RobotDescription.model_validate(manifest)


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
    panda, description = _real_panda_and_manifest()
    slop = collision_model_mesh_slop(panda, description)

    link1 = slop["links"]["panda_link1"]  # type: ignore[index]
    # Faces: sub-millimetre. The OBBs really do hug the meshes.
    assert max(abs(v) for v in link1["face_slop_m"]) < 0.001  # type: ignore[index]
    # Corners: tens of millimetres, and that is the whole point.
    assert 0.030 < link1["corner_slop_m"] < 0.080  # type: ignore[index]
    # Every kernel-checked link resolved, so the budget is not silently partial.
    assert slop["unresolved_links"] == []
    assert float(slop["max_corner_slop_m"]) > 0.020  # type: ignore[arg-type]

    # `has_stage2_hull` (#260), against the real manifest rather than a stub.
    # A `None` overhang is two different facts and only one of them leaves a
    # self-pair unadjudicable, so the block has to say which. `panda_link1`
    # ships no stage-2 hull BY DECISION — #191 withdrew its refined envelope
    # for moving link1's own stops by 0.0003 mm — and every other kernel-checked
    # link carries one.
    assert link1["has_stage2_hull"] is False  # type: ignore[index]
    assert link1["hull_overhang_m"] is None  # type: ignore[index]
    hulled = {
        name: blk
        for name, blk in slop["links"].items()  # type: ignore[union-attr]
        if name != "panda_link1"
    }
    assert hulled, "the manifest must still carry hulls to contrast against"
    for name, blk in hulled.items():
        assert blk["has_stage2_hull"] is True, name
        assert blk["hull_overhang_m"] is not None, name


def test_the_snapshot_publishes_the_slop_block_where_the_adjudicator_reads_it() -> None:
    """`has_stage2_hull` has to arrive under ``adjudication_budget``, not at the top level.

    The producer is pinned above and the consumer
    (``tools/validation_matrix.py::_lacks_stage2_hull``) is pinned in
    ``tests/unit/test_self_pair_box_budget.py``, but both stop at the edges of
    the record and nothing asserted the path between them: the snapshot nests
    the block at ``adjudication_budget.collision_model_slop``, and a reader
    that looks for a top-level ``collision_model_slop`` finds ``None`` and
    concludes the field was never published. That happened while reading the
    live 2026-09-10 round
    (``docs/reference/data/estop-snapshot-has-stage2-hull-2026-09-10.jsonl``),
    where the field was in fact present.

    Real MjModel, real manifest, real snapshot builder — the same components the
    live round used, minus the ROS graph.
    """
    panda, description = _real_panda_and_manifest()
    data = mujoco.MjData(panda)
    mujoco.mj_forward(panda, data)

    snapshot = estop_ground_truth_snapshot(
        panda,
        data,
        robot_body_ids=frozenset(range(panda.nbody)),
        description=description,
    )

    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict), "the snapshot published no adjudication budget at all"
    slop = budget["collision_model_slop"]
    assert isinstance(slop, dict), (
        "`collision_model_slop` is absent from the budget block — the adjudicator reads it "
        "from there, so a self pair naming a hull-less link would go unadjudicated (#260)"
    )
    links = slop["links"]
    assert links["panda_link1"]["has_stage2_hull"] is False
    assert links["panda_link1"]["hull_overhang_m"] is None
    for name, blk in links.items():
        if name == "panda_link1":
            continue
        assert blk["has_stage2_hull"] is True, name
    # The other half of what `_link_link_hull_gap_m` needs from the same block.
    assert float(budget["link_link"]["admissible_gap_box_m"]) > 0.0
    # Nothing top-level: a reader that expects it there is reading the wrong key.
    assert "collision_model_slop" not in snapshot


def test_a_solid_surface_behind_a_visual_shell_is_not_read_as_decoration() -> None:
    """A cell holding both a visual shell and the slab it wraps reads `solid_world`.

    `mj_ray` reports only the NEAREST strike, so a probe that stops at the first
    surface sees only the shell — and adjudicates the cell
    `noncollidable_world`, i.e. "the map disagrees with the world", when a
    collidable slab is millimetres behind it inside the same cell.

    That is not hypothetical. On the 2026-09-06 battery, **6 of the 8 stops that
    carried a backing record at all** came back `noncollidable_world` naming
    `counter_1_right_group_top_visual` — while the certified nearest *collision*
    surface was ~16 mm away, well inside the same 25 mm cell. Since #180 the
    depth cast makes those shells transparent, so the cell was created by the
    collidable surface: the map was right and the diagnostic was wrong.

    The probe now walks past a non-collidable strike and looks again, so both
    are recorded and `solid_world` takes precedence. Without that, this test
    fails with `noncollidable_world`.
    """
    model, data = _model_data()
    record = _backing(model, data, (0.0, -0.325, 0.22))

    assert record["verdict"] == "solid_world", (
        "the collidable slab behind the shell was missed; the stop would be "
        "adjudicated as landing on decoration"
    )
    names = {str(entry["geom"]) for entry in record["backing"]}  # type: ignore[call-overload]
    assert "counter_top" in names, "the slab that explains the cell must be named"
    assert "counter_top_visual" in names, (
        "the shell must still be reported — it is what the depth cast used to "
        "integrate, and dropping it would hide a real map defect"
    )
    classes = {str(entry["class"]) for entry in record["backing"]}  # type: ignore[call-overload]
    assert classes == {"solid_world", "noncollidable_world"}


def test_a_collidable_slab_coincident_with_its_visual_shell_is_not_read_as_decoration() -> None:
    """Coincident geometry, not just geometry behind a shell.

    `test_a_solid_surface_behind_a_visual_shell_is_not_read_as_decoration`
    covers a shell in FRONT of a slab, and the walk-past fix handles it. It does
    not handle the shell and the slab sharing a surface — and that is how
    RoboCasa actually builds every counter top:
    `robocasa/models/fixtures/counter.py` emits one full-span
    `<name>_top_visual` (`contype=0`) and then tiles the *same* volume with
    collidable chunks via `_get_chunks`, identical in `y` and `z` and tiling
    `x`. Stepping `distance + eps` past the shell's face lands *inside* the
    chunk, where the ray reports no further entry surface, so the chunk is never
    seen and the cell reads `noncollidable_world`.

    Measured on `2026-09-07-adr0101-live-1`: 9 of 27 rays struck
    `counter_1_right_group_top_visual` and nothing else, so the tripping cell
    was adjudicated decoration — while the certified probe put the collidable
    chunk `counter_1_right_group_top_0`'s surface at `z = 0.920` with the cell
    spanning `z in [0.900, 0.925]`. The solid geometry was inside the cell the
    whole time, and the map was right again.
    """
    model, data = _model_data()
    record = _backing(model, data, (0.60, 0.0, 0.22))

    assert record["verdict"] == "solid_world", (
        "a collidable chunk coincident with its visual shell was missed; the "
        "stop would be adjudicated as landing on decoration"
    )
    assert record["collidable_overlap_swept"] is True, (
        "the rays found nothing solid, so the overlap sweep must have run"
    )
    names = {str(entry["geom"]) for entry in record["backing"]}  # type: ignore[call-overload]
    assert names & {"counter2_top_0", "counter2_top_1"}, (
        f"the collidable chunk that explains the cell must be named; got {names}"
    )


def test_the_overlap_sweep_does_not_run_when_the_rays_already_found_solid_geometry() -> None:
    """The sweep is a supplement, and must not second-guess a good ray result.

    An AABB overlap can claim a geom whose surface misses the cube, so running
    it unconditionally would let a conservative bound override an exact one.
    It is consulted only when the rays found nothing collidable.
    """
    model, data = _model_data()
    record = _backing(model, data, (0.0, -0.325, 0.22))

    assert record["verdict"] == "solid_world"
    assert record["collidable_overlap_swept"] is False


def test_a_cell_holding_only_robot_geometry_is_swept_for_world_geometry_too() -> None:
    """`self_occupancy_suspect` must not rest on 27 rays having missed the world.

    A cell whose only collidable ray hit is a robot body produces the same
    record whether the world is absent or merely unsampled — and that
    distinction is what separates a self-occupancy stop from ordinary
    quantisation. Measured on the 2026-09-07 `fridge-s2` start-state stop: 15 of
    27 rays struck `robot0_link2_collision`, no world geom appeared, and nothing
    in the record said whether one was there.

    The sweep therefore runs when the rays found no collidable *world*
    geometry, not merely when they found nothing collidable at all. A cell whose
    world backing the rays already found is still left alone.
    """
    model, data = _model_data()
    # The pantry panel is real world geometry; put the probe on it and confirm
    # the rays find it, so the sweep does NOT fire.
    world_only = _backing(model, data, (0.0, 0.36, 0.22))
    assert world_only["collidable_overlap_swept"] is False, (
        "the rays found collidable world geometry; the sweep must not second-guess it"
    )

    # `robot0_link2` sits on the arm. A cell centred on it holds robot geometry
    # and, in this fixture, nothing else — exactly the ambiguous case.
    robot_cell = _backing(model, data, (0.0, 0.0, 0.10))
    classes = {str(e["class"]) for e in robot_cell["backing"]}  # type: ignore[call-overload]
    if not classes or classes == {"noncollidable_world"}:
        pytest.skip("fixture geometry put nothing collidable in this cell")
    if classes <= {"self_occupancy_suspect"}:
        assert robot_cell["collidable_overlap_swept"] is True, (
            "a cell backed only by the robot must be swept for world geometry, "
            "or `self_occupancy_suspect` cannot be told from an unsampled world"
        )


# --- #272: did the cell predate the grasp? -----------------------------------
# `voxel_backing_record`'s `attached_payload` verdict says the payload is in the
# cell NOW. It cannot say whether the payload put it there while it was still
# world geometry, or whether the map gained the cell after the grasp for some
# other reason -- and those have different causes and different fixes. These pin
# the frozen pre-attach set that separates them.


def _occupancy_at(*base_points: tuple[float, float, float]) -> list[int]:
    """A grid whose only occupied cells contain the given base-frame points."""
    occ = [0] * (_GRID_SIZE[0] * _GRID_SIZE[1] * _GRID_SIZE[2])
    for point in base_points:
        occ[_index_at(point)] = 1
    return occ


def _keys(model: Any, data: Any, occupancy: list[int]) -> Any:
    import numpy as np

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _BASE_BODY))
    return occupied_cell_keys(
        occupancy=occupancy,
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        grid_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        base_rot=np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
        base_offset=np.asarray(data.xpos[body_id], dtype=np.float64),
    )


def test_a_cell_frozen_before_the_grasp_is_recognised_after_it() -> None:
    """The whole point: the same physical cell, matched across the attach."""
    model, data = _model_data()
    cell = (0.3, 0.0, 0.145)
    keys, truncated = _keys(model, data, _occupancy_at(cell))
    assert not truncated
    record = _backing(model, data, cell)

    verdict = preattach_verdict(
        world_xyz=record["world_xyz"],  # type: ignore[arg-type]
        resolution=_GRID_RES,
        preattach_keys=keys,
    )
    assert verdict["available"] is True
    assert verdict["preexisting"] is True
    assert verdict["preattach_cells"] == 1


def test_a_cell_absent_before_the_grasp_is_not_claimed_as_preexisting() -> None:
    """A cell the frozen set never held must read as new, not as unknown."""
    model, data = _model_data()
    keys, _ = _keys(model, data, _occupancy_at((0.3, 0.0, 0.145)))
    elsewhere = _backing(model, data, (-0.3, 0.3, 0.145))

    verdict = preattach_verdict(
        world_xyz=elsewhere["world_xyz"],  # type: ignore[arg-type]
        resolution=_GRID_RES,
        preattach_keys=keys,
    )
    assert verdict["preexisting"] is False
    assert verdict["within_one_cell"] is False


def test_the_frozen_set_survives_the_base_driving_away() -> None:
    """The key must name a WORLD cell, not a grid index.

    The published lattice is the source map's, fixed in map/odom; only the
    base-frame ``origin`` slides as the robot moves. An implementation that
    keyed on the voxel index would pass every other test here and then answer
    "not pre-existing" for the very same physical cell the moment the base
    moved -- which is exactly the window a carried payload is stopped in.
    """
    model, data = _model_data()
    world_cell = (0.3, 0.0, 0.145)
    keys, _ = _keys(model, data, _occupancy_at(world_cell))
    before = _backing(model, data, world_cell)

    # Drive the base 0.25 m forward; the same physical point is now a different
    # base-frame position, hence a different grid index.
    data.qpos[0] = 0.25
    mujoco.mj_forward(model, data)
    after = _backing(model, data, (world_cell[0] - 0.25, world_cell[1], world_cell[2]))

    assert after["world_xyz"] == pytest.approx(before["world_xyz"], abs=1e-9)
    assert after["voxel_index"] != before["voxel_index"]
    verdict = preattach_verdict(
        world_xyz=after["world_xyz"],  # type: ignore[arg-type]
        resolution=_GRID_RES,
        preattach_keys=keys,
    )
    assert verdict["preexisting"] is True


def test_no_snapshot_reads_as_unavailable_never_as_a_new_cell() -> None:
    """ "Nobody looked" must not be reported as "the cell is new".

    Collapsing those would let a run with no frozen set read as evidence that
    every stop's cell postdates the grasp -- the exact wrong conclusion, drawn
    from an absence.
    """
    model, data = _model_data()
    record = _backing(model, data, (0.3, 0.0, 0.145))
    for missing in (None, frozenset()):
        verdict = preattach_verdict(
            world_xyz=record["world_xyz"],  # type: ignore[arg-type]
            resolution=_GRID_RES,
            preattach_keys=missing,  # type: ignore[arg-type]
        )
        assert verdict["available"] is False
        assert "preexisting" not in verdict


def test_an_oversized_grid_is_refused_rather_than_partially_sampled() -> None:
    """A truncated set would answer "new" for cells it never looked at."""
    model, data = _model_data()
    import numpy as np

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _BASE_BODY))
    keys, truncated = occupied_cell_keys(
        occupancy=[1] * (_GRID_SIZE[0] * _GRID_SIZE[1] * _GRID_SIZE[2]),
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        grid_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        base_rot=np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
        base_offset=np.asarray(data.xpos[body_id], dtype=np.float64),
        max_cells=16,
    )
    assert truncated is True
    assert keys == frozenset()
    verdict = preattach_verdict(
        world_xyz=[0.0, 0.0, 0.0], resolution=_GRID_RES, preattach_keys=keys
    )
    assert verdict["available"] is False


def test_a_non_unit_grid_quaternion_yields_no_set_rather_than_identity() -> None:
    """Same posture as ``voxel_backing_record``: refuse, never assume identity."""
    model, data = _model_data()
    import numpy as np

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _BASE_BODY))
    keys, truncated = occupied_cell_keys(
        occupancy=_occupancy_at((0.3, 0.0, 0.145)),
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        grid_orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
        base_rot=np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
        base_offset=np.asarray(data.xpos[body_id], dtype=np.float64),
    )
    assert keys == frozenset()
    assert truncated is False


def test_a_payload_in_the_cell_must_not_hide_the_world_surface_behind_it() -> None:
    """A carried payload is neither world nor robot, and the sweep must know it.

    The coincident-shell sweep runs only when the ray fans found no collidable
    **world** geometry, because a cell whose world backing the rays already
    found needs no second look. The condition that implements that excludes
    ``robot_body_ids`` — and an attached payload is in neither set, so a
    collidable payload geom suppressed the sweep exactly as a world geom would,
    while satisfying none of the reasoning that makes suppression safe.

    The consequence is a **false** ``attached_payload`` verdict: the cell reads
    as holding only the carried object when it also holds a counter slab the
    fans cannot see through its own visual shell. That is the verdict #272 is
    built on, so it has to mean what it says — a payload resting on a surface
    is the *normal* case, not a corner one, and it is exactly when both are in
    one cell.
    """
    model, data = _model_data()
    cup_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carried_cup"))
    adr = int(model.jnt_qposadr[int(model.body_jntadr[cup_body])])
    # Into the cell that holds `counter_2_right_group`'s coincident
    # visual-plus-collidable pair — the shape the sweep exists for.
    cell_base = (0.625, 0.025, 0.225)
    data.qpos[adr : adr + 3] = [cell_base[0], cell_base[1], cell_base[2] + 0.18]
    mujoco.mj_forward(model, data)

    record = _backing(model, data, cell_base, attached=frozenset({cup_body}))
    names = {str(entry["geom"]) for entry in record["backing"]}  # type: ignore[index,union-attr]

    assert "cup_body" in names, "the payload really is in this cell"
    assert record["collidable_overlap_swept"] is True, (
        "a payload is not world geometry, so its presence must not suppress the sweep"
    )
    assert "counter2_top_0" in names, "the collidable slab sharing the cell must be found"
    assert record["verdict"] == "solid_world", (
        "real geometry outranks the payload: this cell is explained by the counter"
    )


# --- #275: decode the kernel's index against the grid the KERNEL held --------


def test_the_decode_grid_is_chosen_by_stamp_not_by_arrival() -> None:
    """Newest grid at or before the evidence stamp — never a later one.

    The published window is snapped to the octree's cell boundaries, so a
    base drift across one boundary shifts the whole window by a cell and the
    same index names a different cell in the next message. Decoding against
    the latest grid is then one full cell wrong, silently, on every record.
    """
    hist = [
        {"stamp_ns": 100, "origin": (0.0, 0.0, 0.0)},
        {"stamp_ns": 200, "origin": (0.0, 0.0, 0.0)},
        {"stamp_ns": 300, "origin": (-0.05, 0.0, 0.0)},  # window shifted one cell
    ]
    assert grid_current_at(hist, 250)["stamp_ns"] == 200
    assert grid_current_at(hist, 200)["stamp_ns"] == 200, "at-or-before, inclusive"
    assert grid_current_at(hist, 999)["stamp_ns"] == 300
    assert grid_current_at(hist, 50) is None, "nothing old enough: say so, do not use a newer one"
    assert grid_current_at([], 250) is None
    # Out-of-order arrival must not matter: stamp decides.
    assert grid_current_at(list(reversed(hist)), 250)["stamp_ns"] == 200


def test_a_one_cell_window_shift_moves_the_decoded_cell_by_exactly_one_cell() -> None:
    """The arithmetic the whole hazard rests on, pinned once.

    Same index, two grids whose origins differ by one resolution in x: the
    decoded world positions differ by exactly that. That is the error every
    `world_xyz` carries when the wrong grid is used, and it is one whole cell.
    """
    model, data = _model_data()
    index = _index_at((0.3, 0.0, 0.145))
    shifted = (_GRID_ORIGIN[0] - _GRID_RES, _GRID_ORIGIN[1], _GRID_ORIGIN[2])
    a = voxel_backing_record(
        model,
        data,
        voxel_index=index,
        grid_origin=_GRID_ORIGIN,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    b = voxel_backing_record(
        model,
        data,
        voxel_index=index,
        grid_origin=shifted,
        grid_resolution=_GRID_RES,
        grid_size=_GRID_SIZE,
        robot_body_ids=_robot_bodies(model),
        base_frame_body=_BASE_BODY,
    )
    delta = [b["world_xyz"][k] - a["world_xyz"][k] for k in range(3)]  # type: ignore[index]
    assert delta == pytest.approx([-_GRID_RES, 0.0, 0.0], abs=1e-9)
