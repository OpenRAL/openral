# SPDX-License-Identifier: Apache-2.0
"""E-stop ground truth for UNATTACHED (pre-grasp) stops, not just carried payloads.

The 2026-08-13 post-fix matrix stopped four times; three were pre-grasp
arm↔world stops (``panda_link7`` −15.06 mm at predicted-horizon step 0,
``panda_link7`` −4.79 mm reactive, ``panda_link1`` −17.28 mm reactive) and the
snapshot early-returned on all three because nothing was attached — so none of
them could be adjudicated real-vs-false.

These tests drive :func:`openral_hal.sim_sensor_bridge.estop_ground_truth_snapshot`
against a real compiled ``MjModel``/``MjData`` (no mocks, CLAUDE.md §1.11) in the
three shapes a stop actually takes: an arm already inside a fixture, an arm a few
millimetres away from one (the margin stop, where MuJoCo reports NO contact at
all), and a carried payload (the pre-existing attached case, which must keep
reporting exactly what it did).
"""

from __future__ import annotations

import pytest
from openral_core import JointState
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import candidate_chunk_digest, estop_ground_truth_snapshot

mujoco = pytest.importorskip("mujoco")

# Robosuite/RoboCasa body + joint naming (``robot0_*`` / ``gripper0_*``), so the
# real ``robot_self_body_ids`` prefix derivation resolves the robot exactly as it
# does on a composed kitchen scene. ``counter_main`` is the world fixture the arm
# stops against; ``sink_cup`` is the free payload it may be carrying.
_MJCF = """
<mujoco model="estop_ground_truth">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot0_base">
      <geom name="base_col" type="cylinder" size="0.06 0.05"/>
      <body name="robot0_link1" pos="0 0 0.2">
        <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
        <geom name="link1_col" type="capsule" fromto="0 0 0 0 0 0.15" size="0.05"/>
        <body name="robot0_link7" pos="0 0 0.3">
          <joint name="robot0_joint7" type="hinge" axis="0 1 0"/>
          <geom name="link7_col" type="sphere" size="0.04"/>
          <body name="gripper0_finger" pos="0 -0.06 0">
            <joint name="gripper0_finger_joint" type="slide" axis="0 1 0"/>
            <geom name="finger_col" type="box" size="0.01 0.02 0.03"/>
          </body>
        </body>
      </body>
    </body>
    <body name="counter_main" pos="{counter_x} 0 0.5">
      <geom name="counter_top" type="box" size="0.2 0.4 0.02"/>
    </body>
    <body name="sink_cup" pos="{cup_y_pos}">
      <freejoint name="sink_cup_joint"/>
      <geom name="cup_body" type="cylinder" size="0.03 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# Sphere centre (0, 0, 0.5) with r=0.04; the counter's near face is at
# ``counter_x - 0.2``. 0.23 → 10 mm interpenetration, 0.245 → a 5 mm gap
# (the margin stop: real, but invisible to ``data.ncon``).
_CONTACT_COUNTER_X = 0.23
_NEAR_MISS_COUNTER_X = 0.245
_NEAR_MISS_GAP_M = 0.005
_PENETRATION_M = -0.01
# Payload parked clear of the arm, or overlapping the finger by ~10 mm.
_CUP_CLEAR = "0.8 0.4 0.5"
_CUP_IN_GRIPPER = "0 -0.06 0.5"


def _model_data(*, counter_x: float, cup_pos: str) -> tuple[object, object]:
    """Compile the MJCF and settle kinematics + contacts at ``robot0_joint1=0.3``."""
    model = mujoco.MjModel.from_xml_string(_MJCF.format(counter_x=counter_x, cup_y_pos=cup_pos))
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_joint1")
    data.qpos[int(model.jnt_qposadr[jid])] = 0.3
    mujoco.mj_forward(model, data)
    return model, data


def _robot_bodies(model: object) -> frozenset[int]:
    """The robot's own body ids, via the same helper the bridge's self-filter uses."""
    return robot_self_body_ids(model, ["robot0_joint1", "robot0_joint7", "gripper0_finger_joint"])


def _joint_state(model: object, data: object) -> JointState:
    """The measured joint vector the kernel seeds ``q_meas`` from, read off MuJoCo."""
    names = ["robot0_joint1", "robot0_joint7", "gripper0_finger_joint"]
    positions = []
    velocities = []
    for name in names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        positions.append(float(data.qpos[int(model.jnt_qposadr[jid])]))
        velocities.append(float(data.qvel[int(model.jnt_dofadr[jid])]))
    return JointState(
        name=names,
        position=positions,
        velocity=velocities,
        effort=[0.0] * len(names),
        stamp_ns=1_755_100_000_000_000_000,
    )


def _body_ids(model: object, *names: str) -> frozenset[int]:
    return frozenset(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) for name in names
    )


def test_unattached_arm_world_contact_is_reported() -> None:
    """An arm inside a fixture with nothing carried still yields full ground truth."""
    model, data = _model_data(counter_x=_CONTACT_COUNTER_X, cup_pos=_CUP_CLEAR)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
        joint_state=_joint_state(model, data),
    )

    assert snapshot["stop_class"] == "robot_world"
    assert snapshot["attached_bodies"] == []
    assert snapshot["payload_contacts"] == []
    contacts = snapshot["robot_world_contacts"]
    assert isinstance(contacts, list) and contacts, "the arm is inside the counter"
    hit = next(c for c in contacts if c["body_b"] == "counter_main")
    assert hit["body_a"] == "robot0_link7"
    assert hit["geom_a"] == "link7_col"
    assert hit["geom_b"] == "counter_top"
    assert hit["distance_m"] == pytest.approx(_PENETRATION_M, abs=1e-3)
    assert len(hit["position_xyz"]) == 3
    assert hit["body_a_world_xyz"] == [0.0, 0.0, 0.5]


def test_unattached_margin_stop_reports_near_miss_distance() -> None:
    """A 5 mm gap: no MuJoCo contact exists, so the near-miss probe carries the truth."""
    model, data = _model_data(counter_x=_NEAR_MISS_COUNTER_X, cup_pos=_CUP_CLEAR)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
        joint_state=_joint_state(model, data),
    )

    assert snapshot["robot_world_contacts"] == []  # margin stop — ncon says nothing
    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list) and pairs
    closest = pairs[0]
    assert closest["body_a"] == "robot0_link7"
    assert closest["body_b"] == "counter_main"
    assert closest["distance_m"] == pytest.approx(_NEAR_MISS_GAP_M, abs=5e-4)
    assert [p["distance_m"] for p in pairs] == sorted(p["distance_m"] for p in pairs)


def test_unattached_snapshot_carries_joint_state_and_base_tf() -> None:
    """The FK inputs an offline predicted-horizon reconstruction needs are present."""
    model, data = _model_data(counter_x=_NEAR_MISS_COUNTER_X, cup_pos=_CUP_CLEAR)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
        joint_state=_joint_state(model, data),
    )

    joints = snapshot["robot_joint_state"]
    assert isinstance(joints, dict)
    assert joints["name"] == ["robot0_joint1", "robot0_joint7", "gripper0_finger_joint"]
    assert joints["position"][0] == pytest.approx(0.3)
    assert joints["stamp_ns"] == 1_755_100_000_000_000_000
    base_tf = snapshot["base_frame_tf"]
    assert isinstance(base_tf, dict)
    assert base_tf["body"] == "robot0_base"
    assert base_tf["world_xyz"] == [0.0, 0.0, 0.0]
    assert base_tf["world_quat_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert snapshot["sim_time_s"] == 0.0


def test_attached_payload_stop_keeps_reporting_payload_contacts() -> None:
    """The carried-payload case is unchanged: bodies + payload contacts still emitted."""
    model, data = _model_data(counter_x=_NEAR_MISS_COUNTER_X, cup_pos=_CUP_IN_GRIPPER)
    attached = _body_ids(model, "sink_cup")
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=attached,
        base_frame_body="robot0_base",
        joint_state=_joint_state(model, data),
    )

    assert snapshot["stop_class"] == "attached_payload"
    bodies = snapshot["attached_bodies"]
    assert isinstance(bodies, list)
    assert [b["name"] for b in bodies] == ["sink_cup"]
    payload_contacts = snapshot["payload_contacts"]
    assert isinstance(payload_contacts, list) and payload_contacts
    # The payload is normalised onto the ``_a`` side of every record.
    assert {c["body_a"] for c in payload_contacts} == {"sink_cup"}
    assert {c["body_b"] for c in payload_contacts} <= {"gripper0_finger", "robot0_link7"}
    # The payload is not double-counted as a world obstacle.
    world_contacts = snapshot["robot_world_contacts"]
    assert isinstance(world_contacts, list)
    assert all(c["body_b"] != "sink_cup" and c["body_a"] != "sink_cup" for c in world_contacts)
    assert all(
        p["body_a"] != "sink_cup" and p["body_b"] != "sink_cup"
        for p in snapshot["nearest_robot_world_pairs"]  # type: ignore[union-attr]
    )


def test_candidate_chunk_digest_reshapes_a_cartesian_delta_chunk() -> None:
    """A CARTESIAN_DELTA chunk logs its per-tick rows, which is what FK replays."""
    digest = candidate_chunk_digest(
        stamp_ns=1_755_100_000_000_000_000,
        control_mode=5,  # CARTESIAN_DELTA, per CONTROL_MODE_TO_UINT8
        horizon=2,
        n_dof=7,
        flat=[
            0.05,
            0.0,
            -0.02,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.04,
            0.0,
            -0.03,
            0.0,
            0.0,
            0.0,
            -1.0,
        ],
        cartesian_delta_scale=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0],
        ee_name="panda_hand_tcp",
        frame_id="panda_link0",
        rskill_id="pi05_libero",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        tick_index=12,
    )

    assert digest["control_mode"] == "cartesian_delta"
    assert digest["control_mode_uint8"] == 5
    assert digest["ticks"] == [
        [0.05, 0.0, -0.02, 0.0, 0.0, 0.0, -1.0],
        [0.04, 0.0, -0.03, 0.0, 0.0, 0.0, -1.0],
    ]
    assert "shape_mismatch" not in digest
    assert digest["ee_name"] == "panda_hand_tcp"
    assert digest["rskill_id"] == "pi05_libero"
    assert digest["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_candidate_chunk_digest_flags_a_shape_mismatch() -> None:
    """A flat payload that disagrees with horizon×n_dof is flagged, never truncated."""
    digest = candidate_chunk_digest(
        stamp_ns=1_755_100_000_000_000_000,
        control_mode=0,  # JOINT_POSITION
        horizon=3,
        n_dof=7,
        flat=[0.0, 0.1, 0.2],
    )

    assert digest["shape_mismatch"] is True
    assert digest["flat"] == [0.0, 0.1, 0.2]
    assert "ticks" not in digest


# ── near-miss probe coverage (the 2026-08-14 drawer_utensil adjudication) ─────
#
# The ``panda_link1`` −17.28 mm stop was adjudicated FALSE on the strength of a
# silence: the snapshot's 47 near-miss pairs named ``mobilebase0_*`` and
# ``robot0_link0`` but never ``robot0_link1``, so "nothing physical within
# 100 mm of link1" was read straight off the report. That silence was an
# artifact. A RoboCasa kitchen ships FOUR plane geoms (``floor_1_room_g0`` and
# ``floor_1_backing_room_g0``, each with a ``_vis`` twin) and MuJoCo gives a
# plane ``geom_rbound == 0``; the probe treated that as an infinite radius, so
# every robot↔floor pair scored ``-inf`` and sorted ahead of every finite pair.
# With ~70 geoms on a robosuite mobile Panda that is ~280 pairs against a
# 256-call budget: the arm's real near-misses were never probed at all.
#
# The fixture below is that shape in miniature — planes plus a cabinet panel a
# known 20 mm off ``robot0_link1`` — and it is a real compiled ``MjModel``.

_ROOM_PLANES = ("floor_1_room_g0", "floor_1_room_g0_vis", "floor_1_backing_room_g0")
_PANEL_GAP_M = 0.020

_KITCHEN_MJCF_HEAD = """
<mujoco model="estop_probe_coverage">
  <option gravity="0 0 0"/>
  <worldbody>
"""
_KITCHEN_MJCF_TAIL = """
    <body name="robot0_base" pos="0 0 0.4">
      <geom name="robot0_base_col" type="cylinder" size="0.06 0.05"/>
{filler}
      <body name="robot0_link1" pos="0 0 0.2">
        <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
        <geom name="robot0_link1_col" type="capsule" fromto="0 0 0 0 0 0.15" size="0.05"/>
      </body>
    </body>
    <body name="stack_2_left_group_3_door_main" pos="{panel_x} 0 0.65">
      <geom name="stack_2_left_group_3_door_g0" type="box" size="0.01 0.3 0.15"/>
    </body>
  </worldbody>
</mujoco>
"""


def _kitchen_model_data(*, n_filler_geoms: int) -> tuple[object, object]:
    """A real ``MjModel`` with RoboCasa's plane count and a panel 20 mm off link1.

    ``n_filler_geoms`` stands in for the visual geoms robosuite hangs off the
    base (``robot0_g8_vis`` … ``mobilebase0_g7_vis``): they matter here only
    because each one multiplies the robot↔plane pair count.
    """
    planes = "\n".join(
        f'    <geom name="{name}" type="plane" size="5 5 0.1" pos="0 0 {-0.02 * i}"/>'
        for i, name in enumerate(_ROOM_PLANES)
    )
    filler = "\n".join(
        f'      <geom name="robot0_g{i}_vis" type="sphere" size="0.01" '
        f'pos="{0.02 * i - 0.05:.3f} 0.09 0" contype="0" conaffinity="0"/>'
        for i in range(n_filler_geoms)
    )
    # Panel near face at ``panel_x - 0.01``; link1's capsule radius is 0.05.
    panel_x = 0.05 + _PANEL_GAP_M + 0.01
    xml = (
        _KITCHEN_MJCF_HEAD
        + planes
        + _KITCHEN_MJCF_TAIL.format(filler=filler, panel_x=f"{panel_x:.4f}")
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _kitchen_snapshot(model: object, data: object, *, max_calls: int) -> dict[str, object]:
    return estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=robot_self_body_ids(model, ["robot0_joint1"]),
        base_frame_body="robot0_base",
        max_pairs=32,
        max_calls=max_calls,
    )


def test_near_miss_probe_reports_link1_when_planes_outnumber_the_budget() -> None:
    """The arm's real near-miss survives a budget the scene's planes could eat.

    Twelve robot geoms against three planes is 36 pairs that a radius-``inf``
    ranking puts first; a 36-call budget then leaves nothing for the panel
    20 mm off ``robot0_link1``. The probe must still report it.
    """
    model, data = _kitchen_model_data(n_filler_geoms=10)
    snapshot = _kitchen_snapshot(model, data, max_calls=36)

    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list)
    link1 = [p for p in pairs if p["body_a"] == "robot0_link1"]
    assert link1, (
        "the near-miss probe never looked at link1 — an absent pair is being "
        f"reported as an absent obstacle; got {[p['body_a'] for p in pairs]}"
    )
    assert link1[0]["body_b"] == "stack_2_left_group_3_door_main"
    assert link1[0]["distance_m"] == pytest.approx(_PANEL_GAP_M, abs=1e-3)


def test_near_miss_probe_reports_its_own_coverage() -> None:
    """An empty pair list is only evidence of clearance when the probe was complete."""
    model, data = _kitchen_model_data(n_filler_geoms=10)
    snapshot = _kitchen_snapshot(model, data, max_calls=36)

    coverage = snapshot["nearest_probe_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["max_calls"] == 36
    assert coverage["distmax_m"] == pytest.approx(0.10)
    assert coverage["side_geoms"] == coverage["side_geoms_probed"], (
        "some robot geom was never probed, so its silence means nothing"
    )
    assert coverage["probed_pairs"] <= 36
    assert coverage["truncated"] is (coverage["candidate_pairs"] > coverage["probed_pairs"])


def test_scene_planes_no_longer_outrank_every_finite_pair() -> None:
    """A plane is bounded exactly, so it competes on distance like any other geom.

    ``geom_rbound == 0`` means "no bounding sphere", not "infinitely large":
    the exact plane-to-sphere bound is ``|n·(c-p)| - r``. Without this a floor
    5 m away outranks a cabinet 20 mm away and consumes the probe budget.
    """
    import numpy as np
    from openral_hal.sim_sensor_bridge import _pair_distance_lower_bound

    model, data = _kitchen_model_data(n_filler_geoms=4)
    side = robot_self_body_ids(model, ["robot0_joint1"])
    body_of_geom = np.asarray(model.geom_bodyid, dtype=np.int64)
    in_side = np.isin(body_of_geom, np.fromiter(side, dtype=np.int64, count=len(side)))
    side_geoms = np.flatnonzero(in_side)
    other_geoms = np.flatnonzero(~in_side)

    gap = _pair_distance_lower_bound(model, data, side_geoms, other_geoms)
    assert np.isfinite(gap).all(), "a plane pair is still scoring -inf"

    link1_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot0_link1_col"))
    panel_geom = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stack_2_left_group_3_door_g0")
    )
    row = int(np.flatnonzero(side_geoms == link1_geom)[0])
    panel_col = int(np.flatnonzero(other_geoms == panel_geom)[0])
    plane_cols = [
        int(np.flatnonzero(other_geoms == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n))[0])
        for n in _ROOM_PLANES
    ]
    assert all(gap[row, panel_col] < gap[row, col] for col in plane_cols), (
        "the panel 20 mm away must rank ahead of the floor half a metre below"
    )
    # And the bound stays a LOWER bound: never above the true distance.
    truth = float(mujoco.mj_geomDistance(model, data, link1_geom, panel_geom, 1.0, None))
    assert gap[row, panel_col] <= truth + 1e-9
