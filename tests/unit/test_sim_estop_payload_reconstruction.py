# SPDX-License-Identifier: Apache-2.0
"""A recorded stop must place the CARRIED OBJECT again, not just the robot.

``sim.estop_ground_truth_snapshot`` reconstructs the robot exactly — its
``robot_joint_state`` and ``base_frame_tf`` were verified to 0.000–0.863 mm
across the four checked-in stops — while ``attached_bodies`` carried only
``world_xyz``. A position is not a pose, so driving a recorded snapshot back
into a live model left the payload at its *reset* attitude, and every
payload-side distance in every recorded round described a different
configuration than the one that stopped (#172). That is a CLAUDE.md §1.8
violation in the producer: a trace that cannot replay the geometry it
adjudicates is not a trace.

These tests drive the real snapshot against a real compiled ``MjModel`` /
``MjData`` (no mocks, CLAUDE.md §1.11) and then *reconstruct* from the record
alone, which is the only thing that can hold the property. The payload here
carries a handle geom offset 50 mm from its body frame — the "geoms are not
body-aligned" case — because a body-centred geom cannot move under a rotation
about its own origin and so cannot tell a placed payload from a misplaced one.
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import JointState
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import estop_ground_truth_snapshot

mujoco = pytest.importorskip("mujoco")

# Robosuite/RoboCasa naming (``robot0_*``), so the real ``robot_self_body_ids``
# prefix derivation resolves the robot exactly as it does on a kitchen scene.
# ``mug_main`` is the carried payload; its handle is deliberately off-axis.
_MJCF = """
<mujoco model="payload_pose_reconstruction">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot0_base">
      <geom name="base_col" type="cylinder" size="0.06 0.05"/>
      <body name="robot0_link7" pos="0 0 0.3">
        <joint name="robot0_joint7" type="hinge" axis="0 1 0"/>
        <geom name="link7_col" type="sphere" size="0.04"/>
      </body>
    </body>
    <body name="counter_main" pos="0.6 0 0.5">
      <geom name="counter_top" type="box" size="0.2 0.4 0.02"/>
    </body>
    <body name="mug_main" pos="0.25 0 0.42">
      <freejoint name="mug_joint"/>
      <geom name="mug_cup" type="cylinder" size="0.03 0.05"/>
      <geom name="mug_handle" type="box" pos="0.05 0 0" size="0.012 0.006 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

# A pose no reset would produce: tilted 40 deg about x, then yawed 55 deg about
# z. Both matter — a cylinder is symmetric about its own axis, so the yaw alone
# is invisible on ``mug_cup`` and only the handle records it.
_CARRIED_XYZ = (0.312, -0.087, 0.455)
_CARRIED_RPY = (np.deg2rad(40.0), 0.0, np.deg2rad(55.0))

_PAYLOAD_GEOMS = ("mug_cup", "mug_handle")
# The snapshot rounds to 6 decimals, so a µm is the floor. The robot side of
# the same record reconstructs to 0.863 mm; hold the payload to 10 µm.
_RECONSTRUCTION_TOL_M = 1e-5


def _quat_wxyz(rpy: tuple[float, float, float]) -> np.ndarray:
    """Roll-pitch-yaw → MuJoCo ``(w, x, y, z)``, via MuJoCo's own conversion."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.asarray(rpy, dtype=np.float64), "xyz")
    return quat


def _model() -> object:
    return mujoco.MjModel.from_xml_string(_MJCF)


def _place_payload(model: object, data: object, xyz: object, quat_wxyz: object) -> None:
    """Put ``mug_main`` at a world pose through its free joint, then settle FK."""
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint"))
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr : adr + 3] = np.asarray(xyz, dtype=np.float64)
    data.qpos[adr + 3 : adr + 7] = np.asarray(quat_wxyz, dtype=np.float64)
    mujoco.mj_forward(model, data)


def _geom_xpos(model: object, data: object, name: str) -> np.ndarray:
    gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    return np.asarray(data.geom_xpos[gid], dtype=np.float64).copy()


def _joint_state(model: object, data: object) -> JointState:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_joint7"))
    return JointState(
        name=["robot0_joint7"],
        position=[float(data.qpos[int(model.jnt_qposadr[jid])])],
        velocity=[float(data.qvel[int(model.jnt_dofadr[jid])])],
        effort=[0.0],
        stamp_ns=1_755_100_000_000_000_000,
    )


def _carried_snapshot() -> tuple[object, dict[str, object], dict[str, np.ndarray]]:
    """Compile, carry the mug at ``_CARRIED_*``, and snapshot that stop."""
    model = _model()
    data = mujoco.MjData(model)
    _place_payload(model, data, _CARRIED_XYZ, _quat_wxyz(_CARRIED_RPY))
    truth = {name: _geom_xpos(model, data, name) for name in _PAYLOAD_GEOMS}
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=robot_self_body_ids(model, ["robot0_joint7"]),
        attached_body_ids=frozenset(
            {int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_main"))}
        ),
        base_frame_body="robot0_base",
        joint_state=_joint_state(model, data),
    )
    return model, snapshot, truth


def _payload_record(snapshot: dict[str, object]) -> dict[str, object]:
    bodies = snapshot["attached_bodies"]
    assert isinstance(bodies, list) and len(bodies) == 1
    record = bodies[0]
    assert isinstance(record, dict)
    assert record["name"] == "mug_main"
    return record


def test_attached_body_record_carries_an_orientation() -> None:
    """``attached_bodies`` records a pose — position AND orientation."""
    _, snapshot, _ = _carried_snapshot()
    record = _payload_record(snapshot)

    assert list(record["world_xyz"]) == pytest.approx(list(_CARRIED_XYZ), abs=1e-6)
    quat = record["world_quat_wxyz"]
    assert isinstance(quat, list) and len(quat) == 4
    assert quat == pytest.approx(list(_quat_wxyz(_CARRIED_RPY)), abs=1e-6)
    assert float(np.linalg.norm(np.asarray(quat, dtype=np.float64))) == pytest.approx(1.0, abs=1e-5)


def test_the_record_alone_places_every_payload_geom_again() -> None:
    """Replaying the record into a fresh model puts the payload back exactly.

    This is the property #172 says a trace must have: the recorded stop, and
    nothing else, has to reproduce the geometry the stop was adjudicated on.
    """
    model, snapshot, truth = _carried_snapshot()
    record = _payload_record(snapshot)

    replay = mujoco.MjData(model)
    _place_payload(model, replay, record["world_xyz"], record["world_quat_wxyz"])

    for name, expected in truth.items():
        assert _geom_xpos(model, replay, name) == pytest.approx(
            expected, abs=_RECONSTRUCTION_TOL_M
        ), name


def test_position_alone_cannot_place_the_payload() -> None:
    """The negative control: ``world_xyz`` without the orientation misplaces it.

    Without this the test above would keep passing against a record that had
    silently dropped the quaternion again, as long as the fixture happened to
    be axis-aligned. Replaying position-only is exactly what every consumer of
    a pre-#172 record was forced to do.
    """
    model, snapshot, truth = _carried_snapshot()
    record = _payload_record(snapshot)

    replay = mujoco.MjData(model)
    _place_payload(model, replay, record["world_xyz"], (1.0, 0.0, 0.0, 0.0))

    error_m = float(np.linalg.norm(_geom_xpos(model, replay, "mug_handle") - truth["mug_handle"]))
    # The handle sits 50 mm off the body frame, so the attitude is worth tens of
    # millimetres on its own — an order of magnitude past the 21.65 mm voxel
    # half-diagonal every payload-side verdict is adjudicated against.
    assert error_m > 0.02
