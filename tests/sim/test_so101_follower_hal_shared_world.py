"""Two SO-101 arms in ONE MuJoCo world, each behind its own HAL (issue #666).

Real MuJoCo physics, the real ``robots/so101_follower`` manifest and the real
``scenes/worlds/so101_handover_table.yaml`` fixture; no mocks. The two HALs live
in this one process here (in a deploy each is in its own graph, on its own ROS
domain); the world's server and its Unix-socket wire are the real ones.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

try:
    import mujoco  # noqa: F401
    from robot_descriptions import so_arm101_mj_description as _so101

    _ = _so101.MJCF_PATH  # triggers the lazy clone / cache lookup
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _SIM_ERROR: str | None = str(exc)
else:
    _SIM_ERROR = None

import numpy as np
from numpy.typing import NDArray
from openral_core import Action, ControlMode, RobotDescription, SharedWorldScene
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError, ROSEStopRequested
from openral_hal import build_hal
from openral_hal.peer_voxels import (
    occupancy_grid,
    peer_boxes,
    peer_geoms,
    peer_targets,
    servo_target,
    swept_peer_boxes,
)
from openral_hal.shared_world import SharedWorldArmHAL, SharedWorldServer, robot_root_bodies

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_SIM_ERROR is not None, reason=f"mujoco / SO-101 MJCF: {_SIM_ERROR}"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_YAML = REPO_ROOT / "scenes" / "worlds" / "so101_handover_table.yaml"
ROBOT_YAML = str(REPO_ROOT / "robots" / "so101_follower" / "robot.yaml")
TICK_S = 1.0 / 30.0  # the SO-101's control period

# An arena whose cube slides: a slippery 20 degree ramp, downhill along world +X,
# clear of both arms.
_RAMP_ARENA = """<mujoco><compiler angle="radian"/><worldbody>
  <geom name="ramp" type="box" pos="-0.45 0 0.1" size="0.3 0.2 0.01" euler="0 0.35 0"
        friction="0.1 0.005 0.0001"/>
  <body name="handover_cube" pos="-0.6 0 0.2">
    <freejoint name="handover_cube_joint"/>
    <geom type="box" size="0.015 0.015 0.015" mass="0.02" friction="0.1 0.005 0.0001"/>
  </body>
</worldbody></mujoco>"""


class _World:
    """A served world and one connected HAL per robot."""

    def __init__(self, world: SharedWorldScene, arena: Path, tmp_path: Path) -> None:
        self.address = str(tmp_path / "w.sock")
        self.server = SharedWorldServer(
            world, arena, {r.name: ROBOT_YAML for r in world.robots}, self.address
        )
        self.server.start()
        self.hals: dict[str, SharedWorldArmHAL] = {}
        for robot in world.robots:
            self.hals[robot.name] = self.hal(robot.name)
            self.hals[robot.name].connect()

    def hal(self, name: str) -> SharedWorldArmHAL:
        hal = build_hal(
            RobotDescription.from_yaml(ROBOT_YAML),
            mode="sim",
            transport={"shared_world": self.address, "shared_world_robot": name},
        )
        assert isinstance(hal, SharedWorldArmHAL)
        return hal

    def close(self) -> None:
        for hal in self.hals.values():
            hal.disconnect()
        self.server.close()


@pytest.fixture
def table(tmp_path: Path) -> Iterator[_World]:
    """The committed fixture: two arms facing each other across the table."""
    world = SharedWorldScene.from_yaml(str(WORLD_YAML))
    served = _World(world, WORLD_YAML.parent / world.arena_mjcf, tmp_path)
    yield served
    served.close()


def _action(targets: list[float]) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION, joint_targets=[targets], horizon=1, dt_s=TICK_S
    )


def _cube_xyz(hal: SharedWorldArmHAL) -> NDArray[np.float64]:
    handles = hal.mujoco_handles()
    assert handles is not None
    return np.array(handles[1].body("handover_cube").xpos)


def test_each_robot_sees_the_world_in_its_own_base_frame(table: _World) -> None:
    """Each arm has the cube ahead (-Y) to its right, and the peer's base twice as far."""
    for name, peer in (("a", "b"), ("b", "a")):
        model, data = table.hals[name].mujoco_handles()  # type: ignore[misc]
        np.testing.assert_allclose(data.body("handover_cube").xpos, [-0.1, -0.27, 0.015], atol=1e-6)
        np.testing.assert_allclose(data.body(f"{peer}_base").xpos, [-0.2, -0.54, 0], atol=1e-6)
        # Its own elements are unprefixed: the manifest's camera names hold.
        cameras = {model.camera(i).name for i in range(model.ncam)}
        assert {"top", "wrist", f"{peer}_top", f"{peer}_wrist"} <= cameras


def test_one_commanding_robot_keeps_one_action_one_period(table: _World) -> None:
    """A alone drives the clock: 30 actions are 1 s for both graphs, and B sees A move."""
    a, b = table.hals["a"], table.hals["b"]
    start = table.server.sim_time_s()
    target = [0.3, -0.4, 0.4, 0.2, 0.0, 0.5]
    for _ in range(30):
        a.send_action(_action(target))
    assert table.server.sim_time_s() - start == pytest.approx(30 * TICK_S, abs=1e-9)
    # B is idle: its idle step must not advance a world A is commanding.
    assert b.idle_step(wall_dt_s=TICK_S) is True
    assert b.sim_time_ns() == a.sim_time_ns()
    assert table.server.sim_time_s() - start == pytest.approx(30 * TICK_S, abs=1e-9)
    a_joints = np.array(a.read_state().position[:5])
    assert abs(a_joints[0] - target[0]) < 0.25 and abs(a_joints[0]) > 0.02  # moving to target
    _, b_data = b.mujoco_handles()  # type: ignore[misc]
    np.testing.assert_allclose(b_data.qpos[6:11], a_joints, atol=1e-9)  # A's block in B's view
    # ... and A's targets, so the forces B's mirror computes for A are the world's.
    np.testing.assert_allclose(b_data.ctrl[6:11], target[:5], atol=1e-9)
    _, a_data = a.mujoco_handles()  # type: ignore[misc]
    np.testing.assert_allclose(b_data.qfrc_actuator[6:11], a_data.qfrc_actuator[:5], atol=1e-6)
    # B did not move (it only sags a little under gravity at its zero targets).
    np.testing.assert_allclose(b.read_state().position[:5], 0.0, atol=0.06)
    assert abs(b.read_state().position[0]) < 1e-4


def test_two_commanding_robots_advance_in_lockstep(table: _World) -> None:
    """Both command 20 actions once both are commanding: the world advances 20 periods, not 40."""
    targets = {"a": [0.3, -0.4, 0.4, 0.2, 0.0, 0.5], "b": [-0.3, -0.2, 0.2, 0.1, 0.0, 0.0]}
    commanding = threading.Barrier(2)
    measured = threading.Barrier(2)
    marks: dict[str, float] = {}

    def run(name: str) -> None:
        table.hals[name].send_action(_action(targets[name]))  # in the barrier from here
        commanding.wait()
        if measured.wait() == 0:  # both are commanding: start the clock once
            marks["start"] = table.server.sim_time_s()
        for _ in range(20):
            table.hals[name].send_action(_action(targets[name]))
        if commanding.wait() == 0:
            marks["end"] = table.server.sim_time_s()
        table.hals[name].idle_step(wall_dt_s=TICK_S)  # stop commanding: release the peer

    threads = [threading.Thread(target=run, args=(n,)) for n in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    # In lockstep: 20 paired actions are 20 periods (a robot's first may close a round
    # its peer's warm-up action opened, so 19 at the least).
    assert 19 * TICK_S - 1e-9 <= marks["end"] - marks["start"] <= 21 * TICK_S + 1e-9
    for name, target in targets.items():
        assert abs(table.hals[name].read_state().position[0] - target[0]) < 0.2


def test_reset_to_pose_and_estop_stay_local_to_one_robot(table: _World) -> None:
    """B's pose snap shows in A's view; B's e-stop leaves A's world running."""
    a, b = table.hals["a"], table.hals["b"]
    pose = [0.2, -0.3, 0.3, 0.1, 0.0, 0.0]
    b.reset_to_pose(pose)
    np.testing.assert_allclose(b.read_state().position[:5], pose[:5], atol=1e-9)
    a.send_action(_action([0.0] * 6))
    _, a_data = a.mujoco_handles()  # type: ignore[misc]
    np.testing.assert_allclose(a_data.qpos[6:11], pose[:5], atol=0.05)  # held, one period on
    with pytest.raises(ROSEStopRequested):
        b.estop()
    start = table.server.sim_time_s()
    a.send_action(_action([0.1] * 6))  # no barrier wait on the stopped robot
    assert table.server.sim_time_s() - start == pytest.approx(TICK_S, abs=1e-9)
    np.testing.assert_allclose(a_data.qpos[6:11], pose[:5], atol=0.05)  # B's targets are held
    # A restarted HAL for B joins the same world: it takes the held targets, B's arm stays put.
    again = table.hal("b")
    again.connect()
    table.hals["b"] = again
    for _ in range(30):
        a.send_action(_action([0.1] * 6))
    np.testing.assert_allclose(a_data.qpos[6:11], pose[:5], atol=0.05)
    again.idle_step(wall_dt_s=TICK_S)
    np.testing.assert_allclose(again.read_state().position[:5], pose[:5], atol=0.05)


def test_the_world_refuses_what_it_cannot_serve(table: _World, tmp_path: Path) -> None:
    """A second HAL for an attached robot, an unknown robot, a per-robot reset, no world."""
    with pytest.raises(ROSCapabilityMismatch, match="shared world"):
        table.hals["a"].reset_episode()
    # ...and the HAL node does not advertise /openral/<robot>/reset_episode for it.
    assert table.hals["a"].episode_reset_supported is False
    with pytest.raises(ROSConfigError, match="already attached"):
        table.hal("a").connect()
    with pytest.raises(ROSConfigError, match="robots are"):
        table.hal("c").connect()
    orphan = build_hal(
        RobotDescription.from_yaml(ROBOT_YAML),
        mode="sim",
        transport={"shared_world": str(tmp_path / "none.sock"), "shared_world_robot": "a"},
    )
    with pytest.raises(ROSConfigError, match="no shared world"):
        orphan.connect()


def test_the_world_refuses_a_malformed_request_and_closes_promptly(tmp_path: Path) -> None:
    """Invalid JSON gets a refusal, not a dropped connection; close() returns at once."""
    from multiprocessing.connection import Client

    world = SharedWorldScene.from_yaml(str(WORLD_YAML))
    address = str(tmp_path / "w.sock")
    server = SharedWorldServer(
        world, WORLD_YAML.parent / world.arena_mjcf, {"a": ROBOT_YAML, "b": ROBOT_YAML}, address
    )
    server.start()
    conn = Client(address, family="AF_UNIX")
    conn.send_bytes(b"{not json")
    reply = json.loads(conn.recv_bytes())
    assert reply["ok"] is False
    conn.send_bytes(b'{"op": "hello"}')  # the connection is still served
    assert json.loads(conn.recv_bytes())["ok"] is True
    conn.close()
    started = time.monotonic()
    server.close()  # no client is connecting: the acceptor must still be woken and joined
    assert time.monotonic() - started < 2.0
    assert server._acceptor is not None and not server._acceptor.is_alive()


def test_reset_to_pose_leaves_what_it_does_not_name(table: _World) -> None:
    """Only the named joints are written: the peer's state is the world's, untouched."""
    a, b = table.hals["a"], table.hals["b"]
    b.send_action(_action([0.3, -0.3, 0.3, 0.1, 0.0, 0.2]))  # B moves; A's mirror is behind
    for _ in range(10):
        b.send_action(_action([0.3, -0.3, 0.3, 0.1, 0.0, 0.2]))
    _, b_data = b.mujoco_handles()  # type: ignore[misc]
    b_joints = np.array(b_data.qpos[:6])
    a.reset_to_pose([0.1, -0.1, 0.1, 0.0, 0.0, 0.0])  # from A's stale mirror
    b.idle_step(wall_dt_s=TICK_S)
    np.testing.assert_allclose(b_data.qpos[:6], b_joints, atol=0.02)  # B not rewound
    np.testing.assert_allclose(a.read_state().position[:5], [0.1, -0.1, 0.1, 0.0, 0.0], atol=1e-9)


def test_a_stopped_or_silent_robot_neither_moves_on_nor_freezes_the_world(
    tmp_path: Path,
) -> None:
    """An e-stopped arm holds where it is; a commanding robot gone silent ages out."""
    world = SharedWorldScene.from_yaml(str(WORLD_YAML)).model_copy(
        update={"barrier_timeout_s": 0.3}
    )
    served = _World(world, WORLD_YAML.parent / world.arena_mjcf, tmp_path)
    try:
        a, b = served.hals["a"], served.hals["b"]
        _, a_data = a.mujoco_handles()  # type: ignore[misc]
        b.send_action(_action([1.0, -1.0, 1.0, 0.5, 0.0, 0.5]))  # one period towards a far target
        with pytest.raises(ROSEStopRequested):
            b.estop()
        a.idle_step(wall_dt_s=TICK_S)
        stopped = np.array(a_data.qpos[6:11])
        assert abs(stopped[0]) < 0.3  # nowhere near the 1 rad target yet
        for _ in range(60):
            a.send_action(_action([0.0] * 6))
        np.testing.assert_allclose(a_data.qpos[6:11], stopped, atol=0.08)  # and it stays there

        # A is still "commanding" but sends nothing more. B's restarted HAL only idles.
        again = served.hal("b")
        again.connect()
        served.hals["b"] = again
        time.sleep(0.4)  # past barrier_timeout_s
        again.idle_step(wall_dt_s=TICK_S)  # drops A from the barrier; wall time starts here
        start = served.server.sim_time_s()
        time.sleep(0.1)
        again.idle_step(wall_dt_s=TICK_S)
        advanced = served.server.sim_time_s() - start
        assert 0.05 < advanced < 0.2  # the 0.1 s that passed, not a capped fast-forward
    finally:
        served.close()


def test_each_robot_sees_its_peer_as_occupied_cells(table: _World) -> None:
    """The peer's collision geoms, in this robot's base frame, are occupied cells.

    And the cells catch the collision the issue is about: A panned +0.9 rad from rest
    puts A's own geoms into cells B occupies; A panned -0.4 rad stays clear. A peer
    being driven towards A is occupied along the way before it moves.
    """
    import mujoco as mj

    so101 = RobotDescription.from_yaml(ROBOT_YAML)
    ball = ((0.0, -0.2, 0.15), 0.45, 0.015)  # (centre, radius, cell): covers both arms
    resolution = ball[2]
    for name, peer in (("a", "b"), ("b", "a")):
        model, data = table.hals[name].mujoco_handles()  # type: ignore[misc]
        geoms = peer_geoms(model, robot_root_bodies(peer, so101, view=name))
        assert len(geoms) == 13  # the SO-101's collision meshes
        boxes = peer_boxes(model, data, geoms, int(model.body("base").id))
        origin, (n, _, _), cells = occupancy_grid(boxes, *ball)
        occupied = cells.reshape(n, n, n)
        for g in geoms:  # every vertex of every peer collision mesh is in an occupied cell
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            points = model.mesh_vert[start : start + count] @ data.geom_xmat[g].reshape(3, 3).T
            index = np.floor((points + data.geom_xpos[g] - origin) / resolution).astype(int)
            inside = np.all((index >= 0) & (index < n), axis=1)
            assert inside.any()
            assert occupied[index[inside, 2], index[inside, 1], index[inside, 0]].all()

    model, data = table.hals["a"].mujoco_handles()  # type: ignore[misc]
    base = int(model.body("base").id)
    b_roots = robot_root_bodies("b", so101, view="a")
    b_geoms = peer_geoms(model, b_roots)
    own = peer_geoms(model, robot_root_bodies("a", so101, view="a"))  # A's own links

    def cells_of(geoms: Any, qpos: Any) -> Any:
        probe = mj.MjData(model)
        probe.qpos[:] = qpos
        mj.mj_kinematics(model, probe)
        return occupancy_grid(peer_boxes(model, probe, geoms, base), *ball)[2]

    b_now = cells_of(b_geoms, data.qpos)
    for pan, hits in ((-0.4, False), (0.9, True)):
        a_moved = np.array(data.qpos)
        a_moved[0] = pan
        assert bool((cells_of(own, a_moved) & b_now).any()) is hits

    # B commanded to pan 0.9 rad towards A (its servo target), still at rest:
    # A's links at rest are clear of where B is, not of where B is going.
    a_rest = cells_of(own, data.qpos)
    assert not (a_rest & b_now).any()
    state = {"qpos": list(data.qpos), "qvel": [0.0] * model.nv, "ctrl": list(data.ctrl)}
    b_targets = peer_targets(model, b_roots)
    assert len(b_targets) == 6  # five joints and the jaw, each a position servo
    pan_qadr, pan_actuator = b_targets[0]
    assert pan_qadr == 6  # B's block follows A's six joints in A's view
    assert servo_target(model, pan_actuator, 0.9) == pytest.approx(0.9)  # unit-gear kp servo
    at_rest = swept_peer_boxes(model, mj.MjData(model), state, b_geoms, b_targets, base)
    assert not (occupancy_grid(at_rest, *ball)[2] & a_rest).any()
    commanded = {**state, "ctrl": list(state["ctrl"])}
    commanded["ctrl"][pan_actuator] = 0.9
    swept = swept_peer_boxes(model, mj.MjData(model), commanded, b_geoms, b_targets, base)
    assert (occupancy_grid(swept, *ball)[2] & a_rest).any()
    # B already at its target but still swinging towards A (an overshoot in progress):
    # swept along its measured velocity for the grid's lifetime.
    moving = {**state, "qvel": list(state["qvel"])}
    moving["qvel"][pan_qadr] = 6.0  # rad/s: 1.2 rad over the 0.2 s horizon
    drifting = swept_peer_boxes(model, mj.MjData(model), moving, b_geoms, b_targets, base)
    assert (occupancy_grid(drifting, *ball)[2] & a_rest).any()


def test_an_arena_that_breaks_a_robot_view_is_refused_at_startup(tmp_path: Path) -> None:
    """`base` is an SO-101 body: unprefixed in each robot's own view, so it clashes there."""
    arena = tmp_path / "clash.xml"
    arena.write_text('<mujoco><worldbody><body name="base" pos="0 0 1"/></worldbody></mujoco>')
    world = SharedWorldScene.from_yaml(str(WORLD_YAML))
    with pytest.raises(ROSConfigError, match="does not compile"):
        SharedWorldServer(world, arena, {"a": ROBOT_YAML, "b": ROBOT_YAML}, str(tmp_path / "s"))


def test_free_bodies_are_mapped_into_each_view(tmp_path: Path) -> None:
    """A sliding cube: every view's pose and velocity are the world's, in its base frame."""
    arena = tmp_path / "ramp.xml"
    arena.write_text(_RAMP_ARENA)
    world = SharedWorldScene.from_yaml(str(WORLD_YAML))
    world = world.model_copy(
        update={
            "robots": [
                world.robots[0],
                # Asymmetric on purpose: a mirrored layout hides a sign error.
                world.robots[1].model_copy(
                    update={"base_xyz": (0.2, -0.3, 0.0), "base_yaw_deg": 90.0}
                ),
            ]
        }
    )
    served = _World(world, arena, tmp_path)
    try:
        a, b = served.hals["a"], served.hals["b"]
        for name, hal in served.hals.items():  # at rest: the composed view IS the state
            model, data = hal.mujoco_handles()  # type: ignore[misc]
            np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12, err_msg=name)
        before = {n: _cube_xyz(h) for n, h in served.hals.items()}
        for _ in range(10):
            a.send_action(_action([0.0] * 6))
        b.idle_step(wall_dt_s=TICK_S)
        after = {n: _cube_xyz(h) for n, h in served.hals.items()}
        slid = after["a"] - before["a"]
        assert np.linalg.norm(slid[:2]) > 0.01  # it really slid
        # A at yaw 0: world axes. B at yaw +90: world +X is B's -Y.
        np.testing.assert_allclose(
            after["b"] - before["b"], [slid[1], -slid[0], slid[2]], atol=1e-9
        )
        for hal in served.hals.values():  # the reported velocity is the pose's derivative
            model, data = hal.mujoco_handles()  # type: ignore[misc]
            adr = model.joint("handover_cube_joint").dofadr[0]
            b.idle_step(wall_dt_s=TICK_S)  # B's mirror follows the world on its idle steps
            p0, v0 = _cube_xyz(hal), np.array(data.qvel[adr : adr + 3])
            a.send_action(_action([0.0] * 6))
            b.idle_step(wall_dt_s=TICK_S)
            mean_velocity = (v0 + np.array(data.qvel[adr : adr + 3])) / 2.0  # constant acceleration
            assert np.linalg.norm(mean_velocity[:2]) > 0.1
            np.testing.assert_allclose((_cube_xyz(hal) - p0) / TICK_S, mean_velocity, atol=0.03)
    finally:
        served.close()
