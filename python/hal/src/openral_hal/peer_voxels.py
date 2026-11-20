"""Peer robots as obstacles for the safety kernel's world check (issue #666).

Robots in one workcell must know about each other. The safety kernel already
refuses a command that drives a robot link into an occupied cell of its world
grid (``openral_msgs/OccupancyVoxels`` in the robot's base frame, refused when
stale). This module puts the **other** robots' links into such a grid, on
``PEER_VOXELS_TOPIC``, which the deploy launch connects to the kernel (and to
nothing else: the other ``/openral/world_voxels`` readers expect a perceived map).

The grid is built from oriented boxes in the robot's base frame (``PeerBox``),
whatever their source:

- **shared MuJoCo world (implemented):** ``peer_boxes`` reads the peers' collision
  geoms from this robot's view of the world (``SharedWorldServer``, ``openral
  deploy world``), both where each peer is and along the way to the joint targets
  it is being driven to, and ``main`` publishes them.
- **real robots in one workcell (follow-up):** the same boxes from each peer's
  manifest ``collision_geometry``, posed by its ``/joint_states`` and its known
  base pose in this robot's base frame.

Every grid is built from a fresh read of the peers and stamped with that read's
time (``source_stamp``). When the source stops answering, publication stops (and
the deploy graph shuts down); the kernel's receipt deadline and data-age budget
refuse every command meanwhile. A stale peer is never taken for a clear workspace.
"""

from __future__ import annotations

import argparse
import array
import math
import signal
import threading
import time
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

__all__ = [
    "PEER_VOXELS_TOPIC",
    "PeerBox",
    "cells_per_axis",
    "main",
    "occupancy_grid",
    "own_base_body",
    "peer_boxes",
    "peer_geoms",
    "peer_targets",
    "servo_target",
    "swept_peer_boxes",
]

#: The kernel's world grid in a shared world (its ``/openral/world_voxels`` is remapped here).
PEER_VOXELS_TOPIC = "/openral/peer_voxels"

#: Largest joint step between two swept poses, rad (or m for a slide joint).
_SWEEP_STEP = 0.05
#: How far past its target a peer is swept: a P servo overshoots (the SO-101 elbow
#: commanded +0.8 rad peaks 17% past it in this sim), so 25% more of the move.
_SWEEP_OVERSHOOT = 1.25
#: How long ahead a peer is swept at its measured joint velocity, s: the kernel's
#: acceptance of a peer grid (the launch's ``_PEER_GRID_BUDGET_S``). Covers a peer
#: still moving where its target says it has arrived (an overshoot in progress).
_SWEEP_HORIZON_S = 0.2


class PeerBox(NamedTuple):
    """An oriented box a peer robot occupies, in this robot's base frame.

    Attributes:
        centre: Box centre, metres.
        rotation: Box axes as the columns of a 3x3 rotation matrix.
        half_extents: Half sizes along those axes, metres.
    """

    centre: NDArray[np.float64]
    rotation: NDArray[np.float64]
    half_extents: NDArray[np.float64]


def cells_per_axis(radius_m: float, resolution_m: float) -> int:
    """Cells per axis of the grid around a coverage ball: the kernel's budget is its cube.

    The deploy launch's ``_world_voxel_max_cells`` is this, cubed.

    Example:
        >>> cells_per_axis(0.2, 0.05)
        9
    """
    return math.floor(2.0 * radius_m / resolution_m + 0.5 + 1e-9) + 1


def occupancy_grid(
    boxes: list[PeerBox], centre: tuple[float, float, float], radius_m: float, resolution_m: float
) -> tuple[NDArray[np.float64], tuple[int, int, int], NDArray[np.uint8]]:
    """Rasterize ``boxes`` into the axis-aligned grid covering a ball, conservatively.

    The grid is the box around the ball (``centre``, ``radius_m``) with
    ``floor(2 r / res + 1/2) + 1`` cells per axis, the size the launch reserves in
    the kernel for that ball (``world_voxel_max_cells``). A cell is occupied when
    its centre lies in a box grown by half a cell diagonal on every axis: every cell
    a box touches is marked (and a few more), never fewer.

    Returns:
        ``(origin, (sx, sy, sz), occupancy)``: the minimum corner of cell
        ``(0, 0, 0)``, the shape, and the cells row-major with x fastest
        (``OccupancyVoxels``' layout), ``1`` for occupied.

    Raises:
        ROSRuntimeError: A box is not finite (it would otherwise vanish from the grid).

    Example:
        >>> box = PeerBox(np.array([0.1, 0.0, 0.0]), np.eye(3), np.array([0.01, 0.01, 0.01]))
        >>> origin, shape, cells = occupancy_grid([box], (0.0, 0.0, 0.0), 0.2, 0.05)
        >>> shape, int(cells.sum())
        ((9, 9, 9), 27)
    """
    n = cells_per_axis(radius_m, resolution_m)
    origin = np.asarray(centre, dtype=np.float64) - n * resolution_m / 2.0
    grid = np.zeros((n, n, n), dtype=np.uint8)  # indexed [z, y, x]: x fastest when flattened
    grow = resolution_m * math.sqrt(3.0) / 2.0
    for box in boxes:
        if not all(np.isfinite(v).all() for v in box):
            raise ROSRuntimeError(f"peer_voxels: a peer box is not finite: {box}")
        half = np.abs(box.half_extents) + grow
        corner_reach = np.abs(box.rotation) @ half  # the grown box's axis-aligned half size
        lo = np.floor((box.centre - corner_reach - origin) / resolution_m).astype(int)
        hi = np.ceil((box.centre + corner_reach - origin) / resolution_m).astype(int)
        lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
        if np.any(hi <= lo):
            continue  # wholly outside the grid: beyond this robot's reach
        axes = [origin[i] + (np.arange(lo[i], hi[i]) + 0.5) * resolution_m for i in range(3)]
        zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        local = (np.stack([xx, yy, zz], axis=-1) - box.centre) @ box.rotation
        inside = np.all(np.abs(local) <= half, axis=-1)
        grid[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] |= inside.astype(np.uint8)
    return origin, (n, n, n), grid.reshape(-1)


def peer_geoms(model: Any, peer_roots: set[str]) -> NDArray[np.int64]:
    """The collision geoms (a ``contype`` or ``conaffinity``) of the bodies under ``peer_roots``.

    ``peer_roots`` are the peers' top-level body names in this view
    (``shared_world.robot_root_bodies``): a body belongs to a peer by its kinematic
    root, never by its name, so this robot's own ``upper_arm`` is not peer ``upper``'s.

    Example:
        >>> geoms = peer_geoms(model, {"b_base"})  # doctest: +SKIP
    """
    roots = [int(model.body(name).id) for name in peer_roots]
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    in_peer = np.isin(model.body_rootid[model.geom_bodyid], roots)
    return np.flatnonzero(collides & in_peer)


def peer_targets(model: Any, peer_roots: set[str]) -> list[tuple[int, int]]:
    """``(qpos address, actuator)`` of every peer joint driven by a position servo.

    A position servo is a joint-transmission actuator with an affine bias whose
    position term is negative (MuJoCo's ``<position>`` and the SO-ARM ``general``
    servos); ``servo_target`` gives the joint position it drives to. Other actuators
    are left out, so their joints are taken where they are, not swept.
    """
    import mujoco as mj

    roots = {int(model.body(name).id) for name in peer_roots}
    out: list[tuple[int, int]] = []
    for u in range(int(model.nu)):
        if model.actuator_trntype[u] != mj.mjtTrn.mjTRN_JOINT:
            continue
        joint = int(model.actuator_trnid[u, 0])
        if int(model.body_rootid[model.jnt_bodyid[joint]]) not in roots:
            continue
        servo = model.actuator_biastype[u] == mj.mjtBias.mjBIAS_AFFINE
        if servo and model.actuator_biasprm[u, 1] < 0:
            out.append((int(model.jnt_qposadr[joint]), u))
    return out


def servo_target(model: Any, actuator: int, ctrl: float) -> float:
    """The joint position a position servo settles at for ``ctrl``.

    Equilibrium of ``gain * ctrl + b0 + b1 * length = 0`` (``length = gear * q``),
    with ``ctrl`` clamped to ``ctrlrange`` when the actuator is ``ctrllimited``.

    Example:
        >>> servo_target(model, 6, 0.9)  # doctest: +SKIP
        0.9
    """
    if model.actuator_ctrllimited[actuator]:
        low, high = model.actuator_ctrlrange[actuator]
        ctrl = min(max(ctrl, float(low)), float(high))
    gain = float(model.actuator_gainprm[actuator, 0])
    b0, b1 = float(model.actuator_biasprm[actuator, 0]), float(model.actuator_biasprm[actuator, 1])
    return -(gain * ctrl + b0) / b1 / float(model.actuator_gear[actuator, 0])


def peer_boxes(model: Any, data: Any, geoms: NDArray[np.int64], base_body: int) -> list[PeerBox]:
    """``geoms``' local bounding boxes (``geom_aabb``) in the frame of body ``base_body``.

    ``model`` / ``data`` are a robot's view of a shared world with kinematics
    computed; ``base_body`` is the body its ``base_frame`` names.
    """
    base_rot = np.asarray(data.xmat[base_body], dtype=np.float64).reshape(3, 3)
    base_pos = np.asarray(data.xpos[base_body], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geoms], dtype=np.float64).reshape(-1, 3, 3)
    aabb = np.asarray(model.geom_aabb[geoms], dtype=np.float64)
    centres = np.asarray(data.geom_xpos[geoms]) + np.einsum("gij,gj->gi", rot, aabb[:, :3])
    local_centres = (centres - base_pos) @ base_rot
    local_rot = np.einsum("ji,gjk->gik", base_rot, rot)
    return [PeerBox(c, r, h) for c, r, h in zip(local_centres, local_rot, aabb[:, 3:], strict=True)]


def swept_peer_boxes(
    model: Any,
    data: Any,
    state: dict[str, Any],
    geoms: NDArray[np.int64],
    targets: list[tuple[int, int]],
    base_body: int,
) -> list[PeerBox]:
    """The peers' boxes where they are AND along the way to their servo targets.

    ``state`` is a world read in this robot's view (``qpos``, ``ctrl``). Poses are
    sampled at most ``_SWEEP_STEP`` apart per joint, up to ``_SWEEP_OVERSHOOT`` of
    the move, so a peer commanded into the space between the arms is occupied there
    before it arrives; and along its measured joint velocity for ``_SWEEP_HORIZON_S``,
    so a peer still moving past its target is too. ``data`` is scratch.
    """
    import mujoco as mj

    current = np.asarray(state["qpos"], dtype=np.float64)
    if not (np.isfinite(current).all() and np.isfinite(state["qvel"]).all()):
        raise ROSRuntimeError("peer_voxels: the world read is not finite")
    goal = current.copy()
    for qadr, u in targets:
        goal[qadr] = servo_target(model, u, float(state["ctrl"][u]))
    # Hinge / slide joints only (refused otherwise), so a qpos slot is a qvel slot.
    drift = np.zeros_like(current)
    for qadr, _ in targets:
        drift[qadr] = _SWEEP_HORIZON_S * float(state["qvel"][qadr])
    boxes: list[PeerBox] = []
    if not (np.isfinite(goal).all() and np.isfinite(drift).all()):
        raise ROSRuntimeError("peer_voxels: a peer's target or velocity is not finite")
    for reach in (_SWEEP_OVERSHOOT * (goal - current), drift):
        samples = int(np.ceil(np.max(np.abs(reach), initial=0.0) / _SWEEP_STEP)) + 1
        for f in np.linspace(0.0, 1.0, samples):
            data.qpos[:] = current + f * reach
            mj.mj_kinematics(model, data)
            boxes += peer_boxes(model, data, geoms, base_body)
    return boxes


def own_base_body(model: Any, base_frame: str, own_roots: set[str]) -> int:
    """The body this robot's ``base_frame`` names in its own view, never a peer's or the arena's.

    The body named ``base_frame`` when it is one of this robot's own (under
    ``own_roots``), else this robot's single root body.

    Raises:
        ROSConfigError: Neither exists (the grid's frame would be a guess).
    """
    roots = {int(model.body(name).id) for name in own_roots}
    for b in range(int(model.nbody)):
        if model.body(b).name == base_frame and int(model.body_rootid[b]) in roots:
            return b
    if len(roots) == 1:
        return next(iter(roots))
    raise ROSConfigError(f"peer_voxels: no own body for base frame {base_frame!r}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", required=True, help="the SharedWorldServer's Unix socket")
    parser.add_argument("--robot", required=True, help="this robot's SharedWorldRobot.name")
    parser.add_argument("--frame", required=True, help="this robot's base frame (frame_id)")
    parser.add_argument("--centre", type=float, nargs=3, required=True, help="coverage ball, m")
    parser.add_argument("--radius", type=float, required=True, help="coverage ball radius, m")
    parser.add_argument("--resolution", type=float, required=True, help="cell size, m")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="wall-clock publish rate")
    return parser.parse_args(argv)


def _voxels_msg(
    msg: Any,
    grid: tuple[NDArray[np.float64], tuple[int, int, int], NDArray[np.uint8]],
    args: argparse.Namespace,
    sim_time_s: float,
) -> Any:
    """Fill an ``OccupancyVoxels`` with ``grid``, stamped with the world time it was read at."""
    origin, (sx, sy, sz), cells = grid
    sim_ns = round(sim_time_s * 1e9)
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(sim_ns, 1_000_000_000)
    msg.header.frame_id = args.frame
    msg.source_stamp = msg.header.stamp  # the data age the kernel budgets
    msg.origin.x, msg.origin.y, msg.origin.z = (float(v) for v in origin)
    msg.orientation.w = 1.0
    msg.resolution = float(args.resolution)
    msg.size_x, msg.size_y, msg.size_z = sx, sy, sz
    msg.occupancy = array.array("B", cells.tobytes())
    return msg


def main(argv: list[str] | None = None) -> None:
    """Publish a shared world's peers on ``PEER_VOXELS_TOPIC`` for one robot's kernel.

    Started by the deploy launch for a robot of ``openral deploy world``. Exits
    non-zero when the world stops answering (the launch then shuts the graph down);
    a tick whose boxes are not finite is not published, so the kernel fails closed.
    """
    import mujoco as mj
    import rclpy
    from openral_msgs.msg import OccupancyVoxels
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions

    from openral_hal.shared_world import _ask, open_view, robot_root_bodies

    args = _parse_args(argv)
    conn, world, descriptions, model = open_view(args.world, args.robot)
    data = mj.MjData(model)
    peers = [r.name for r in world.robots if r.name != args.robot]
    roots = set().union(*(robot_root_bodies(p, descriptions[p], view=args.robot) for p in peers))
    geoms = peer_geoms(model, roots)
    targets = peer_targets(model, roots)
    base_body = own_base_body(
        model, args.frame, robot_root_bodies(args.robot, descriptions[args.robot], view=args.robot)
    )
    base_name = model.body(base_body).name

    # SIGINT / SIGTERM (the launch's teardown) end the loop and the process exits 0;
    # only a lost world (exit 1) or a crash is a failure the launch shuts the graph on.
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"openral_peer_voxels_{args.robot}")
    # The kernel's world-grid subscription: RELIABLE, VOLATILE, keep last 1.
    qos = QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(OccupancyVoxels, PEER_VOXELS_TOPIC, qos)
    node.get_logger().info(
        f"peer_voxels: robot {args.robot!r} sees peers {peers} ({len(geoms)} collision geoms, "
        f"{len(targets)} servo targets) on {PEER_VOXELS_TOPIC} in {args.frame} (body "
        f"{base_name!r}), ball r={args.radius} m about {args.centre}, {args.resolution} m cells"
    )
    period = 1.0 / args.rate_hz
    try:
        while rclpy.ok() and not stop.is_set():
            started = time.monotonic()
            # Every grid comes from a fresh read: an unanswered read ends the loop.
            state = _ask(conn, {"op": "observe", "robot": args.robot})
            try:
                boxes = swept_peer_boxes(model, data, state, geoms, targets, base_body)
                grid = occupancy_grid(boxes, tuple(args.centre), args.radius, args.resolution)
            except ROSRuntimeError as exc:  # not published: the kernel's deadline refuses
                node.get_logger().error(f"{exc}; grid withheld", throttle_duration_sec=1.0)
            else:
                pub.publish(_voxels_msg(OccupancyVoxels(), grid, args, float(state["time"])))
            rclpy.spin_once(node, timeout_sec=max(0.0, period - (time.monotonic() - started)))
    except ExternalShutdownException:
        return  # the graph is shutting down: a clean exit, not a lost peer source
    except (EOFError, OSError, ROSConfigError) as exc:
        node.get_logger().error(
            f"peer_voxels: shared world at {args.world!r} went away ({exc}): robot "
            f"{args.robot!r} no longer sees its peers, so its kernel refuses every command"
        )
        raise SystemExit(1) from exc
    finally:
        # The teardown signals arrive more than once (the launch, and `deploy sim`
        # forwarding Ctrl-C to its group). Python restores the default handlers when it
        # finalizes, so a repeat during the exit would kill the process by signal and
        # read as a lost peer source; ignored ones stay ignored.
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        conn.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
