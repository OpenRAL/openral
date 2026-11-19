"""Peer robots as obstacles for the safety kernel's world check (issue #666).

Robots in one workcell must know about each other. The safety kernel already
refuses a command that drives a robot link into an occupied cell of
``/openral/world_voxels`` (``openral_msgs/OccupancyVoxels``, in the robot's base
frame, fail-closed when stale). This module puts the **other** robots' links into
such a grid, so each robot's unmodified kernel sees its peers.

The grid is built from oriented boxes in the robot's base frame (``PeerBox``),
whatever their source:

- **shared MuJoCo world (implemented):** ``mujoco_peer_boxes`` reads the peers'
  collision geoms from this robot's view of the world (``SharedWorldServer``,
  ``openral deploy world``), and ``main`` publishes them.
- **real robots in one workcell (follow-up):** the same boxes from each peer's
  manifest ``collision_geometry``, posed by its ``/joint_states`` and its known
  base pose in this robot's base frame.

Every grid is built from a fresh read of the peers' state and stamped with that
state's time (``source_stamp``). When the source stops answering, publication
stops, and the kernel's receipt deadline and data-age budget refuse every command
(fail-closed): a stale peer is never taken for a clear workspace.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from openral_core import RobotDescription, SharedWorldScene
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

__all__ = ["PeerBox", "main", "mujoco_peer_boxes", "occupancy_grid"]


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

    Example:
        >>> box = PeerBox(np.array([0.1, 0.0, 0.0]), np.eye(3), np.array([0.01, 0.01, 0.01]))
        >>> origin, shape, cells = occupancy_grid([box], (0.0, 0.0, 0.0), 0.2, 0.05)
        >>> shape, int(cells.sum())
        ((9, 9, 9), 27)
    """
    n = math.floor(2.0 * radius_m / resolution_m + 0.5 + 1e-9) + 1
    origin = np.asarray(centre, dtype=np.float64) - n * resolution_m / 2.0
    grid = np.zeros((n, n, n), dtype=np.uint8)  # indexed [z, y, x]: x fastest when flattened
    grow = resolution_m * math.sqrt(3.0) / 2.0
    for box in boxes:
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


def mujoco_peer_boxes(model: Any, data: Any, peers: list[str]) -> list[PeerBox]:
    """The peers' collision geoms in a robot's view of a shared world, as boxes.

    ``model`` / ``data`` are that robot's view (``compose_shared_world(view=...)``,
    kinematics computed): its world frame is the robot's base frame and each peer's
    bodies are named ``<peer>_…``. A geom is a collision geom when it has a
    ``contype`` or ``conaffinity``; its box is its local bounding box (``geom_aabb``).

    Example:
        >>> boxes = mujoco_peer_boxes(model, data, ["b"])  # doctest: +SKIP
    """
    prefixes = tuple(f"{peer}_" for peer in peers)
    boxes: list[PeerBox] = []
    for g in range(int(model.ngeom)):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = model.body(int(model.geom_bodyid[g])).name
        if not body.startswith(prefixes):
            continue
        rotation = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        centre = np.asarray(data.geom_xpos[g], dtype=np.float64) + rotation @ aabb[:3]
        boxes.append(PeerBox(centre, rotation, aabb[3:]))
    return boxes


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
    """Publish a shared world's peers on ``/openral/world_voxels`` for one robot's kernel.

    Started by the deploy launch for a robot of ``openral deploy world``. Exits
    (non-zero) when the world stops answering, so the kernel fails closed.
    """
    import mujoco as mj
    import rclpy
    from openral_msgs.msg import OccupancyVoxels
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    from openral_hal.shared_world import _compile, compose_shared_world

    args = _parse_args(argv)
    try:
        conn = Client(args.world, family="AF_UNIX")
    except OSError as exc:
        raise ROSConfigError(f"peer_voxels: no shared world at {args.world!r}: {exc}") from exc

    def request(msg: dict[str, object]) -> dict[str, Any]:
        conn.send_bytes(json.dumps(msg).encode())
        reply: dict[str, Any] = json.loads(conn.recv_bytes())
        if not reply["ok"]:
            raise ROSRuntimeError(f"peer_voxels: {reply['error']}")
        return reply

    hello = request({"op": "hello"})
    world = SharedWorldScene.model_validate(hello["world"])
    descriptions = {n: RobotDescription.from_yaml(p) for n, p in hello["robot_yamls"].items()}
    spec, _ = compose_shared_world(world, Path(hello["arena_mjcf"]), descriptions, args.robot)
    model = _compile(spec, world.id)
    data = mj.MjData(model)
    peers = [r.name for r in world.robots if r.name != args.robot]

    rclpy.init()
    node = rclpy.create_node(f"openral_peer_voxels_{args.robot}")
    # The kernel's subscription: RELIABLE, VOLATILE, keep last 1.
    qos = QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(OccupancyVoxels, "/openral/world_voxels", qos)
    node.get_logger().info(
        f"peer_voxels: robot {args.robot!r} sees peers {peers} on /openral/world_voxels "
        f"({args.frame}, ball r={args.radius} m about {args.centre}, {args.resolution} m cells)"
    )
    period = 1.0 / args.rate_hz
    try:
        while rclpy.ok():
            started = time.monotonic()
            # Every grid comes from a fresh read: an unanswered read ends the loop.
            state = request({"op": "observe", "robot": args.robot})
            data.qpos[:] = state["qpos"]
            mj.mj_kinematics(model, data)
            grid = occupancy_grid(
                mujoco_peer_boxes(model, data, peers),
                tuple(args.centre),
                args.radius,
                args.resolution,
            )
            pub.publish(_voxels_msg(OccupancyVoxels(), grid, args, float(state["time"])))
            rclpy.spin_once(node, timeout_sec=max(0.0, period - (time.monotonic() - started)))
    except (EOFError, OSError) as exc:
        raise ROSRuntimeError(f"peer_voxels: shared world at {args.world!r} went away") from exc
    finally:
        conn.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
