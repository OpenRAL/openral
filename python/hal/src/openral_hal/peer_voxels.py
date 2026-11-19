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
import time
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

__all__ = [
    "PEER_VOXELS_TOPIC",
    "PeerBox",
    "main",
    "occupancy_grid",
    "peer_boxes",
    "peer_geoms",
    "peer_targets",
    "swept_peer_boxes",
]

#: The kernel's world grid in a shared world (its ``/openral/world_voxels`` is remapped here).
PEER_VOXELS_TOPIC = "/openral/peer_voxels"

#: Poses sampled from where a peer is to the targets it is driven to (both ends included).
_SWEEP_SAMPLES = 5


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

    Raises:
        ROSRuntimeError: A box is not finite (it would otherwise vanish from the grid).

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
    servos): its ``ctrl`` is the joint target. Other actuators are left out, so their
    joints are taken where they are, not swept.
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

    ``state`` is a world read in this robot's view (``qpos``, ``ctrl``). A peer
    commanded into the space between the arms is occupied there before it arrives.
    ``data`` is scratch: its kinematics are overwritten.
    """
    import mujoco as mj

    current = np.asarray(state["qpos"], dtype=np.float64)
    goal = current.copy()
    for qadr, u in targets:
        goal[qadr] = state["ctrl"][u]
    boxes: list[PeerBox] = []
    for f in np.linspace(0.0, 1.0, _SWEEP_SAMPLES):
        data.qpos[:] = current + f * (goal - current)
        mj.mj_kinematics(model, data)
        boxes += peer_boxes(model, data, geoms, base_body)
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
    """Publish a shared world's peers on ``PEER_VOXELS_TOPIC`` for one robot's kernel.

    Started by the deploy launch for a robot of ``openral deploy world``. Exits
    non-zero when the world stops answering (the launch then shuts the graph down);
    a tick whose boxes are not finite is not published, so the kernel fails closed.
    """
    import mujoco as mj
    import rclpy
    from openral_msgs.msg import OccupancyVoxels
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    from openral_hal.depth_cloud import resolve_base_frame_body_name
    from openral_hal.shared_world import _ask, open_view, robot_root_bodies

    args = _parse_args(argv)
    conn, world, descriptions, model = open_view(args.world, args.robot)
    data = mj.MjData(model)
    peers = [r.name for r in world.robots if r.name != args.robot]
    roots = set().union(*(robot_root_bodies(p, descriptions[p], view=args.robot) for p in peers))
    geoms = peer_geoms(model, roots)
    targets = peer_targets(model, roots)
    base_name = resolve_base_frame_body_name(model, description=descriptions[args.robot])
    if base_name is None:
        raise ROSConfigError(f"peer_voxels: no body for {args.frame!r} in {args.robot!r}'s view")
    base_body = int(model.body(base_name).id)

    rclpy.init()
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
        while rclpy.ok():
            started = time.monotonic()
            # Every grid comes from a fresh read: an unanswered read ends the loop.
            state = _ask(conn, {"op": "observe", "robot": args.robot})
            boxes = swept_peer_boxes(model, data, state, geoms, targets, base_body)
            try:
                grid = occupancy_grid(boxes, tuple(args.centre), args.radius, args.resolution)
            except ROSRuntimeError as exc:  # not published: the kernel's deadline refuses
                node.get_logger().error(f"{exc}; grid withheld", throttle_duration_sec=1.0)
            else:
                pub.publish(_voxels_msg(OccupancyVoxels(), grid, args, float(state["time"])))
            rclpy.spin_once(node, timeout_sec=max(0.0, period - (time.monotonic() - started)))
    except (EOFError, OSError, ROSConfigError) as exc:
        node.get_logger().error(
            f"peer_voxels: shared world at {args.world!r} went away ({exc}): robot "
            f"{args.robot!r} no longer sees its peers, so its kernel refuses every command"
        )
        raise SystemExit(1) from exc
    finally:
        conn.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
