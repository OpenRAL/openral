"""Sweep a live graph's Nav2 costmaps for cells marked inside the robot itself.

Issue #108: neither costmap should mark `LETHAL` inside the robot's own
silhouette or a carried payload's — since PR #186 made Nav2 base-only
(ADR-0099) the scan filter is the only thing preventing that. Complements
`tests/integration/test_nav2_scan_filter_live.py` (a deterministic synthetic-
ring test that runs on every CI build): this tool adds real RoboCasa scenes,
real `synthesize_laser_scan_2d` returns, real SLAM, a base that drives.

Private (`_` prefix, like `_nav2_mppi_loop_probe.py`): attaches to an
already-launched graph, evidence tooling not a shipped entry point.

Measures: transforms every costmap cell centre into `base_frame` and tests it
against the manifest's `base_footprint_polygon` and every attached object on
`/openral/world_state_fast` (placed via TF at its attach link, projected onto
the costmap plane by sampling z extent) — the same predicates
`payload_scan_filter_node` uses.

Non-vacuity guards (decide `verdict` and exit code, not advisory):
`lethal_cells_anywhere` (0 -> empty map, proves nothing), `base_travel_m` (~0
-> nothing rolled through the window), `payload_samples` /
`payload_partial_samples` / `attached_objects_seen` (payload verdict requires
every declared object placed, else `payload_silhouette_measured: false`).

Cannot attribute causation: in sim, self-returns never enter `/scan` at all
(`synthesize_laser_scan_2d` re-casts through the robot's own kinematic tree),
so a clean silhouette here is the end state being correct, not proof
`payload_scan_filter_node` caused it — that's what the live-lane test is for.

Usage (with a deploy-sim graph already up and Nav2 ACTIVE)::

    python tools/_nav2_costmap_silhouette_probe.py --scene robocasa_deliver_straw \
        --drive 1.5 --seconds 120 >> docs/reference/data/<file>.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
import rclpy
from nav2_msgs.action import (  # reason: generated ROS action package; see mypy.ini [mypy-nav2_msgs.*]
    NavigateToPose,
)
from nav2_msgs.msg import Costmap
from openral_core import RobotDescription
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_msgs.msg import WorldStateStamped
from openral_nav2_bringup._footprint_geometry import base_footprint_polygon
from openral_nav2_bringup.payload_scan_filter_node import (
    points_in_convex_polygon,
    points_in_primitive,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

#: ``nav2_costmap_2d``'s ``LETHAL_OBSTACLE``. Deliberately not 253
#: (``INSCRIBED_INFLATED_OBSTACLE``): the claim is that no cell inside the robot
#: was **marked**, and 253 is what an inflation layer writes near a legitimate
#: obstacle somewhere else.
LETHAL_OBSTACLE = 254

#: Vertical step used to project an attached primitive onto the costmap plane.
#: Half the costmap resolution, so a primitive thinner than one cell still gets
#: several samples through it.
_Z_STEP_M = 0.025


def _pose_matrix(pose: Any) -> Any:
    q, p = pose.orientation, pose.position
    return homogeneous_from_quat_xyz(
        (float(p.x), float(p.y), float(p.z)), (float(q.x), float(q.y), float(q.z), float(q.w))
    )


def _transform_matrix(tf: Any) -> Any:
    t, q = tf.transform.translation, tf.transform.rotation
    return homogeneous_from_quat_xyz(
        (float(t.x), float(t.y), float(t.z)), (float(q.x), float(q.y), float(q.z), float(q.w))
    )


def _primitive_z_span(shape_dimensions: list[float], transform: Any) -> tuple[float, float]:
    """A conservative z interval, in the transform's frame, that contains the shape.

    The bound is the primitive's centre plus/minus the norm of its own
    dimensions, which over-covers every shape type this schema has. Over-covering
    is the safe direction here: it can only make the projected silhouette
    larger, i.e. the assertion stricter.
    """
    reach = float(np.linalg.norm([abs(d) for d in shape_dimensions]))
    cz = float(transform[2, 3])
    return cz - reach, cz + reach


def payload_mask(
    objects: Any, base_from: Any, points_xy: Any
) -> tuple[Any, int, tuple[float, float] | None]:
    """Project every attached object onto the costmap plane.

    Returns ``(union mask, objects fully placed, z span in the base frame)``.

    Counts per OBJECT, not primitive: MuJoCo bodies decompose into many
    primitives via ``extract_body_primitives`` (a RoboCasa baguette has 16),
    so counting primitives against ``len(attached_objects)`` could never reach
    ``placed == declared`` — the payload half of issue #108 was unmeasurable
    on any scene. An object counts as placed only when every one of its
    primitives projected (a partial placement would otherwise shrink the
    silhouette while its cells still count as "elsewhere on the map"); an
    object with no primitives never counts as placed.
    """
    mask = np.zeros(points_xy.shape[0], dtype=bool)
    placed = 0
    z_lo_seen: float | None = None
    z_hi_seen: float | None = None
    for obj in objects:
        base_from_link = base_from(obj.attach_link)
        if base_from_link is None:
            continue
        link_from_obj = _pose_matrix(obj.pose_in_link)
        object_placed = bool(obj.primitives)
        for prim in obj.primitives:
            transform = base_from_link @ link_from_obj @ _pose_matrix(prim.pose_in_object)
            dims = [float(d) for d in prim.shape_dimensions]
            z_lo, z_hi = _primitive_z_span(dims, transform)
            z_lo_seen = z_lo if z_lo_seen is None else min(z_lo_seen, z_lo)
            z_hi_seen = z_hi if z_hi_seen is None else max(z_hi_seen, z_hi)
            steps = max(2, int(math.ceil((z_hi - z_lo) / _Z_STEP_M)) + 1)
            for z in np.linspace(z_lo, z_hi, steps):
                pts = np.concatenate(
                    [points_xy, np.full((points_xy.shape[0], 1), float(z))], axis=1
                )
                try:
                    mask |= points_in_primitive(pts, int(prim.shape_type), dims, transform)
                except ValueError:
                    # A malformed primitive is the producer's problem; it must not
                    # silently shrink the silhouette, so the whole object counts
                    # as unplaced rather than being skipped quietly.
                    object_placed = False
                    break
        if object_placed:
            placed += 1
    span = (
        (round(z_lo_seen, 3), round(z_hi_seen, 3))
        if z_lo_seen is not None and z_hi_seen is not None
        else None
    )
    return mask, placed, span


class SilhouetteProbe(Node):  # type: ignore[misc]  # reason: rclpy ships no py.typed
    """Samples both costmaps and counts marked cells inside the robot silhouette."""

    def __init__(self, robot_yaml: str, *, base_frame: str = "") -> None:
        # The stack runs `use_sim_time:=True`; a wall-clock TF listener asks the
        # sim-time buffer for stamps it has never held.
        super().__init__(
            "nav2_costmap_silhouette_probe",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        description = RobotDescription.from_yaml(robot_yaml)
        self.chassis = base_footprint_polygon(description)
        self.base_frame = base_frame or description.base_frame

        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.state: WorldStateStamped | None = None
        self.attached_seen = 0
        #: Placed payload's z extent in ``base_frame``: decides whether a
        #: marked cell under its projection could even be a payload return
        #: (both costmaps only observe the planar scan). Measured on a
        #: baguette carry: payload at z=1.43 m in odom vs 0.30 m scan plane
        #: (1.13 m clear), while flagged cells were 4-9 cm past the chassis
        #: edge (the counter) — this field lets a reader tell those apart.
        self.payload_z_span: tuple[float, float] | None = None
        self.track: list[tuple[float, float]] = []

        self.create_subscription(
            WorldStateStamped,
            "/openral/world_state_fast",
            self._on_state,
            QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
        self.results: dict[str, dict[str, Any]] = {}
        for which in ("global", "local"):
            self.results[which] = {
                "samples": 0,
                "samples_with_tf": 0,
                "payload_samples": 0,
                "worst_marked_in_silhouette": 0,
                "total_marked_in_silhouette": 0,
                "nearest_marked_m": None,
                "worst_sample_cells": [],
                "chassis_cells_per_sample": 0,
                "lethal_cells_anywhere_max": 0,
                "lethal_cells_anywhere_last": 0,
                "max_cost_seen": 0,
                "nonzero_cells_max": 0,
                "payload_partial_samples": 0,
            }
            self.create_subscription(
                Costmap, f"/{which}_costmap/costmap_raw", self._make_cb(which), 1
            )

    def _on_state(self, msg: WorldStateStamped) -> None:
        self.state = msg
        self.attached_seen = max(self.attached_seen, len(msg.attached_objects))

    def _make_cb(self, which: str) -> Any:
        def _cb(msg: Costmap) -> None:
            self._score(which, msg)

        return _cb

    def _base_from(self, frame: str) -> Any | None:
        """``base_frame <- frame`` at the buffer's own newest common stamp.

        Asking for "latest" (`Time()`) races the writer on a sim-time buffer that
        holds only a sliver of history, so the newest *common* time is used and a
        miss is reported rather than retried into a stale answer.
        """
        try:
            stamp = self.buf.get_latest_common_time(self.base_frame, frame)
            return _transform_matrix(self.buf.lookup_transform(self.base_frame, frame, stamp))
        except Exception:
            return None

    def _payload_masks(self, points_xy: Any) -> tuple[Any, int, int]:
        """Union mask of the attached objects' projections, placed and declared counts.

        Thin wrapper over :func:`payload_mask` so the counting rule it got wrong
        can be tested without a graph; see that function.
        """
        state = self.state
        if state is None:
            return np.zeros(points_xy.shape[0], dtype=bool), 0, 0
        mask, placed, z_span = payload_mask(state.attached_objects, self._base_from, points_xy)
        if z_span is not None:
            self.payload_z_span = z_span
        return mask, placed, len(state.attached_objects)

    def _score(self, which: str, msg: Costmap) -> None:
        out = self.results[which]
        out["samples"] += 1
        base_from_map = self._base_from(msg.header.frame_id)
        if base_from_map is None:
            return
        out["samples_with_tf"] += 1

        meta = msg.metadata
        xs = meta.origin.position.x + (np.arange(meta.size_x) + 0.5) * meta.resolution
        ys = meta.origin.position.y + (np.arange(meta.size_y) + 0.5) * meta.resolution
        gx, gy = np.meshgrid(xs, ys)
        flat = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        in_base = (flat @ base_from_map[:3, :3].T) + base_from_map[:3, 3]
        points_xy = in_base[:, :2]

        silhouette = points_in_convex_polygon(points_xy, self.chassis)
        out["chassis_cells_per_sample"] = int(silhouette.sum())
        payload, placed, declared = self._payload_masks(points_xy)
        if placed:
            silhouette = silhouette | payload
        if declared and placed == declared:
            out["payload_samples"] += 1
        elif declared:
            out["payload_partial_samples"] += 1

        data = np.asarray(msg.data, dtype=np.int32)
        lethal = data == LETHAL_OBSTACLE
        # Non-vacuity: a costmap with nothing marked anywhere makes "nothing
        # marked inside the robot" true for free.
        outside = int((lethal & ~silhouette).sum())
        out["lethal_cells_anywhere_last"] = outside
        out["lethal_cells_anywhere_max"] = max(out["lethal_cells_anywhere_max"], outside)
        # Distinguishes "the map marked nothing" from "the map marked only
        # inflation": a global costmap that never exceeds 0 was never fed.
        out["max_cost_seen"] = max(out["max_cost_seen"], int(data.max()))
        out["nonzero_cells_max"] = max(out["nonzero_cells_max"], int((data > 0).sum()))
        hits = np.flatnonzero(silhouette & lethal)
        n = int(hits.size)
        out["total_marked_in_silhouette"] += n
        if n:
            radii = np.linalg.norm(points_xy[hits], axis=1)
            nearest = float(radii.min())
            prev = out["nearest_marked_m"]
            out["nearest_marked_m"] = nearest if prev is None else min(prev, nearest)
            if n > out["worst_marked_in_silhouette"]:
                out["worst_marked_in_silhouette"] = n
                out["worst_sample_cells"] = [
                    [round(float(points_xy[i][0]), 3), round(float(points_xy[i][1]), 3)]
                    for i in hits[:8]
                ]


def _spin(node: Node, seconds: float, *, track: bool = False) -> None:
    end = time.monotonic() + seconds
    next_sample = 0.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
        if track and time.monotonic() >= next_sample:
            next_sample = time.monotonic() + 0.5
            try:
                stamp = node.buf.get_latest_common_time("map", node.base_frame)
                tf = node.buf.lookup_transform("map", node.base_frame, stamp)
            except Exception:
                continue
            node.track.append(
                (float(tf.transform.translation.x), float(tf.transform.translation.y))
            )


def _drive(node: SilhouetteProbe, metres: float) -> dict[str, Any]:
    """Send one ``NavigateToPose`` goal ``metres`` ahead, so the costmaps roll."""
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    if not client.wait_for_server(timeout_sec=20.0):
        return {"drive": "no navigate_to_pose server"}
    tf = None
    for _ in range(60):
        _spin(node, 0.5)
        try:
            stamp = node.buf.get_latest_common_time("map", node.base_frame)
            tf = node.buf.lookup_transform("map", node.base_frame, stamp)
            break
        except Exception:
            continue
    if tf is None:
        return {"drive": "no map->base TF"}
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x = tf.transform.translation.x + metres
    goal.pose.pose.position.y = tf.transform.translation.y
    goal.pose.pose.orientation.w = 1.0
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=20.0)
    if fut.result() is None or not fut.result().accepted:
        return {"drive": "goal rejected"}
    return {"drive": "accepted", "goal_dx_m": metres}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-yaml", default="robots/panda_mobile/robot.yaml")
    ap.add_argument("--scene", default="")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--settle", type=float, default=15.0)
    ap.add_argument("--drive", type=float, default=0.0, help="metres ahead to NavigateToPose")
    args = ap.parse_args()

    rclpy.init()
    node = SilhouetteProbe(args.robot_yaml)
    _spin(node, args.settle)  # /clock + TF + the first costmap updates
    driven = _drive(node, args.drive) if args.drive else {"drive": "none"}
    _spin(node, args.seconds, track=True)
    travel = sum(math.dist(node.track[i], node.track[i + 1]) for i in range(len(node.track) - 1))

    verdict: dict[str, Any] = {
        "scene": args.scene,
        "base_frame": node.base_frame,
        "chassis_polygon": [[round(x, 3), round(y, 3)] for x, y in node.chassis],
        "seconds": args.seconds,
        "attached_objects_seen": node.attached_seen,
        "payload_z_span_in_base_m": (
            list(node.payload_z_span) if node.payload_z_span is not None else None
        ),
        "base_travel_m": round(travel, 3),
        "base_net_displacement_m": (
            round(math.dist(node.track[0], node.track[-1]), 3) if len(node.track) > 1 else 0.0
        ),
        **driven,
    }
    #: A run that never moved the base swept no new geometry, so a clean sweep
    #: says little about the filter. Small but non-zero, because a pure rotation
    #: still rolls the window.
    min_travel_m = 0.10
    verdicts: list[str] = []
    for which, out in node.results.items():
        out["payload_silhouette_measured"] = (
            out["payload_samples"] > 0 and out["payload_partial_samples"] == 0
        )
        out["non_vacuous"] = out["lethal_cells_anywhere_max"] > 0
        # "Clean" is only a result when the map had something to mark and the
        # robot moved. Without both, zero cells inside the silhouette is a fact
        # about an empty measurement, not about the filter.
        out["clean"] = out["samples_with_tf"] > 0 and out["total_marked_in_silhouette"] == 0
        out["verdict"] = (
            "MARKED CELLS INSIDE SILHOUETTE"
            if not out["clean"]
            else "clean"
            if out["non_vacuous"]
            else "VACUOUS - the costmap marked nothing anywhere"
        )
        verdicts.append(str(out["verdict"]))
        verdict[f"{which}_costmap"] = out
    if travel < min_travel_m:
        verdicts.append(f"VACUOUS - the base moved only {round(travel, 3)} m")
    worst = next((v for v in verdicts if v != "clean"), "clean")
    verdict["verdict"] = worst
    verdict["verdict_per_costmap"] = {which: out["verdict"] for which, out in node.results.items()}
    print(json.dumps(verdict))
    rclpy.shutdown()
    return 0 if worst == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
