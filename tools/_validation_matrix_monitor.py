"""Record attachment / voxel / E-stop evidence during a live ``deploy sim`` run.

Private helper of ``tools/validation_matrix.py``, which spawns it alongside each
scene's ROS graph. Recovered verbatim (bar this docstring and the node name)
from ``attach_monitor4.py``, the fourth generation of the monitor that lived in
``spark:~/openral-runs/<round>/scripts/``; it is in the repo so a round no longer
depends on one machine's home directory.

What it records, beyond the plain attachment stream:

* the kernel's own collision verdict, parsed out of ``FailureTrigger.evidence_json``
  on ``/openral/failure/safety`` (``min_distance_m`` + the ``voxel_<idx>`` evidence
  cell);
* a full ``.npz`` snapshot at every E-stop *and* periodically while a payload is
  carried: the last ``OccupancyVoxels`` grid, the last ``AttachmentState`` payload
  primitives, and the TF that places the attach link in the grid frame —
  everything needed to recompute payload residue offline. Carry snapshots exist
  because an E-stop-only monitor records nothing at all on a run that passes;
* the ``PlaceDeclaration`` and its producer-measured region, keyed on identity +
  liveness + region *shape* so a region whose pose is re-measured every tick does
  not emit a record per tick;
* the occupied-cell **set** hash, not merely the count — a frozen map keeps an
  identical set, which a count alone cannot distinguish from a live one.

Requires a sourced ROS 2 environment and the built overlay. Writes one JSON
object per line to argv[1]; snapshots to argv[2]/``<tag>_<n>.npz``. Runs until
SIGINT/SIGTERM.

Run::

    python tools/_validation_matrix_monitor.py <monitor.jsonl> <snapshot_dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from typing import Any

import numpy as np
import rclpy
import tf2_ros
from openral_msgs.msg import (
    AttachmentState,
    FailureTrigger,
    OccupancyVoxels,
    WorldStateStamped,
)
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, UInt64


def _transient(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _safety_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _witness(obj: Any) -> dict[str, Any] | None:
    """The SupportContactWitness on an attached object, if it carries one.

    Returns ``None`` when the producer attested nothing — which is the
    pre-witness behaviour and means the kernel exempts no payload-vs-world
    contact at all.
    """
    if not getattr(obj, "support_contact_valid", False):
        return None
    w = obj.support_contact
    return {
        "support_id": w.support_id,
        "contact_point_in_object": [
            round(float(w.contact_point_in_object.x), 6),
            round(float(w.contact_point_in_object.y), 6),
            round(float(w.contact_point_in_object.z), 6),
        ],
        "contact_normal_in_object": [
            round(float(w.contact_normal_in_object.x), 6),
            round(float(w.contact_normal_in_object.y), 6),
            round(float(w.contact_normal_in_object.z), 6),
        ],
        "patch_radius_m": round(float(w.patch_radius_m), 6),
        "max_penetration_m": round(float(w.max_penetration_m), 6),
        "confidence": round(float(w.confidence), 4),
        "evidence_kind": w.evidence_kind,
        "evidence_ref": w.evidence_ref,
        "stamp_ns": int(w.stamp_ns),
    }


def _declaration(msg: Any) -> dict[str, Any] | None:
    """The PlaceDeclaration riding an AttachmentState / WorldStateStamped.

    ``None`` when no declaration is in force, which is the pre-ADR-0097
    behaviour: no place witness and no approach allowance. The region is the
    2026-08-14 amendment's producer-measured box; ``region`` is ``None``
    whenever ``region_valid`` is false, which is the "no allowance, unchanged
    margin" case.
    """
    if not getattr(msg, "place_declaration_valid", False):
        return None
    d = msg.place_declaration
    out: dict[str, Any] = {
        "target_id": d.target_id,
        "object_id": d.object_id,
        "rskill_id": d.rskill_id,
        "trace_id": d.trace_id,
        "timeout_s": round(float(d.timeout_s), 3),
        "stamp_ns": int(d.stamp_ns),
        "active": bool(d.active),
        "region_valid": bool(getattr(d, "region_valid", False)),
        "region": None,
    }
    if out["region_valid"]:
        r = d.region
        out["region"] = {
            "frame_id": r.frame_id,
            "pose_xyz": [
                round(float(r.pose.position.x), 6),
                round(float(r.pose.position.y), 6),
                round(float(r.pose.position.z), 6),
            ],
            "pose_quat_xyzw": [
                round(float(r.pose.orientation.x), 6),
                round(float(r.pose.orientation.y), 6),
                round(float(r.pose.orientation.z), 6),
                round(float(r.pose.orientation.w), 6),
            ],
            "half_extents": [
                round(float(r.half_extents.x), 6),
                round(float(r.half_extents.y), 6),
                round(float(r.half_extents.z), 6),
            ],
            "volume_m3": round(
                8.0 * float(r.half_extents.x) * float(r.half_extents.y) * float(r.half_extents.z),
                6,
            ),
            "evidence_ref": r.evidence_ref,
            "stamp_ns": int(r.stamp_ns),
        }
    return out


def _decl_key(decl: dict[str, Any] | None) -> str:
    """Change key: identity + liveness + region SHAPE, never the refreshed pose.

    The region's pose is re-measured in the (moving) base frame on every
    publication, so keying on it would emit a record per tick. The half-extents
    are measured once from the model subtree, so a change in them is real.
    """
    if decl is None:
        return "none"
    region = decl["region"]
    shape = "none" if region is None else ",".join(f"{v:.6f}" for v in region["half_extents"])
    return (
        f"{decl['target_id']}|{decl['object_id']}|{decl['active']}|{decl['region_valid']}|{shape}"
    )


class Monitor:
    def __init__(self, node: Any, sink: Any, snap_dir: str) -> None:
        self._node = node
        self._sink = sink
        self._snap_dir = snap_dir
        self._t0 = time.time()
        self._last_ws_attached: str | None = None
        self._last_voxel_log = 0.0
        self._voxel_counts: list[int] = []
        self._last_voxels: OccupancyVoxels | None = None
        self._last_attachment: AttachmentState | None = None
        self._n_snap = 0
        # Round-6 addition: periodic snapshots WHILE a payload is carried,
        # so the attach-sweep question ('did the stale shell clear, and did
        # the widened reach close once the payload moved a padding-distance')
        # is answerable on a run that never E-stops. attach_monitor2 only
        # snapshotted at an E-stop, which a passing run never reaches.
        self._last_carry_snap = 0.0
        # Round-7: place declaration + producer-measured region tracking.
        self._last_decl_key: str | None = None
        self._last_ws_decl_key: str | None = None
        self._last_decl: dict[str, Any] | None = None
        self._carry_period_s = float(os.environ.get("CARRY_SNAP_PERIOD_S", "1.0"))

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        node.create_subscription(
            AttachmentState, "/openral/attachment_state", self._on_attachment, _transient()
        )
        node.create_subscription(
            UInt64, "/openral/attachment_state_applied", self._on_ack, _transient()
        )
        node.create_subscription(
            WorldStateStamped, "/openral/world_state_fast", self._on_world_state, 10
        )
        node.create_subscription(OccupancyVoxels, "/openral/world_voxels", self._on_voxels, 10)
        node.create_subscription(Empty, "/openral/estop", self._on_estop, 10)
        node.create_subscription(
            FailureTrigger, "/openral/failure/safety", self._on_failure, _safety_qos()
        )

    def _emit(self, event: str, **fields: Any) -> None:
        record = {"t": round(time.time() - self._t0, 3), "event": event, **fields}
        self._sink.write(json.dumps(record, sort_keys=True) + "\n")
        self._sink.flush()

    # ---------------------------------------------------------------- inputs

    def _on_attachment(self, msg: AttachmentState) -> None:
        self._last_attachment = msg
        decl = _declaration(msg)
        self._last_decl = decl
        key = _decl_key(decl)
        if key != self._last_decl_key:
            self._last_decl_key = key
            self._emit(
                "place_declaration",
                source="attachment_state",
                revision=int(msg.revision),
                declaration=decl,
            )
        self._emit(
            "attachment_state",
            revision=int(msg.revision),
            n_objects=len(msg.objects),
            objects=[
                {
                    "id": obj.object_id,
                    "link": obj.attach_link,
                    "touch_links": list(obj.touch_links),
                    "n_primitives": len(obj.primitives),
                    "shapes": [int(p.shape_type) for p in obj.primitives],
                    "evidence_kind": obj.evidence_kind,
                    "evidence_ref": obj.evidence_ref,
                    "confidence": round(float(obj.confidence), 4),
                    "mass_kg": round(float(obj.mass_kg), 5) if obj.mass_valid else None,
                    "support_contact_valid": bool(getattr(obj, "support_contact_valid", False)),
                    "support_contact": _witness(obj),
                }
                for obj in msg.objects
            ],
        )

    def _on_ack(self, msg: UInt64) -> None:
        self._emit("attachment_applied", revision=int(msg.data))

    def _on_world_state(self, msg: WorldStateStamped) -> None:
        ids = sorted(obj.object_id for obj in msg.attached_objects)
        key = ",".join(ids)
        if key != self._last_ws_attached:
            self._last_ws_attached = key
            self._emit("world_state_attached", n=len(ids), object_ids=ids)
        decl = _declaration(msg)
        ws_key = _decl_key(decl)
        if ws_key != self._last_ws_decl_key:
            self._last_ws_decl_key = ws_key
            self._emit("place_declaration", source="world_state", declaration=decl)

    def _on_voxels(self, msg: OccupancyVoxels) -> None:
        self._last_voxels = msg
        self._maybe_carry_snapshot()
        occ = np.frombuffer(bytes(msg.occupancy), dtype=np.uint8)
        idx = np.flatnonzero(occ)
        count = int(idx.size)
        self._voxel_counts.append(count)
        now = time.time()
        if now - self._last_voxel_log >= 2.0:
            self._last_voxel_log = now
            # Freshness evidence: a frozen map keeps an identical occupied-cell
            # SET, not merely an identical count. Hash the set, and track the
            # x-index-33 column where the stale self-silhouette plane sat.
            sx, sy = int(msg.size_x), int(msg.size_y)
            xs = (idx % sx) if sx else idx
            self._emit(
                "world_voxels",
                count=count,
                set_sha1=hashlib.sha1(idx.tobytes()).hexdigest()[:16],
                n_x33=int((xs == 33).sum()),
                frame_id=msg.header.frame_id,
                resolution=round(float(msg.resolution), 5),
                dims=[int(msg.size_x), int(msg.size_y), int(msg.size_z)],
                origin=[
                    round(float(msg.origin.x), 5),
                    round(float(msg.origin.y), 5),
                    round(float(msg.origin.z), 5),
                ],
                attached_ws=self._last_ws_attached,
            )

    def _maybe_carry_snapshot(self) -> None:
        """Snapshot the published grid while a payload is attached.

        Same npz payload `_snapshot` writes at an E-stop, so `residue.py`
        reads it unchanged. Gated on there actually being an attached
        object, and throttled, so an idle run writes nothing.
        """
        att = self._last_attachment
        if att is None or not att.objects or self._last_voxels is None:
            return
        now = time.time()
        if now - self._last_carry_snap < self._carry_period_s:
            return
        self._last_carry_snap = now
        self._snapshot("carry")

    def _on_failure(self, msg: FailureTrigger) -> None:
        try:
            evidence = json.loads(msg.evidence_json) if msg.evidence_json else None
        except json.JSONDecodeError:
            evidence = {"_unparsed": msg.evidence_json}
        self._emit(
            "failure_safety",
            kind=int(msg.kind),
            severity=int(msg.severity),
            rskill_id=msg.rskill_id,
            evidence=evidence,
        )
        if int(msg.kind) == 10:  # KIND_COLLISION
            self._snapshot("collision")

    def _on_estop(self, _msg: Empty) -> None:
        self._emit("estop", voxel_count=self._voxel_counts[-1] if self._voxel_counts else None)
        self._snapshot("estop")

    # -------------------------------------------------------------- snapshot

    def _lookup_tf(self, target: str, source: str) -> dict[str, Any]:
        try:
            tf = self._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        t = tf.transform.translation
        q = tf.transform.rotation
        return {
            "ok": True,
            "target": target,
            "source": source,
            "translation": [float(t.x), float(t.y), float(t.z)],
            "rotation_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
        }

    def _snapshot(self, tag: str) -> None:
        self._n_snap += 1
        path = os.path.join(self._snap_dir, f"{tag}_{self._n_snap:02d}.npz")
        payload: dict[str, Any] = {"tag": tag, "t": time.time() - self._t0}

        grid = self._last_voxels
        att = self._last_attachment
        grid_frame = grid.header.frame_id if grid is not None else ""

        if grid is not None:
            payload["grid_origin"] = np.array(
                [grid.origin.x, grid.origin.y, grid.origin.z], dtype=np.float64
            )
            payload["grid_resolution"] = np.float64(grid.resolution)
            payload["grid_dims"] = np.array([grid.size_x, grid.size_y, grid.size_z], dtype=np.int64)
            payload["grid_occupancy"] = np.frombuffer(bytes(grid.occupancy), dtype=np.uint8).copy()
            payload["grid_frame_id"] = np.str_(grid_frame)

        objs: list[dict[str, Any]] = []
        if att is not None:
            payload["attachment_revision"] = np.int64(att.revision)
            for obj in att.objects:
                p = obj.pose_in_link.position
                o = obj.pose_in_link.orientation
                prims = []
                for prim in obj.primitives:
                    pp = prim.pose_in_object.position
                    po = prim.pose_in_object.orientation
                    prims.append(
                        {
                            "shape_type": int(prim.shape_type),
                            "dims": [float(d) for d in prim.shape_dimensions],
                            "pos": [float(pp.x), float(pp.y), float(pp.z)],
                            "quat_xyzw": [float(po.x), float(po.y), float(po.z), float(po.w)],
                        }
                    )
                objs.append(
                    {
                        "object_id": obj.object_id,
                        "attach_link": obj.attach_link,
                        "touch_links": list(obj.touch_links),
                        "pose_in_link": {
                            "pos": [float(p.x), float(p.y), float(p.z)],
                            "quat_xyzw": [float(o.x), float(o.y), float(o.z), float(o.w)],
                        },
                        "primitives": prims,
                        "evidence_kind": obj.evidence_kind,
                        "evidence_ref": obj.evidence_ref,
                        "confidence": float(obj.confidence),
                        "support_contact_valid": bool(getattr(obj, "support_contact_valid", False)),
                        "support_contact": _witness(obj),
                    }
                )
        payload["attached_objects_json"] = np.str_(json.dumps(objs, sort_keys=True))
        payload["place_declaration_json"] = np.str_(
            json.dumps(_declaration(att) if att is not None else None, sort_keys=True)
        )

        # TF that places each attach link in the grid frame, plus the common
        # candidates for the known base_link/base-frame ambiguity. Robot links
        # are always looked up, not just attach links: a stop that happens
        # before any grasp (arm-vs-world) still needs the frame offset measured.
        tfs: dict[str, Any] = {}
        links = sorted(
            {o["attach_link"] for o in objs}
            | {f"panda_link{i}" for i in range(8)}
            | {"panda_hand", "panda_hand_tcp", "panda_link0"}
        )
        frames = [f for f in (grid_frame, "base_link", "odom", "map") if f]
        for link in links:
            for frame in frames:
                tfs[f"{frame}<-{link}"] = self._lookup_tf(frame, link)
        payload["tf_json"] = np.str_(json.dumps(tfs, sort_keys=True))
        try:
            payload["tf_frames_yaml"] = np.str_(self._tf_buffer.all_frames_as_yaml())
        except Exception:
            payload["tf_frames_yaml"] = np.str_("")

        np.savez_compressed(path, **payload)
        self._emit(
            "snapshot",
            tag=tag,
            path=path,
            has_grid=grid is not None,
            grid_frame=grid_frame,
            n_attached=len(objs),
            attach_links=links,
        )


def main() -> int:
    path = sys.argv[1]
    snap_dir = sys.argv[2]
    os.makedirs(snap_dir, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("validation_matrix_monitor")
    with open(path, "w", encoding="utf-8") as sink:
        monitor = Monitor(node, sink, snap_dir)
        monitor._emit("monitor_started")
        stop = False

        def _handle(_sig: int, _frm: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
        monitor._emit("monitor_stopped")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
