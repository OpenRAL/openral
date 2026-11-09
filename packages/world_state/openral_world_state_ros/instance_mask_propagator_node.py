"""EXPERIMENT (exp/vision-voxel-fusion): SAM2 video instance masks lifted into voxel cells.

Streams SAM2.1 video propagation over one camera's RGB, one object per live TARGET /
DESTINATION track on ``/openral/tracked_objects``. Each new track is seeded from its box (plus a
positive point off the robot and a negative one on it) through SAM2 image mode. The video session
is always seeded with masks, so it carries existing objects across a re-seed. Every processed
frame lifts each mask, minus robot pixels, through the aligned depth into the voxel lattice of
``/openral/world_voxels``. A cell keeps its track label while it stays occupied, up to
``memory_s``. The labels go out per track on ``/openral/instance_cells``
(``openral_msgs/InstanceCells``); the tracker fits those tracks' boxes to them
(``ObjectTracker.apply_instance_cells``). Experiment only: the deploy launch starts it beside
the tracker with ``DeployRuntime.instance_cells`` (sim-only).

Usage:
    ros2 run openral_world_state instance_mask_propagator_node --camera top_depth
"""

from __future__ import annotations

import argparse
import contextlib
import threading
import time
from typing import Any

import numpy as np
from openral_core.tracking_thresholds import load_tracking_thresholds

# a released or lost target is still followed: vision continuity is what re-finds it
# Followed while the tracker holds them. SPLIT too: its map is ambiguous, which is when vision
# matters, and a split track goes LOST next, never re-seeded (its box is stale), so a mask
# dropped at the split is gone for good (A/B ab1b_po seed 5: an orange pushed 10 cm split,
# lost its mask, and the jaw stopped on it 47 s later).
LIVE = {"confirmed", "occluded", "bound", "released", "lost", "split"}
ROLES = {"target", "destination"}
_V = load_tracking_thresholds().vision  # memory_s, orphan_s, re-seed and drift rules
WEIGHTS = "facebook/sam2.1-hiera-small"


def quat_xyzw_to_r(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w).

    Example:
        >>> np.allclose(quat_xyzw_to_r((0.0, 0.0, 0.0, 1.0)), np.eye(3))
        True
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def image_array(msg: Any) -> np.ndarray:
    """A ``sensor_msgs/Image`` as an array: float depth, or RGB (BGR channels swapped)."""
    if msg.encoding == "32FC1":
        return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}[msg.encoding]
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, ch)
    if msg.encoding.startswith("bgr"):
        a = a[..., [2, 1, 0]]
    return a[..., :3] if ch >= 3 else a[..., 0]


def erode(m: np.ndarray, n: int) -> np.ndarray:
    """M shrunk by n pixels (4-neighbour), so edge pixels never carry background depth.

    Example:
        >>> m = np.zeros((5, 5), bool)
        >>> m[1:4, 1:4] = True
        >>> int(erode(m, 1).sum())
        1
    """
    out = m.copy()
    for _ in range(n):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        e[0, :] = e[-1, :] = False
        e[:, 0] = e[:, -1] = False
        out = e
    return out


def stamp_ns(st: Any) -> int:
    """A ``builtin_interfaces/Time`` in nanoseconds."""
    return int(st.sec) * 1_000_000_000 + int(st.nanosec)


def lift_cells(
    mask: np.ndarray,
    depth: np.ndarray,
    k: tuple[float, float, float, float],
    r_gc: np.ndarray,
    t_gc: np.ndarray,
    origin: np.ndarray,
    r_grid: np.ndarray,
    res: float,
) -> np.ndarray:
    """Unique lattice cells (N x 3 int) the masked depth pixels fall in.

    Example:
        >>> m = np.zeros((2, 2), bool)
        >>> m[0, 0] = True
        >>> d = np.ones((2, 2), np.float32)
        >>> lift_cells(
        ...     m, d, (1.0, 1.0, 0.0, 0.0), np.eye(3), np.zeros(3), np.zeros(3), np.eye(3), 1.0
        ... )
        array([[0, 0, 1]])
    """
    fx, fy, cx, cy = k
    sel = mask & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(sel)
    z = depth[sel].astype(np.float64)
    pc = np.stack(((u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z), axis=1)
    pg = pc @ r_gc.T + t_gc
    ijk = np.floor((pg - origin) @ r_grid / res).astype(np.int64)
    return np.unique(ijk, axis=0) if len(ijk) else np.zeros((0, 3), np.int64)


def main() -> None:  # noqa: PLR0915  # reason: one node, its callbacks share the worker state
    """Run the propagator on ``--camera`` until shutdown."""
    import rclpy
    import rclpy.executors
    import tf2_ros
    import torch
    from openral_msgs.msg import InstanceCells, OccupancyVoxels, TrackedObjectSet
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from transformers import (
        Sam2Model,
        Sam2Processor,
        Sam2VideoModel,
        Sam2VideoProcessor,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    a, _ = ap.parse_known_args()
    base = f"/openral/cameras/{a.camera}"
    dev, dt = ("cuda", torch.bfloat16) if torch.cuda.is_available() else ("cpu", torch.float32)
    vmodel = Sam2VideoModel.from_pretrained(WEIGHTS).to(dev, dtype=dt)
    vproc = Sam2VideoProcessor.from_pretrained(WEIGHTS)
    imodel = Sam2Model.from_pretrained(WEIGHTS).to(dev, dtype=dt)
    iproc = Sam2Processor.from_pretrained(WEIGHTS)

    rclpy.init()
    node = Node("instance_mask_propagator")
    log = node.get_logger()
    reliable_1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    tf_buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buf, node)
    pub = node.create_publisher(InstanceCells, "/openral/instance_cells", reliable_1)
    lock = threading.Lock()
    st: dict[str, Any] = {"rgb": None, "depth": {}, "robot": {}, "k": None, "grid": None,
                          "tracked": {}, "revision": -1}  # fmt: skip

    def keep(d: dict, t: int, v: Any, n: int = 30) -> None:
        d[t] = v
        while len(d) > n:
            del d[min(d)]

    def on_rgb(m: Any) -> None:
        with lock:
            st["rgb"] = m

    def on_depth(m: Any) -> None:
        with lock:
            keep(st["depth"], stamp_ns(m.header.stamp), (image_array(m), m.header.frame_id))

    def on_robot(m: Any) -> None:
        with lock:
            keep(st["robot"], stamp_ns(m.header.stamp), image_array(m) > 0)

    def on_info(m: Any) -> None:
        with lock:
            st["k"] = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))

    def on_grid(m: Any) -> None:
        o = m.orientation
        g = {
            "frame": m.header.frame_id, "origin": np.array([m.origin.x, m.origin.y, m.origin.z]),
            "r": quat_xyzw_to_r((o.x, o.y, o.z, o.w)), "res": float(m.resolution),
            "size": (int(m.size_x), int(m.size_y), int(m.size_z)),
            "occ": np.frombuffer(bytes(m.occupancy), np.uint8).astype(bool),
        }  # fmt: skip
        with lock:
            st["grid"] = g

    def on_tracked(m: Any) -> None:
        with lock:
            if int(m.revision) < st["revision"]:
                st["reset"] = True  # a new episode's tracker: start over
            st["revision"] = int(m.revision)
            st["tracked"] = {o.object_id: o for o in m.objects}

    node.create_subscription(Image, f"{base}/image", on_rgb, qos_profile_sensor_data)
    node.create_subscription(Image, f"{base}/depth/image", on_depth, qos_profile_sensor_data)
    node.create_subscription(Image, f"{base}/robot_mask", on_robot, qos_profile_sensor_data)
    node.create_subscription(
        CameraInfo, f"{base}/depth/camera_info", on_info, qos_profile_sensor_data
    )
    node.create_subscription(OccupancyVoxels, "/openral/world_voxels", on_grid, reliable_1)
    node.create_subscription(TrackedObjectSet, "/openral/tracked_objects", on_tracked, reliable_1)

    # worker state
    ws: dict[str, Any] = {"session": None, "ids": [], "masks": {}, "empty": {}, "labels": {},
                          "frames": 0, "last_t": None}  # fmt: skip

    def seed_mask(img: np.ndarray, box: list[float], robot: np.ndarray) -> np.ndarray | None:
        """SAM2 image mode on a box, a robot-free positive point and a robot negative point."""
        x0, y0, x1, y1 = (round(v) for v in box)
        sub = ~robot[y0 : y1 + 1, x0 : x1 + 1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts, labs = [], []
        for want, lab in ((sub, 1), (~sub, 0)):
            if want.any():
                ys, xs = np.nonzero(want)
                i = int(np.argmin((xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2))
                pts.append([float(xs[i] + x0), float(ys[i] + y0)])
                labs.append(lab)
        kw: dict[str, Any] = {"input_boxes": [[box]]}
        if pts:
            kw.update(input_points=[[pts]], input_labels=[[labs]])
        inp = iproc(images=img, return_tensors="pt", **kw).to(dev)
        with torch.no_grad():
            args = {k: (v.to(dt) if v.dtype.is_floating_point else v) for k, v in inp.items()}
            out = imodel(**args, multimask_output=False)
        m = iproc.post_process_masks(out.pred_masks.float().cpu(), inp["original_sizes"].cpu())[0]
        m = m[0, 0].numpy().astype(bool) & ~robot
        return m if m.sum() >= 20 else None

    def project_box(
        o: Any, k: tuple, r_gc: np.ndarray, t_gc: np.ndarray, hw: tuple[int, int]
    ) -> list[float] | None:
        b = o.box
        q = b.pose.orientation
        r = quat_xyzw_to_r((q.x, q.y, q.z, q.w))
        s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        c = (s * [b.half_extents.x, b.half_extents.y, b.half_extents.z]) @ r.T + [
            b.pose.position.x, b.pose.position.y, b.pose.position.z]  # fmt: skip
        pc = (c - t_gc) @ r_gc
        if (pc[:, 2] <= 0).any():
            return None
        u = k[0] * pc[:, 0] / pc[:, 2] + k[2]
        v = k[1] * pc[:, 1] / pc[:, 2] + k[3]
        h, w = hw
        x0, x1 = float(np.clip(u.min(), 0, w - 1)), float(np.clip(u.max(), 0, w - 1))
        y0, y1 = float(np.clip(v.min(), 0, h - 1)), float(np.clip(v.max(), 0, h - 1))
        return [x0, y0, x1, y1] if x1 - x0 >= 3 and y1 - y0 >= 3 else None

    def step() -> bool:  # noqa: PLR0912, PLR0915
        with lock:
            m = st["rgb"]
            st["rgb"] = None
            if m is None or st["k"] is None or st["grid"] is None:
                return False
            if st.pop("reset", False):
                ws.update(session=None, ids=[], masks={}, empty={}, labels={}, frames=0, seen={},
                      seed_px={}, drift={})  # fmt: skip
            t = stamp_ns(m.header.stamp)
            if not st["depth"]:
                return False
            td = min(st["depth"], key=lambda s: abs(s - t))
            depth, optical = st["depth"][td]
            robot = st["robot"][min(st["robot"], key=lambda s: abs(s - t))] if st["robot"] else None
            k, g, tracked = st["k"], st["grid"], dict(st["tracked"])
        if abs(td - t) > 150_000_000:
            return False
        img = image_array(m)
        if robot is None or robot.shape != depth.shape:
            robot = np.zeros(depth.shape, bool)
        try:
            tf = tf_buf.lookup_transform(g["frame"], optical, rclpy.time.Time(nanoseconds=td))
        except Exception:
            try:
                tf = tf_buf.lookup_transform(g["frame"], optical, rclpy.time.Time())
            except Exception:
                return False
        q, tr_ = tf.transform.rotation, tf.transform.translation
        r_gc, t_gc = quat_xyzw_to_r((q.x, q.y, q.z, q.w)), np.array([tr_.x, tr_.y, tr_.z])

        seen = ws.setdefault("seen", {})
        for i, o in tracked.items():
            if o.status in LIVE:
                seen[i] = td
        want = [i for i, o in tracked.items() if o.status in LIVE and o.role in ROLES]
        want += [
            i for i in ws["ids"] if i not in tracked and td - seen.get(i, td) <= _V.orphan_s * 1e9
        ]
        new = [
            i
            for i in want
            if i not in ws["ids"] and i in tracked and tracked[i].status == "confirmed"
        ]
        gone = [i for i in ws["ids"] if i not in want]
        stale = [i for i in ws["ids"] if i in tracked and tracked[i].status != "bound" and (
            (ws["empty"].get(i, 0) >= _V.empty_reseed_frames and tracked[i].status == "confirmed")
            or ws.setdefault("drift", {}).get(i, 0) >= _V.drift_frames)]  # fmt: skip
        if ws["session"] is None or new or gone or stale or ws["frames"] >= _V.reseed_frames:
            seeds: dict[str, np.ndarray] = {}
            for i in [*ws["ids"], *new]:
                if i in gone or i not in want:
                    continue
                cur = ws["masks"].get(i)
                if i in stale or i in new or cur is None or cur.sum() < 20:
                    if i not in tracked:
                        continue  # an orphan whose mask is gone: nothing left to follow
                    box = project_box(tracked[i], k, r_gc, t_gc, img.shape[:2])
                    cur = seed_mask(img, box, robot) if box is not None else None
                if cur is not None:
                    seeds[i] = cur
                    if i in stale or i in new or i not in ws.get("seed_px", {}):
                        ws.setdefault("seed_px", {})[i] = int(cur.sum())
            ws["ids"] = list(seeds)
            ws["drift"] = {}
            ws["empty"] = {}
            ws["frames"] = 0
            ws["masks"] = {}
            if not seeds:
                ws["session"] = None
                return True
            s = vproc.init_video_session(inference_device=dev, dtype=dt)
            inp = vproc(images=img, device=dev, return_tensors="pt")
            vproc.add_inputs_to_inference_session(
                inference_session=s, frame_idx=0, obj_ids=list(range(1, len(seeds) + 1)),
                input_masks=[v.astype(np.float32) for v in seeds.values()],
                original_size=inp.original_sizes[0],
            )  # fmt: skip
            ws["session"] = s
            log.info(f"instance_masks.seeded ids={ws['ids']} new={new} gone={gone} stale={stale}")
        else:
            inp = vproc(images=img, device=dev, return_tensors="pt")
        with torch.no_grad():
            out = vmodel(inference_session=ws["session"], frame=inp.pixel_values[0])
        masks = vproc.post_process_masks(
            [out.pred_masks], original_sizes=inp.original_sizes, binarize=True
        )[0]
        masks = masks[:, 0].cpu().numpy().astype(bool)
        ws["frames"] += 1
        sx, sy, sz = g["size"]
        labels = ws["labels"]
        for i, raw in zip(ws["ids"], masks, strict=True):
            mk = erode(raw, _V.erode_px) & ~robot  # silhouette pixels take the depth behind it
            ws["masks"][i] = mk
            ws["empty"][i] = ws["empty"].get(i, 0) + 1 if mk.sum() < 20 else 0
            seed_px = ws.get("seed_px", {}).get(i, 0)
            drift = seed_px > 0 and mk.sum() > _V.drift_ratio * seed_px
            ws.setdefault("drift", {})[i] = ws["drift"].get(i, 0) + 1 if drift else 0
            cells = lift_cells(mk, depth, k, r_gc, t_gc, g["origin"], g["r"], g["res"])
            ok = (
                (cells >= 0).all(axis=1)
                & (cells[:, 0] < sx)
                & (cells[:, 1] < sy)
                & (cells[:, 2] < sz)
            )
            for c in cells[ok]:
                if g["occ"][c[0] + sx * (c[1] + sy * c[2])]:
                    labels[(int(c[0]), int(c[1]), int(c[2]))] = (i, td)
        for c, (i, tl) in list(labels.items()):  # kept while occupied, and not past memory_s
            occupied = (
                0 <= c[0] < sx
                and 0 <= c[1] < sy
                and 0 <= c[2] < sz
                and g["occ"][c[0] + sx * (c[1] + sy * c[2])]
            )
            if not occupied or td - tl > _V.memory_s * 1e9 or i not in ws["ids"]:
                del labels[c]
        by: dict[str, list[list[int]]] = {}
        for c, (i, _) in labels.items():
            by.setdefault(i, []).append(list(c))
        msg = InstanceCells()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(td), 10**9)
        msg.header.frame_id = g["frame"]
        msg.origin.x, msg.origin.y, msg.origin.z = (float(v) for v in g["origin"])
        msg.resolution = float(g["res"])
        msg.object_ids = list(by)
        msg.cell_counts = [len(v) for v in by.values()]
        msg.cells = [int(x) for v in by.values() for c in v for x in c]
        pub.publish(msg)
        return True

    def worker() -> None:
        n, t0 = 0, time.monotonic()
        while rclpy.ok():
            try:
                did = step()
            except Exception as e:
                log.error(f"instance_masks.step_failed {type(e).__name__}: {e}")
                ws["session"] = None
                did = False
            if not did:
                time.sleep(0.01)
                continue
            n += 1
            if n % 50 == 0:
                hz = n / (time.monotonic() - t0)
                log.info(
                    f"instance_masks.rate hz={hz:.1f} ids={ws['ids']} labels={len(ws['labels'])}"
                )

    threading.Thread(target=worker, daemon=True).start()
    log.info(f"instance_masks.started camera={a.camera} device={dev}")
    with contextlib.suppress(KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        rclpy.spin(node)


if __name__ == "__main__":
    main()
