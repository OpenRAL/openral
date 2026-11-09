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

A lifecycle node: ``configure`` loads SAM2 and subscribes, ``activate`` starts propagating,
``deactivate`` / ``cleanup`` drop every mask, label and video session.

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

from openral_world_state_ros.object_tracker_node import image_array

# A released or lost target is still followed: vision continuity is what re-finds it.
# Followed while the tracker holds them. TENTATIVE too: a contact hold ages a pushed object
# CONFIRMED -> TENTATIVE -> LOST, and dropping its mask there lost it for good. SPLIT too: its
# map is ambiguous, which is when vision matters, and a split track goes LOST next, never
# re-seeded (its box is stale), so a mask dropped at the split is gone for good (A/B ab1b_po
# seed 5: an orange pushed 10 cm split, lost its mask, and the jaw stopped on it 47 s later).
LIVE = {"tentative", "confirmed", "occluded", "bound", "released", "lost", "split"}
#: Only a confirmed track's box is fresh enough to seed SAM from; any other keeps its mask.
SEEDABLE = "confirmed"
#: A tracked_objects revision jump past this is a re-activated tracker (its revision base is the
#: wall-clock ns at activation; the store itself bumps a few per frame): a new episode.
REVISION_RESEED_JUMP = 1_000_000_000
#: A sensor stamp this far behind the newest kept one is a clock reset (a new sim episode).
CLOCK_RESET_NS = 1_000_000_000

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


def keep(d: dict[int, Any], t: int, v: Any, n: int = 30) -> bool:
    """Keep ``v`` at stamp ``t``, the ``n`` newest; ``False`` (and ``d`` emptied) on a clock reset.

    Example:
        >>> d = {}
        >>> keep(d, 5_000_000_000, "a"), keep(d, 6_000_000_000, "b"), sorted(d)
        (True, True, [5000000000, 6000000000])
        >>> keep(d, 0, "c"), sorted(d)
        (False, [0])
    """
    reset = bool(d) and t < max(d) - CLOCK_RESET_NS
    if reset:
        d.clear()
    d[t] = v
    while len(d) > n:
        del d[min(d)]
    return not reset


def new_tracker_session(prev: int, rev: int) -> bool:
    """Whether revision ``rev`` after ``prev`` comes from a new tracker session (a new episode).

    Example:
        >>> new_tracker_session(-1, 10**18), new_tracker_session(10**18, 10**18 + 7)
        (False, False)
        >>> new_tracker_session(10**18, 10**18 + 5 * 10**9), new_tracker_session(10, 3)
        (True, True)
    """
    return prev >= 0 and (rev < prev or rev - prev > REVISION_RESEED_JUMP)


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


def plan(
    ids: list[str],
    status: dict[str, tuple[str, str]],
    seen: dict[str, int],
    td: int,
    empty: dict[str, int],
    drift: dict[str, int],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Which tracks to follow, seed, drop and re-seed now: ``(want, new, gone, stale)``.

    ``ids``: followed now; ``status``: per tracked id, its (status, role); ``seen``: the last
    stamp each was LIVE (updated here); ``empty`` / ``drift``: per followed id, its run of
    empty / drifted masks. Only a ``SEEDABLE`` (confirmed) track is ever seeded or re-seeded
    from its box; a drifted mask with no fresh box is dropped (a held one rides the hand).

    Example:
        >>> s = {"trk:1": ("tentative", "target"), "trk:2": ("confirmed", "target")}
        >>> plan(["trk:1"], s, {}, 0, {}, {})  # a contact hold ages trk:1: still followed
        (['trk:1', 'trk:2'], ['trk:2'], [], [])
        >>> s = {"trk:1": ("lost", "target")}
        >>> plan(["trk:1"], s, {}, 0, {"trk:1": 99}, {})  # empty: no re-seed on a stale box
        (['trk:1'], [], [], [])
        >>> plan(["trk:1"], s, {}, 0, {}, {"trk:1": 99})  # drifted with no fresh box: dropped
        (['trk:1'], [], ['trk:1'], [])
    """
    for i, (st, _) in status.items():
        if st in LIVE:
            seen[i] = td
    want = [i for i, (st, role) in status.items() if st in LIVE and role in ROLES]
    want += [i for i in ids if i not in status and td - seen.get(i, td) <= _V.orphan_s * 1e9]
    seedable = {i for i, (st, _) in status.items() if st == SEEDABLE}
    new = [i for i in want if i not in ids and i in seedable]
    drifted = {i for i in ids if drift.get(i, 0) >= _V.drift_frames}
    held = {i for i, (st, _) in status.items() if st == "bound"}
    gone = [i for i in ids if i not in want or (i in drifted and i not in seedable | held)]
    stale = [
        i
        for i in ids
        if i in seedable and (empty.get(i, 0) >= _V.empty_reseed_frames or i in drifted)
    ]
    return want, new, gone, stale


def main() -> None:  # noqa: PLR0915  # reason: one node, its callbacks share the worker state
    """Run the propagator lifecycle node on ``--camera`` until shutdown."""
    import rclpy
    import rclpy.executors
    import tf2_ros
    import torch
    from openral_msgs.msg import InstanceCells, OccupancyVoxels, TrackedObjectSet
    from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
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
    sam: dict[str, Any] = {}

    rclpy.init()
    lock = threading.Lock()
    st: dict[str, Any] = {}
    ws: dict[str, Any] = {}

    def clear() -> None:
        """Forget every frame, mask, label and video session (a new episode or a transition)."""
        st.update(rgb=None, depth={}, robot={}, k=None, grid=None, tracked={}, revision=-1)
        ws.clear()
        ws.update(session=None, ids=[], masks={}, empty={}, labels={}, frames=0, seen={},
                  seed_px={}, drift={})  # fmt: skip

    clear()

    class Propagator(LifecycleNode):  # type: ignore[misc]  # reason: rclpy is untyped
        """The propagator's lifecycle (module docstring)."""

        def __init__(self) -> None:
            super().__init__("instance_mask_propagator")
            self.active = False
            self.pub: Any = None
            self.subs: list[Any] = []
            self.tf_buf: Any = None
            self.tf_listener: Any = None

        def on_configure(self, state: object) -> TransitionCallbackReturn:
            if not sam:
                sam["v"] = Sam2VideoModel.from_pretrained(WEIGHTS).to(dev, dtype=dt)
                sam["vp"] = Sam2VideoProcessor.from_pretrained(WEIGHTS)
                sam["i"] = Sam2Model.from_pretrained(WEIGHTS).to(dev, dtype=dt)
                sam["ip"] = Sam2Processor.from_pretrained(WEIGHTS)
            reliable_1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
            self.pub = self.create_publisher(InstanceCells, "/openral/instance_cells", reliable_1)
            sd = qos_profile_sensor_data
            self.subs = [
                self.create_subscription(Image, f"{base}/image", on_rgb, sd),
                self.create_subscription(Image, f"{base}/depth/image", on_depth, sd),
                self.create_subscription(Image, f"{base}/robot_mask", on_robot, sd),
                self.create_subscription(CameraInfo, f"{base}/depth/camera_info", on_info, sd),
                self.create_subscription(
                    OccupancyVoxels, "/openral/world_voxels", on_grid, reliable_1
                ),
                self.create_subscription(
                    TrackedObjectSet, "/openral/tracked_objects", on_tracked, reliable_1
                ),
            ]
            log.info(f"instance_masks.configured camera={a.camera} device={dev}")
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: object) -> TransitionCallbackReturn:
            with lock:
                clear()
                self.active = True
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: object) -> TransitionCallbackReturn:
            with lock:
                self.active = False
                clear()
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: object) -> TransitionCallbackReturn:
            with lock:
                self.active = False
                clear()
            for sub in self.subs:
                self.destroy_subscription(sub)
            self.subs = []
            if self.pub is not None:
                self.destroy_publisher(self.pub)
                self.pub = None
            self.tf_listener = None
            self.tf_buf = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: object) -> TransitionCallbackReturn:
            return self.on_cleanup(state)

    node = Propagator()
    log = node.get_logger()

    def on_rgb(m: Any) -> None:  # reason: rclpy message
        with lock:
            st["rgb"] = m

    def on_depth(m: Any) -> None:  # reason: rclpy message
        arr = image_array(m)
        with lock:
            if not keep(st["depth"], stamp_ns(m.header.stamp), (arr, m.header.frame_id)):
                st["robot"].clear()
                st["reset"] = True  # the clock went back: a new episode

    def on_robot(m: Any) -> None:  # reason: rclpy message
        arr = image_array(m) > 0
        with lock:
            keep(st["robot"], stamp_ns(m.header.stamp), arr)

    def on_info(m: Any) -> None:  # reason: rclpy message
        with lock:
            st["k"] = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))

    def on_grid(m: Any) -> None:  # reason: rclpy message
        o = m.orientation
        g = {
            "frame": m.header.frame_id, "origin": np.array([m.origin.x, m.origin.y, m.origin.z]),
            "r": quat_xyzw_to_r((o.x, o.y, o.z, o.w)), "res": float(m.resolution),
            "size": (int(m.size_x), int(m.size_y), int(m.size_z)),
            "occ": np.frombuffer(bytes(m.occupancy), np.uint8).astype(bool),
        }  # fmt: skip
        with lock:
            st["grid"] = g

    def on_tracked(m: Any) -> None:  # reason: rclpy message
        with lock:
            if new_tracker_session(st["revision"], int(m.revision)):
                st["reset"] = True  # a new episode's tracker: its ids restart at trk:1
            st["revision"] = int(m.revision)
            st["tracked"] = {o.object_id: o for o in m.objects}

    def seed_mask(img: np.ndarray, box: list[float], robot: np.ndarray) -> np.ndarray | None:
        """SAM2 image mode on a box, a robot-free positive point and a robot negative point."""
        iproc, imodel = sam["ip"], sam["i"]
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
        o: Any,  # reason: rclpy message
        k: tuple[float, float, float, float],
        r_gc: np.ndarray,
        t_gc: np.ndarray,
        hw: tuple[int, int],
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

    def step() -> bool:  # noqa: PLR0911, PLR0912, PLR0915
        vproc, vmodel = sam["vp"], sam["v"]
        with lock:
            if not node.active:
                return False
            if st.pop("reset", False):
                depth_kept, robot_kept = st["depth"], st["robot"]
                clear()
                st["depth"], st["robot"] = depth_kept, robot_kept  # already the new episode's
                log.info("instance_masks.episode_reset")
            m = st["rgb"]
            st["rgb"] = None
            if m is None or st["k"] is None or st["grid"] is None:
                return False
            t = stamp_ns(m.header.stamp)
            if not st["depth"]:
                return False
            td = min(st["depth"], key=lambda s: abs(s - t))
            depth, optical = st["depth"][td]
            robot = st["robot"][min(st["robot"], key=lambda s: abs(s - t))] if st["robot"] else None
            k, g, tracked = st["k"], st["grid"], dict(st["tracked"])
            tf_buf = node.tf_buf
        if abs(td - t) > 150_000_000 or tf_buf is None:
            return False
        img = image_array(m)
        if robot is None or robot.shape != depth.shape:
            robot = np.zeros(depth.shape, bool)
        try:
            # The camera pose AT the depth stamp: never the latest (on a moving camera that
            # lifts the mask through the wrong pose); no transform yet = skip the frame.
            tf = tf_buf.lookup_transform(g["frame"], optical, rclpy.time.Time(nanoseconds=td))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:  # fmt: skip
            log.debug(f"instance_masks.no_tf {type(e).__name__}")
            return False
        q, tr_ = tf.transform.rotation, tf.transform.translation
        r_gc, t_gc = quat_xyzw_to_r((q.x, q.y, q.z, q.w)), np.array([tr_.x, tr_.y, tr_.z])

        status = {i: (o.status, o.role) for i, o in tracked.items()}
        want, new, gone, stale = plan(ws["ids"], status, ws["seen"], td, ws["empty"], ws["drift"])
        if ws["session"] is None or new or gone or stale or ws["frames"] >= _V.reseed_frames:
            seeds: dict[str, np.ndarray] = {}
            for i in [*ws["ids"], *new]:
                if i in gone or i not in want:
                    continue
                cur = ws["masks"].get(i)
                if i in stale or i in new or cur is None or cur.sum() < 20:
                    # Only a confirmed track's box is where the object is: a released, lost or
                    # split one's box is stale, and SAM on it segments whatever is there now.
                    if status.get(i, ("", ""))[0] != SEEDABLE:
                        continue  # nothing fresh to follow: its mask is gone
                    box = project_box(tracked[i], k, r_gc, t_gc, img.shape[:2])
                    cur = seed_mask(img, box, robot) if box is not None else None
                if cur is not None:
                    seeds[i] = cur
                    if i in stale or i in new or i not in ws["seed_px"]:
                        ws["seed_px"][i] = int(cur.sum())
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
            seed_px = ws["seed_px"].get(i, 0)
            drift = seed_px > 0 and mk.sum() > _V.drift_ratio * seed_px
            ws["drift"][i] = ws["drift"].get(i, 0) + 1 if drift else 0
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
        with lock:
            if node.active and node.pub is not None:
                node.pub.publish(msg)
        return True

    def worker() -> None:
        n, t0 = 0, time.monotonic()
        while rclpy.ok():
            try:
                did = step()
            except Exception as e:  # reason: a SAM failure drops the session, logged
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
    log.info(f"instance_masks.started camera={a.camera} (unconfigured)")
    with contextlib.suppress(KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        rclpy.spin(node)


if __name__ == "__main__":
    main()
