"""Scripted OpenArm pick/place driver for the Isaac deploy-sim validation run.

Registered as policy family ``isaac_pick_place_scripted`` in the runner process by
``tools/isaac_pick_place/sitecustomize.py``; never installed, never on a real robot. It
is NOT a policy: a perception-driven script whose only world inputs are what the deploy
graph's perception publishes on ROS:

* ``/openral/world_voxels`` (the octomap the kernel checks against) -> the support
  plane, object clusters and free support spots;
* ``/openral/attachment_state`` (the vision attachment leg) -> the grasp region and
  attach / detach events.

Kinematics come from the robot's own MJCF (``tools/isaac_pick_place/kin.py``): robot
model only, no scene state, no simulator ground truth. Every command leaves through the
runner's normal path (adapter.step -> Action -> candidate_action -> safety kernel ->
safe_action -> HAL), so the run exercises the grasp-target exemption end to end.

Environment:

* ``PICK_PLACE_LOG`` — JSONL event log (default ``./pick_place_driver.jsonl``).
* ``PICK_PLACE_PICKS`` — objects to pick and place (default 2).
* ``PICK_PLACE_FIRST_SIDE`` — ``left`` / ``right``: try that hand's candidates first.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

_Arr = NDArray[np.float64]

if TYPE_CHECKING:
    from openral_core import VLASpec

_LOG_PATH = os.environ.get("PICK_PLACE_LOG", "pick_place_driver.jsonl")
_N_PICKS = int(os.environ.get("PICK_PLACE_PICKS", "2"))
_HANDS = ("left", "right")  # the hand nearest each target picks it (_choose_target)
_CART_SPEED = 0.05  # m/s
_JOINT_STEP = 0.02  # rad/tick
_GRIP_STEP = 0.12  # rad/tick (3.6 rad/s at 30 Hz, under the 4 rad/s limit)
_TIP_CLEAR = (
    0.020  # fingertip centre above the measured support cell top: the support is never exempt
)
_PLACE_EXTRA = 0.010  # release the payload this much above where it was grasped
_PREGRASP = 0.12  # TCP back along the fingers on the way out after a place
_HOVER = 0.085  # TCP back along the fingers while waiting for the grasp region
# Re-aim the jaws to the measured region's minor axis only past this, and only when
# the region covers the scanned footprint (region_covers_footprint): a rotation at
# hover occludes the target and can outlive the region's life.
_REPLAN_MIN_DEG = 30.0
_GRASP_CLEAR = 0.020  # TCP over the region floor at the grasp
_LOOK_UP = 0.05  # TCP over the target top while it is measured: the hand's approach box must hold its top layer
_HOVER_WAIT_S = (
    40.0  # wait this long at hover for the grasp region, then descend anyway (the kernel decides)
)
_DESCEND_SPEED = (
    0.30  # m/s (planned at 30 Hz): inside the region's life once the hand occludes the target
)
_REGION_FRESH_S = 0.6  # descend only on a region measured this recently
_LIFT = 0.12
_W_UP = 0.05  # gentle pull of the gripper toward vertical
_MAX_TILT = 75.0  # deg from vertical, tilting about the closing axis only


def _log(event: str, **kw: object) -> None:
    kw = {"event": event, "wall": round(time.time(), 3), **kw}
    with open(_LOG_PATH, "a") as fh:
        fh.write(json.dumps(kw, default=lambda o: np.asarray(o).round(4).tolist()) + "\n")


# ── pure geometry (unit-tested in tests/unit/test_isaac_pick_place_driver.py) ──


def footprint_minor_yaw(columns_xy: _Arr) -> float:
    """Yaw of the narrow horizontal axis of a set of occupied columns (one row per column).

    Each occupied (x, y) column counts once. A 3-D PCA over every cell leans toward the
    walls the camera sees (the map holds a box's top layer plus only its camera-facing
    walls), which tilted an axis-aligned brick by about 9 degrees, mirrored per side.
    """
    xy = np.asarray(columns_xy, dtype=np.float64)
    xy = xy - xy.mean(0)
    _evals, evecs = np.linalg.eigh(xy.T @ xy + 1e-9 * np.eye(2))
    minor = evecs[:, 0]
    return float(np.arctan2(minor[1], minor[0]))


def grasp_centre(region_xy: _Arr, cluster_xy: _Arr, closing_yaw: float) -> _Arr:
    """Jaw centre: the measured region's centre ACROSS the jaws, the map cluster's ALONG them.

    The region is fitted while the hand hovers over the target and hides its far part
    from the head camera, so its centre falls short along the jaws' width; across the
    jaws it is the better measurement (the map cluster is coarser there).
    """
    c = np.array([math.cos(closing_yaw), math.sin(closing_yaw)])
    rxy, txy = np.asarray(region_xy, dtype=np.float64), np.asarray(cluster_xy, dtype=np.float64)
    return txy + c * float(np.dot(rxy - txy, c))


def region_covers_footprint(region: dict[str, Any], columns_xy: _Arr, res: float) -> bool:
    """Whether the yaw-only region box holds every scanned footprint column (one cell of slack).

    A region fitted from a partial view (the hovering hand hides part of the target) has
    an unreliable yaw; only a region that saw the whole footprint may re-aim the jaws.
    """
    yaw = float(region["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(columns_xy, dtype=np.float64) - np.asarray(region["xyz"][:2], dtype=np.float64)
    local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    half = np.asarray(region["half"][:2], dtype=np.float64) + res
    return bool(np.all(np.abs(local) <= half))


# ── perception ──────────────────────────────────────────────────────────────────


class Perception:
    """World voxels + attachment state, on a dedicated node in its own executor thread."""

    def __init__(self) -> None:
        from openral_msgs.msg import AttachmentState, OccupancyVoxels
        from openral_observability.rclpy_spin import spin_executor_until_shutdown
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self.node = Node("pick_place_driver_perception")
        self.lock = threading.Lock()
        self.voxel_msg: Any = None
        self.voxel_rx = 0.0
        self.attachment: Any = None
        self.node.create_subscription(
            OccupancyVoxels,
            "/openral/world_voxels",
            self._on_voxels,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            ),
        )
        self.node.create_subscription(
            AttachmentState,
            "/openral/attachment_state",
            self._on_attachment,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(
            target=spin_executor_until_shutdown,
            args=(self.executor,),
            daemon=True,
            name="pick-place-perception",
        ).start()

    def _on_voxels(self, msg: object) -> None:
        with self.lock:
            self.voxel_msg = msg
            self.voxel_rx = time.monotonic()

    def _on_attachment(self, msg: object) -> None:
        with self.lock:
            self.attachment = msg

    def lattice(self) -> Any:
        from openral_hal._grasp_target_leg import lattice_from_msg

        with self.lock:
            msg, age = self.voxel_msg, time.monotonic() - self.voxel_rx
        if msg is None or age > 1.0:
            return None
        return lattice_from_msg(msg)

    def held(self, side: str) -> list[str]:
        with self.lock:
            msg = self.attachment
        if msg is None:
            return []
        return [o.object_id for o in msg.objects if f"_{side}_" in o.attach_link]

    def grasp_region(self, side: str) -> dict[str, Any] | None:
        """The producer-measured grasp region for this hand, if the envelope carries one."""
        with self.lock:
            msg = self.attachment
        if msg is None or not msg.grasp_declaration_valid:
            return None
        d = msg.grasp_declaration
        if not d.region_valid or f"openarm_{side}_finger_pair" not in list(d.contact_links):
            return None
        p, h, o = d.region.pose.position, d.region.half_extents, d.region.pose.orientation
        # The fit's box is gravity-aligned (yaw-only): its jaws-closing axis is the minor horizontal one.
        ry = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        now_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        return {
            "target": d.target_id,
            "xyz": [p.x, p.y, p.z],
            "half": [h.x, h.y, h.z],
            "frame": d.region.frame_id,
            "yaw": ry,
            "minor_yaw": ry if h.x <= h.y else ry + math.pi / 2,
            "stamp_ns": d.region.stamp_ns,
            "age_s": (now_ns - d.region.stamp_ns) / 1e9,
        }

    def any_attached(self) -> list[tuple[str, str]]:
        with self.lock:
            msg = self.attachment
        return [] if msg is None else [(o.object_id, o.attach_link) for o in msg.objects]


def scene_from_lattice(lat: Any) -> dict[str, Any]:
    """Support plane + object clusters + support cells, all measured from the voxel map.

    Cells are indexed from the lattice origin (``floor((centre - origin) / res)``), never
    by rounding centres, which sit on half-cells and alias under banker's rounding.
    """
    c = lat.occupied_centers()
    res = float(lat.resolution)
    origin = np.asarray(lat.origin, dtype=np.float64)
    # ponytail: a fixed workspace box in front of the OpenArm (base frame); a robot-agnostic
    # driver would take it from the scene.
    w = c[
        (c[:, 0] > 0.05)
        & (c[:, 0] < 0.62)
        & (np.abs(c[:, 1]) < 0.5)
        & (c[:, 2] > -0.6)
        & (c[:, 2] < 0.1)
    ]
    if len(w) == 0:
        return {}
    idx = np.floor((w - origin) / res + 1e-6).astype(int)
    vals, counts = np.unique(idx[:, 2], return_counts=True)
    kt = int(vals[np.argmax(counts)])
    table_top = float(origin[2] + (kt + 1) * res)
    table = w[idx[:, 2] == kt]
    sel = (idx[:, 2] > kt) & (w[:, 2] < table_top + 0.20)
    above, above_idx = w[sel], idx[sel]
    keys = {tuple(k) for k in above_idx}
    centre_of = {tuple(k): p for k, p in zip(above_idx, above, strict=True)}
    clusters, seen = [], set()
    for k in keys:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                        if nb in keys and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
        pts = np.array([centre_of[kk] for kk in comp])
        ext = pts.max(0) - pts.min(0)
        if len(comp) >= 3 and ext[0] < 0.25 and ext[1] < 0.25:
            # Each occupied column once, from the lattice index (not by rounding centres).
            cols = np.unique(np.array(comp)[:, :2], axis=0)
            foot = origin[:2] + (cols + 0.5) * res
            lo = pts.min(0) - res / 2
            hi = pts.max(0) + res / 2
            clusters.append(
                {
                    "minor_yaw": footprint_minor_yaw(foot),
                    "foot": foot,
                    "lo": lo,
                    "hi": hi,
                    "xy": (lo[:2] + hi[:2]) / 2.0,
                    "top": float(hi[2]),
                    "bottom": float(lo[2]),
                    "n": len(comp),
                }
            )
    return {"table_top": table_top, "res": res, "table": table, "objects": clusters, "above": above}


# ── driver ──────────────────────────────────────────────────────────────────────


class Driver:
    def __init__(self, env_cfg: Any) -> None:
        # A sibling module on PYTHONPATH at run time, not an installed package.
        from kin import Kin  # type: ignore[import-not-found]

        self.spec: VLASpec = env_cfg.vla
        self.device = "cpu"
        self.action_dim = 16
        self.kin = Kin(env_cfg.robot_description)
        self.perc: Perception | None = None
        self.reset()

    # PolicyAdapter protocol ---------------------------------------------------------
    def reset(self) -> None:
        self.des: _Arr = np.zeros(16)
        self.seeded = False
        self.bias = np.zeros(16)
        self.q_prev: _Arr | None = None
        self.queue: deque[_Arr] = deque()
        self.phase = "init"
        self.phase_t = time.monotonic()
        self.picks_done = 0
        self.placed: list[_Arr] = []
        self.target: dict[str, Any] = {}
        self.target_top = 0.0
        self.plan: dict[str, Any] = {}
        self.place: _Arr = np.zeros(2)
        self.place_tcp: _Arr = np.zeros(3)
        self.scene: dict[str, Any] = {}
        self.yaw = 0.0
        self.side = "right"
        self.tick = 0
        self.region_yaw_done = False
        self.lost_logged = False
        self._jaw_dev_logged = 0.0

    def close(self) -> None:
        return None

    def step(self, observation: dict[str, Any], instruction: str) -> NDArray[np.float32]:
        q = np.asarray(observation["state"], dtype=np.float64).copy()
        if self.perc is None:
            # The warm-up call also lands here; the runner resets the adapter after it,
            # so the command re-seeds from the first real measured state.
            self.perc = Perception()
            _log("driver_started", instruction=instruction, hands=_HANDS, picks=_N_PICKS)
        if not self.seeded:
            self.des, self.seeded = q.copy(), True
            _log("measured_start", q=q)
        self.tick += 1
        if self.tick % 30 == 0 or self.phase in ("descend", "close"):
            err = q - self.des
            gi = self.kin.grip[self.side]
            _log(
                "track",
                phase=self.phase,
                queue=len(self.queue),
                err_right=err[8:16],
                err_left=err[0:8],
                grip=q[gi],
                grip_cmd=self.des[gi],
                tick=self.tick,
            )
        try:
            self._advance(q)
        except Exception as exc:  # reason: a script fault holds the last command, logged
            import traceback

            _log(
                "driver_exception",
                phase=self.phase,
                error=repr(exc),
                tb=traceback.format_exc()[-1500:],
            )
            self.phase = "hold"
            self.queue.clear()
        if self.queue:
            self.des = self.queue.popleft()
        else:
            # Outer-loop sag compensation on the ARM joints from measured position only:
            # the twin's position servos hold a gravity-dependent steady-state error, so
            # once the arm is stationary the command is nudged until the measured joints
            # reach the desired ones. Never on the gripper (its stall short of the
            # command is the grasp trigger), and never while the hand is in contact:
            # there the error is the contact, and integrating it presses the arm into
            # the target.
            moving = self.q_prev is not None and float(np.max(np.abs(q - self.q_prev))) > 2e-3
            if not moving and self.phase not in ("close", "lift_wait", "lift"):
                idx = self.kin.arm["left"] + self.kin.arm["right"]
                self.bias[idx] = np.clip(
                    self.bias[idx] + 0.3 * (self.des[idx] - q[idx]), -0.15, 0.15
                )
        self.q_prev = q
        return (self.des + self.bias).astype(np.float32)

    # helpers ---------------------------------------------------------------------------
    def _set_phase(self, name: str, **kw: object) -> None:
        self.phase = name
        self.phase_t = time.monotonic()
        _log("phase", phase=name, pick=self.picks_done + 1, **kw)

    def _log_pose(self, event: str, side: str, q: _Arr) -> None:
        """Measured TCP and closing yaw vs the planned ones."""
        p, r = self.kin.tcp(q, side)
        yaw = float(np.arctan2(r[1, 1], r[0, 1]))
        d = (yaw - self.yaw + np.pi / 2) % np.pi - np.pi / 2
        _log(
            event,
            side=side,
            tcp=p,
            plan_tcp=self.plan["grasp"],
            yaw=yaw,
            plan_yaw=self.yaw,
            yaw_err_deg=np.degrees(d),
            approach=-r[:, 2],
            bias=self.bias[self.kin.arm[side]],
        )

    def _elapsed(self) -> float:
        return time.monotonic() - self.phase_t

    def _settled(self, q: _Arr, tol: float = 0.012, tcp_tol: float = 0.015) -> bool:
        """Queue drained and the arms at the command — per joint, or every TCP within
        ``tcp_tol`` while stationary (the sag bias at its clip can leave one joint well off
        with the TCP close)."""
        if self.queue:
            return False
        idx = self.kin.arm["left"] + self.kin.arm["right"]
        if float(np.max(np.abs(q[idx] - self.des[idx]))) < tol:
            return True
        still = self.q_prev is not None and float(np.max(np.abs(q - self.q_prev))) <= 2e-3
        return still and all(
            float(np.linalg.norm(self.kin.tcp(q, s)[0] - self.kin.tcp(self.des, s)[0])) < tcp_tol
            for s in ("left", "right")
        )

    def _joint_move(self, goal: _Arr, step: float = _JOINT_STEP) -> None:
        start = self.queue[-1] if self.queue else self.des
        n = int(np.ceil(np.max(np.abs(goal - start)) / step)) or 1
        for i in range(1, n + 1):
            self.queue.append(start + (goal - start) * i / n)

    def _grip(self, side: str, value: float) -> None:
        start = (self.queue[-1] if self.queue else self.des).copy()
        gi = self.kin.grip[side]
        n = int(np.ceil(abs(value - start[gi]) / _GRIP_STEP)) or 1
        for i in range(1, n + 1):
            nxt = start.copy()
            nxt[gi] = start[gi] + (value - start[gi]) * i / n
            self.queue.append(nxt)

    # planning (robot kinematics only; targets are fingertip-centre "TCP" points) --------
    def _cart_plan(
        self, side: str, start_q: _Arr, goal_tcp: _Arr, speed: float = _CART_SPEED
    ) -> tuple[list[_Arr], float]:
        p0, _ = self.kin.tcp(start_q, side)
        n = max(1, int(np.ceil(np.linalg.norm(goal_tcp - p0) / (speed / 30.0))))
        q = start_q.copy()
        out, worst = [], 0.0
        for i in range(1, n + 1):
            q, pe, _ye, _tilt, _ok = self.kin.ik_tcp(
                q,
                side,
                p0 + (goal_tcp - p0) * i / n,
                self.yaw,
                iters=60,
                w_up=_W_UP,
                max_tilt_deg=_MAX_TILT,
            )
            worst = max(worst, pe)
            out.append(q.copy())
        return out, worst

    def _cart_move(self, side: str, goal_tcp: _Arr, speed: float = _CART_SPEED) -> float:
        """Queue a straight TCP line at the fixed closing yaw; returns the worst IK error on it."""
        start_q = self.queue[-1] if self.queue else self.des
        line, worst = self._cart_plan(side, start_q, goal_tcp, speed)
        self.queue.extend(line)
        return worst

    def _lift_move(self, side: str, dz: float, speed: float) -> float:
        """Queue a straight vertical line holding the measured hand orientation exactly.

        Re-solving at the nominal closing yaw/tilt would rotate the hand back after the
        squeeze rotated it, with the payload still at the support. Full-orientation IK
        keeps the payload rigid in the hand.
        """
        start_q = self.queue[-1] if self.queue else self.des
        p0, r0 = self.kin.fk(start_q, side)
        n = max(1, int(np.ceil(dz / (speed / 30.0))))
        q, worst, out = start_q.copy(), 0.0, []
        for i in range(1, n + 1):
            q, pe, _re = self.kin.ik(
                q, side, p0 + np.array([0, 0, dz * i / n]), r0, iters=60, full=True
            )
            worst = max(worst, pe)
            out.append(q.copy())
        self.queue.extend(out)
        return worst

    def _solve(
        self, side: str, tcp: _Arr, yaw: float, near: _Arr | None = None
    ) -> tuple[_Arr, float] | None:
        """TCP at ``tcp``, jaws closing along ``yaw`` (or yaw+pi), tilt <= _MAX_TILT; best of many seeds."""
        k = self.kin
        arm = k.arm[side]
        base = (self.queue[-1] if self.queue else self.des) if near is None else near
        rng = np.random.default_rng(11)
        seeds = [base]
        for _ in range(24):
            sd = base.copy()
            sd[arm] = rng.uniform(k.lo[arm] * 0.9, k.hi[arm] * 0.9)
            seeds.append(sd)
        best = None
        for sd in seeds:
            for yy in (yaw, yaw + np.pi):
                q, _pe, _ye, tilt, ok = k.ik_tcp(
                    sd, side, tcp, yy, iters=120, w_up=_W_UP, max_tilt_deg=_MAX_TILT
                )
                if ok:
                    cost = float(np.linalg.norm((q - base)[arm])) + 0.01 * tilt
                    if best is None or cost < best[0]:
                        best = (cost, q, yy)
        return None if best is None else (best[1], float(best[2]))

    def _plan_pick(self, side: str, tgt: dict[str, Any], table_top: float) -> dict[str, Any] | None:
        """Grasp pose + its approach line, closing along the footprint's narrow axis or a few
        degrees off it (the jaws open well past the target, so +-12 deg buys reachability)."""
        for off_deg in (0.0, 6.0, -6.0, 12.0, -12.0):
            plan = self._plan_pick_at(side, tgt, table_top, tgt["minor_yaw"] + np.radians(off_deg))
            if plan is not None:
                return plan
        return None

    def _plan_pick_at(
        self, side: str, tgt: dict[str, Any], table_top: float, closing_yaw: float
    ) -> dict[str, Any] | None:
        """Grasp pose + its approach line along the fingers, all IK-checked, or None.

        The hand waits straight over the target (``hover``), not on the approach line:
        that line runs along the head camera's ray to the target, so a hand on it hides
        the target's near face and the region never reaches the support.
        """
        k = self.kin
        grasp = np.array([*tgt["xy"], table_top + _TIP_CLEAR])
        start = (self.queue[-1] if self.queue else self.des).copy()
        sol = self._solve(side, grasp, closing_yaw, near=start)
        if sol is None:
            return None
        q_g, yaw = sol
        _, r = k.tcp(q_g, side)
        approach = -r[:, 2]  # the fingers point along -z_ee
        pre = grasp - approach * _PREGRASP
        hover = grasp - approach * _HOVER
        look = np.array([grasp[0], grasp[1], tgt["top"] + _LOOK_UP])
        if self._solve(side, look, yaw, near=q_g) is not None:
            hover = look
        sol_pre = self._solve(side, pre, yaw, near=q_g)
        if sol_pre is None:
            return None
        q_pre = sol_pre[0]
        saved, self.yaw = self.yaw, yaw
        _line, worst = self._cart_plan(side, q_pre, grasp, speed=0.05)
        self.yaw = saved
        if worst > 0.005:
            return None
        return {
            "grasp": grasp,
            "pre": pre,
            "hover": hover,
            "above": hover,
            "q_pre": q_pre,
            "approach": approach,
            "yaw": yaw,
        }

    def _replan_from_region(self, side: str, region: dict[str, Any]) -> bool:
        """Re-aim the jaws along the accepted region's minor horizontal axis, when it is trustworthy.

        Only a region in the base frame that covers the whole scanned footprint, and only
        past ``_REPLAN_MIN_DEG``. True when the hand is now rotating to the new hover (the
        jaws are open, so this is a small joint move).
        """
        want = region["minor_yaw"]
        d = (want - self.yaw + np.pi / 2) % np.pi - np.pi / 2  # jaws are symmetric: mod pi
        covers = region_covers_footprint(region, self.target["foot"], self.scene["res"])
        if (
            region["frame"] not in ("openarm_base", "base_link", "base")
            or not covers
            or abs(d) < np.radians(_REPLAN_MIN_DEG)
        ):
            _log(
                "region_yaw_kept",
                side=side,
                closing_yaw=self.yaw,
                region_minor_yaw=want,
                diff_deg=np.degrees(d),
                covers_footprint=covers,
                frame=region["frame"],
            )
            return False
        # Hover over the scanned top, not the region's: the published region is bloated by
        # the grasp-target margin, and a hover that much higher never re-arms the approach.
        tgt = {"xy": np.asarray(region["xyz"][:2]), "minor_yaw": want, "top": self.target_top}
        plan = self._plan_pick_at(side, tgt, self.scene["table_top"], want)
        sol = self._solve(side, plan["hover"], plan["yaw"], near=self.des) if plan else None
        if plan is None or sol is None:
            _log(
                "region_yaw_unreachable",
                side=side,
                closing_yaw=self.yaw,
                region_minor_yaw=want,
                diff_deg=np.degrees(d),
            )
            return False
        _log(
            "region_yaw_replan",
            side=side,
            old_yaw=self.yaw,
            new_yaw=plan["yaw"],
            region_minor_yaw=want,
            region_xy=region["xyz"][:2],
            old_grasp=self.plan["grasp"],
            new_grasp=plan["grasp"],
        )
        self.plan, self.yaw = plan, plan["yaw"]
        self._joint_move(sol[0])
        self._set_phase("hover", tcp_goal=plan["hover"], replan="region_minor_axis")
        return True

    def _choose_target(self, scene: dict[str, Any]) -> tuple[Any, Any, Any]:
        """(object, hand, plan): the measured cluster/hand pair with the hand closest to it.

        Whichever hand is nearer the object picks it; a pair whose grasp the IK cannot
        reach falls through to the next nearest.
        """
        cands = []
        for o in scene["objects"]:
            if any(np.linalg.norm(o["xy"] - p) < 0.06 for p in self.placed):
                continue
            for side in _HANDS:
                tcp, _ = self.kin.tcp(self.des, side)
                cands.append((float(np.linalg.norm(o["xy"] - tcp[:2])), o, side))
        first = os.environ.get("PICK_PLACE_FIRST_SIDE", "")
        for _, o, side in sorted(cands, key=lambda t: (t[2] != first if first else False, t[0]))[
            :4
        ]:
            plan = self._plan_pick(side, o, scene["table_top"])
            _log("target_candidate", side=side, xy=o["xy"], reachable=plan is not None)
            if plan is not None:
                return o, side, plan
        return None, None, None

    def _choose_place(
        self, scene: dict[str, Any], side: str, picked_xy: _Arr, start_q: _Arr
    ) -> Any:
        """A measured free spot on the support surface this hand can carry the payload to."""
        sgn = -1.0 if side == "right" else 1.0
        obj_xy = scene["above"][:, :2] if len(scene["above"]) else np.zeros((0, 2))
        table_xy = scene["table"][:, :2]
        p_lift, _ = self.kin.tcp(start_q, side)
        cands = []
        for x in np.arange(0.18, 0.41, 0.02):
            for y in np.arange(0.04, 0.40, 0.02):
                xy = np.array([x, sgn * y])
                if np.linalg.norm(xy - picked_xy) < 0.12:
                    continue
                clear = float(np.min(np.linalg.norm(obj_xy - xy, axis=1))) if len(obj_xy) else 1.0
                if clear < 0.12:
                    continue
                support = int(np.sum(np.linalg.norm(table_xy - xy, axis=1) < 0.04))
                if support < 12:
                    continue
                score = clear - 0.5 * float(np.linalg.norm(xy - picked_xy))
                cands.append((score, xy, clear, support))
        for cand in sorted(cands, key=lambda c: -c[0])[:8]:
            xy = cand[1]
            over = np.array([xy[0], xy[1], p_lift[2]])
            down = np.array([xy[0], xy[1], scene["table_top"] + _TIP_CLEAR + _PLACE_EXTRA])
            l1, w1 = self._cart_plan(side, start_q, over, speed=0.05)
            _l2, w2 = self._cart_plan(side, l1[-1], down, speed=0.03)
            if max(w1, w2) < 0.005:
                return cand
        return None

    # state machine ---------------------------------------------------------------------
    def _advance(self, q: _Arr) -> None:  # noqa: PLR0911
        perc = self.perc
        assert perc is not None  # step() builds it before the first _advance
        side = self.side
        k = self.kin
        if self.phase == "init":
            if perc.lattice() is None:
                return
            self._set_phase("scan")
        elif self.phase == "scan":
            if self._elapsed() < 1.0:
                return  # let the map refresh
            lat = perc.lattice()
            if lat is None:
                return
            self.scene = scene_from_lattice(lat)
            if not self.scene:
                self._set_phase("hold", reason="the voxel map has no support in the workspace")
                return
            _log(
                "scene",
                table_top=self.scene["table_top"],
                objects=[
                    {k2: o[k2] for k2 in ("xy", "top", "bottom", "n", "minor_yaw", "lo", "hi")}
                    for o in self.scene["objects"]
                ],
            )
            tgt, side, plan = self._choose_target(self.scene)
            if tgt is None:
                self._set_phase("hold", reason="no graspable target in the voxel map")
                return
            self.target, self.side, self.plan, self.yaw = tgt, side, plan, plan["yaw"]
            arm = k.arm[side]
            # Park every other hand out to the side at shoulder height: an idle hand
            # left near the support is "approaching" too, and the producer arms no
            # hand while two approach at once. The active arm unfolds through the
            # same sideways pose (shoulder roll), clear of what is in front of it.
            park = self.des.copy()
            for other in _HANDS:
                if other == side:
                    continue
                oa = k.arm[other]
                park[oa] = 0.0
                park[oa[1]] = 1.5 if other == "right" else -1.5
            if float(np.max(np.abs(self.des[arm]))) < 0.3:
                park[arm[1]] = 1.5 if side == "right" else -1.5
                park[k.grip[side]] = k.open[side]
            if float(np.max(np.abs(park - self.des))) > 1e-3:
                # Away from the support first (straight up for a hand over it), then sideways.
                for other in _HANDS:
                    if other != side and float(np.max(np.abs(self.des[k.arm[other]]))) > 0.3:
                        p_o, r_o = k.tcp(self.des, other)
                        saved_yaw = self.yaw
                        self.yaw = float(np.arctan2(r_o[1, 1], r_o[0, 1]))  # its own closing yaw
                        self._cart_move(other, p_o + np.array([0.0, 0.0, 0.10]), speed=0.05)
                        self.yaw = saved_yaw
                self._joint_move(park)
            goal = (self.queue[-1] if self.queue else self.des).copy()
            goal[arm] = plan["q_pre"][arm]
            goal[k.grip[side]] = k.open[side]
            self._joint_move(goal)
            self.target_top = tgt["top"]  # the scanned cluster's top: a re-aimed hover stands on it
            _log(
                "target",
                side=side,
                xy=tgt["xy"],
                top=tgt["top"],
                n=tgt["n"],
                closing_yaw=self.yaw,
                grasp_tcp=plan["grasp"],
                pregrasp_tcp=plan["pre"],
                approach=plan["approach"],
            )
            self.region_yaw_done = False
            self._set_phase("pregrasp")
        elif self.phase == "pregrasp":
            if self._settled(q):
                # Wait at hover — over the target, its approach box reaching the support —
                # for a fresh grasp region, then descend in one go (inside its life).
                worst = self._cart_move(side, self.plan["hover"], speed=0.03)
                self._set_phase("hover", tcp_goal=self.plan["hover"], worst_ik_err=worst)
            elif self._elapsed() > 40:
                self._set_phase("hold", reason="pregrasp not reached", q=q, cmd=self.des)
        elif self.phase == "hover":
            if not self._settled(q):
                return
            region = perc.grasp_region(side)
            if region is not None and region["age_s"] > _REGION_FRESH_S:
                region = None  # wait for a fresh measurement: the descent must fit its life
            if region is not None and not self.region_yaw_done:
                self.region_yaw_done = True
                if self._replan_from_region(side, region):
                    return  # rotating over the target; the next settled hover re-checks the region
            if region is not None or self._elapsed() > _HOVER_WAIT_S:
                # The driver never vetoes contact itself: with no measured region the
                # kernel is what allows or stops the fingers entering the target's cells.
                _log(
                    "grasp_region_seen" if region is not None else "descend_without_region",
                    side=side,
                    region=region,
                )
                if region is not None:
                    # The measured region starts above the support layer, which is never
                    # exempt: keep the fingers _GRASP_CLEAR above the region's bottom by
                    # backing the grasp point up its own approach line.
                    floor = region["xyz"][2] - region["half"][2] + _GRASP_CLEAR
                    g, a = self.plan["grasp"], self.plan["approach"]
                    gxy = grasp_centre(
                        np.asarray(region["xyz"][:2]), np.asarray(self.target["xy"]), self.yaw
                    )
                    _log(
                        "grasp_centre",
                        side=side,
                        region_xy=region["xyz"][:2],
                        cluster_xy=self.target["xy"],
                        grasp_xy=gxy,
                    )
                    g = np.array([gxy[0], gxy[1], g[2]])
                    self.plan["grasp"] = g
                    if g[2] < floor and a[2] < -0.1:
                        self.plan["grasp"] = g - a * ((floor - g[2]) / -a[2])
                        _log(
                            "grasp_raised",
                            side=side,
                            old=g,
                            new=self.plan["grasp"],
                            region_floor=floor - _GRASP_CLEAR,
                        )
                self._cart_move(side, self.plan["above"], speed=_DESCEND_SPEED)
                worst = self._cart_move(side, self.plan["grasp"], speed=_DESCEND_SPEED)
                self._jaw_dev_logged = 0.0
                self._set_phase(
                    "descend",
                    tcp_goal=self.plan["grasp"],
                    worst_ik_err=worst,
                    region_measured=region is not None,
                )
        elif self.phase == "descend":
            gi = k.grip[side]
            dev = abs(float(q[gi] - self.des[gi]))
            if dev > 0.02 and dev > self._jaw_dev_logged + 0.05:
                self._jaw_dev_logged = dev
                _log(
                    "jaw_pushed",
                    side=side,
                    dev=dev,
                    gripper=q[gi],
                    cmd=self.des[gi],
                    queue=len(self.queue),
                    tcp_goal=self.plan["grasp"],
                )
            if self._settled(q):
                # Close from the measured pose with no integrated bias (see the bias update).
                arm = k.arm[side]
                self.des[arm] = q[arm]
                self.bias[arm] = 0.0
                self._grip(side, 0.0)
                self._log_pose("close_pose", side, q)
                self._set_phase("close")
        elif self.phase == "close":
            held = perc.held(side)
            if held:
                _log(
                    "attached_seen",
                    held=held,
                    gripper=q[k.grip[side]],
                    all=perc.any_attached(),
                )
                # Re-anchor the arm's command on its measured pose: pressed on the target,
                # position control leaves a standing error, and the kernel checks the
                # payload where the command puts it. The jaw keeps its closed command.
                arm = k.arm[side]
                gap = float(np.max(np.abs(self.des[arm] - q[arm])))
                self.queue.clear()
                self.des[arm] = q[arm]
                self.bias[arm] = 0.0
                _log("reanchored", side=side, max_joint_gap=gap)
                self._log_pose("attach_pose", side, q)
                self._set_phase("lift_wait")
            elif self._elapsed() > 25:
                self._set_phase(
                    "hold", reason="no attach from the vision leg", gripper=q[k.grip[side]]
                )
        elif self.phase == "lift_wait":
            if self._elapsed() > 0.5:
                worst = self._lift_move(side, _LIFT, speed=0.04)
                self._set_phase("lift", worst_ik_err=worst)
        elif self.phase == "lift":
            if not perc.held(side) and not self.lost_logged:
                self.lost_logged = True
                _log("lost_attachment_in_lift", all=perc.any_attached())
            if self._settled(q):
                self.lost_logged = False
                lat = perc.lattice()
                if lat is None:
                    return
                scene = scene_from_lattice(lat)
                choice = (
                    self._choose_place(scene, side, self.target["xy"], self.des) if scene else None
                )
                if choice is None:
                    self._set_phase("hold", reason="no reachable free measured surface spot")
                    return
                _, xy, clear, support = choice
                self.place = xy
                p, _ = k.tcp(self.des, side)
                worst = self._cart_move(side, np.array([xy[0], xy[1], p[2]]), speed=0.05)
                self.place_tcp = np.array(
                    [xy[0], xy[1], scene["table_top"] + _TIP_CLEAR + _PLACE_EXTRA]
                )
                _log(
                    "place_spot",
                    xy=xy,
                    clearance=clear,
                    support_cells=support,
                    table_top=scene["table_top"],
                    place_tcp=self.place_tcp,
                    worst_ik_err=worst,
                )
                self._set_phase("transport")
        elif self.phase == "transport":
            if self._settled(q):
                worst = self._cart_move(side, self.place_tcp, speed=0.03)
                self._set_phase("lower", tcp_goal=self.place_tcp, worst_ik_err=worst)
        elif self.phase == "lower":
            if self._settled(q):
                self._grip(side, k.open[side])
                self._set_phase("open")
        elif self.phase == "open":
            if self._settled(q) and self._elapsed() > 1.0:
                _log(
                    "after_open",
                    held=perc.held(side),
                    all=perc.any_attached(),
                    gripper=q[k.grip[side]],
                )
                p, r = k.tcp(self.des, side)
                self._cart_move(side, p + r[:, 2] * _PREGRASP + np.array([0, 0, 0.04]), speed=0.03)
                self._set_phase("retreat")
        elif self.phase == "retreat":
            if self._settled(q):
                _log("after_retreat", held=perc.held(side), all=perc.any_attached())
                self.placed.append(self.place)
                self.picks_done += 1
                if self.picks_done >= _N_PICKS:
                    self._set_phase("done")
                else:
                    self._set_phase("scan")
        # hold / done: keep the last command


def register() -> None:
    """Register the driver as policy family ``isaac_pick_place_scripted`` (idempotent)."""
    from openral_sim.registry import POLICIES

    if "isaac_pick_place_scripted" in POLICIES:
        return

    @POLICIES.register("isaac_pick_place_scripted", install_groups=(), required_imports=())
    def _build(env_cfg: Any) -> Driver:
        return Driver(env_cfg)
