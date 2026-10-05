"""Kinematics-only IK for the scripted driver, on the OpenArm's own MJCF (``openarm_base`` frame).

Robot model only, no scene state. ``tcp``/``ik_tcp`` target the closed fingertip centre
with the jaws' closing axis held horizontal along a yaw, tilting about it allowed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

_Arr = NDArray[np.float64]

_TIP = 0.165  # ee_base_link origin -> closed fingertip, metres (from the robot MJCF)
_URDF = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "openarm.urdf"


class Kin:
    def __init__(self, description: Any) -> None:
        from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

        self.m = mujoco.MjModel.from_xml_path(ensure_openarm_v2_mjcf())
        self.d = mujoco.MjData(self.m)
        jt = mujoco.mjtObj.mjOBJ_JOINT
        ids = [mujoco.mj_name2id(self.m, jt, j.sim_joint_name) for j in description.joints]
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([self.m.jnt_dofadr[i] for i in ids])
        self.lo = np.array([j.position_limits[0] for j in description.joints])
        self.hi = np.array([j.position_limits[1] for j in description.joints])
        # Intersect the arm joints with the URDF the Isaac scene imports (joint7: URDF
        # +-0.785 vs manifest +-1.571): a target outside it is clamped by the simulator.
        lim = {
            j.get("name"): j.find("limit")
            for j in ET.parse(_URDF).getroot().iter("joint")
            if j.get("type")
        }
        for i, j in enumerate(description.joints):
            el = lim.get(j.sim_joint_name)
            if el is not None and j.role != "gripper":
                self.lo[i] = max(self.lo[i], float(el.attrib["lower"]))
                self.hi[i] = min(self.hi[i], float(el.attrib["upper"]))
        self.ee = {
            s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{s}_ee_base_link")
            for s in ("left", "right")
        }
        self.arm = {"left": list(range(7)), "right": list(range(8, 15))}
        self.grip = {"left": 7, "right": 15}
        # Fully open (manifest units): the outer finger's shadow on the target, seen from
        # the head camera while the hand waits over it, is the strip the region loses.
        self.open = {"left": 0.7854, "right": -0.7854}

    def _set(self, q: _Arr) -> None:
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def fk(self, q: _Arr, side: str) -> tuple[_Arr, _Arr]:
        self._set(q)
        b = self.ee[side]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def tcp(self, q: _Arr, side: str) -> tuple[_Arr, _Arr]:
        p, r = self.fk(q, side)
        return p + r @ np.array([0.0, 0.0, -_TIP]), r

    def ik(
        self,
        q0: _Arr,
        side: str,
        p_tgt: _Arr,
        r_tgt: _Arr,
        iters: int = 200,
        lam: float = 0.02,
        w_rot: float = 0.5,
        full: bool = False,
    ) -> tuple[_Arr, float, float]:
        """Damped least squares on ee_base_link: ``full`` matches all three axes, else only z."""
        q = np.array(q0, float).copy()
        idx = self.arm[side]
        cols = self.dadr[idx]
        b = self.ee[side]
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self._set(q)
            p = self.d.xpos[b]
            r = self.d.xmat[b].reshape(3, 3)
            ep = np.asarray(p_tgt) - p
            er = (
                (0.5 * sum(np.cross(r[:, i], r_tgt[:, i]) for i in range(3)))
                if full
                else np.cross(r[:, 2], r_tgt[:, 2])
            )
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jacBody(self.m, self.d, jp, jr, b)
            jm = np.vstack([jp[:, cols], w_rot * jr[:, cols]])
            e = np.concatenate([ep, w_rot * er])
            dq = jm.T @ np.linalg.solve(jm @ jm.T + lam**2 * np.eye(6), e)
            q[idx] = np.clip(
                q[idx] + np.clip(dq, -0.15, 0.15), self.lo[idx] + 0.02, self.hi[idx] - 0.02
            )
        p, r = self.fk(q, side)
        tilt = float(np.degrees(np.arccos(np.clip(r[:, 2] @ r_tgt[:, 2], -1, 1))))
        return q, float(np.linalg.norm(np.asarray(p_tgt) - p)), tilt

    def ik_tcp(
        self,
        q0: _Arr,
        side: str,
        tcp_target: _Arr,
        yaw: float,
        iters: int = 150,
        lam: float = 0.02,
        w_up: float = 0.15,
        max_tilt_deg: float = 40.0,
    ) -> tuple[_Arr, float, float, float, bool]:
        """TCP at ``tcp_target``, closing axis horizontal along ``yaw`` (2 DOF), gentle pull to vertical."""
        q = np.array(q0, float).copy()
        idx = self.arm[side]
        cols = self.dadr[idx]
        b = self.ee[side]
        y_t = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self._set(q)
            r = self.d.xmat[b].reshape(3, 3)
            pt = self.d.xpos[b] + r @ np.array([0.0, 0.0, -_TIP])
            ep = np.asarray(tcp_target) - pt
            ey = np.cross(r[:, 1], y_t)
            ez = w_up * np.cross(r[:, 2], up)
            if (
                np.linalg.norm(ep) < 2e-4
                and np.linalg.norm(ey) < 2e-3
                and np.linalg.norm(ez) < 1e-3
            ):
                break
            mujoco.mj_jac(self.m, self.d, jp, jr, pt, b)
            jm = np.vstack([jp[:, cols], jr[:, cols], jr[:, cols]])
            e = np.concatenate([ep, ey, ez])
            dq = jm.T @ np.linalg.solve(jm @ jm.T + lam**2 * np.eye(9), e)
            q[idx] = np.clip(
                q[idx] + np.clip(dq, -0.15, 0.15), self.lo[idx] + 0.02, self.hi[idx] - 0.02
            )
        pt, r = self.tcp(q, side)
        tilt = float(np.degrees(np.arccos(np.clip(r[:, 2] @ up, -1, 1))))
        yerr = float(np.degrees(np.arcsin(np.clip(np.linalg.norm(np.cross(r[:, 1], y_t)), 0, 1))))
        perr = float(np.linalg.norm(np.asarray(tcp_target) - pt))
        ok = perr < 0.002 and yerr < 3.0 and tilt <= max_tilt_deg
        return q, perr, yerr, tilt, ok
