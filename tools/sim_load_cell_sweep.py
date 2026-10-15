"""Simulated load-cell sweep in the OpenArm v2 cell (#373 step 3, sim stand-in for the real cell).

The right hand is placed above the cell table at ``(x, y)``, lowered until the table pushes
back, then commanded 0-60 mm below that height so the native position actuators press it into
the table. Per depth, the "load cell" is MuJoCo's contact force between the right arm and
``cell_table_col`` (3-D, averaged over the hold), compared with the static estimate an effort
reader can make: ``F = pinv(J_tcp^T) (tau - g(q))`` from ``qfrc_actuator`` (what
``/joint_states.effort`` carries) and a gravity model. ``hw`` draws Coulomb friction anywhere in
its +-2 %-of-rated band (a resting joint's friction is indeterminate), noise and 12-bit
quantisation, and uses a +10 % mass/inertia model error. ``cond(J)`` flags poses near a
singularity or joint limit, where the estimate is not trustworthy.

Usage: ``uv run python tools/sim_load_cell_sweep.py --xy 0.40 -0.20``
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from numpy.typing import NDArray
from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

_RATED_NM = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])
_DEPTHS_MM = (-10, 0, 2, 5, 10, 15, 20, 30, 40, 60)
_HW_DRAWS = 20


class _Cell:
    """The cell scene with the right arm's indices resolved."""

    def __init__(self) -> None:
        path = os.path.join(os.path.dirname(ensure_openarm_v2_mjcf()), "cell.xml")
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.model_hw = mujoco.MjModel.from_xml_path(path)
        self.model_hw.body_mass[:] *= 1.10
        self.model_hw.body_inertia[:] *= 1.10
        m = self.m
        joints = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
            for i in range(1, 8)
        ]
        self.dofs = np.array([m.jnt_dofadr[j] for j in joints])
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.lo, self.hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        self.acts = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint{i}_ctrl")
            for i in range(1, 8)
        ]
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_base_link")
        self.table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cell_table_col")
        self.arm_bodies = {
            b
            for b in range(m.nbody)
            if "openarm_right" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        }
        self.d.ctrl[0] = self.d.qpos[0]  # hold the lifter where it is

    def ik(self, target: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Damped least-squares position IK for the hand body, clipped to joint limits."""
        k = mujoco.MjData(self.m)
        k.qpos[:] = self.d.qpos
        jac = np.zeros((3, self.m.nv))
        for _ in range(300):
            k.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, k)
            mujoco.mj_comPos(self.m, k)
            err = target - k.xpos[self.ee]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacBody(self.m, k, jac, None, self.ee)
            jr = jac[:, self.dofs]
            step = jr.T @ np.linalg.solve(jr @ jr.T + 1e-4 * np.eye(3), err)
            q = np.clip(q + step, self.lo, self.hi)
        return q

    def table_contact(self) -> tuple[float, NDArray[np.float64] | None]:
        """Arm-on-table contact force magnitude (N) and mean contact point (world)."""
        f6, total, pts = np.zeros(6), np.zeros(3), []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.table not in (c.geom1, c.geom2):
                continue
            other = c.geom1 if c.geom2 == self.table else c.geom2
            if int(self.m.geom_bodyid[other]) not in self.arm_bodies:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            total += c.frame.reshape(3, 3).T @ f6[:3]
            pts.append(c.pos.copy())
        return float(np.linalg.norm(total)), (np.mean(pts, axis=0) if pts else None)

    def hold(
        self, q_target: NDArray[np.float64], settle_s: float
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None]:
        """Command, settle, then average effort and load cell over 200 steps."""
        self.d.ctrl[self.acts] = q_target
        for _ in range(int(settle_s / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        taus, forces, pt = [], [], None
        for _ in range(200):
            mujoco.mj_step(self.m, self.d)
            taus.append(self.d.qfrc_actuator[self.dofs].copy())
            f, p = self.table_contact()
            forces.append(f)
            pt = p if p is not None else pt
        return np.mean(taus, axis=0), float(np.mean(forces)), pt

    def estimate(
        self,
        tau: NDArray[np.float64],
        tcp: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> float:
        """Static EE force estimate at the TCP; ``rng`` set = hardware-like corruption."""
        model = self.m if rng is None else self.model_hw
        k = mujoco.MjData(model)
        k.qpos[:] = self.d.qpos
        mujoco.mj_forward(model, k)
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jac, None, tcp, self.ee)
        t = tau.copy()
        if rng is not None:
            t += rng.uniform(-1.0, 1.0, 7) * 0.02 * _RATED_NM + rng.normal(0, 0.002, 7) * _RATED_NM
            lsb = _RATED_NM / 2048
            t = np.round(t / lsb) * lsb
        return float(
            np.linalg.norm(np.linalg.pinv(jac[:, self.dofs].T) @ (t - k.qfrc_bias[self.dofs]))
        )

    def cond(self, tcp: NDArray[np.float64]) -> float:
        """Condition number of the TCP translational Jacobian over the arm joints."""
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jac, None, tcp, self.ee)
        sv = np.linalg.svd(jac[:, self.dofs], compute_uv=False)
        return float(sv[0] / sv[-1])


def sweep(x: float, y: float) -> int:
    """Run the sweep at hand position ``(x, y)`` and print one row per commanded depth."""
    cell = _Cell()
    q_above = cell.ik(np.array([x, y, 1.18]), np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]))
    cell.d.qpos[cell.qadr] = q_above
    mujoco.mj_forward(cell.m, cell.d)
    cell.hold(q_above, 1.0)
    z = 1.18
    while z > 0.95:  # descend 2 mm at a time until the table pushes back
        _, f, _ = cell.hold(cell.ik(np.array([x, y, z]), q_above), 0.4)
        if f > 0.3:
            break
        z -= 0.002
    z_contact = z + 0.002
    _, _, pt = cell.hold(cell.ik(np.array([x, y, z_contact - 0.01]), q_above), 1.5)
    if pt is None:
        print(f"no table contact reachable at ({x}, {y})")
        return 1
    rot = cell.d.xmat[cell.ee].reshape(3, 3)
    tcp_local = rot.T @ (pt - cell.d.xpos[cell.ee])  # TCP fixed on the hand
    rng = np.random.default_rng(0)
    print(f"site ({x}, {y}): contact at commanded z {z_contact:.3f} m, TCP {tcp_local.round(3)}")
    print(f"{'depth mm':>8s} {'cond(J)':>7s} {'load cell N':>11s} {'ideal N':>8s} {'hw N':>14s}")
    for mm in _DEPTHS_MM:
        tau, f, _ = cell.hold(cell.ik(np.array([x, y, z_contact - mm / 1000]), q_above), 1.5)
        tcp = cell.d.xpos[cell.ee] + cell.d.xmat[cell.ee].reshape(3, 3) @ tcp_local
        hw = [cell.estimate(tau, tcp, rng) for _ in range(_HW_DRAWS)]
        print(
            f"{mm:8d} {cell.cond(tcp):7.1f} {f:11.2f} {cell.estimate(tau, tcp, None):8.2f} "
            f"{np.mean(hw):7.2f} +-{np.std(hw):5.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--xy X Y`` picks the table site (cell base frame, metres)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xy", type=float, nargs=2, default=(0.40, -0.20), metavar=("X", "Y"))
    args = ap.parse_args(argv)
    return sweep(*args.xy)


if __name__ == "__main__":
    raise SystemExit(main())
