"""Force/contact layer (#373) feasibility tools on the MuJoCo OpenArm v2 right arm.

Both score what ``/joint_states.effort`` can resolve: the sim's ``qfrc_actuator`` (exactly
what the MuJoCo HALs publish, ``openral_hal._mujoco_arm.joint_actuator_effort``) plus a model.
Analysis only — not on any control path.

``probe`` — an offline momentum observer over free sinusoidal motion, with a known push on
the right hand (``xfrc_applied``) over ``t in [3.0, 3.5)`` s, scored as an EE-equivalent force.

``load-cell`` — the sim stand-in for the real-cell load-cell sweep, in the cell scene
(``cell.xml``): the hand is lowered onto the cell table until it pushes back, then commanded
0-60 mm below so the position actuators press it in. Per depth the "load cell" (MuJoCo's
arm-on-table contact force, averaged over the hold) is compared with the static estimate
``pinv(J_tcp^T) (tau - g(q))``. ``cond(J)`` flags poses near a singularity or joint limit.

``hw`` cases corrupt the torque as a current-sensed Damiao drive would (``hw_effort``:
unmodelled friction, noise, 12-bit quantisation; ``probe`` adds slow bias drift) and give the
estimator a +10 % mass/inertia model error. Motor ratings come from the OpenArm manifest.

Usage::

    uv run python tools/force_layer_feasibility.py probe --force-n 10
    uv run python tools/force_layer_feasibility.py load-cell --xy 0.40 -0.20
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from openral_core import RobotDescription
from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

_MANIFEST = Path(__file__).resolve().parents[1] / "robots" / "openarm" / "robot.yaml"

_COULOMB_FRAC = 0.02  # unmodelled Coulomb friction, fraction of rated torque
_VISCOUS_FRAC = 0.01  # unmodelled viscous friction, fraction of rated torque per rad/s
_NOISE_FRAC = 0.002  # white current noise (1 sigma), fraction of rated torque
_LSB_COUNTS = 2048  # 12-bit MIT-mode feedback over +-T_MAX
_MODEL_MASS_ERROR = 1.10  # the observer's model is 10 % heavy


def rated_nm() -> NDArray[np.float64]:
    """Right-arm joint 1-7 motor ratings (N*m) from the OpenArm manifest's ``effort_limit``."""
    desc = RobotDescription.from_yaml(str(_MANIFEST))
    by_name = {j.name: j.effort_limit for j in desc.joints}
    limits = [by_name[f"right_joint{i}"] for i in range(1, 8)]
    if any(v is None for v in limits):
        raise ValueError(f"{_MANIFEST}: a right_joint effort_limit is missing")
    return np.array(limits, dtype=np.float64)


@dataclass(frozen=True)
class RightArm:
    """The OpenArm v2 right arm's indices in one MuJoCo model."""

    dofs: NDArray[np.int64]
    qadr: NDArray[np.int64]
    acts: list[int]
    ee: int
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]

    @classmethod
    def resolve(cls, m: mujoco.MjModel) -> RightArm:
        """Resolve joints ``openarm_right_joint1..7``, their actuators and the hand body."""
        joints = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
            for i in range(1, 8)
        ]
        acts = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint{i}_ctrl")
            for i in range(1, 8)
        ]
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_base_link")
        if min(*joints, *acts, ee) < 0:
            raise ValueError("OpenArm v2 MJCF lacks a right-arm joint, actuator or hand body")
        return cls(
            dofs=np.array([m.jnt_dofadr[j] for j in joints]),
            qadr=np.array([m.jnt_qposadr[j] for j in joints]),
            acts=acts,
            ee=ee,
            lo=m.jnt_range[joints, 0].copy(),
            hi=m.jnt_range[joints, 1].copy(),
        )


def heavy_model(path: str) -> mujoco.MjModel:
    """The observer's model: the same MJCF with ``_MODEL_MASS_ERROR`` on mass and inertia."""
    m = mujoco.MjModel.from_xml_path(path)
    m.body_mass[:] *= _MODEL_MASS_ERROR
    m.body_inertia[:] *= _MODEL_MASS_ERROR
    return m


def hw_effort(
    tau: NDArray[np.float64],
    rated: NDArray[np.float64],
    rng: np.random.Generator,
    qd: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Corrupt sim joint torque as a current-sensed drive would.

    Moving (``qd`` given): Coulomb (smoothed sign) + viscous friction. At rest (``qd`` None)
    static friction is indeterminate, so it is drawn anywhere in its band. Both add white
    noise and quantise to ``rated / _LSB_COUNTS``.
    """
    if qd is None:
        friction = rng.uniform(-1.0, 1.0, tau.shape) * _COULOMB_FRAC * rated
    else:
        friction = _COULOMB_FRAC * rated * np.tanh(qd / 0.02) + _VISCOUS_FRAC * rated * qd
    out = tau + friction + rng.normal(0.0, _NOISE_FRAC, tau.shape) * rated
    lsb = rated / _LSB_COUNTS
    return np.asarray(np.round(out / lsb) * lsb, dtype=np.float64)


_CONTACT_S = (3.0, 3.5)
_DURATION_S = 5.0
_OBSERVER_GAIN = 50.0  # 1/s, ~20 ms time constant


def run(force_n: float, *, hw: bool, seed: int = 0) -> NDArray[np.float64]:
    """Simulate one episode; rows are ``(t, |F_est| at the EE in N)``."""
    path = ensure_openarm_v2_mjcf()
    plant = mujoco.MjModel.from_xml_path(path)
    model = heavy_model(path) if hw else mujoco.MjModel.from_xml_path(path)
    pd, od = mujoco.MjData(plant), mujoco.MjData(model)
    arm = RightArm.resolve(plant)
    dofs, ee = arm.dofs, arm.ee
    rated = rated_nm()
    q0 = np.array([0.0, 0.3, 0.0, 1.2, 0.0, 0.0, 0.0])
    amp = np.array([0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.5])
    freq = np.array([0.3, 0.25, 0.35, 0.3, 0.4, 0.45, 0.5])
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0, 0.005, 7) * rated  # slow current-offset drift over the episode
    dt = float(plant.opt.timestep)
    pd.qpos[arm.qadr] = q0
    mujoco.mj_forward(plant, pd)
    r, integ = np.zeros(7), np.zeros(7)
    p0: NDArray[np.float64] | None = None
    m_prev: NDArray[np.float64] | None = None
    full_m = np.zeros((model.nv, model.nv))
    jacp = np.zeros((3, plant.nv))
    rows: list[tuple[float, float]] = []
    for k in range(int(_DURATION_S / dt)):
        t = k * dt
        pd.ctrl[arm.acts] = q0 + amp * np.sin(2 * np.pi * freq * t)
        on = _CONTACT_S[0] <= t < _CONTACT_S[1]
        pd.xfrc_applied[ee, :3] = (0.0, 0.0, -force_n) if on else (0.0, 0.0, 0.0)
        mujoco.mj_step(plant, pd)
        # mj_step leaves qfrc_actuator at the pre-step state; recompute it at the
        # post-step (q, q_dot) so torque and state are one sample, as a driver reads them.
        mujoco.mj_forward(plant, pd)
        qd = pd.qvel[dofs]
        tau = pd.qfrc_actuator[dofs].copy()
        if hw:
            tau = hw_effort(tau, rated, rng, qd) + drift * (t / _DURATION_S)
        od.qpos[:], od.qvel[:] = pd.qpos, pd.qvel
        mujoco.mj_forward(model, od)
        mujoco.mj_fullM(model, full_m, od.qM)
        m = full_m[np.ix_(dofs, dofs)]
        m_dot = np.zeros((7, 7)) if m_prev is None else (m - m_prev) / dt
        m_prev = m
        p = m @ qd
        if p0 is None:
            p0 = p.copy()
        # p_dot = tau + passive - bias + M_dot q_dot + tau_ext  (C^T q_dot = M_dot q_dot - C q_dot)
        integ += (tau + od.qfrc_passive[dofs] - od.qfrc_bias[dofs] + m_dot @ qd + r) * dt
        r = _OBSERVER_GAIN * (p - p0 - integ)
        # xfrc_applied acts at the body's centre of mass: map through the CoM Jacobian.
        mujoco.mj_jacBodyCom(plant, pd, jacp, None, ee)
        f_est = np.linalg.pinv(jacp[:, dofs].T) @ r
        rows.append((t, float(np.linalg.norm(f_est))))
    return np.asarray(rows)


def probe(force_n: float) -> int:
    """Print the free-motion floor and detection result for the ideal and hw-like cases."""
    for hw in (False, True):
        t, fmag = run(force_n, hw=hw).T
        free = (t > 0.5) & ((t < _CONTACT_S[0]) | (t > _CONTACT_S[1] + 0.2))
        on = (t >= _CONTACT_S[0]) & (t < _CONTACT_S[1])
        thr = 1.2 * float(fmag[free].max())
        hit = np.flatnonzero(on & (fmag > thr))
        latency = f"{(t[hit[0]] - _CONTACT_S[0]) * 1e3:.0f} ms" if hit.size else "missed"
        print(
            f"[{'hw-like' if hw else 'ideal'}] F={force_n:.1f} N | "
            f"free-motion max {fmag[free].max():.2f} N -> threshold {thr:.2f} N | "
            f"in-contact median {np.median(fmag[on & (t > _CONTACT_S[0] + 0.1)]):.2f} N | "
            f"detect: {latency}"
        )
    return 0


_DEPTHS_MM = (-10, 0, 2, 5, 10, 15, 20, 30, 40, 60)
_HW_DRAWS = 20


class _Cell:
    """The cell scene with the right arm's indices resolved."""

    def __init__(self) -> None:
        path = os.path.join(os.path.dirname(ensure_openarm_v2_mjcf()), "cell.xml")
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.model_hw = heavy_model(path)
        self.rated = rated_nm()
        m = self.m
        arm = RightArm.resolve(m)
        self.dofs, self.qadr, self.acts, self.ee = arm.dofs, arm.qadr, arm.acts, arm.ee
        self.lo, self.hi = arm.lo, arm.hi
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
        t = tau if rng is None else hw_effort(tau, self.rated, rng)  # at rest: friction band
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
    z, touched = 1.18, False
    while z > 0.95:  # descend 2 mm at a time until the table pushes back
        _, f, _ = cell.hold(cell.ik(np.array([x, y, z]), q_above), 0.4)
        if f > 0.3:
            touched = True
            break
        z -= 0.002
    if not touched:
        print(f"no table contact found descending to z 0.95 m at ({x}, {y})")
        return 1
    if z == 1.18:  # touching before any descent: no free space above the site
        print(f"arm already in contact at the start height at ({x}, {y}); not a table site")
        return 1
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
    """CLI entry: ``probe [--force-n N]`` or ``load-cell [--xy X Y]``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="momentum observer over free motion with a known push")
    p.add_argument("--force-n", type=float, default=10.0, help="applied EE force (N)")
    lc = sub.add_parser("load-cell", help="press the hand into the cell table")
    lc.add_argument("--xy", type=float, nargs=2, default=(0.40, -0.20), metavar=("X", "Y"))
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        return probe(args.force_n)
    return sweep(*args.xy)


if __name__ == "__main__":
    raise SystemExit(main())
