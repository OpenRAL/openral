"""Offline momentum-observer feasibility probe on the MuJoCo OpenArm v2 right arm (#373 step 2).

Free sinusoidal motion under the native position actuators; a known external force pushes the
right hand down (``xfrc_applied``) over ``t in [3.0, 3.5)`` s. The observer sees only q, q-dot
and the measured actuator torque (``qfrc_actuator``, what ``/joint_states.effort`` carries) plus
a model (M, bias, damping), and is scored as an end-effector-equivalent force.

``hw`` adds what a current-sensed Damiao drive adds and the sim lacks: unmodelled Coulomb +
viscous friction (2 % / 1 % of rated torque), white noise, slow bias drift, 12-bit quantisation,
and a +10 % mass/inertia error in the observer's model. Analysis only — not on any control path.

Usage: ``uv run python tools/force_observer_probe.py --force-n 10``
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from numpy.typing import NDArray
from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

_RATED_NM = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])  # DM8009 x2, DM4340 x2, DM4310 x3
_CONTACT_S = (3.0, 3.5)
_DURATION_S = 5.0
_OBSERVER_GAIN = 50.0  # 1/s, ~20 ms time constant


def run(force_n: float, *, hw: bool, seed: int = 0) -> NDArray[np.float64]:
    """Simulate one episode; rows are ``(t, |F_est| at the EE in N)``."""
    path = ensure_openarm_v2_mjcf()
    plant, model = mujoco.MjModel.from_xml_path(path), mujoco.MjModel.from_xml_path(path)
    if hw:
        model.body_mass[:] = plant.body_mass * 1.10
        model.body_inertia[:] = plant.body_inertia * 1.10
    pd, od = mujoco.MjData(plant), mujoco.MjData(model)
    joints = [
        mujoco.mj_name2id(plant, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
        for i in range(1, 8)
    ]
    dofs = np.array([plant.jnt_dofadr[j] for j in joints])
    qadr = np.array([plant.jnt_qposadr[j] for j in joints])
    acts = [
        mujoco.mj_name2id(plant, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint{i}_ctrl")
        for i in range(1, 8)
    ]
    ee = mujoco.mj_name2id(plant, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_base_link")
    q0 = np.array([0.0, 0.3, 0.0, 1.2, 0.0, 0.0, 0.0])
    amp = np.array([0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.5])
    freq = np.array([0.3, 0.25, 0.35, 0.3, 0.4, 0.45, 0.5])
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0, 0.005, 7) * _RATED_NM
    dt = float(plant.opt.timestep)
    pd.qpos[qadr] = q0
    mujoco.mj_forward(plant, pd)
    r, integ = np.zeros(7), np.zeros(7)
    p0: NDArray[np.float64] | None = None
    m_prev: NDArray[np.float64] | None = None
    full_m = np.zeros((model.nv, model.nv))
    jacp = np.zeros((3, plant.nv))
    rows: list[tuple[float, float]] = []
    for k in range(int(_DURATION_S / dt)):
        t = k * dt
        pd.ctrl[acts] = q0 + amp * np.sin(2 * np.pi * freq * t)
        on = _CONTACT_S[0] <= t < _CONTACT_S[1]
        pd.xfrc_applied[ee, :3] = (0.0, 0.0, -force_n) if on else (0.0, 0.0, 0.0)
        mujoco.mj_step(plant, pd)
        qd = pd.qvel[dofs]
        tau = pd.qfrc_actuator[dofs].copy()
        if hw:
            tau += 0.02 * _RATED_NM * np.tanh(qd / 0.02) + 0.01 * _RATED_NM * qd
            tau += rng.normal(0.0, 0.002, 7) * _RATED_NM + drift * (t / _DURATION_S)
            lsb = _RATED_NM / 2048
            tau = np.round(tau / lsb) * lsb
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
        mujoco.mj_jacBody(plant, pd, jacp, None, ee)
        f_est = np.linalg.pinv(jacp[:, dofs].T) @ r
        rows.append((t, float(np.linalg.norm(f_est))))
    return np.asarray(rows)


def main(argv: list[str] | None = None) -> int:
    """Print the free-motion floor and detection result for the ideal and hw-like cases."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force-n", type=float, default=10.0, help="applied EE force (N)")
    args = ap.parse_args(argv)
    for hw in (False, True):
        t, fmag = run(args.force_n, hw=hw).T
        free = (t > 0.5) & ((t < _CONTACT_S[0]) | (t > _CONTACT_S[1] + 0.2))
        on = (t >= _CONTACT_S[0]) & (t < _CONTACT_S[1])
        thr = 1.2 * float(fmag[free].max())
        hit = np.flatnonzero(on & (fmag > thr))
        latency = f"{(t[hit[0]] - _CONTACT_S[0]) * 1e3:.0f} ms" if hit.size else "missed"
        print(
            f"[{'hw-like' if hw else 'ideal'}] F={args.force_n:.1f} N | "
            f"free-motion max {fmag[free].max():.2f} N -> threshold {thr:.2f} N | "
            f"in-contact median {np.median(fmag[on & (t > _CONTACT_S[0] + 0.1)]):.2f} N | "
            f"detect: {latency}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
