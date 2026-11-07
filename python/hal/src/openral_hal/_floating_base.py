"""Shared floating-base helpers for the humanoid MuJoCo twins (G1, H2).

Two pieces every pinned-humanoid twin needs, kept in ONE place:

* **ADR-0087 kinematic glide** — the free joint is re-pinned upright every
  physics step to an ``(x, y, yaw, z0)`` pose; a ``BODY_TWIST`` action
  Euler-integrates ``(vx, vy, wz)`` into that pose. No balance, no gait: the pin
  is what keeps a gravity-on twin standing without an S0 controller.
* **Software PD over ``motor`` actuators** — Unitree's official MJCFs ship
  torque actuators (``ctrl`` is N·m), so a position-target HAL runs
  ``tau = kp (q* - q) - kd dq`` every step, clamped to ``ctrlrange`` — the
  same split as ``unitree_sdk2`` (torque at the bus, position + Kp/Kd at the
  user API).

Sim-only: nothing here is reachable from a real HAL.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from openral_core import BODY_TWIST_DIM
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import Action

__all__ = [
    "BasePin",
    "apply_software_pd",
    "capture_base_pin",
    "integrate_planar_twist",
    "validate_planar_body_twist",
    "write_base_pin",
]

#: ``(x, y, yaw, z0)`` world pose the free joint is pinned to.
BasePin = tuple[float, float, float, float]

# Twist components a kinematic-glide base does not actuate (linear-z,
# angular-x, angular-y).  Mirrors ``panda_mobile._PLANAR_TWIST_EPS``.
_PLANAR_TWIST_EPS = 1e-9


def capture_base_pin(qpos: Sequence[float]) -> BasePin:
    """Read ``(x, y, yaw, z)`` from a free joint's ``qpos[0:7]`` (``xyz`` + ``wxyz``).

    Example:
        >>> capture_base_pin([1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        (1.0, 2.0, 0.0, 0.8)
    """
    qw, qx, qy, qz = (float(qpos[i]) for i in range(3, 7))
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return (float(qpos[0]), float(qpos[1]), yaw, float(qpos[2]))


def integrate_planar_twist(pin: BasePin, vx: float, vy: float, wz: float, dt: float) -> BasePin:
    """Euler-integrate a body-frame planar twist into ``pin`` over ``dt`` seconds.

    The body-frame ``(vx, vy)`` is rotated into world by the current yaw (the
    ``PandaMobileHAL`` convention); yaw is wrapped to ``[-pi, pi)``.

    Example:
        >>> integrate_planar_twist((0.0, 0.0, 0.0, 0.8), 1.0, 0.0, 0.0, 0.5)
        (0.5, 0.0, 0.0, 0.8)
    """
    x, y, yaw, z0 = pin
    cy, sy = math.cos(yaw), math.sin(yaw)
    x += (cy * vx - sy * vy) * dt
    y += (sy * vx + cy * vy) * dt
    yaw = (yaw + wz * dt + math.pi) % (2.0 * math.pi) - math.pi
    return (x, y, yaw, z0)


def write_base_pin(data: Any, pin: BasePin) -> None:  # noqa: ANN401  # reason: mujoco.MjData
    """Write the upright pinned pose into ``data.qpos[0:7]`` and zero ``qvel[0:6]``.

    Roll and pitch are clamped to zero (upright); x / y / yaw / z come from ``pin``.
    """
    x, y, yaw, z0 = pin
    half = 0.5 * yaw
    data.qpos[0:7] = (x, y, z0, math.cos(half), 0.0, 0.0, math.sin(half))
    data.qvel[0:6] = 0.0


def validate_planar_body_twist(action: Action, hal_name: str) -> tuple[float, float, float]:
    """Return ``(vx, vy, wz)`` of a ``BODY_TWIST`` action, refusing non-planar rows.

    Raises:
        ROSConfigError: Empty ``body_twist``, a row of the wrong width, or a
            non-zero linear-z / angular-x / angular-y component (the glide base
            is planar).
    """
    if not action.body_twist:
        raise ROSConfigError(f"{hal_name}.send_action: empty Action.body_twist for BODY_TWIST.")
    row = action.body_twist[0]
    if len(row) != BODY_TWIST_DIM:
        raise ROSConfigError(
            f"{hal_name}: BODY_TWIST expects {BODY_TWIST_DIM} floats "
            f"(vx, vy, vz, wx, wy, wz); got {len(row)}."
        )
    if any(abs(row[i]) > _PLANAR_TWIST_EPS for i in (2, 3, 4)):
        raise ROSConfigError(
            f"{hal_name}: BODY_TWIST row carries non-zero linear-z / "
            "angular-x / angular-y; the glide base is planar — only "
            "vx, vy, wz are actuated."
        )
    return (float(row[0]), float(row[1]), float(row[5]))


def apply_software_pd(
    model: Any,  # noqa: ANN401  # reason: mujoco.MjModel
    data: Any,  # noqa: ANN401  # reason: mujoco.MjData
    targets: Mapping[str, float],
    gains: Mapping[str, tuple[float, float]],
    actuator_index: Mapping[str, int],
    qpos_addr: Mapping[str, int],
    qvel_addr: Mapping[str, int],
) -> None:
    """Write ``tau = kp (target - q) - kd dq``, clamped to ``ctrlrange``, into ``ctrl``.

    One call per ``mj_step``: the torque is recomputed from the current state so
    a torque-mode (``motor``) actuator tracks a *position* target. Joints with
    no actuator are skipped.
    """
    for name, target in targets.items():
        act_idx = actuator_index.get(name)
        if act_idx is None:
            continue
        kp, kd = gains[name]
        q = float(data.qpos[qpos_addr[name]])
        dq = float(data.qvel[qvel_addr[name]])
        tau = kp * (float(target) - q) - kd * dq
        low, high = model.actuator_ctrlrange[act_idx]
        data.ctrl[act_idx] = max(float(low), min(float(high), tau))
