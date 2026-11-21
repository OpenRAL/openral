"""Shared planar-base helpers for the sim HAL shell and the MuJoCo twins.

* **The planar-twist rule** — ``validate_planar_body_twist`` is the one check of a
  ``BODY_TWIST`` a sim base can execute (a 6-float row, no linear-z / angular-x /
  angular-y, no lateral ``vy`` on a differential base), with one tolerance,
  ``PLANAR_TWIST_EPS``; ``integrate_planar_twist`` is the one Euler integrator of a
  holonomic planar base.
* **ADR-0087 kinematic glide** — the free joint is re-pinned upright every
  physics step to an ``(x, y, yaw, z0)`` pose (``capture_base_pin`` /
  ``write_base_pin``); ``mujoco_twin._GlideTwin`` is the twin that does it.

Sim-only: nothing here is reachable from a real HAL.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from openral_core import BODY_TWIST_DIM
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import Action

__all__ = [
    "PLANAR_TWIST_EPS",
    "BasePin",
    "capture_base_pin",
    "integrate_planar_twist",
    "validate_planar_body_twist",
    "write_base_pin",
]

#: ``(x, y, yaw, z0)`` world pose the free joint is pinned to.
BasePin = tuple[float, float, float, float]

#: Tolerance for a twist component a planar sim base does not actuate (linear-z,
#: angular-x, angular-y; lateral vy on a differential base): float noise, not a
#: command. The value of ``panda_mobile`` and the real Unitree HAL.
PLANAR_TWIST_EPS = 1e-9


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


def validate_planar_body_twist(
    action: Action, backend: str, *, differential: bool = False
) -> tuple[float, float, float]:
    """Return ``(vx, vy, wz)`` of a ``BODY_TWIST`` action, refusing what a planar base cannot do.

    Args:
        action: A ``BODY_TWIST`` action (its first row is the command).
        backend: The HAL class name the messages carry.
        differential: The base is differential (``base_kinematics: differential``):
            a lateral ``vy`` is refused too, and ``vy`` is returned as ``0.0``.

    Raises:
        ROSConfigError: Empty ``body_twist``, a row of the wrong width, a non-zero
            linear-z / angular-x / angular-y component, or (``differential``) a
            non-zero ``vy`` — beyond ``PLANAR_TWIST_EPS``.

    Example:
        >>> twist = Action(control_mode="body_twist", body_twist=[[0.2, 0.0, 0.0, 0.0, 0.0, 0.5]])
        >>> validate_planar_body_twist(twist, "X", differential=True)
        (0.2, 0.0, 0.5)
    """
    if not action.body_twist:
        raise ROSConfigError(f"{backend}.send_action: empty Action.body_twist for BODY_TWIST.")
    row = action.body_twist[0]
    if len(row) != BODY_TWIST_DIM:
        raise ROSConfigError(
            f"{backend}: BODY_TWIST expects {BODY_TWIST_DIM} floats "
            f"(vx, vy, vz, wx, wy, wz); got {len(row)}."
        )
    if any(abs(row[i]) > PLANAR_TWIST_EPS for i in (2, 3, 4)):
        raise ROSConfigError(
            f"{backend}: BODY_TWIST row carries non-zero linear-z / "
            "angular-x / angular-y; only vx, vy, wz are actuated (a holonomic planar base model)."
        )
    if differential:
        if abs(row[1]) > PLANAR_TWIST_EPS:
            raise ROSConfigError(
                f"{backend}: a differential base cannot execute a BODY_TWIST with "
                f"vy={float(row[1])}; only vx and wz are actuated."
            )
        return (float(row[0]), 0.0, float(row[5]))
    return (float(row[0]), float(row[1]), float(row[5]))
