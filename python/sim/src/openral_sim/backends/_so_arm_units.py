"""Shared unit + cadence conversions for the raw-MuJoCo SO-ARM bench scenes.

The native SO-101 scenes (``so101_eraser``, ``so101_box``) drive the same
upstream MJCF from the same LeRobot-trained checkpoint convention, so the two
conversions live here once:

* **Control cadence** — one policy action covers a control PERIOD of physics
  (``round(1 / (control_hz * timestep))`` ``mj_step`` calls), not a single
  2 ms tick. A checkpoint recorded at 30 FPS issues absolute joint targets
  that each assume ~33 ms of travel; stepping one tick per action gives the
  position actuators 1/17th of that, proprio never progresses, and the policy
  re-issues near-home commands forever.
* **LeRobot degrees mode** — the five arm channels are servo degrees bridged
  by the per-joint calibration affine (``lerobot_deg = signs * mujoco_deg +
  offsets``), but the gripper channel is NOT degrees: LeRobot SO-ARM datasets
  store it normalised ``[0, 100]`` over the jaw's calibrated travel. Mapping
  it through ``radians()`` like the arm joints lands ~10° open at the closed
  end (the MJCF jaw range is [-10°, +100°]) — exactly the grasp-critical
  regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError


def steps_per_control_period(timestep_s: float, control_hz: float, *, scene: str) -> int:
    """Physics steps that make up one control period.

    Args:
        timestep_s: The compiled model's ``opt.timestep``, seconds.
        control_hz: Rate the policy's actions are issued at, Hz.
        scene: Scene id, for the error message.

    Returns:
        ``round(1 / (control_hz * timestep_s))``, at least 1.

    Raises:
        ROSConfigError: If ``control_hz`` is not positive.
    """
    if control_hz <= 0.0:
        raise ROSConfigError(f"{scene}: control_hz must be > 0; got {control_hz}.")
    return max(1, round(1.0 / (control_hz * float(timestep_s))))


def lerobot_action_to_radians(
    action: NDArray[np.float64],
    *,
    joint_signs: NDArray[np.float64],
    joint_offsets_deg: NDArray[np.float64],
    gripper_range: tuple[float, float],
) -> NDArray[np.float64]:
    """LeRobot degrees-mode action → MuJoCo radian ctrl targets.

    Arm channels invert the calibration affine then convert to radians; the
    last (gripper) channel maps its ``[0, 100]`` fraction onto the jaw's
    radian range — the same ``[0, 1]``-style surface the deploy HAL exposes.
    """
    cmd: NDArray[np.float64] = np.radians(joint_signs * (action - joint_offsets_deg))
    g_lo, g_hi = gripper_range
    cmd[-1] = g_lo + np.clip(action[-1] / 100.0, 0.0, 1.0) * (g_hi - g_lo)
    return cmd


def radians_to_lerobot_state(
    qpos: NDArray[np.float64],
    *,
    joint_signs: NDArray[np.float64],
    joint_offsets_deg: NDArray[np.float64],
    gripper_range: tuple[float, float],
) -> NDArray[np.float64]:
    """MuJoCo radian qpos → LeRobot degrees-mode proprio (inverse of the action map).

    ``degrees(qpos)`` on the gripper would report the manifest home
    (lerobot ~2) as -7.9 — below the checkpoint normalizer's observed minimum.
    """
    state: NDArray[np.float64] = joint_signs * np.degrees(qpos) + joint_offsets_deg
    g_lo, g_hi = gripper_range
    state[-1] = (qpos[-1] - g_lo) / (g_hi - g_lo) * 100.0
    return state
