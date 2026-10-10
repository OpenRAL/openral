"""The deploy-sim time base: one ``env.step`` / ``send_action`` is one control period.

The skill runner ticks at ``action_spec.control_freq_hz`` and steps the sim once
per tick, so a backend whose step advances the sim by any other interval
traverses every trajectory at the wrong speed and every kernel velocity /
tracking check reads the wrong clock (issues #355, #358). Every sim backend
derives its physics steps per tick here and every sim HAL holds the backend to
the robot's rate here, so the rule lives in exactly one place.
"""

from __future__ import annotations

import math

from openral_core.exceptions import ROSConfigError

#: Tolerance on ``sim_dt_per_tick_s * control_freq_hz == 1`` (1 %).
TIME_BASE_TOL = 0.01


def physics_steps_per_tick(control_freq_hz: float, physics_dt_s: float) -> tuple[int, float]:
    """Physics steps per control tick, and the physics dt that makes them exact.

    The control period must be a whole number of physics steps. When the
    engine's dt already divides it the pair is ``(period / dt, dt)``; otherwise
    the dt is **shrunk** (never grown) to the largest value at or below
    ``physics_dt_s`` that divides the period, so the integration is at least as
    fine as the model's author chose. Shrinking by less than one step is what
    MuJoCo's default 0.002 s needs at 30 Hz (1 / 30 s is 16.67 steps).

    Args:
        control_freq_hz: The robot's ``action_spec.control_freq_hz``.
        physics_dt_s: The engine's physics time step (``MjModel.opt.timestep``,
            robosuite's ``SIMULATION_TIMESTEP``).

    Returns:
        ``(steps, dt_s)`` with ``steps * dt_s == 1 / control_freq_hz``.

    Raises:
        ROSConfigError: a rate or dt that is not finite and positive.

    Example:
        >>> physics_steps_per_tick(20.0, 0.002)
        (25, 0.002)
        >>> steps, dt = physics_steps_per_tick(30.0, 0.002)
        >>> steps, round(dt, 6)
        (17, 0.001961)
    """
    if not (math.isfinite(control_freq_hz) and control_freq_hz > 0):
        raise ROSConfigError(f"control_freq_hz must be finite and > 0, got {control_freq_hz!r}")
    if not (math.isfinite(physics_dt_s) and physics_dt_s > 0):
        raise ROSConfigError(f"physics_dt_s must be finite and > 0, got {physics_dt_s!r}")
    period = 1.0 / control_freq_hz
    exact = period / physics_dt_s
    steps = round(exact) if math.isclose(exact, round(exact), rel_tol=1e-9) else math.ceil(exact)
    return max(1, steps), period / max(1, steps)


def check_time_base(
    sim_dt_per_tick_s: float, control_freq_hz: float | None, *, backend: str, robot: str
) -> None:
    """Refuse a sim backend whose step is not one control period of the robot.

    Args:
        sim_dt_per_tick_s: Sim seconds one step / action advances.
        control_freq_hz: ``RobotDescription.control_rate_hz``; ``None`` when the
            manifest declares no rate.
        backend: Name for the message (the env / HAL class).
        robot: ``RobotDescription.name`` for the message.

    Raises:
        ROSConfigError: the manifest declares no control rate, or the tick
            differs from ``1 / control_freq_hz`` by more than ``TIME_BASE_TOL``
            (a NaN tick is refused too).

    Example:
        >>> check_time_base(1 / 30, 30.0, backend="x", robot="r")
        >>> check_time_base(1 / 60, 30.0, backend="x", robot="r")
        Traceback (most recent call last):
        ...
        openral_core.exceptions.ROSConfigError: ...
    """
    if control_freq_hz is None:
        raise ROSConfigError(
            f"{backend}: steps {sim_dt_per_tick_s:.6f} s of sim time per action, but robot "
            f"{robot!r} declares no action_spec.control_freq_hz to hold it to. Declare the "
            "control rate in the manifest."
        )
    # ``not <=`` so a NaN tick is refused too (NaN compares False both ways).
    if not abs(sim_dt_per_tick_s * control_freq_hz - 1.0) <= TIME_BASE_TOL:
        raise ROSConfigError(
            f"{backend}: steps {sim_dt_per_tick_s:.6f} s of sim time per action, but robot "
            f"{robot!r} ticks at {control_freq_hz:g} Hz ({1.0 / control_freq_hz:.6f} s). One "
            "step must be one control period, or the rehearsal runs at the wrong speed; fix "
            "the backend's physics steps per tick, not the robot's rate."
        )
