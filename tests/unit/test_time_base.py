"""``openral_core.time_base``: one sim step is one control period (issues #355, #358).

Pure helpers, exercised directly plus a hypothesis sweep over the rate x dt
space; the backends that call them are covered by the sim tier.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import TIME_BASE_TOL, ROSConfigError, check_time_base, physics_steps_per_tick


@pytest.mark.parametrize(
    ("rate", "dt", "steps", "dt_out"),
    [
        (20.0, 0.002, 25, 0.002),  # robosuite native: 1/20 s is 25 steps, dt kept
        (50.0, 0.002, 10, 0.002),  # ALOHA bimanual
        (60.0, 1 / 60, 1, 1 / 60),  # Isaac at its own rate
        (30.0, 1 / 60, 2, 1 / 60),  # Isaac 30 Hz on 60 Hz physics
        (30.0, 0.002, 17, 1 / 510),  # MuJoCo default at 30 Hz: 16.67 -> 17 steps, dt shrunk
        (25.0, 0.002, 20, 0.002),  # ALOHA AgileX
    ],
)
def test_steps_per_tick_is_exact_and_never_coarser(
    rate: float, dt: float, steps: int, dt_out: float
) -> None:
    got_steps, got_dt = physics_steps_per_tick(rate, dt)
    assert got_steps == steps
    assert got_dt == pytest.approx(dt_out, rel=1e-12)
    assert got_steps * got_dt == pytest.approx(1.0 / rate, rel=1e-12)
    assert got_dt <= dt + 1e-15


@given(
    rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    dt=st.floats(min_value=1e-4, max_value=0.05, allow_nan=False),
)
def test_steps_per_tick_invariants(rate: float, dt: float) -> None:
    steps, out = physics_steps_per_tick(rate, dt)
    assert steps >= 1
    assert out <= dt * (1 + 1e-12)  # shrunk, never grown
    assert steps * out == pytest.approx(1.0 / rate, rel=1e-9)
    check_time_base(steps * out, rate, backend="t", robot="r")  # the pair always passes the guard


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_steps_per_tick_refuses_bad_inputs(bad: float) -> None:
    with pytest.raises(ROSConfigError):
        physics_steps_per_tick(bad, 0.002)
    with pytest.raises(ROSConfigError):
        physics_steps_per_tick(30.0, bad)


def test_check_time_base_accepts_one_control_period_within_tolerance() -> None:
    check_time_base(1 / 30, 30.0, backend="b", robot="r")
    check_time_base((1 / 30) * (1 + TIME_BASE_TOL * 0.9), 30.0, backend="b", robot="r")


def test_check_time_base_refuses_the_1p5x_and_2x_symptoms_and_nan() -> None:
    with pytest.raises(ROSConfigError, match=r"20 Hz|0\.050000"):
        check_time_base(0.05, 20.0 * 1.5, backend="b", robot="r")  # RoboCasa 20 Hz under 30 Hz
    with pytest.raises(ROSConfigError, match=r"30 Hz"):
        check_time_base(1 / 60, 30.0, backend="b", robot="r")  # Isaac 1/60 under 30 Hz
    with pytest.raises(ROSConfigError):
        check_time_base(math.nan, 30.0, backend="b", robot="r")


def test_check_time_base_refuses_a_manifest_without_a_rate() -> None:
    with pytest.raises(ROSConfigError, match=r"declares no action_spec\.control_freq_hz"):
        check_time_base(1 / 30, None, backend="b", robot="r")
