"""The bare MuJoCo twin: one ``send_action`` is one control period (issue #358).

Real ``MujocoArmHAL`` on the real SO-101 manifest and MJCF (no mocks, CLAUDE.md
§1.11). Before this the twin advanced one 0.002 s physics step per action and
relied on the wall-time idle stepper to keep its clock moving.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

_SO101 = "robots/so101_follower/robot.yaml"


def _connected(**kwargs: object):  # type: ignore[no-untyped-def]  # reason: test helper
    from openral_core import RobotDescription
    from openral_hal._mujoco_arm import MujocoArmHAL

    desc = RobotDescription.from_yaml(_SO101)
    assert desc.control_rate_hz == 30.0
    hal = MujocoArmHAL.from_description(desc, **kwargs)  # type: ignore[arg-type]  # reason: kwargs passthrough
    hal.connect()
    return hal


def _hold(hal):  # type: ignore[no-untyped-def]  # reason: test helper
    from openral_core import Action, ControlMode

    return Action(
        control_mode=ControlMode.JOINT_POSITION, joint_targets=[list(hal.read_state().position)]
    )


def test_connect_derives_one_control_period_per_action() -> None:
    hal = _connected()
    try:
        model, data = hal.mujoco_handles()
        # 1/30 s is 16.67 default steps: 17 steps at a dt shrunk to 1/510 s.
        assert hal._settle_steps == 17
        assert float(model.opt.timestep) == pytest.approx(1 / 510, rel=1e-9)
        assert hal.sim_dt_per_tick_s == pytest.approx(1 / 30, rel=1e-9)
        assert hal.clock_authority().timestep_s == pytest.approx(1 / 30, rel=1e-9)
        # The runner's ticks drive the clock now: the idle stepper yields to them.
        assert hal._step_while_active is False
        t0 = float(data.time)
        for _ in range(30):
            hal.send_action(_hold(hal))
        assert float(data.time) - t0 == pytest.approx(1.0, rel=1e-6)
    finally:
        hal.disconnect()


def test_a_0p1_rad_s_ramp_moves_at_0p1_rad_s_in_sim_time() -> None:
    """The #355/#358 symptom on the twin: velocity measured on the sim clock."""
    from openral_core import Action, ControlMode

    hal = _connected()
    try:
        _, data = hal.mujoco_handles()
        for _ in range(10):
            hal.send_action(_hold(hal))
        start = list(hal.read_state().position)
        v, n, rate = 0.1, 60, 30.0
        samples: list[tuple[float, float]] = []
        for k in range(1, n + 1):
            target = list(start)
            target[1] = start[1] + v * k / rate
            hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target]))
            samples.append((float(data.time), float(hal.read_state().position[1])))
        (t0, q0), (t1, q1) = samples[19], samples[-1]
        assert (q1 - q0) / (t1 - t0) == pytest.approx(v, rel=0.05)
    finally:
        hal.disconnect()


def test_a_pinned_settle_count_is_honoured_and_stays_wall_paced() -> None:
    hal = _connected(settle_steps=5)
    try:
        model, data = hal.mujoco_handles()
        assert hal._settle_steps == 5
        assert float(model.opt.timestep) == pytest.approx(0.002)  # untouched
        assert hal.sim_dt_per_tick_s is None
        assert hal._step_while_active is True
        t0 = float(data.time)
        hal.send_action(_hold(hal))
        assert float(data.time) - t0 == pytest.approx(0.010, rel=1e-6)
    finally:
        hal.disconnect()
