"""Deploy-sim continuous mode is the RoboCasa backend's job, not a scene-id check.

``SimAttachedHAL`` calls ``enable_continuous()`` on whatever rollout it wraps;
RoboCasa's must set robosuite's ``ignore_done`` on the live env, or the first
step past ``horizon`` hard-raises "executing action in terminated episode" and
the HAL resets the kitchen under a running policy. ``sim_bringup`` used to
inject the option by matching ``scene.id.startswith("robocasa")``; that check
is gone, so this test is what holds the behaviour.

No mocks (CLAUDE.md §1.11): the real tracked deploy scene, the real deploy
bring-up path, a real robosuite/MuJoCo kitchen.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)


@pytest.mark.sim
def test_enable_continuous_survives_the_horizon() -> None:
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml("scenes/deploy/robocasa_pnp.yaml")
    try:
        env.reset(seed=seed)
        rs = env._env  # the robosuite kitchen env
        assert rs.ignore_done is False  # bring-up no longer injects it
        env.enable_continuous()
        assert rs.ignore_done is True
        # Jump to the horizon: without ignore_done this step latches `done`
        # and the next one raises.
        rs.timestep = rs.horizon
        zero = np.zeros(env.action_dim, dtype=np.float32)
        env.step(zero)
        env.step(zero)
        assert rs.done is False
    finally:
        env.close()


@pytest.mark.sim
def test_one_step_is_one_control_period_in_sim_time() -> None:
    """Regression for issue #358: a deploy kitchen steps 1/30 s, not robosuite's 1/20 s.

    The deploy scene pins ``control_freq_hz: 30.0`` (the panda_mobile rate), the
    factory shrinks robosuite's physics dt to divide the period, and the env
    reports the tick ``SimAttachedHAL.connect`` holds to the manifest. Before
    this, 30 steps advanced 1.5 s and every rehearsal ran at 1.5x speed.
    """
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml("scenes/deploy/robocasa_pnp.yaml")
    try:
        env.reset(seed=seed)
        assert env.sim_dt_per_tick_s == pytest.approx(1 / 30, rel=0.01)
        zero = np.zeros(env.action_dim, dtype=np.float32)
        env.step(zero)
        t0 = env.sim_time_ns()
        for _ in range(30):
            env.step(zero)
        t1 = env.sim_time_ns()
        assert t0 is not None and t1 is not None
        assert (t1 - t0) / 1e9 == pytest.approx(1.0, rel=0.01)
        # The guard that refused the 20 Hz kitchen under the 30 Hz robot now passes.
        hal = SimAttachedHAL(
            env, RobotDescription.from_yaml("robots/panda_mobile/robot.yaml"), env_reset_seed=seed
        )
        hal.connect()
        assert hal.clock_authority().timestep_s == pytest.approx(1 / 30, rel=0.01)
    finally:
        env.close()
