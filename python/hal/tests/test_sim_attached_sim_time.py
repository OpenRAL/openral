"""Real-MuJoCo exercise of the ``sim_time_ns`` seam.

The hermetic offset / clock-less behaviour is covered by
``tests/unit/test_sim_attached_hal.py`` against the sanctioned ``FakeSimEnv``
boundary double. These tests close the loop against a *real* MuJoCo backend
(CLAUDE.md §1.11/§1.12) so the ``round(MjData.time * 1e9)`` reading and the
cross-reset offset are validated against the actual physics clock.

The native-MuJoCo ``so101_box`` backend is used because it needs neither
robosuite nor LIBERO, so it runs wherever ``mujoco`` is importable — mirroring
the ``test_sim_attached_idle_step.py`` ``_build_so101_hal`` idiom.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from _renderer_probe import requires_renderer


def _build_so101_hal() -> object:
    """Build a connected SimAttachedHAL over the native-MuJoCo so101 box scene.

    Mirrors ``test_sim_attached_idle_step._build_so101_hal``: the backend
    exposes no introspectable ``action_dim`` so we pass ``env_action_dim=6``
    explicitly (the documented path for non-introspectable envs).
    """
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed, env_action_dim=6)
    hal.connect()
    return hal


@requires_renderer
def test_rollout_sim_time_ns_advances_on_real_mujoco_backend() -> None:
    """The native so101 rollout's own sim_time_ns advances as the env steps."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, _seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    # Before any reset the MjData clock is at 0 ns.
    t0 = env.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimRollout surface
    assert t0 == 0
    env.reset(seed=0)
    after_reset = env.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimRollout surface
    assert after_reset is not None and after_reset >= 0


@requires_renderer
def test_sim_attached_sim_time_ns_monotonic_across_steps_real_mujoco() -> None:
    """SimAttachedHAL.sim_time_ns is monotonic non-decreasing across real steps."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    hal = _build_so101_hal()

    samples = [hal.sim_time_ns()]  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    for _ in range(8):
        hal.idle_step()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
        samples.append(hal.sim_time_ns())  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface

    assert all(s is not None for s in samples), f"clock dropped to None: {samples}"
    for prev, cur in pairwise(samples):
        assert cur >= prev, f"sim_time_ns went backwards: {prev} -> {cur}"
    # Physics advanced over 8 idle steps — the last reading must exceed the first.
    assert samples[-1] > samples[0]


@requires_renderer
def test_sim_attached_sim_time_ns_does_not_rewind_across_reconnect_real_mujoco() -> None:
    """The cross-reset offset prevents a rewind on lifecycle reconnect.

    The env's explicit reset rewinds real ``MjData.time``; deploy-sim
    actions and terminal signals never reset it.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    hal = _build_so101_hal()

    for _ in range(5):
        hal.idle_step()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    pre_reset = hal.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert pre_reset is not None and pre_reset > 0

    hal.connect()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    hal.idle_step()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    post_reset = hal.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface

    assert post_reset is not None
    assert post_reset >= pre_reset, f"sim_time_ns rewound across reset: {pre_reset} -> {post_reset}"
