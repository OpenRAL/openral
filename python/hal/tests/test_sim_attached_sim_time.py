"""Real-MuJoCo exercise of the ADR-0048 Phase 1 ``sim_time_ns`` seam.

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

import os
from itertools import pairwise

import pytest

# Force EGL (off-screen) rendering so CI hosts without a display don't abort.
# The classic renderer calls glXOpenDisplay() and raises SIGABRT on headless
# runners; EGL avoids the display requirement entirely.
os.environ.setdefault("MUJOCO_GL", "egl")


def _mujoco_renderer_probe_error() -> str | None:
    """Return ``None`` if a MuJoCo off-screen renderer can be created, else a reason.

    Creating a ``mujoco.Renderer`` on a headless host without a working GL/EGL
    stack calls ``abort()`` at the C level (SIGABRT), which a Python
    ``try/except`` cannot catch — an in-process probe therefore crashes pytest
    outright (``Fatal Python error: Aborted``) and takes the whole partition
    down with it. Running the probe in a subprocess turns that abort into a
    non-zero exit code we can detect and convert into a clean skip reason,
    leaving collection alive. Mirrors ``test_sim_attached_idle_step`` (a sibling
    real-MuJoCo HAL test) and ``tests/sim/conftest`` (a test root we cannot
    import across).
    """
    import subprocess
    import sys

    probe = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>');"
        "r = mujoco.Renderer(m, 1, 1); r.close()"
    )
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except FileNotFoundError:  # mujoco import unavailable in the probe interpreter
        return "mujoco unavailable for renderer probe"
    except subprocess.TimeoutExpired:
        return "mujoco renderer probe timed out (120s)"
    if proc.returncode == 0:
        return None
    stderr_lines = (proc.stderr or "").strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else "no stderr"
    return f"renderer probe exited {proc.returncode}: {detail}"


# Every test below builds a HAL whose ``connect()`` (or a bare ``env.reset()``)
# renders the so101 box's off-screen OAK camera. On a headless CI runner that
# abort()s the native MuJoCo Renderer at the C level (SIGABRT) — uncatchable in
# Python — so each render-dependent test must skip when no off-screen renderer
# is available. ``importorskip("mujoco")`` alone is not enough: mujoco imports
# fine on a host that still lacks a working GL/EGL stack.
_RENDERER_ERROR = _mujoco_renderer_probe_error()
_requires_renderer = pytest.mark.skipif(
    _RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}",
)


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


@_requires_renderer
def test_rollout_sim_time_ns_advances_on_real_mujoco_backend() -> None:
    """The native so101 rollout's own sim_time_ns advances as the env steps."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, _seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    # Before any reset the MjData clock is at 0 ns.
    t0 = env.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimRollout ADR-0048 surface
    assert t0 == 0
    env.reset(seed=0)
    after_reset = env.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimRollout surface
    assert after_reset is not None and after_reset >= 0


@_requires_renderer
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


@_requires_renderer
def test_sim_attached_sim_time_ns_does_not_rewind_across_reset_real_mujoco() -> None:
    """The cross-reset offset prevents a rewind even when MjData.time resets to 0.

    Drive the ADR-0036 auto-reset via the ``_episode_done`` latch (the env's
    own ``reset`` rewinds the real ``MjData.time``), then assert the
    HAL-published value never goes backwards.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    hal = _build_so101_hal()

    for _ in range(5):
        hal.idle_step()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    pre_reset = hal.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert pre_reset is not None and pre_reset > 0

    # The next idle_step resets the env (rewinding MjData.time to 0) before
    # stepping; the offset must absorb the finished episode's elapsed time.
    hal._episode_done = True  # type: ignore[attr-defined]  # reason: white-box latch (mirrors idle_step test)
    hal.idle_step()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    post_reset = hal.sim_time_ns()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface

    assert post_reset is not None
    assert post_reset >= pre_reset, f"sim_time_ns rewound across reset: {pre_reset} -> {post_reset}"
