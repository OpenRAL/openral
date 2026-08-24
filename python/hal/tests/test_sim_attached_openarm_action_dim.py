"""SimAttachedHAL probes the OpenArm v2 tabletop env's true action width.

Split out of ``test_sim_attached_action_dim.py``, which is a **LIBERO** file:
LIBERO pins ``robosuite==1.4`` while the ``openarm_tabletop_pnp`` scene needs
``robosuite>=1.5`` (the ``robocasa`` dependency group — see
``openral_sim._deps._openarm_robosuite_plan``). The two pins are mutually
exclusive in one venv (CLAUDE.md — "LIBERO↔RoboCasa groups are mutually
exclusive"), and ``tools/test_selection.toml`` assigns a dependency lane
**per file**, so a file holding both halves is always wrong for one of them.

Concretely: run inside the ``libero`` lane, the LIBERO tests above this one
imported robosuite 1.4, then this test's ``ensure_backend_deps`` probe found
the wrong robosuite and ``_assert_no_live_dependency_swap`` correctly refused
to swap the package under live objects — so the test failed instead of
running, on every PR whose diff selected ``python/hal/tests/``. Keeping it in
its own file lets ``requirement_globs.robocasa`` give it the robosuite>=1.5
env it actually needs.
"""

from __future__ import annotations

import os

import pytest

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
    leaving collection alive. Mirrors ``test_sim_attached_action_dim`` /
    ``test_sim_attached_idle_step`` / ``tests/sim/conftest`` (sibling test
    roots we cannot import across).
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


# The openarm_tabletop backend renders an RGB observation inside ``connect()``;
# on a headless runner without a GL stack that SIGABRTs the process.
_RENDERER_ERROR = _mujoco_renderer_probe_error()
_requires_renderer = pytest.mark.skipif(
    _RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}",
)


@_requires_renderer
def test_openarm_tabletop_action_dim_matches_state_dim() -> None:
    """Native openarm_tabletop_pnp reports its bimanual state_dim; the HAL probe resolves it.

    Built through the deploy-sim ``build_sim_env_from_yaml`` loader (robosuite
    MJCF wrapper). The ``openarm_tabletop_pnp`` scene mandates a ``base_pose``
    at compose time; the loader now propagates the
    SimScene YAML's ``base_pose`` into the composed ``SimEnvironment``,
    so the scene builds through the loader exactly as it does through the direct
    factory path.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/openarm_tabletop.yaml", robot_id_fallback="openarm"
    )
    # state_dim is derived from the manifest joint count (bimanual OpenArm v2).
    assert env.action_dim == env._state_dim
    assert env.action_dim > 0

    desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    assert hal._env_action_dim == env.action_dim
