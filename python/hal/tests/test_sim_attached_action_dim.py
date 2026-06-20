"""SimAttachedHAL probes the LIBERO env's true action width (7), not the robocasa fallback (11).

ADR-0036 — a cartesian rSkill's slot-packed action must be sized to the env's
action space. The LIBERO Franka env (robosuite with an OSC_POSE controller)
accepts a 7-D action (6-D end-effector delta + gripper). Before this fix
``_probe_env_action_dim`` missed it (the action width is only reachable as
``sum(r.action_dim for r in env._env.robots)``, not via the two ``_env.action_dim``
probe paths) and fell back to 11, so ``env.step`` rejected the packed action
("expected 7, got 11"). The backend now exposes ``action_dim`` so the probe's
first path resolves.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    leaving collection alive. Mirrors ``test_sim_attached_idle_step`` /
    ``tests/sim/conftest`` (sibling test roots we cannot import across).
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


# The native tabletop_push / openarm_tabletop backends render an RGB observation
# inside ``connect()``; on a headless runner that SIGABRTs the process. Skip
# those two per-test (the other tests here don't render and must still run).
_RENDERER_ERROR = _mujoco_renderer_probe_error()
_requires_renderer = pytest.mark.skipif(
    _RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}",
)


class _NonIntrospectableEnv:
    """A real ``SimRollout`` whose action width cannot be introspected.

    Conforms to the ``openral_sim.rollout.SimRollout`` Protocol (``reset`` /
    ``step`` / ``render`` / ``close``) but deliberately exposes neither
    ``action_dim`` nor an inner ``_env`` — the case the probe must reject
    loudly instead of guessing 11. This is a genuine env at the process
    boundary, not a behavioural mock of the HAL.
    """

    scene: Any = None
    task: Any = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return {"state": np.zeros(1, dtype=np.float32)}

    def step(self, action: NDArray[np.float32]) -> Any:
        raise AssertionError("step must never be reached — connect should raise first.")

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_is_terminated_episode_error_matches_robosuite_guard_only() -> None:
    """Pure matcher (no sim): recover ONLY on robosuite's terminal guard.

    The raised-terminal recovery in ``_step_and_cache`` keys off this predicate.
    It must be True for robosuite's exact "executing action in terminated
    episode" message (so deploy-sim recovers) and False for any other ``step``
    fault — a NaN/dimension/contact error must NOT be silently swallowed by a
    reset (CLAUDE.md §1.4 observability / §1 truth-over-plausibility).
    """
    from openral_hal.sim_attached import is_terminated_episode_error

    assert is_terminated_episode_error(ValueError("executing action in terminated episode"))
    # Case-insensitive, substring-tolerant (robust to wrapper re-phrasing).
    assert is_terminated_episode_error(RuntimeError("Executing action in TERMINATED EPISODE!"))
    # Real faults must propagate, not trigger a reset.
    assert not is_terminated_episode_error(ValueError("expected action dim 7, got 11"))
    assert not is_terminated_episode_error(RuntimeError("mujoco: qpos contains NaN"))


@_requires_renderer
def test_libero_action_dim_is_seven() -> None:
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/libero_spatial.yaml", robot_id_fallback="franka_panda"
    )
    # The backend exposes its true action width (LIBERO OSC_POSE = 7).
    assert env.action_dim == 7

    # And the HAL's probe picks it up (not the robosuite-mobile fallback of 11).
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    assert hal._env_action_dim == 7


@_requires_renderer
def test_send_action_auto_resets_after_episode_termination() -> None:
    """ADR-0036 — a terminated episodic backend (LIBERO) is reset, not re-stepped.

    Without the auto-reset, ``env.step`` on a terminated robosuite episode
    raises "executing action in terminated episode" and the deploy-sim
    continuous-control loop spams failures. Forcing the terminal latch and
    sending one more action must reset the env and step cleanly instead.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    from openral_core import RobotDescription
    from openral_core.schemas import Action, ControlMode
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/libero_spatial.yaml", robot_id_fallback="franka_panda"
    )
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()

    # Simulate the env having reported a terminal step on the previous tick.
    hal._episode_done = True
    zero_delta = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        ee="panda_hand",
        frame="panda_hand",
    )
    # Must NOT raise — the auto-reset branch resets the env before stepping.
    hal.send_action(zero_delta)
    # A single zero-delta step on a fresh episode does not terminate, so the
    # latch is cleared.
    assert hal._episode_done is False


@_requires_renderer
def test_send_action_recovers_when_env_terminal_but_latch_clear() -> None:
    """ADR-0036 follow-up — recover from a *raised* terminal, not just a returned one.

    Raw-robosuite backends (LIBERO, ``so100_robosuite``)
    run with ``ignore_done=False`` and HARD-RAISE
    ``ValueError("executing action in terminated episode")`` on a post-terminal
    ``step`` (robosuite ``environments/base.py``) instead of returning a terminal
    ``StepResult``. When that happens the returned-flag latch never fires, so
    ``_episode_done`` stays ``False`` while robosuite is internally ``done`` —
    and every subsequent ``send_action`` re-raises, freezing the arm and spamming
    ``send_action … env.step failed: executing action in terminated episode``
    (the ``openral deploy sim`` symptom on LIBERO scenes).

    The HAL must treat a *raised* terminal the same as a *returned* one: reset
    once and re-step so the continuous deploy-sim twin keeps driving. Gymnasium /
    native backends never raise, so this only engages where it is needed.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    from openral_core import RobotDescription
    from openral_core.schemas import Action, ControlMode
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/libero_spatial.yaml", robot_id_fallback="franka_panda"
    )
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()

    # Drive the REAL robosuite env into its terminal state, exactly as the
    # horizon/success guard does at episode end: robosuite.step() raises while
    # ``done`` is set, and reset() clears it (environments/base.py). This is the
    # genuine desync the deploy-sim flow hits — NOT a returned terminal, so the
    # HAL's returned-flag latch is (correctly) still clear.
    robosuite_env = hal._env._env.env  # _LiberoSim → LiberoEnv → robosuite
    robosuite_env.done = True
    assert hal._episode_done is False  # the latch never saw this terminal

    zero_delta = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        ee="panda_hand",
        frame="panda_hand",
    )
    # Pre-fix: ROSRuntimeError("… env.step failed: executing action in terminated
    # episode"). Post-fix: the raised terminal is caught, the env reset, and the
    # action re-stepped — no raise.
    hal.send_action(zero_delta)

    # Recovery actually happened: robosuite is live again and the latch reflects a
    # fresh, non-terminal episode (so the next tick proceeds normally).
    assert robosuite_env.done is False
    assert hal._episode_done is False
    hal.send_action(zero_delta)  # and a follow-up tick still steps cleanly


# ── ADR-0034 probe-gap fix — native MuJoCo backends expose their own action_dim ──
#
# Before the fix ``_probe_env_action_dim`` fell back to a hardcoded 11 for any
# backend that didn't expose ``action_dim``. The native MuJoCo backends
# (so101_box → 6, tabletop_push → robot actuator count, openarm_tabletop_pnp →
# bimanual state_dim) require a specific width and raise on a 11-wide
# ``env.step``. Each now reports its true width via an ``action_dim`` property,
# so the probe resolves the authoritative width for a HAL built with NO
# explicit ``env_action_dim`` — used by both ``send_action`` and ``idle_step``.


@_requires_renderer
def test_so101_box_action_dim_is_six() -> None:
    """Native so101_box reports 6 and the HAL probe resolves it (not the 11 fallback)."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    # The backend exposes its true action width (SO-101 = 6 joint targets).
    assert env.action_dim == 6

    # And the HAL's probe picks it up with no explicit env_action_dim override.
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    assert hal._env_action_dim == 6


@_requires_renderer
def test_tabletop_push_action_dim_matches_robot_actuator_count() -> None:
    """Native tabletop_push reports the robot's actuator count; the HAL probe resolves it."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/tabletop_cube_push.yaml", robot_id_fallback="so101_follower"
    )
    # Robot-agnostic scene: the action width is the compiled model's actuator
    # count (the robot's nu). The scene adds no actuators, so for the so101
    # flag this is the SO-101's 6.
    assert env.action_dim == 6

    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    assert hal._env_action_dim == env.action_dim


@_requires_renderer
def test_openarm_tabletop_action_dim_matches_state_dim() -> None:
    """Native openarm_tabletop_pnp reports its bimanual state_dim; the HAL probe resolves it.

    Built through the deploy-sim ``build_sim_env_from_yaml`` loader (robosuite
    MJCF wrapper). The ``openarm_tabletop_pnp`` scene mandates a ``base_pose``
    at compose time (ADR-0002 Amendment 3); the loader now propagates the
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


def test_probe_raises_for_non_introspectable_env_without_override() -> None:
    """A backend exposing no ``action_dim`` (and no override) fails loud, not at 11.

    The safety net: ``_probe_env_action_dim`` refuses to guess a width. With a
    real ``SimRollout`` that genuinely cannot be introspected and no
    ``env_action_dim`` override, ``connect`` raises ``ROSConfigError`` naming the
    backend — a loud boot-time failure beats a wrong-width mid-run E-stop. This
    is the regression guard against re-introducing the silent 11 fallback.
    """
    pytest.importorskip("openral_sim")
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError
    from openral_hal.sim_attached import SimAttachedHAL

    env = _NonIntrospectableEnv()
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc)
    with pytest.raises(ROSConfigError, match="_NonIntrospectableEnv"):
        hal.connect()

    # But an explicit override still resolves — the escape hatch for an env
    # whose action space genuinely isn't introspectable.
    hal_override = SimAttachedHAL(_NonIntrospectableEnv(), desc, env_action_dim=6)
    hal_override.connect()
    assert hal_override._env_action_dim == 6
