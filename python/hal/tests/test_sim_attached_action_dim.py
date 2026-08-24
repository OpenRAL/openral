"""SimAttachedHAL probes the LIBERO env's true action width (7), not the robocasa fallback (11).

A cartesian rSkill's slot-packed action must be sized to the env's
action space. The LIBERO Franka env (robosuite with an OSC_POSE controller)
accepts a 7-D action (6-D end-effector delta + gripper). Before this fix
``_probe_env_action_dim`` missed it (the action width is only reachable as
``sum(r.action_dim for r in env._env.robots)``, not via the two ``_env.action_dim``
probe paths) and fell back to 11, so ``env.step`` rejected the packed action
("expected 7, got 11"). The backend now exposes ``action_dim`` so the probe's
first path resolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from _renderer_probe import requires_renderer

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


@requires_renderer
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


@requires_renderer
def test_send_action_surfaces_terminal_guard_without_reset() -> None:
    """A backend terminal guard is surfaced; deploy sim never resets it."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSRuntimeError
    from openral_core.schemas import Action, ControlMode
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/libero_spatial.yaml", robot_id_fallback="franka_panda"
    )
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()

    robosuite_env = hal._env._robosuite_env()
    robosuite_env.ignore_done = False
    robosuite_env.done = True

    zero_delta = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        ee="panda_hand",
        frame="panda_hand",
    )
    with pytest.raises(ROSRuntimeError, match="terminated episode"):
        hal.send_action(zero_delta)
    assert robosuite_env.done is True


# ── Probe-gap fix — native MuJoCo backends expose their own action_dim ──
#
# Before the fix ``_probe_env_action_dim`` fell back to a hardcoded 11 for any
# backend that didn't expose ``action_dim``. The native MuJoCo backends
# (so101_box → 6, tabletop_push → robot actuator count; and, in the sibling
# `test_sim_attached_openarm_action_dim.py`, openarm_tabletop_pnp → bimanual
# state_dim) require a specific width and raise on a 11-wide
# ``env.step``. Each now reports its true width via an ``action_dim`` property,
# so the probe resolves the authoritative width for a HAL built with NO
# explicit ``env_action_dim`` — used by both ``send_action`` and ``idle_step``.


@requires_renderer
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


@requires_renderer
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


# NB: the openarm_tabletop_pnp counterpart of the two tests above deliberately
# does NOT live here — see `test_sim_attached_openarm_action_dim.py`. This is a
# LIBERO file (robosuite==1.4); the OpenArm scene needs robosuite>=1.5, and
# tools/test_selection.toml assigns a dependency lane per FILE, so a file
# holding both is always wrong for one of them.


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
