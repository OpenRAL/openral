"""Resolve a :class:`SimEnvironment` config into concrete env + policy objects.

The factory is the single place the eval runner asks: "given this YAML, what
do I actually instantiate?".  It dispatches to the right registry entry, runs
the optional rSkill loader for bare rSkill references, and returns objects
ready for :class:`openral_sim.SimRunner` to tick against.

Adding a new backend is one decorator line in
:mod:`openral_sim.{policies,backends}`; nothing in this file needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openral_sim.registry import POLICIES, ROBOTS, SCENES

if TYPE_CHECKING:
    from openral_core import RobotDescription, SimEnvironment

    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import SimRollout


def make_env(env_cfg: SimEnvironment) -> SimRollout:
    """Build the simulated environment described by ``env_cfg``.

    Args:
        env_cfg: Validated :class:`openral_core.SimEnvironment` config.

    Returns:
        A :class:`openral_sim.rollout.SimRollout` ready for
        :meth:`reset` / :meth:`step`.

    Raises:
        openral_core.exceptions.ROSConfigError: If
            ``env_cfg.scene.id`` is not registered.
    """
    factory = SCENES.get(env_cfg.scene.id)
    return factory(env_cfg)


def make_policy(env_cfg: SimEnvironment) -> PolicyAdapter:
    """Build the VLA / policy adapter described by ``env_cfg.vla``.

    Args:
        env_cfg: Validated :class:`openral_core.SimEnvironment` config.

    Returns:
        A :class:`openral_sim.policy.PolicyAdapter` ready for
        :meth:`reset` / :meth:`step`.

    Raises:
        openral_core.exceptions.ROSConfigError: If
            ``env_cfg.vla.id`` is not registered.
    """
    factory = POLICIES.get(env_cfg.vla.id)
    return factory(env_cfg)


def make_robot(env_cfg: SimEnvironment) -> RobotDescription | None:
    """Resolve the robot description for ``env_cfg.robot_id`` if registered.

    Returns ``None`` when no robot factory is registered — the eval layer
    does not require a :class:`RobotDescription` to run, but having one
    lets the runner emit capability checks against the
    :class:`~openral_core.RSkillManifest` of the loaded skill.

    Args:
        env_cfg: Validated :class:`openral_core.SimEnvironment` config.

    Returns:
        The matching :class:`openral_core.RobotDescription` or ``None``.
    """
    if env_cfg.robot_id not in ROBOTS:
        return None
    return ROBOTS.get(env_cfg.robot_id)()
