"""robosuite integration for the Hugging Face SO-100 follower.

Importing this package as a side effect:

* registers :class:`~.model.SO100` in robosuite's robot factory so
  ``robosuite.make(robots=["SO100"])`` (and downstream env constructors)
  works;
* registers :class:`~.model.SO100Gripper` in robosuite's gripper factory
  so ``gripper_types="default"`` (or an explicit ``"SO100Gripper"``)
  resolves cleanly.

The XMLs robosuite consumes are generated lazily from the upstream
DeepMind ``mujoco_menagerie`` MJCF — see :mod:`._assets` for the
rewrite. Tests under ``tests/sim/test_so100_robosuite_lift.py`` exercise
the full pipeline end-to-end.

Example:
    >>> from openral_sim.backends.so100_robosuite import make_so100_lift_env
    >>> env = make_so100_lift_env(has_offscreen_renderer=False, use_camera_obs=False)
    >>> _obs = env.reset()
    >>> env.close()
"""

from __future__ import annotations

from openral_sim.backends.so100_robosuite.env import (
    make_so100_lift_env,
    so100_osc_controller_config,
)
from openral_sim.backends.so100_robosuite.model import SO100, SO100Gripper

__all__ = [
    "SO100",
    "SO100Gripper",
    "make_so100_lift_env",
    "so100_osc_controller_config",
]
