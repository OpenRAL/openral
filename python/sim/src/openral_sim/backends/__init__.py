"""Built-in sim backends for eval — registered at import time.

Mirrors :mod:`openral_sim.policies` but for the scene half of the
``(robot × scene × task × VLA)`` quad. To add a new sim backend, drop a
module here and register a factory in :data:`openral_sim.SCENES`.

The factories themselves are responsible for lazily importing heavy backends
(robosuite, libero, metaworld, mujoco, …) so installing ``openral-sim``
never pulls those transitively.

Two scene categories (ADR-0033)
-------------------------------
A scene is one of two kinds, set by the ``fixed_robot=`` argument to
``@SCENES.register`` — the runtime source of truth (``SCENES.fixed_robot(id)``
returns the bound robot or ``None``):

* **Multi-robot (free-axis)** — registered WITHOUT ``fixed_robot``. The robot
  is a flag: it comes from the YAML ``robot_id`` (or ``--robot``) and the scene
  composes around whatever compatible robot is named (the base MJCF resolved
  from that robot's manifest ``assets.mjcf``). **New robot-flexible scenes
  belong here.** Today: ``tabletop_push`` (the greenfield robot-agnostic native
  scene — composes its table/cube/goal world onto any position-controlled arm
  via MjSpec; ADR-0033), ``maniskill3``, ``openarm_robosuite``, ``simpler_env``,
  ``isaac_sim`` (Isaac Lab env behind an out-of-process py3.11 sidecar; ADR-0045).
* **Single-robot (fixed)** — registered WITH ``fixed_robot="<id>"``. The robot
  is baked into the scene (its own MJCF / a benchmark world); the CLI rejects
  ``--robot``. These reproduce a specific embodiment + reward. Today:
  ``libero`` / ``libero_custom_bddl`` (franka), ``metaworld`` (sawyer),
  ``robocasa`` (panda_mobile), ``aloha``, ``pusht``, ``so101_box`` (so101 — the
  box/tube task is coupled to the so_arm101 MJCF schema; ADR-0033 finding).
"""

from __future__ import annotations


def _register_backends() -> None:
    """Import side-effect modules that register scene factories.

    Each module's ``@SCENES.register`` declares its category via ``fixed_robot``
    (multi-robot / free-axis when absent; single-robot when set) — see the
    module docstring above for the taxonomy + current membership.
    """
    from openral_sim.backends import (
        aloha,
        isaac_sim,
        libero,
        libero_custom_bddl,
        maniskill3,
        metaworld,
        openarm_robosuite,
        pusht,
        robocasa,
        simpler_env,
        so101_box,
        tabletop_push,
    )


_register_backends()
