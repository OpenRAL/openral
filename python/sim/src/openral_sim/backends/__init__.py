"""Built-in sim backends for eval — registered at import time.

Mirrors :mod:`openral_sim.policies` but for the scene half of the
``(robot × scene × task × VLA)`` quad. To add a new sim backend, drop a
module here and register a factory in :data:`openral_sim.SCENES`.

The factories themselves are responsible for lazily importing heavy backends
(robosuite, libero, metaworld, mujoco, …) so installing ``openral-sim``
never pulls those transitively.

Two scene categories
---------------------
A scene is one of two kinds, set by the ``fixed_robot=`` argument to
``@SCENES.register`` — the runtime source of truth (``SCENES.fixed_robot(id)``
returns the bound robot or ``None``):

* **Multi-robot (free-axis)** — registered WITHOUT ``fixed_robot``. The robot
  is a flag: it comes from the YAML ``robot_id`` (or ``--robot``) and the scene
  composes around whatever compatible robot is named (the base MJCF resolved
  from that robot's manifest ``assets.mjcf``). **New robot-flexible scenes
  belong here.** Today: ``tabletop_push`` (the greenfield robot-agnostic native
  scene — composes its table/cube/goal world onto any position-controlled arm
  via MjSpec), ``maniskill3``, ``openarm_robosuite``, ``simpler_env``,
  ``isaac_sim`` (Isaac Lab env behind an out-of-process py3.11 sidecar).
* **Single-robot (fixed)** — registered WITH ``fixed_robot="<id>"``. The robot
  is baked into the scene (its own MJCF / a benchmark world); the CLI rejects
  ``--robot``. These reproduce a specific embodiment + reward. Today:
  ``libero`` (franka), ``metaworld`` (sawyer),
  ``robocasa`` (panda_mobile), ``aloha``, ``pusht``, ``so101_box`` (so101 — the
  box/tube task is coupled to the so_arm101 MJCF schema),
  ``rlbench`` (franka_panda — CoppeliaSim/PyRep tasks behind an out-of-process
  py3.10 sidecar), ``robotwin`` (aloha_agilex — the RoboTwin 2.0
  SAPIEN dual-arm benchmark behind a py3.10 sidecar), ``vlabench``
  (franka_panda — native in-process on lerobot's VLABenchEnv), ``behavior``
  (r1pro — the BEHAVIOR-1K 2026 OmniGibson evaluator behind an
  out-of-process sidecar in the official BEHAVIOR Python environment).

Declaring out-of-tree provisioning
----------------------------------
A backend whose first run does something genuinely slow before it can build —
clone a repo, pull a multi-GB asset bundle, construct a sidecar venv, or
refuse with an "install it yourself" recipe — MUST also pass ``provision=`` to
``@SCENES.register``. Without it that work lands inside the HAL's
``on_configure``, which ``tools/lifecycle_autostart.py`` bounds at 300 s and
the nav2 palette re-seed helper at 120 s: on a fresh machine both time out,
the HAL never reaches ACTIVE, and the reasoner silently loses skills. With it,
``openral deploy sim`` runs the hook in front of ``ros2 launch``, where the
download is ordinary foreground work and the license prompt has a real TTY.

The hook must be idempotent and must be the same callable the factory itself
invokes, so the preflight and the build cannot drift — see
``robocasa.provision_robocasa`` (deps + assets) and
``isaac_sim.provision_isaac_sim`` (deps + sidecar venv) for the two shapes.
Backends whose only setup is a ``uv sync`` dependency group (LIBERO, MetaWorld,
ManiSkill3, gym-aloha, gym-pusht, the native MuJoCo scenes) leave it unset.
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
        behavior,
        isaac_sim,
        libero,
        maniskill3,
        metaworld,
        openarm_robosuite,
        pusht,
        rlbench,
        robocasa,
        robotwin,
        simpler_env,
        so101_box,
        tabletop_push,
        vlabench,
    )


_register_backends()
