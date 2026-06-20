"""Shared base for the Isaac Sim sidecar scenes (ADR-0045).

Runs under the Isaac Sim py3.11 venv only (imported by the scene modules, which
``isaac_sidecar.py`` imports after ``SimulationApp`` is live). Owns the bits both
the ``lift_cube`` and ``bowl_plate`` scenes share — the obs/step lifecycle,
RGBA→HWC frame grabbing, the warmup + physics-substep loop, and the eval-layer
observation assembly — so a new layout is a few template-method overrides rather
than a third copy of the skeleton.

Subclasses implement the divergent parts:

* :meth:`build` — construct the stage (robot, props, cameras, controllers);
* :meth:`_apply_action` — translate one policy action into actuator commands;
* :meth:`_images` — return the ``{name: HWC uint8}`` camera dict;
* :meth:`_state` — return the 1-D float32 proprioception vector;
* :meth:`_reward_terminated` — return ``(reward, terminated)`` for the step.

and may override :meth:`_on_reset` (per-episode randomization), the class
attributes :data:`warmup_steps` / :data:`physics_substeps`, and set
``self.action_dim`` in ``__init__``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _franka_dof_to_manifest(vec: NDArray[np.float32]) -> NDArray[np.float32]:
    """Map an Isaac Franka 9-DOF vector to the manifest's 8 joints.

    Isaac's articulation orders ``[panda_joint1..7, finger1, finger2]``; the
    OpenRAL manifest collapses the two fingers into one ``panda_gripper`` joint.
    Returns ``[arm0..6, gripper]`` (gripper = mean of the two finger entries) in
    manifest order so ``SimAttachedHAL.read_state`` can index it against
    ``description.joints``. Works for positions or velocities.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    arm = v[:7]
    if v.shape[0] >= 9:
        gripper = float(np.mean(v[7:9]))
    elif v.shape[0] > 7:
        gripper = float(v[7])
    else:
        gripper = 0.0
    return np.concatenate([arm, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)


def franka_joint_positions(franka: Any) -> NDArray[np.float32]:
    """Franka joint angles in manifest order (8 = 7 arm + gripper)."""
    return _franka_dof_to_manifest(np.asarray(franka.get_joint_positions(), dtype=np.float32))


def franka_joint_velocities(franka: Any) -> NDArray[np.float32]:
    """Franka joint velocities in manifest order (8 = 7 arm + gripper)."""
    return _franka_dof_to_manifest(np.asarray(franka.get_joint_velocities(), dtype=np.float32))


class IsaacSceneBase:
    """Lifecycle + obs skeleton common to the Isaac Sim sidecar scenes."""

    #: Physics steps to settle after a reset before the first observation.
    warmup_steps: int = 4
    #: Physics steps per policy action so the controller tracks its target
    #: (LIBERO/robosuite likewise run several sim steps per policy step).
    physics_substeps: int = 1

    def __init__(
        self,
        *,
        obs_height: int,
        obs_width: int,
        instruction: str,
        success_key: str,
        max_steps: int,
    ) -> None:
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.instruction = instruction
        self.success_key = success_key
        self.max_steps = max_steps
        self.action_dim = 0  # subclass sets the real value
        self._step_idx = 0
        self._last_rgb: NDArray[np.uint8] | None = None
        self._world: Any = None

    # ── public sidecar contract ──────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Per-episode reset: randomize, reset physics, warm up, observe."""
        self._on_reset(np.random.default_rng(seed))
        self._world.reset()
        self._step_idx = 0
        for _ in range(self.warmup_steps):
            self._before_render()
            self._world.step(render=True)
        return self._observe()

    def step(self, action: NDArray[np.float32]) -> dict[str, Any]:
        """Apply one action, advance physics, and return a StepResult dict."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.shape[0]))
        self._apply_action(action)
        # Render only the final substep — it is the frame the obs reads.
        for _ in range(max(0, self.physics_substeps - 1)):
            self._world.step(render=False)
        self._before_render()
        self._world.step(render=True)
        self._step_idx += 1
        reward, terminated = self._reward_terminated()
        info: dict[str, Any] = {self.success_key: bool(terminated)}
        info.update(self._extra_info())
        return {
            "observation": self._observe(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(self._step_idx >= self.max_steps),
            "info": info,
            # ADR-0048 Phase 2 — elapsed sim time so the deploy-sim ROS graph
            # can run on /clock with an Isaac backend (None if unavailable).
            "sim_time_ns": self.sim_time_ns(),
        }

    def sim_time_ns(self) -> int | None:
        """Elapsed simulation time in ns, or ``None`` (ADR-0048 Phase 2).

        Best-effort: prefer the Isaac ``SimulationContext.current_time`` (seconds
        since sim start); else integrate the step count by the physics dt. The
        deploy-sim HAL reads this (via the sidecar reply) to publish ``/clock``.
        """
        world = self._world
        if world is None:
            return None
        current_time = getattr(world, "current_time", None)
        if current_time is not None:
            return int(round(float(current_time) * 1e9))
        dt = getattr(self, "physics_dt", None)
        if dt is None:
            get_dt = getattr(world, "get_physics_dt", None)
            dt = get_dt() if callable(get_dt) else None
        if dt is not None:
            substeps = max(1, int(getattr(self, "physics_substeps", 1)))
            return int(round(self._step_idx * substeps * float(dt) * 1e9))
        return None

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_rgb is None else self._last_rgb.copy()

    # ── template methods (override in subclasses) ────────────────────────────

    def build(self) -> None:
        raise NotImplementedError

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Per-episode randomization hook. Default: nothing to randomize."""

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        raise NotImplementedError

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        raise NotImplementedError

    def _state(self) -> NDArray[np.float32]:
        raise NotImplementedError

    def _reward_terminated(self) -> tuple[float, bool]:
        raise NotImplementedError

    def _extra_info(self) -> dict[str, Any]:
        """Extra keys merged into the step ``info`` dict. Default: none."""
        return {}

    def _before_render(self) -> None:
        """Hook for camera poses that must be updated before a rendered frame."""

    def _joint_positions(self) -> NDArray[np.float32] | None:
        """Robot joint angles in the host manifest's joint order, or None.

        Surfaced as ``obs["joint_positions"]`` so ``openral deploy sim``'s
        ``SimAttachedHAL.read_state`` can publish a real ``/joint_states`` for a
        non-MuJoCo backend (it otherwise reads joints from a MuJoCo handle this
        sidecar has none of). Default None → the HAL falls back to zeros.
        """
        return None

    def _joint_velocities(self) -> NDArray[np.float32] | None:
        """Robot joint velocities in manifest order, or None (→ HAL zeros)."""
        return None

    # ── shared helpers ───────────────────────────────────────────────────────

    def _observe(self) -> dict[str, Any]:
        images = self._images()
        if images:
            self._last_rgb = next(iter(images.values()))
        obs: dict[str, Any] = {
            "images": images,
            "state": self._state(),
            "task": self.instruction,
        }
        joints = self._joint_positions()
        if joints is not None:
            obs["joint_positions"] = np.asarray(joints, dtype=np.float32)
        joint_vel = self._joint_velocities()
        if joint_vel is not None:
            obs["joint_velocities"] = np.asarray(joint_vel, dtype=np.float32)
        return obs

    def _grab(self, cam: Any) -> NDArray[np.uint8]:
        """Return an HWC uint8 RGB frame from a camera; zeros until it warms up."""
        rgba = cam.get_rgba()
        if rgba is None or np.asarray(rgba).size == 0:
            return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)
        return np.asarray(rgba, dtype=np.uint8)[:, :, :3]
