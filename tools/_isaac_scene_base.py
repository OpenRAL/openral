"""Shared base for the Isaac Sim sidecar scenes.

Runs under the Isaac Sim interpreter only (imported by the scene module, which
``isaac_sidecar.py`` imports after ``SimulationApp`` is live). Owns the
obs/step lifecycle, RGBA→HWC frame grabbing, the warmup + physics-substep loop,
and the eval-layer observation assembly; ``isaac_manifest_scene`` fills in the
template methods.

Subclasses implement the divergent parts:

* ``build`` — construct the stage (robot, props, cameras, controllers);
* ``_apply_action`` — translate one policy action into actuator commands;
* ``_images`` — return the ``{name: HWC uint8}`` camera dict;
* ``_state`` — return the 1-D float32 proprioception vector;
* ``_reward_terminated`` — return ``(reward, terminated)`` for the step.

and may override ``_on_reset`` (per-episode randomization), the class
attributes ``warmup_steps`` / ``physics_substeps``, and set
``self.action_dim`` in ``__init__``.

Time base: one ``step()`` is one control period of the robot
(``action_spec.control_freq_hz``), so a deploy-sim tick advances the world by
exactly ``1 / control_freq_hz`` — ``build`` calls ``resolve_time_base`` with
the world's physics dt, which derives ``physics_substeps`` (issue #355).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Tolerance on ``physics_hz / control_freq_hz`` being an integer (1 %).
_SUBSTEP_RATIO_TOL = 0.01
#: Isaac's default physics rate, the rate the finger contacts were validated at;
#: the physics rate never drops below it.
_MIN_PHYSICS_HZ = 60.0


def physics_dt_for(control_freq_hz: float) -> float:
    """Physics dt for a robot: the smallest multiple of its control rate at or above 60 Hz.

    So one control period is always a whole number of physics steps: a 30 Hz robot
    keeps Isaac's default 60 Hz (2 steps per tick), a 50 Hz robot runs at 100 Hz,
    a 25 Hz one at 75 Hz — instead of refusing every rate that does not divide 60.

    Example:
        >>> [round(1 / physics_dt_for(hz)) for hz in (30.0, 50.0, 25.0, 60.0, 120.0)]
        [60, 100, 75, 60, 120]
    """
    if not control_freq_hz > 0.0:
        raise ValueError(f"control rate {control_freq_hz!r} Hz must be > 0")
    n = max(1, math.ceil(_MIN_PHYSICS_HZ / control_freq_hz - 1e-9))
    return 1.0 / (n * control_freq_hz)


def derive_physics_substeps(physics_dt_s: float, control_freq_hz: float) -> int:
    """Physics steps per control tick so one ``step()`` is one control period.

    Refuses (``ValueError``) a ratio that is not an integer within 1 % instead
    of rounding silently: a 25 Hz robot on 60 Hz physics has no faithful
    substep count, and the message says which rate to change.

    Example:
        >>> derive_physics_substeps(1 / 60, 30.0)
        2
        >>> derive_physics_substeps(1 / 200, 50.0)
        4
    """
    if physics_dt_s <= 0.0 or control_freq_hz <= 0.0:
        raise ValueError(
            f"physics dt {physics_dt_s!r} s and control rate {control_freq_hz!r} Hz must be > 0"
        )
    ratio = 1.0 / (physics_dt_s * control_freq_hz)
    n = round(ratio)
    if n < 1 or abs(ratio - n) > _SUBSTEP_RATIO_TOL * n:
        raise ValueError(
            f"physics rate {1.0 / physics_dt_s:.4g} Hz is not an integer multiple of the "
            f"control rate {control_freq_hz:.4g} Hz (ratio {ratio:.3f}); one scene step must be "
            "one control period. Build the scene's World at physics_dt_for(control_freq_hz) "
            "(a pinned physics_substeps is still held to one control period by "
            "SimAttachedHAL.connect)."
        )
    return n


class IsaacSceneBase:
    """Lifecycle + obs skeleton common to the Isaac Sim sidecar scenes."""

    #: Physics steps to settle after a reset before the first observation.
    warmup_steps: int = 4
    #: Physics steps per policy action. ``None`` (default) = derived by
    #: ``resolve_time_base`` so one step is one control period; an explicit
    #: int is a scene-level override that wins and is logged at build.
    physics_substeps: int | None = None

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
        # Physics dt the world runs at; set by ``resolve_time_base`` in ``build``.
        self._physics_dt_s: float | None = None

    # ── public sidecar contract ──────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Per-episode reset: randomize, reset physics, warm up, observe."""
        self._on_reset(np.random.default_rng(seed))
        self._world.reset()
        self._after_world_reset()
        self._step_idx = 0
        for _ in range(self.warmup_steps):
            self._before_render()
            self._world.step(render=True)
        return self._observe()

    def step(self, action: NDArray[np.float32]) -> dict[str, Any]:
        """Apply one action, advance physics, and return a StepResult dict."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            # NaN, not 0: slots are absolute targets, and 0 is a legal one ("drive
            # to 0 rad", "close the gripper"). NaN holds; the scene reads a NaN
            # base twist as 0 (stop).
            action = np.pad(action, (0, self.action_dim - action.shape[0]), constant_values=np.nan)
        self._apply_action(action)
        # Render only the final substep — it is the frame the obs reads, so the
        # camera observation stays at the control rate.
        for _ in range(self._substeps() - 1):
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
            # Elapsed sim time so the deploy-sim ROS graph
            # can run on /clock with an Isaac backend (None if unavailable).
            "sim_time_ns": self.sim_time_ns(),
        }

    def resolve_time_base(self, physics_dt_s: float, control_freq_hz: float) -> int:
        """Fix the scene's time base: physics dt + substeps so one step = one tick.

        Called by ``build`` once the world exists. Derives ``physics_substeps``
        from the physics dt and the robot's ``action_spec.control_freq_hz``
        (``derive_physics_substeps``) unless the class pinned an explicit
        value, which wins and is reported as an override. Returns the substep
        count. Raises ``ValueError`` when no faithful integer count exists.
        """
        self._physics_dt_s = float(physics_dt_s)
        pinned = type(self).physics_substeps
        if pinned is None:
            self.physics_substeps = derive_physics_substeps(physics_dt_s, control_freq_hz)
            how = "derived"
        else:
            self.physics_substeps = int(pinned)
            how = "pinned by the scene class (override)"
        tick = self.physics_substeps * self._physics_dt_s
        print(
            f"[{type(self).__name__}] time base: physics {1.0 / self._physics_dt_s:.4g} Hz, "
            f"control {control_freq_hz:.4g} Hz, physics_substeps={self.physics_substeps} ({how}), "
            f"sim_dt_per_tick_s={tick:.6f}",
            flush=True,
        )
        return self.physics_substeps

    def _substeps(self) -> int:
        """Resolved physics substeps; refuses to step before ``resolve_time_base``."""
        if self.physics_substeps is None:
            raise RuntimeError(
                f"{type(self).__name__}: physics_substeps unresolved — build() must call "
                "resolve_time_base() before the scene is stepped."
            )
        return max(1, int(self.physics_substeps))

    @property
    def sim_dt_per_tick_s(self) -> float | None:
        """Simulation seconds one ``step()`` advances, or ``None`` before ``build``.

        Reported in the sidecar ``ping`` reply so ``SimAttachedHAL.connect``
        can refuse a tick that is not one control period.
        """
        if self._physics_dt_s is None or self.physics_substeps is None:
            return None
        return self._substeps() * self._physics_dt_s

    def sim_time_ns(self) -> int | None:
        """Elapsed simulation time in ns, or ``None`` if unavailable.

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
        dt = self._physics_dt_s
        if dt is None:
            get_dt = getattr(world, "get_physics_dt", None)
            dt = get_dt() if callable(get_dt) else None
        if dt is not None and self.physics_substeps is not None:
            return int(round(self._step_idx * self._substeps() * float(dt) * 1e9))
        return None

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_rgb is None else self._last_rgb.copy()

    # ── template methods (override in subclasses) ────────────────────────────

    def build(self) -> None:
        raise NotImplementedError

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Per-episode randomization hook. Default: nothing to randomize."""

    def _after_world_reset(self) -> None:
        """Hook run right after ``world.reset()``, before the warmup steps.

        For state ``world.reset()`` overwrites (e.g. a robot root placed away
        from its import pose). Default: nothing.
        """

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
