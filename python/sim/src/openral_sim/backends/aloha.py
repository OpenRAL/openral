"""gym-aloha scene adapter — wraps ``gym_aloha/AlohaTransferCube-v0``.

Driven by a :class:`openral_core.SimEnvironment` config, this adapter is
the canonical entry point for gym-aloha rollouts (used by
``tests/sim/test_aloha_bimanual_act_aloha.py``).

gym-aloha is opt-in: the underlying ``gym-aloha`` package depends on MuJoCo
and is heavy. The factory imports lazily so this module is safe to import
on any platform; the failure surfaces at ``make_env()`` time.

Task ID convention
------------------
``"<env_id>"`` where ``<env_id>`` is one of ``transfer_cube``, ``insertion``.
The default env_id when ``scene.id == "aloha_transfer_cube"`` is
``gym_aloha/AlohaTransferCube-v0``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_ALOHA_SCENES: dict[str, str] = {
    "aloha_transfer_cube": "gym_aloha/AlohaTransferCube-v0",
    "aloha_insertion": "gym_aloha/AlohaInsertion-v0",
}


@dataclass
class _AlohaSim:
    """Thin :class:`SimRollout` wrapper around a ``gym_aloha`` env."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # gymnasium env
    _last_pixels: NDArray[np.uint8] | None = None

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        a_raw: NDArray[np.float32] = np.asarray(action, dtype=np.float32).reshape(-1)
        # Pass the raw joint-radian command straight through. gym-aloha's
        # declared ``action_space=Box(-1, 1)`` is misleading — dm_control
        # actually accepts unnormalized joint targets, and the published
        # ACT checkpoint's un-normalized output (e.g. 1.17 rad on joint 2,
        # paired with state.shape=(14,) in real radians) lives modestly
        # outside that nominal box. Clipping to [-1, 1] destroys the
        # grasp + handoff commands and tanks aloha_transfer_cube success
        # (seed 0–4: 1/5 with the clip, 4/5 without).
        obs, reward, terminated, truncated, info = self._env.step(a_raw)
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self._last_pixels is None:
            return None
        return self._last_pixels.copy()

    def close(self) -> None:
        self._env.close()

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Reach through gym-aloha + dm_control to the raw MuJoCo handles.

        gym-aloha wraps a dm_control ``Environment``; its ``physics``
        attribute exposes ``model`` / ``data`` as dm_control wrappers
        whose ``.ptr`` is the underlying ``mujoco.MjModel`` / ``MjData``.
        Returns ``None`` if the chain is broken.

        Note on ``--view``: the historical failure mode here was a
        viewer window opening blank (``GLFWError 65546: cannot swap
        buffers``) because dm_control monopolised GLFW for its
        offscreen pixel-rendering pipeline. The fix lives in
        :func:`_build_aloha_scene` below: it forces ``MUJOCO_GL=egl``
        before importing gym_aloha so dm_control renders offscreen
        via EGL, leaving GLFW free for ``mujoco.viewer``'s onscreen
        window. ``mujoco.viewer.launch_passive`` ignores the
        ``MUJOCO_GL`` plugin selector for its own window creation
        (it calls ``glfw.init()`` directly), so the viewer paints
        cleanly even when offscreen rendering is on EGL.
        """
        env = getattr(self._env, "unwrapped", self._env)
        inner = getattr(env, "_env", None)
        physics = getattr(inner, "physics", None) if inner is not None else None
        if physics is None:
            return None
        model = getattr(getattr(physics, "model", None), "ptr", None)
        data = getattr(getattr(physics, "data", None), "ptr", None)
        if model is None or data is None:
            return None
        return model, data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1), or None.

        Reads dm_control's ``MjData.time`` off :meth:`mujoco_handles`. Monotonic
        within an episode; rewinds on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        """Translate gym-aloha obs into the eval-layer Observation schema."""
        pixels = obs.get("pixels", {})
        top = np.asarray(pixels.get("top")) if "top" in pixels else None
        if top is not None:
            self._last_pixels = top.astype(np.uint8)

        agent_pos = obs.get("agent_pos")
        state = (
            np.asarray(agent_pos, dtype=np.float32)
            if agent_pos is not None
            else np.zeros(14, dtype=np.float32)
        )

        images: dict[str, NDArray[np.uint8]] = {}
        if top is not None:
            images["top"] = top.astype(np.uint8)

        return {
            "images": images,
            "state": state,
            "task": self.task.instruction,
            # Preserve the raw nested obs for adapters that need it.
            "raw": obs,
        }


def _build_aloha_scene(env_cfg: SimEnvironment) -> _AlohaSim:
    """Lazily import ``gym_aloha`` and build a :class:`_AlohaSim`.

    Forces ``MUJOCO_GL=egl`` before importing gym_aloha so dm_control's
    ``Physics`` offscreen pixel pipeline runs on EGL rather than
    monopolising GLFW. Without this, ``mujoco.viewer.launch_passive``
    creates a window whose OpenGL context never binds — the symptom
    is a blank window over the desktop background, no scene ever
    paints (``GLFWError 65546: cannot swap buffers of a window that
    has no OpenGL or OpenGL ES context``). The viewer itself still
    uses GLFW for its onscreen window because
    ``mujoco.viewer.launch_passive`` calls ``glfw.init()`` directly
    and ignores ``MUJOCO_GL`` for its own context — the two backends
    coexist as long as dm_control gets EGL.

    The override is unconditional rather than ``--view``-gated because
    (a) the scene factory does not see the CLI flag, (b) EGL works
    fine for offscreen-only runs too, and (c) it neutralises the
    upstream ``_resolve_view`` egl→glfw override in
    ``openral_sim.cli`` for this specific backend (which is correct
    for every other MuJoCo-backed scene but actively harmful for
    aloha).
    """
    if env_cfg.scene.id not in _ALOHA_SCENES:
        raise ROSConfigError(
            f"aloha scene id must be one of {sorted(_ALOHA_SCENES)}, got {env_cfg.scene.id!r}"
        )

    # Must precede the gym_aloha import — dm_control reads MUJOCO_GL
    # exactly once when it picks its rendering plugin. ``os.environ``
    # assignment after the first dm_control import is a no-op.
    os.environ["MUJOCO_GL"] = "egl"

    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("aloha")

    try:
        import gym_aloha  # noqa: F401  (registers gym envs as a side effect)
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - tested via runtime error path
        # ensure_backend_deps re-probes after running its plan; this
        # branch is only reached when the install ran but gym_aloha
        # still refuses to import.
        raise ROSConfigError(
            "gym-aloha backend installed but `gym_aloha` still refuses to "
            "import. Inspect the auto-install output above and re-run: "
            "uv sync --all-packages --group sim --inexact"
        ) from exc

    env_id = _ALOHA_SCENES[env_cfg.scene.id]
    # gym-aloha's TimeLimit wrapper caps at 300 by default. Honour the
    # eval-layer ``task.max_steps`` so users can ask for longer rollouts
    # (the ACT cube-transfer reference run is ≈ 400 sim steps).
    env = gym.make(
        env_id,
        obs_type="pixels_agent_pos",
        render_mode=None,
        max_episode_steps=int(env_cfg.task.max_steps or 500),
    )
    return _AlohaSim(scene=env_cfg.scene, task=env_cfg.task, _env=env)


for _scene_id in _ALOHA_SCENES:
    # gym-aloha hard-wires the ALOHA bimanual rig; the scene rejects
    # mismatched robot_id values.
    SCENES.register(_scene_id, fixed_robot="aloha_bimanual")(_build_aloha_scene)
