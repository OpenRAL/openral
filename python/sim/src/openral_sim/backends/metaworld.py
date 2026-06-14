"""MetaWorld scene adapter — wraps :class:`lerobot.envs.metaworld.MetaworldEnv`.

Wiring matches ``scenes/benchmark/metaworld_push.yaml``:
1× RGB camera at 480×480 → resized to ``scene.observation_height/width``,
4-D ``agent_pos`` proprioception, 4-D action (delta XYZ + gripper).

Task ID convention
------------------
``"metaworld/<task-name>"`` e.g. ``"metaworld/push-v3"``,
``"metaworld/pick-place-v3"``, ``"metaworld/reach-v3"``. The ``scene.id`` MUST
be ``"metaworld"`` for cross-field validation to succeed.
"""

from __future__ import annotations

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


_METAWORLD_SCENE_ID = "metaworld"
_METAWORLD_RENDER_SIZE = 480


def _parse_task_id(task_id: str) -> str:
    """Parse ``"metaworld/<task-name>"`` and return ``<task-name>``."""
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0] != _METAWORLD_SCENE_ID:
        raise ROSConfigError(f"metaworld task id must be 'metaworld/<task-name>', got {task_id!r}")
    return parts[1]


@dataclass
class _MetaworldSim:
    """Thin :class:`SimRollout` wrapper around ``MetaworldEnv``."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # MetaworldEnv (lazy-imported)
    _last_image: np.ndarray | None = None  # type: ignore[type-arg]  # reason: shape-flexible image

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_np = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self._last_image is None:
            return None
        return self._last_image.astype(np.uint8).copy()

    def close(self) -> None:
        self._env.close()

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Reach through lerobot's ``MetaworldEnv`` to the underlying MuJoCo handles.

        MetaWorld's gymnasium env exposes ``model`` (``mujoco.MjModel``) and
        ``data`` (``mujoco.MjData``) directly on the unwrapped env. Returns
        ``None`` if the chain is broken (e.g. a future MetaWorld release
        renames the attrs).
        """
        candidates = (
            getattr(self._env, "unwrapped", None),
            getattr(self._env, "_env", None),
            self._env,
        )
        env = next(
            (c for c in candidates if c is not None and getattr(c, "model", None) is not None),
            None,
        )
        if env is None:
            return None
        model = getattr(env, "model", None)
        data = getattr(env, "data", None)
        if model is None or data is None:
            return None
        return model, data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1), or None.

        Reads ``MjData.time`` off :meth:`mujoco_handles`. Monotonic within an
        episode; rewinds on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        """Translate MetaworldEnv obs into the eval-layer Observation schema."""
        image = obs.get("pixels")
        if image is None:
            image = np.zeros(
                (self.scene.observation_height, self.scene.observation_width, 3),
                dtype=np.uint8,
            )
        else:
            image = np.asarray(image, dtype=np.uint8)
        self._last_image = image

        agent_pos = np.asarray(obs.get("agent_pos", np.zeros(4)), dtype=np.float32)

        return {
            "images": {"camera1": image},
            "state": agent_pos,
            "task": self.task.instruction,
        }


@SCENES.register(_METAWORLD_SCENE_ID, fixed_robot="sawyer")
def _build_metaworld_scene(env_cfg: SimEnvironment) -> _MetaworldSim:
    """Lazily import ``lerobot.envs.metaworld`` and build a :class:`_MetaworldSim`."""
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("metaworld")

    try:
        from lerobot.envs.metaworld import (
            MetaworldEnv,  # reason: heavy import deferred until used
        )
    except ImportError as exc:  # pragma: no cover - tested via runtime error path
        # ensure_backend_deps re-probes after running its plan; this
        # branch is only reached when the install ran but
        # lerobot.envs.metaworld / metaworld still refuse to import.
        raise ROSConfigError(
            "MetaWorld backend installed but `lerobot.envs.metaworld` still "
            "refuses to import. Inspect the auto-install output above and "
            "re-run: uv sync --all-packages --group metaworld --inexact && "
            "uv pip install metaworld==3.0.0 --no-deps"
        ) from exc

    task_name = _parse_task_id(env_cfg.task.id)
    # MetaworldEnv has no ``episode_length`` kwarg in lerobot's current API; the
    # per-episode cap is enforced by the runner loop via ``task.max_steps``.
    env = MetaworldEnv(
        task=task_name,
        obs_type="pixels_agent_pos",
        observation_height=_METAWORLD_RENDER_SIZE,
        observation_width=_METAWORLD_RENDER_SIZE,
    )
    return _MetaworldSim(scene=env_cfg.scene, task=env_cfg.task, _env=env)
