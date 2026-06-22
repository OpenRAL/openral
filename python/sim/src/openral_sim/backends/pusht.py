"""gym-pusht scene adapter — wraps ``gym_pusht/PushT-v0``.

Driven by a :class:`openral_core.SimEnvironment` config, this adapter is
the canonical entry point for PushT (pymunk 2-D rigid-body) rollouts (used
by ``tests/sim/test_pusht_2d_diffusion_pusht.py``).

gym-pusht is opt-in: the underlying ``gym_pusht`` package is heavy and only
linux-supported. The factory imports lazily so this module is safe to
import on any platform; the failure surfaces at ``make_env()`` time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_PUSHT_SCENE_ID = "pusht"
_PUSHT_ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class _PushTSim:
    """Thin :class:`SimRollout` wrapper around ``gym_pusht/PushT-v0``."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # gymnasium env
    _last_pixels: NDArray[np.uint8] | None = None
    # Live-view pygame window populated by enable_intrinsic_viewer().
    # ``render_mode="human"`` is *not* an option here: gym_pusht zeroes
    # the pixel obs when human-mode owns the window, which breaks the
    # Diffusion Policy adapter (it consumes ``observation.image``). We
    # therefore keep the env in rgb_array mode and paint each frame to
    # our own window from inside reset / step.
    _view_screen: Any = None
    _view_pygame: Any = None

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        wrapped = self._wrap_obs(obs)
        self._paint_view()
        return wrapped

    def step(self, action: NDArray[np.float32]) -> StepResult:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self._env.step(a)
        wrapped = self._wrap_obs(obs)
        self._paint_view()
        return StepResult(
            observation=wrapped,
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
        if self._view_pygame is not None:
            try:
                self._view_pygame.display.quit()
            finally:
                self._view_screen = None
                self._view_pygame = None
        self._env.close()

    def enable_intrinsic_viewer(self) -> None:
        # Open a pygame window we drive from _paint_view(). SimRunner
        # calls this once at activate() time, before the first reset, so
        # the window exists for every paint that follows.
        import pygame

        pygame.init()
        pygame.display.init()
        # gym_pusht renders at 96x96 by default; upscale 5x so the
        # rollout is legible in a desktop window without forcing a
        # second resampling step on the policy's 96x96 input.
        size = (480, 480)
        self._view_screen = pygame.display.set_mode(size)
        pygame.display.set_caption("openral sim — gym_pusht (Diffusion Policy)")
        self._view_pygame = pygame

    def _paint_view(self) -> None:
        # No-op when --view is off. When on, blit the last pixel frame
        # (HWC uint8 RGB) scaled to the window. The pixel frame is the
        # same array the policy consumes, so what you see is exactly
        # what the network sees.
        if self._view_screen is None or self._last_pixels is None:
            return
        pygame = self._view_pygame
        # pygame.surfarray expects (W, H, 3); env frames are (H, W, 3).
        surf = pygame.surfarray.make_surface(self._last_pixels.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, self._view_screen.get_size())
        self._view_screen.blit(surf, (0, 0))
        pygame.display.flip()
        # Pump the OS event queue so the window stays responsive (and
        # the WM doesn't mark it "not responding"). We drain quit / X
        # events with the same effect as a deadman switch.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # User closed the window — best-effort: close the window,
                # let the rollout continue offscreen. We deliberately
                # don't raise; killing the policy mid-rollout would
                # invalidate the trace.
                pygame.display.quit()
                self._view_screen = None
                self._view_pygame = None
                return

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        """Translate gym-pusht obs into the eval-layer Observation schema."""
        pixels = obs.get("pixels")
        agent_pos = obs.get("agent_pos")

        pixel_arr: NDArray[np.uint8] | None = None
        if pixels is not None:
            pixel_arr = np.asarray(pixels).astype(np.uint8)
            self._last_pixels = pixel_arr

        # Diffusion Policy adapter reads `images["camera1"]` by default;
        # expose the PushT topdown stream under the scene's first camera
        # (canonical ``top`` per ADR-0069, falling back to ``camera1``).
        images: dict[str, NDArray[np.uint8]] = {}
        if pixel_arr is not None:
            cam0 = self.scene.cameras[0] if self.scene.cameras else "camera1"
            images[cam0] = pixel_arr

        state = (
            np.asarray(agent_pos, dtype=np.float32)
            if agent_pos is not None
            else np.zeros(2, dtype=np.float32)
        )

        return {
            "images": images,
            "state": state,
            "task": self.task.instruction,
            "raw": obs,
        }


def _build_pusht_scene(env_cfg: SimEnvironment) -> _PushTSim:
    """Lazily import ``gym_pusht`` and build a :class:`_PushTSim`."""
    if env_cfg.scene.id != _PUSHT_SCENE_ID:
        raise ROSConfigError(
            f"pusht scene id must be {_PUSHT_SCENE_ID!r}, got {env_cfg.scene.id!r}"
        )

    try:
        import gym_pusht  # noqa: F401  (registers gym envs as a side effect)
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "gym-pusht backend not installed; install with: just sync --all-packages --group sim"
        ) from exc

    env = gym.make(
        _PUSHT_ENV_ID,
        obs_type="pixels_agent_pos",
    )
    return _PushTSim(scene=env_cfg.scene, task=env_cfg.task, _env=env)


# gym-pusht hard-wires the 2-D pusher; the scene rejects mismatched robot_id values.
SCENES.register(_PUSHT_SCENE_ID, fixed_robot="pusht_2d")(_build_pusht_scene)
