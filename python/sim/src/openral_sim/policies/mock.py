"""Mock scene + policies — no physics, no weights.

Used by unit tests and as a wiring smoketest. The mock scene exposes a fixed
observation schema and a configurable action dimensionality, and the mock
policies emit deterministic actions so end-to-end tests can pass on CPU-only
runners without downloading any HF weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openral_sim.registry import POLICIES, SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec

    from openral_sim.rollout import Observation


_MOCK_ACTION_DIM = 7
_MOCK_STATE_DIM = 8


@dataclass
class _MockSim:
    """Tiny gym-like env used for tests.

    The "physics" is just a step counter; success fires after
    :attr:`success_step` steps. Image observations are zeros of the right
    shape so any downstream image preprocessing path runs end-to-end.
    """

    scene: SceneSpec
    task: TaskSpec
    success_step: int = 5
    action_dim: int = _MOCK_ACTION_DIM
    _step: int = 0
    _rng: np.random.Generator | None = None

    def reset(self, seed: int | None = None) -> Observation:
        self._step = 0
        self._rng = np.random.default_rng(seed)
        return self._observe()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"mock scene expects action_dim={self.action_dim}, got {action.shape}")
        self._step += 1
        terminated = self._step >= self.success_step
        success = terminated  # mock success: terminate after N steps
        mock_info: dict[str, object] = {}
        if self.task.success_key is not None:
            mock_info[self.task.success_key] = success
        return StepResult(
            observation=self._observe(),
            reward=1.0 if success else 0.0,
            terminated=terminated,
            truncated=False,
            info=mock_info,
        )

    def render(self) -> NDArray[np.uint8] | None:
        h = self.scene.observation_height
        w = self.scene.observation_width
        return np.zeros((h, w, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        h = self.scene.observation_height
        w = self.scene.observation_width
        return {
            "images": {
                "camera1": np.zeros((h, w, 3), dtype=np.uint8),
            },
            "state": np.zeros(_MOCK_STATE_DIM, dtype=np.float32),
            "task": self.task.instruction,
        }


def _coerce_int(value: object, default: int) -> int:
    """Best-effort int coercion for values pulled from a dict[str, object]."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"cannot coerce {value!r} to int")


@SCENES.register("mock")
def _build_mock_scene(env_cfg: SimEnvironment) -> _MockSim:
    """Build the mock scene from the SimEnvironment.

    The optional ``backend_options`` dict supports:
        ``success_step`` (int): step at which the mock task succeeds.
        ``action_dim``   (int): action vector dimensionality.
    """
    opts = env_cfg.scene.backend_options
    return _MockSim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        success_step=_coerce_int(opts.get("success_step"), 5),
        action_dim=_coerce_int(opts.get("action_dim"), _MOCK_ACTION_DIM),
    )


@dataclass
class _ZeroPolicy:
    """Always emits zero-vector actions of the configured size."""

    spec: VLASpec
    device: str
    action_dim: int = _MOCK_ACTION_DIM

    def reset(self) -> None:
        return None

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        del observation, instruction
        return np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        return None


@dataclass
class _RandomPolicy:
    """Emits actions sampled from a fixed-seed Gaussian."""

    spec: VLASpec
    device: str
    action_dim: int = _MOCK_ACTION_DIM
    _rng: np.random.Generator | None = None

    def reset(self) -> None:
        seed_extra = self.spec.extra.get("seed", 0)
        self._rng = np.random.default_rng(_coerce_int(seed_extra, 0))

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        del observation, instruction
        if self._rng is None:
            self.reset()
        assert self._rng is not None
        return self._rng.standard_normal(self.action_dim).astype(np.float32) * 0.01

    def close(self) -> None:
        return None


def _resolve_action_dim(env_cfg: SimEnvironment) -> int:
    explicit = env_cfg.vla.extra.get("action_dim")
    if explicit is not None:
        return _coerce_int(explicit, _MOCK_ACTION_DIM)
    explicit_scene = env_cfg.scene.backend_options.get("action_dim")
    if explicit_scene is not None:
        return _coerce_int(explicit_scene, _MOCK_ACTION_DIM)
    # Defaults for registered scenes so the mock zero policy works
    # end-to-end without forcing the user to pass `action_dim` overrides.
    scene_default = _SCENE_DEFAULT_ACTION_DIM.get(env_cfg.scene.id)
    if scene_default is not None:
        return scene_default
    # Isaac Sim (ADR-0045) serves two sidecar layouts under one scene id:
    # lift_cube is 8-D (7 arm joint deltas + gripper); bowl_plate is the
    # LIBERO 7-D OSC-pose delta. The dim is layout-, not id-, determined.
    if env_cfg.scene.id == "isaac_sim":
        layout = env_cfg.scene.backend_options.get("layout", "lift_cube")
        return 7 if layout == "bowl_plate" else 8
    # GR1 tabletop scenes share a single 29-D action shape (right arm 7
    # + left arm 7 + waist 3 + right Fourier hand 6 + left Fourier
    # hand 6); special-case the prefix so we don't enumerate all 24
    # task ids in _SCENE_DEFAULT_ACTION_DIM.
    if env_cfg.scene.id.startswith("robocasa/gr1/"):
        return 29
    return _MOCK_ACTION_DIM


# Per-scene action dims used by mock policies (zero/random) when no override
# is supplied. Values match each scene adapter's underlying gym/robosuite env.
_SCENE_DEFAULT_ACTION_DIM: dict[str, int] = {
    "pusht": 2,
    "libero_spatial": 7,
    "libero_object": 7,
    "libero_goal": 7,
    "libero_10": 7,
    "metaworld": 4,
    "aloha_bimanual": 14,
    # isaac_sim is intentionally absent — its dim is layout-determined; see
    # the layout branch in _resolve_action_dim (ADR-0045).
}


@POLICIES.register("zero")
def _build_zero_policy(env_cfg: SimEnvironment) -> _ZeroPolicy:
    return _ZeroPolicy(
        spec=env_cfg.vla,
        device="cpu",
        action_dim=_resolve_action_dim(env_cfg),
    )


@POLICIES.register("random")
def _build_random_policy(env_cfg: SimEnvironment) -> _RandomPolicy:
    return _RandomPolicy(
        spec=env_cfg.vla,
        device="cpu",
        action_dim=_resolve_action_dim(env_cfg),
    )
