"""VLABench scene adapter — wraps :class:`lerobot.envs.configs.VLABenchEnv`.

VLABench (ICCV 2025, OpenMOSS) is a MuJoCo + dm_control language-conditioned
manipulation benchmark on a Franka Panda arm. lerobot 0.6.0 ships a native
``vlabench`` env config; this adapter drives it through OpenRAL's unbatched
:class:`SimRollout` contract.

Env contract (``obs_type="pixels_agent_pos"``, ``n_envs=1``):
- ``obs["pixels"]`` — dict of 3 RGB views ``image`` / ``second_image`` /
  ``wrist_image`` (480×480×3), batched ``(1, H, W, 3)``.
- ``obs["agent_pos"]`` — 7-D proprio, batched ``(1, 7)``.
- action — 7-D end-effector, batched ``(1, 7)``.
- ``info["is_success"]`` — per-env success flag.

``lerobot``'s ``VLABenchEnv.create_envs`` returns a nested
``{task_group: {idx: SyncVectorEnv}}`` mapping; we extract the single
``SyncVectorEnv`` and unwrap the ``n_envs=1`` batch dimension here.

Task ID convention: ``"vlabench/<task-name>"`` (e.g. ``"vlabench/select_fruit"``).
``scene.id`` MUST be ``"vlabench"``.

Provisioning. The Python side (clone + editable install + numpy-2 sim
deps + rrt-algorithms stub) is handled by :func:`ensure_backend_deps` under the
``"vlabench"`` plan, auto-installed on first env build (``OPENRAL_AUTO_INSTALL_DEPS``).
The plan pins MuJoCo 3.2.2 + dm_control 1.0.22, matching the upstream evaluator;
newer loose pairs fail during dm_control model indexing.
The ~12 GB CC-BY asset bundle is a separate one-time Google-Drive fetch that this
backend does NOT auto-download — like the CoppeliaSim/RLBench backend it leaves the
heavy external artefact to the user and raises with the exact recipe when it is
absent. ``VLABENCH_ROOT`` (or ``OPENRAL_VLABENCH_ROOT``) points at the ``VLABench/``
package dir that holds ``assets/``; it defaults to the clone the plan installs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_VLABENCH_SCENE_ID = "vlabench"
# Env agent_pos is 7-D (pos[3]+euler[3]+gripper[1]); the smolvla_vlabench
# normalizer stats + dataset (lerobot/vlabench_unified observation.state) are
# 7-D, so the full vector is fed through (the checkpoint config.json's [6] is
# stale metadata — the runtime normalizer buffer is 7-D).
_VLABENCH_STATE_DIM = 7
_VLABENCH_ACTION_DIM = 7
# lerobot's VLABenchEnv obs["pixels"] sub-keys → OpenRAL canonical camera keys.
# We emit camera1/2/3 (the franka_panda robot's declared sensor keys); the
# smolvla checkpoint's own preprocessor rename (image→camera1, …) then no-ops.
_VLABENCH_CAMERA_MAP = {
    "image": "camera1",
    "second_image": "camera2",
    "wrist_image": "camera3",
}
# Xiaomi's XR-1 evaluator and VLABench's own eval path select the raw camera
# tuple as front(index 2), second/base(index 0), wrist(index 3). LeRobot 0.6.0's
# wrapper currently keeps indices 0,1,2, silently dropping the wrist view.
_VLABENCH_POLICY_CAMERA_INDICES = (2, 0, 3)
_IMAGE_RANK = 3
_GRAYSCALE_CHANNELS = 1
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4
_log = structlog.get_logger(__name__)


def _to_hwc_uint8(frame: Any) -> NDArray[np.uint8]:
    """Normalize one VLABench raw RGB frame without flipping it."""
    arr = np.asarray(frame)
    while arr.ndim > _IMAGE_RANK and arr.shape[0] == 1:
        arr = arr[0]
    channels = (_GRAYSCALE_CHANNELS, _RGB_CHANNELS, _RGBA_CHANNELS)
    if arr.ndim == _IMAGE_RANK and arr.shape[0] in channels and arr.shape[-1] not in channels:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != _IMAGE_RANK:
        raise ROSConfigError(f"VLABench RGB frame must be rank-3, got shape {arr.shape}.")
    if arr.shape[-1] == _GRAYSCALE_CHANNELS:
        arr = np.repeat(arr, _RGB_CHANNELS, axis=-1)
    elif arr.shape[-1] == _RGBA_CHANNELS:
        arr = arr[..., :_RGB_CHANNELS]
    elif arr.shape[-1] != _RGB_CHANNELS:
        raise ROSConfigError(f"VLABench RGB frame must have 1, 3, or 4 channels, got {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _select_policy_cameras(raw_rgb: Any) -> dict[str, NDArray[np.uint8]]:
    """Select front/base/wrist in Xiaomi's policy order; preserve row orientation."""
    frames = [_to_hwc_uint8(frame) for frame in np.asarray(raw_rgb)]
    if len(frames) <= max(_VLABENCH_POLICY_CAMERA_INDICES):
        raise ROSConfigError(
            "VLABench raw observation must expose at least four RGB cameras "
            f"for indices {_VLABENCH_POLICY_CAMERA_INDICES}, got {len(frames)}."
        )
    selected = [frames[index] for index in _VLABENCH_POLICY_CAMERA_INDICES]
    return dict(zip(_VLABENCH_CAMERA_MAP.values(), selected, strict=True))


def _parse_task_id(task_id: str) -> str:
    """Parse ``"vlabench/<task-name>"`` → ``<task-name>``."""
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0] != _VLABENCH_SCENE_ID:
        raise ROSConfigError(f"vlabench task id must be 'vlabench/<task-name>', got {task_id!r}")
    return parts[1]


def _unbatch(value: Any) -> NDArray[Any]:
    """Drop the leading ``n_envs=1`` batch dim from a SyncVectorEnv array."""
    arr = np.asarray(value)
    return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr


@dataclass
class _VLABenchSim:
    """Thin :class:`SimRollout` wrapper around a VLABench ``SyncVectorEnv``."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # gymnasium SyncVectorEnv (n_envs=1), lazy-built
    _last_image: NDArray[np.uint8] | None = None
    _camera_fallback_warned: bool = False

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        # SyncVectorEnv expects a batched (1, action_dim) action.
        action_b = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action_b)
        success = bool(_unbatch(info.get("is_success", False)))
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=float(_unbatch(reward)),
            terminated=bool(_unbatch(terminated)),
            truncated=bool(_unbatch(truncated)),
            info={"is_success": success},
        )

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        cameras = self._policy_cameras(obs)
        self._last_image = cameras["camera1"]
        # Full 7-D agent_pos (pos[3] + euler[3] + gripper[1]) — matches the
        # checkpoint's 7-D normalizer stats + the vlabench_unified dataset's
        # observation.state (the config.json's [6] is stale metadata).
        state = _unbatch(obs["agent_pos"]).astype(np.float32)
        return {"images": cameras, "state": state, "task": self.task.instruction}

    def _policy_cameras(self, obs: dict[str, Any]) -> dict[str, NDArray[np.uint8]]:
        """Read the raw four-camera tuple when available; fall back for old wrappers."""
        envs: Any = getattr(self._env, "envs", ())
        wrapper = envs[0] if len(envs) == 1 else None
        inner = getattr(wrapper, "_env", None)
        if inner is not None:
            raw = inner.get_observation()
            if isinstance(raw, dict) and "rgb" in raw:
                return _select_policy_cameras(raw["rgb"])

        if not self._camera_fallback_warned:
            _log.warning(
                "vlabench_raw_camera_fallback",
                note=(
                    "Could not reach the raw four-camera observation; falling back "
                    "to LeRobot's three-camera mapping, which may omit the wrist view."
                ),
            )
            self._camera_fallback_warned = True
        pixels = obs["pixels"]
        return {
            cam_key: _unbatch(pixels[env_key]).astype(np.uint8)
            for env_key, cam_key in _VLABENCH_CAMERA_MAP.items()
        }

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def close(self) -> None:
        self._env.close()

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        # VLABench wraps a dm_control Physics behind a SyncVectorEnv; no direct
        # (MjModel, MjData) reach-through, so the runner falls back to wall-clock.
        return None


def _resolve_vlabench_root() -> None:
    """Ensure ``VLABENCH_ROOT`` points at the installed VLABench asset dir."""
    if os.environ.get("VLABENCH_ROOT"):
        return
    override = os.environ.get("OPENRAL_VLABENCH_ROOT")
    if override:
        os.environ["VLABENCH_ROOT"] = override
        return
    import VLABench  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in vlabench backend, externally provisioned

    os.environ["VLABENCH_ROOT"] = os.path.dirname(VLABench.__file__)


def _check_vlabench_assets() -> None:
    """Verify the ~12 GB asset bundle is unpacked under ``VLABENCH_ROOT/assets``.

    The clone ships ``assets/base`` + ``assets/robots``, but the object + scene
    meshes (``assets/obj``, ``assets/scenes``) come from the separate Google-Drive
    download and are what every task's MJCF references. We do NOT auto-fetch them
    (a 12 GB gdown pull is too flaky to drive unattended); raise with
    the exact recipe when they are missing so the failure is legible instead of a
    ``FileNotFoundError`` deep in dm_control at ``env.reset()``.
    """
    root = Path(os.environ["VLABENCH_ROOT"])
    obj_dir = root / "assets" / "obj"
    if obj_dir.is_dir() and any(obj_dir.iterdir()):
        return
    scripts = root.parent / "scripts" / "download_assets.py"
    raise ROSConfigError(
        f"VLABench asset bundle missing under {root / 'assets'} (no populated "
        "assets/obj). It is a one-time ~12 GB CC-BY download from Google Drive; "
        f"fetch it with:\n  VLABENCH_ROOT={root} python {scripts}\n"
        "See this module's docstring for the full provisioning recipe."
    )


def provision_vlabench() -> None:
    """Install VLABench and verify its ~12 GB bundle ahead of ``ros2 launch``.

    ``ensure_backend_deps`` clones + installs the package; the asset bundle is
    a separate Google-Drive pull we deliberately never automate, so
    :func:`_check_vlabench_assets` only verifies it and raises the recipe when
    it is missing. Preflighting both means the operator reads that recipe on a
    terminal instead of hitting the HAL's 300 s ``on_configure`` bound.

    Idempotent — the install probe and the ``assets/obj`` check both
    short-circuit once warm.

    Raises:
        ROSConfigError: When the install fails or the asset bundle is absent.
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("vlabench")
    _resolve_vlabench_root()
    _check_vlabench_assets()


def _build_vlabench_scene(env_cfg: SimEnvironment) -> _VLABenchSim:
    """Lazily construct a VLABench primitive-task env via lerobot's config."""
    if env_cfg.scene.id != _VLABENCH_SCENE_ID:
        raise ROSConfigError(
            f"vlabench scene id must be {_VLABENCH_SCENE_ID!r}, got {env_cfg.scene.id!r}"
        )
    task_name = _parse_task_id(env_cfg.task.id)

    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("vlabench")
    try:
        _resolve_vlabench_root()
        from lerobot.envs.configs import VLABenchEnv
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "VLABench backend not installed; the 'vlabench' install plan "
            "(openral_sim._deps) auto-provisions it — re-run with "
            "OPENRAL_AUTO_INSTALL_DEPS=1, or install manually per the "
            "provisioning recipe in this module's docstring and set "
            "VLABENCH_ROOT."
        ) from exc
    _check_vlabench_assets()

    cfg = VLABenchEnv(
        task=task_name,
        obs_type="pixels_agent_pos",
        episode_length=int(env_cfg.task.max_steps or 500),
        render_resolution=(
            env_cfg.scene.observation_height,
            env_cfg.scene.observation_width,
        ),
    )
    env_groups = cfg.create_envs(n_envs=1)
    # create_envs returns {task_group: {idx: SyncVectorEnv}}; take the sole env.
    group = env_groups[next(iter(env_groups))]
    env = group[next(iter(group))]
    return _VLABenchSim(scene=env_cfg.scene, task=env_cfg.task, _env=env)


SCENES.register(_VLABENCH_SCENE_ID, fixed_robot="franka_panda", provision=provision_vlabench)(
    _build_vlabench_scene
)
