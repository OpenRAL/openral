"""ManiSkill3 scene adapter — wraps SAPIEN-backed GPU manipulation envs.

ADR-0014. ManiSkill3 is opt-in via the ``maniskill3`` dependency group
(``just sync --all-packages --group maniskill3``); without it this module still imports
fine but the scene factory raises a typed :class:`ROSConfigError` with
the install hint.

Task ID convention
------------------
``"maniskill3/<env_id>"`` e.g. ``"maniskill3/PickCube-v1"``,
``"maniskill3/StackCube-v1"``. The ``<env_id>`` is passed through to
``gymnasium.make`` so the full ManiSkill3 task catalogue is reachable
without a per-task adapter edit (any future MS3 release adds new tasks
transparently). The ``scene.id`` MUST be ``"maniskill3"`` for
cross-field validation to succeed.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_MANISKILL3_SCENE_ID = "maniskill3"
# Mirrors the constant in :mod:`openral_sim.sim_runner` and
# :mod:`openral_sim.backends.simpler_env`. :meth:`SimRunner.activate`
# sets it to ``"1"`` for the duration of the scene-build window when
# ``openral sim run --view`` is on; we read it here to decide whether to
# build with deferred SAPIEN viewer plumbing.
_VIEW_ENV = "OPENRAL_SIM_VIEW"


def _parse_task_id(task_id: str) -> str:
    """Parse ``"maniskill3/<env_id>"`` and return ``<env_id>``."""
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0] != _MANISKILL3_SCENE_ID:
        raise ROSConfigError(f"maniskill3 task id must be 'maniskill3/<env_id>', got {task_id!r}")
    return parts[1]


def _reconcile_robot_uids(env_id: str, robot_uids: str) -> None:
    """Validate ``robot_uids`` against a ManiSkill3 task's ``SUPPORTED_ROBOTS``.

    MS3's per-task ``SUPPORTED_ROBOTS`` lists *base* agent uids only, so a
    registered camera-variant subclass (e.g. ``panda_wristcam``, a ``Panda``
    that only adds a ``hand_camera``) trips a false "not in the task's list of
    supported robots" warning even though it is genuinely usable. We walk the
    requested agent's MRO: if any base uid is supported, the variant is
    accepted; otherwise raise a typed :class:`ROSCapabilityMismatch` at the
    scene boundary instead of MS3's vague warning + downstream crash.

    Args:
        env_id: The bare ManiSkill3 env id (e.g. ``"PickCube-v1"``).
        robot_uids: The agent uid requested via
            ``scene.backend_options.robot_uids``.

    Raises:
        ROSCapabilityMismatch: ``robot_uids`` is neither a supported base
            robot nor a registered subclass of one for ``env_id``.
    """
    from mani_skill.agents.registration import (  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in dep (--group maniskill3), no py.typed
        REGISTERED_AGENTS,
    )
    from mani_skill.utils.registration import (  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in dep (--group maniskill3), no py.typed
        REGISTERED_ENVS,
    )

    env_spec = REGISTERED_ENVS.get(env_id)
    if env_spec is None:
        # Unknown env id — let gym.make raise its own (clearer) error.
        return
    supported = getattr(env_spec.cls, "SUPPORTED_ROBOTS", None)
    if not supported:
        # Task imposes no allowlist — any robot is acceptable.
        return
    # Entries may be tuples (multi-agent tasks); flatten to a uid set.
    supported_uids: set[str] = set()
    for entry in supported:
        if isinstance(entry, (list, tuple)):
            supported_uids.update(entry)
        else:
            supported_uids.add(entry)
    if robot_uids in supported_uids:
        return
    agent_spec = REGISTERED_AGENTS.get(robot_uids)
    if agent_spec is not None:
        for base in agent_spec.agent_cls.__mro__:
            if getattr(base, "uid", None) in supported_uids:
                # Registered camera-variant of a supported base — usable.
                return
    raise ROSCapabilityMismatch(
        f"ManiSkill3 task {env_id!r} does not support robot_uids={robot_uids!r}. "
        f"Supported base robots: {sorted(supported_uids)}. Set "
        "scene.backend_options.robot_uids to one of these, or to a registered "
        "camera-variant subclass of one (e.g. 'panda_wristcam' for 'panda')."
    )


_UNSUPPORTED_ROBOT_WARNING = "not in the task's list of supported robots"


class _DropUnsupportedRobotWarning(logging.Filter):
    """Drops MS3's false "unsupported robot" warning for validated variants."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _UNSUPPORTED_ROBOT_WARNING not in record.getMessage()


@contextlib.contextmanager
def _suppress_unsupported_robot_warning() -> Iterator[None]:
    """Silence MS3's false "not in the task's list of supported robots" warning.

    Used around ``gym.make`` *after* :func:`_reconcile_robot_uids` has already
    validated the requested robot (so the warning is provably a false positive
    about a registered camera-variant). Only that one message is dropped; every
    other ``mani_skill`` log record passes through untouched.
    """
    logger = logging.getLogger("mani_skill")
    filt = _DropUnsupportedRobotWarning()
    logger.addFilter(filt)
    try:
        yield
    finally:
        logger.removeFilter(filt)


def _scalar_float(value: object) -> float | None:
    """Return the first scalar value from a tensor/array-like object."""
    if value is None:
        return None
    if callable(value):
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return float(arr[0])


def _first_scalar_attr(obj: object, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        scalar = _scalar_float(value)
        if scalar is not None:
            return scalar
    return None


def _sapien_sim_time_ns(env: object) -> int | None:
    """Derive elapsed SAPIEN/ManiSkill control time from a live env.

    ManiSkill3/SimplerEnv expose the elapsed control step counter as an array or
    tensor (`elapsed_steps` / `_elapsed_steps`) and the control period as a
    scalar (`control_timestep`, `control_dt`, or a `control_freq` inverse).
    """
    candidates = (env, getattr(env, "unwrapped", env))
    elapsed_s_names = ("elapsed_time", "time_elapsed", "sim_time", "elapsed_seconds")
    step_names = ("elapsed_steps", "_elapsed_steps")
    dt_names = ("control_timestep", "_control_timestep", "control_dt", "dt")
    freq_names = ("control_freq", "_control_freq")

    for candidate in candidates:
        elapsed_s = _first_scalar_attr(candidate, elapsed_s_names)
        if elapsed_s is not None:
            return round(elapsed_s * 1_000_000_000)

        steps = _first_scalar_attr(candidate, step_names)
        if steps is None:
            continue

        dt_s = _first_scalar_attr(candidate, dt_names)
        if dt_s is None:
            freq_hz = _first_scalar_attr(candidate, freq_names)
            if freq_hz is not None and freq_hz > 0:
                dt_s = 1.0 / freq_hz
        if dt_s is not None:
            return round(steps * dt_s * 1_000_000_000)
    return None


@dataclass
class _ManiSkill3Sim:
    """Thin :class:`SimRollout` wrapper around a ManiSkill3 gym env.

    ManiSkill3 envs default to vectorised GPU rollouts; the eval-layer
    contract is single-env so we pass ``num_envs=1`` at construction and
    unwrap the leading batch dim on every observation / step.
    """

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # mani_skill env (lazy-imported)
    _last_image: NDArray[np.uint8] | None = None
    # Deferred-window mode for ``openral sim run --view`` (mirrors the
    # simpler_env backend in PR #160). When True the env was constructed
    # with ``render_mode=None`` and the SAPIEN viewer hasn't been opened
    # yet; the first :meth:`viewer_render` call promotes
    # ``env.unwrapped.render_mode`` to ``"human"`` so the window opens
    # *after* the slow policy build, not during it.
    _view_pending: bool = False

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        reward_f = float(_unbatch(reward))
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=reward_f,
            terminated=bool(_unbatch(terminated)),
            truncated=bool(_unbatch(truncated)),
            info=_unbatch_info(info),
        )

    def sim_time_ns(self) -> int | None:
        """Elapsed SAPIEN/ManiSkill simulation time in nanoseconds."""
        return _sapien_sim_time_ns(self._env)

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def viewer_render(self) -> None:
        """Pump the SAPIEN live viewer; promotes to ``human`` mode on first call.

        Picked up by :func:`openral_sim.sim_runner._open_viewer_and_pacing`
        as the engine-owns-the-viewer hook (returns a
        ``_SapienViewerProxy`` that the runner ``.sync()``-s after each
        applied step). The first call lazily promotes
        ``env.unwrapped.render_mode`` from ``None`` to ``"human"`` —
        MS3's ``render_human`` (``sapien_env.py``) then creates the
        SAPIEN viewer + runs ``_setup_viewer`` against the
        already-populated scene. Deferring window creation until the
        first ``viewer_render()`` call (i.e. after the runner has built
        the policy and started ticking) prevents the WM from marking an
        empty unresponsive window "Not Responding" during the multi-
        second policy load. Mirrors the simpler_env backend.
        """
        if self._view_pending:
            self._env.unwrapped.render_mode = "human"
            self._view_pending = False
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def _wrap_obs(self, obs: Any) -> Observation:
        """Translate ManiSkill3's nested dict obs into the eval Observation.

        ManiSkill3's ``sensor_data`` sub-dict carries one entry per
        registered camera (e.g. ``base_camera`` + ``hand_camera`` on
        ``panda_wristcam``). We surface them in declaration order as
        ``camera1`` / ``camera2`` / ... to match the scene-side names
        the franka_panda RobotDescription declares and that
        ``image_preprocessing.aliases`` in an rSkill manifest then
        renames to model-side keys (``up`` / ``wrist`` / ...).
        """
        flat = _unbatch_obs(obs)
        images = _extract_rgb_streams(flat)
        state = _extract_state(flat)
        if images:
            self._last_image = next(iter(images.values()))
        h = self.scene.observation_height
        w = self.scene.observation_width
        if not images:
            images = {"camera1": np.zeros((h, w, 3), dtype=np.uint8)}
        return {
            "images": images,
            "state": state,
            "task": self.task.instruction,
            "raw": flat,
        }


def _unbatch(value: Any) -> Any:
    """Strip a leading ``num_envs=1`` dim from a tensor / array / scalar."""
    if hasattr(value, "cpu"):  # torch tensor
        value = value.cpu().numpy()
    arr = np.asarray(value)
    return arr.reshape(-1)[0] if arr.size else arr


def _unbatch_info(info: dict[str, Any]) -> dict[str, Any]:
    """Recursively unbatch a ManiSkill3 info dict — preserves nested keys."""
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = _unbatch_info(v)
        elif hasattr(v, "shape") or isinstance(v, (list, tuple)):
            try:
                out[k] = _unbatch(v)
            except (ValueError, IndexError):
                out[k] = v
        else:
            out[k] = v
    return out


def _unbatch_obs(obs: Any) -> Any:
    """Recursively unbatch a ManiSkill3 obs dict.

    Returns ``Any`` (not ``dict[str, Any]``) because the recursion bottoms
    out on numpy arrays — the eval-layer callers (``_extract_rgb`` /
    ``_extract_state``) ``isinstance(flat, dict)`` before indexing into it.
    """
    if isinstance(obs, dict):
        return {k: _unbatch_obs(v) for k, v in obs.items()}
    if hasattr(obs, "cpu"):
        obs = obs.cpu().numpy()
    arr = np.asarray(obs)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _extract_rgb(flat: Any) -> NDArray[np.uint8] | None:
    """Pull the first RGB camera stream out of a ManiSkill3 obs dict.

    Kept for backwards-compatibility / tests that probe a single stream;
    the multi-camera surface used at runtime is :func:`_extract_rgb_streams`.
    """
    streams = _extract_rgb_streams(flat)
    if not streams:
        return None
    return next(iter(streams.values()))


def _extract_rgb_streams(flat: Any) -> dict[str, NDArray[np.uint8]]:
    """Pull every RGB camera stream from a ManiSkill3 obs dict.

    Returns an ordered ``{"camera1": rgb1, "camera2": rgb2, ...}`` map
    that mirrors the declaration order in ``sensor_data``. The keys are
    the scene-side names the franka_panda RobotDescription declares;
    an rSkill manifest's ``image_preprocessing.aliases`` block renames
    them to model-side keys before preprocessing.
    """
    if not isinstance(flat, dict):
        return {}
    sensor_data = flat.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return {}
    out: dict[str, NDArray[np.uint8]] = {}
    for idx, cam_obs in enumerate(sensor_data.values(), start=1):
        if isinstance(cam_obs, dict):
            rgb = cam_obs.get("rgb")
            if rgb is not None:
                out[f"camera{idx}"] = np.asarray(rgb, dtype=np.uint8)
    return out


def _extract_state(flat: Any) -> NDArray[np.float32]:
    """Concatenate ``agent.qpos`` + ``agent.qvel`` into a 1-D state vector."""
    if not isinstance(flat, dict):
        return np.zeros(0, dtype=np.float32)
    agent = flat.get("agent")
    if not isinstance(agent, dict):
        return np.zeros(0, dtype=np.float32)
    parts: list[NDArray[np.float32]] = []
    for key in ("qpos", "qvel"):
        value = agent.get(key)
        if value is not None:
            parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@SCENES.register(_MANISKILL3_SCENE_ID)
def _build_maniskill3_scene(env_cfg: SimEnvironment) -> _ManiSkill3Sim:
    """Lazily import ``mani_skill`` and build a :class:`_ManiSkill3Sim`."""
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("maniskill3")

    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in via --group maniskill3, registers gym envs
    except ImportError as exc:  # pragma: no cover — tested via runtime error path
        # ensure_backend_deps re-probes after running its plan; this
        # branch is only reached when the install ran but mani_skill
        # still refuses to import (e.g. SAPIEN wheel mismatch).
        raise ROSConfigError(
            "ManiSkill3 backend installed but `mani_skill` still refuses to "
            "import. Inspect the auto-install output above and re-run: "
            "uv sync --all-packages --group maniskill3 --inexact"
        ) from exc

    env_id = _parse_task_id(env_cfg.task.id)
    # Single-env eval — the harness expects one Observation per step, and
    # the per-(task, seed) outer loop in run_benchmark is the right place
    # to parallelise (clear semantics, OTel spans per episode).
    # ``state_dict+rgb`` exposes ``agent.qpos`` / ``agent.qvel`` as nested
    # dicts (what :func:`_extract_state` reads). The flat ``rgb+state``
    # mode collapses those into a single top-level tensor, which the
    # adapter would surface as an empty state vector — see ADR-0014.
    # robot_uids selects the MS3 agent variant — the default `panda` agent
    # carries a single `base_camera`; multi-camera rSkills (e.g. SmolVLA
    # with wrist + overhead views) need `panda_wristcam`, which adds the
    # `hand_camera` mount. Passed through only when set so the default
    # PickCube behaviour for single-camera configs is preserved.
    #
    # Deferred-window mode: when ``OPENRAL_SIM_VIEW=1`` (set by
    # :meth:`SimRunner.activate` for ``openral sim run --view``) we still
    # construct with ``render_mode=None`` and stash a ``_view_pending``
    # flag on the adapter. The first :meth:`viewer_render` call promotes
    # ``env.unwrapped.render_mode`` to ``"human"`` and lazily opens the
    # SAPIEN window — after the policy has loaded, so the WM never sees
    # an empty unresponsive window. Mirrors PR #160's simpler_env path.
    view_pending = os.environ.get(_VIEW_ENV) == "1"
    make_kwargs: dict[str, Any] = {
        "num_envs": 1,
        "obs_mode": env_cfg.scene.backend_options.get("obs_mode", "state_dict+rgb"),
        "control_mode": env_cfg.scene.backend_options.get("control_mode", "pd_ee_delta_pose"),
        "render_mode": None,
        # MS3's gym.register pins PickCube-v1 (and most tabletop tasks)
        # to max_episode_steps=50, which truncates rollouts at step 50
        # regardless of the YAML's task.max_steps. Forward the YAML's
        # value so long-horizon configs aren't silently clipped. Same
        # pattern as PR #160 for simpler_env.
        "max_episode_steps": env_cfg.task.max_steps,
        "sensor_configs": {
            "width": env_cfg.scene.observation_width,
            "height": env_cfg.scene.observation_height,
        },
    }
    robot_uids = env_cfg.scene.backend_options.get("robot_uids")
    if robot_uids is not None:
        # Reconcile against the task's SUPPORTED_ROBOTS: accept registered
        # camera-variants of a supported base (silencing MS3's false warning),
        # raise ROSCapabilityMismatch for genuinely-unsupported robots.
        _reconcile_robot_uids(env_id, str(robot_uids))
        make_kwargs["robot_uids"] = robot_uids
    with _suppress_unsupported_robot_warning():
        env = gym.make(env_id, **make_kwargs)
    return _ManiSkill3Sim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _env=env,
        _view_pending=view_pending,
    )
