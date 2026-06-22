"""LIBERO scene adapter — wraps :class:`lerobot.envs.libero.LiberoEnv`.

Driven by a :class:`openral_core.SimScene` config, this adapter is
the canonical entry point for LIBERO physics-backed rollouts (used by
``scenes/benchmark/libero_spatial.yaml`` and other rSkill-backed configs).

LIBERO is opt-in: the underlying ``lerobot[libero]`` group is heavy and only
linux-supported. The factory imports lazily so this module is safe to import
on any platform; the failure surfaces at ``make_env()`` time with an
actionable message.

Task ID convention
------------------
``"<suite>/<task_id>"`` where ``<suite>`` is one of
``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10`` and
``<task_id>`` is an integer index inside that suite. The suite name MUST
also be the ``scene.id`` so :class:`openral_core.SimEnvironment`
cross-field validation passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

_log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _load_manifest_for_spec(spec: Any) -> Any:
    """Load the rSkill manifest from ``spec.weights_uri`` (bare rSkill reference).

    Mirrors :func:`openral_sim.policies.act._load_manifest_for_spec`. Returns
    ``None`` for explicit-scheme URIs (``hf://`` etc.) which carry no local
    manifest, so a raw-repo VLASpec still works.
    """
    weights_uri = str(getattr(spec, "weights_uri", "") or "")
    if weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        return None
    from openral_rskill.loader import load_rskill_manifest

    return load_rskill_manifest(weights_uri)


def _resolve_control_mode(env_cfg: SimEnvironment) -> str:
    """Resolve the LIBERO OSC controller mode for this (scene, policy) pair.

    Precedence (explicit beats implicit, CLAUDE.md §1.4):

    1. ``scene.backend_options["control_mode"]`` — an explicit per-scene pin.
    2. The policy manifest's ``sim_env_control_mode`` — lets an absolute-control
       policy (xVLA) declare it needs ``"absolute"`` so it runs on the canonical
       ``libero_spatial.yaml`` without a duplicate per-policy scene.
    3. ``"relative"`` — LiberoEnv's OSC delta-EE default (SmolVLA / π0.5 / rldx1
       / molmoact2 / GR00T).
    """
    explicit = env_cfg.scene.backend_options.get("control_mode")
    if explicit is not None:
        return str(explicit)
    manifest = _load_manifest_for_spec(env_cfg.vla)
    declared = getattr(manifest, "sim_env_control_mode", None) if manifest is not None else None
    if declared:
        return str(declared)
    return "relative"


def _parse_task_id(task_id: str, scene_id: str) -> int:
    """Parse ``"<suite>/<int>"`` and validate ``<suite> == scene_id``.

    Args:
        task_id: Composite task identifier from :class:`TaskSpec.id`.
        scene_id: Expected suite name from :class:`SceneSpec.id`.

    Returns:
        The integer task index.

    Raises:
        ROSConfigError: If the format is wrong or the suite mismatches.
    """
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts:
        raise ROSConfigError(f"libero task id must be '<suite>/<int>', got {task_id!r}")
    suite, idx = parts
    if suite != scene_id:
        raise ROSConfigError(
            f"libero task suite ({suite!r}) does not match scene id ({scene_id!r})"
        )
    try:
        return int(idx)
    except ValueError as exc:
        raise ROSConfigError(
            f"libero task index {idx!r} is not an integer (task_id={task_id!r})"
        ) from exc


@dataclass
class _LiberoSim:
    """Thin :class:`SimRollout` wrapper around ``LiberoEnv``."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # LiberoEnv (lazy-imported)
    _last_pixels: dict[str, np.ndarray]  # type: ignore[type-arg]  # reason: heterogeneous numpy dict

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

    @property
    def action_dim(self) -> int:
        """Flat action width the env's ``step`` accepts (LIBERO OSC_POSE = 7).

        ADR-0036 — ``SimAttachedHAL._probe_env_action_dim`` reads this so a
        cartesian rSkill's slot-packed action is sized to the LIBERO action
        space (6-D OSC end-effector delta + gripper) rather than the
        robosuite-mobile-manipulator fallback. Without it, ``openral deploy sim``
        on a standard LIBERO suite scene raised ``ROSConfigError`` at HAL
        ``connect`` and the franka HAL never configured.

        robosuite exposes the per-robot action width on each ``robots`` entry.
        The suite backend's ``self._env`` is lerobot's ``LiberoEnv``, which wraps
        robosuite's ``OffScreenRenderEnv`` on its private ``_env`` attr — so we
        walk the same wrapper chain ``mujoco_handles`` uses to find the env that
        carries ``robots``, robust to robosuite's cross-release re-layering.
        """
        candidates = (
            getattr(self._env, "_env", None),
            self._env,
            getattr(getattr(self._env, "unwrapped", None), "_env", None),
        )
        robots = next(
            (getattr(c, "robots", None) for c in candidates if getattr(c, "robots", None)),
            None,
        )
        if robots is None:
            raise ROSConfigError(
                "_LiberoSim.action_dim: could not reach robosuite `robots` through "
                "the LiberoEnv wrapper chain to resolve the env action width."
            )
        return int(sum(int(r.action_dim) for r in robots))

    def render(self) -> NDArray[np.uint8] | None:
        agentview = self._last_pixels.get("image")
        if agentview is None:
            return None
        return agentview.copy().astype(np.uint8)

    def close(self) -> None:
        self._env.close()

    def mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData] | None:
        """Reach through lerobot+robosuite to expose the underlying MuJoCo handles.

        ``self._env`` is lerobot's :class:`LiberoEnv`, which holds
        robosuite's ``OffScreenRenderEnv`` on its private ``_env`` attr;
        that env exposes ``.sim`` (a ``robosuite.utils.binding_utils.MjSim``)
        whose ``.model._model`` / ``.data._data`` are the raw
        ``mujoco.MjModel`` / ``MjData`` the passive viewer needs.

        Returns ``None`` (and the CLI falls back to offscreen) when any
        link in the chain is missing — robosuite reorganises these paths
        between releases.
        """
        candidates = (
            getattr(self._env, "_env", None),
            self._env,
            getattr(getattr(self._env, "unwrapped", None), "_env", None),
        )
        sim = next((getattr(c, "sim", None) for c in candidates if c is not None), None)
        if sim is None:
            return None
        model = getattr(getattr(sim, "model", None), "_model", None)
        data = getattr(getattr(sim, "data", None), "_data", None)
        if model is None or data is None:
            return None
        return model, data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1), or None.

        Reads ``MjData.time`` off :meth:`mujoco_handles`. Monotonic within an
        episode; the LIBERO env rewinds the clock on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        """Translate LiberoEnv obs into the eval-layer Observation schema."""
        pixels = obs.get("pixels", {})
        self._last_pixels = {k: np.asarray(v) for k, v in pixels.items()}

        # Build 8-D state: eef_pos(3) ‖ quat→axisangle(3) ‖ gripper_qpos(2).
        # Matches the LiberoProcessorStep convention used by lerobot/smolvla_libero.
        robot_state = obs.get("robot_state", {})
        eef = robot_state.get("eef", {})
        eef_pos = eef.get("pos")
        eef_quat = eef.get("quat")
        gripper_qpos = robot_state.get("gripper", {}).get("qpos")

        state_8d: NDArray[np.float32] = np.zeros(8, dtype=np.float32)
        if eef_pos is not None and eef_quat is not None:
            pos = np.asarray(eef_pos, dtype=np.float32)
            axisangle = _quat_to_axisangle(np.asarray(eef_quat, dtype=np.float32))
            gr = (
                np.asarray(gripper_qpos, dtype=np.float32)
                if gripper_qpos is not None
                else np.zeros(2, dtype=np.float32)
            )
            state_8d = np.concatenate([pos, axisangle, gr]).astype(np.float32)

        return {
            "images": {
                "camera1": self._last_pixels.get("image", np.zeros((256, 256, 3), dtype=np.uint8)),
                "camera2": self._last_pixels.get("image2", np.zeros((256, 256, 3), dtype=np.uint8)),
            },
            "state": state_8d,
            "task": getattr(self._env, "task_description", self.task.instruction),
            # Preserve the raw nested LiberoEnv observation for adapters that
            # need the full robot_state dict (e.g. xVLA's env preprocessor
            # consumes eef.mat / joints.pos / gripper.qvel directly).
            "raw": obs,
        }


def _quat_to_axisangle(quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a (4,) ``[x, y, z, w]`` quaternion to a (3,) axis-angle vector."""
    eps = 1e-10
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den > eps:
        angle = 2.0 * np.arccos(w)
        axis = quat[:3] / den
        out: NDArray[np.float32] = (axis * angle).astype(np.float32)
        return out
    return np.zeros(3, dtype=np.float32)


def _ensure_libero_config_matches_active_install() -> None:
    """Repoint LIBERO's global ``~/.libero/config.yaml`` at the ACTIVE install.

    LIBERO caches absolute benchmark / init-state / asset paths in a global
    ``~/.libero/config.yaml`` (or ``$LIBERO_CONFIG_PATH``). That file is
    written once from whichever ``libero`` package first generated it, so when
    a *different* venv imports libero — a sibling git worktree, or a deploy-sim
    launch subprocess whose venv differs from the one that seeded the config —
    ``get_libero_path("init_states")`` keeps resolving to the OTHER venv's path.
    If that path lacks the ``*.pruned_init`` files (e.g. an incomplete install),
    ``LiberoEnv`` construction dies with a ``FileNotFoundError`` deep in
    ``get_task_init_states``.

    Fix: when the configured ``init_states`` path is missing, regenerate the
    config from the *active* package's own location (``get_default_path_dict()``
    defaults to the dir of the imported ``libero.__file__``). Idempotent once the
    config already matches a complete install. This is a repair of a known
    pre-existing cross-venv path bug, not new behaviour.
    """
    import os

    from libero import libero as libero_pkg  # the active install's package

    try:
        init_states = libero_pkg.get_libero_path("init_states")
    except Exception:  # reason: unreadable/absent config → regenerate
        init_states = ""
    if init_states and os.path.isdir(init_states):
        return  # config already valid for this install

    import yaml

    defaults = libero_pkg.get_default_path_dict()
    config_file = libero_pkg.config_file
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(defaults, f)
    _log.warning(
        "libero.config_repointed_to_active_install",
        config_file=config_file,
        init_states=defaults["init_states"],
        reason="configured init_states path was missing (stale cross-venv config)",
    )


def _build_libero_scene(env_cfg: SimEnvironment) -> _LiberoSim:
    """Lazily import ``lerobot.envs.libero`` and build a :class:`_LiberoSim`."""
    if env_cfg.scene.id not in _LIBERO_SUITES:
        raise ROSConfigError(
            f"libero scene id must be one of {_LIBERO_SUITES}, got {env_cfg.scene.id!r}"
        )

    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("libero")

    try:
        from lerobot.envs.libero import LiberoEnv, _get_suite
    except ImportError as exc:  # pragma: no cover - tested via runtime error path
        # ensure_backend_deps re-probes after running its plan so this
        # branch is only reached when the install ran but
        # lerobot.envs.libero still refuses to import (e.g. partial
        # C-extension compile failure). Keep the typed error so the
        # user has somewhere to look.
        raise ROSConfigError(
            "LIBERO backend installed but lerobot.envs.libero still refuses "
            "to import. Inspect the auto-install output above and re-run: "
            "CC=/usr/bin/gcc just sync --all-packages --group libero"
        ) from exc

    # Repair a stale cross-venv ~/.libero/config.yaml before any path lookup
    # (deploy-sim launch subprocesses + sibling worktrees seed it to the wrong
    # venv, breaking get_task_init_states with a FileNotFoundError).
    _ensure_libero_config_matches_active_install()

    task_index = _parse_task_id(env_cfg.task.id, env_cfg.scene.id)
    suite = _get_suite(env_cfg.scene.id)
    # ``control_mode`` is policy-dependent: SmolVLA / π0.5 emit deltas
    # (LiberoEnv default ``"relative"``) while xVLA emits absolute targets.
    # Resolved from the scene pin, else the policy manifest's
    # ``sim_env_control_mode``, else the relative default — so xVLA runs on the
    # canonical libero_spatial.yaml without a per-policy scene variant.
    control_mode = _resolve_control_mode(env_cfg)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_index,
        task_suite_name=env_cfg.scene.id,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        observation_height=env_cfg.scene.observation_height,
        observation_width=env_cfg.scene.observation_width,
        episode_length=env_cfg.task.max_steps or 100,
        control_mode=control_mode,
    )
    return _LiberoSim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _env=env,
        _last_pixels={},
    )


for _suite in _LIBERO_SUITES:
    # LIBERO's MuJoCo physics hard-wire the Franka Panda; the scene rejects
    # any robot_id that disagrees so users get a typed ROSConfigError rather
    # than a silently-swapped robot.
    SCENES.register(_suite, fixed_robot="franka_panda")(_build_libero_scene)
