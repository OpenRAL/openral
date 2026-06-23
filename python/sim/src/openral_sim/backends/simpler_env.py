"""SimplerEnv scene adapter — wraps the real-to-sim correlator envs.

ADR-0014. SimplerEnv is opt-in via the ``simpler-env`` dependency group
(``just sync --all-packages --group simpler-env``); without it this module still imports
fine but the scene factory raises a typed :class:`ROSConfigError` with
the install hint.

The metric SimplerEnv targets is the real-to-sim correlation (MMRV /
Pearson) — see Li et al. CoRL 2024. It exposes the same Google Robot +
WidowX setups used by RT-1, RT-2, Octo and OpenVLA. On the upstream
``maniskill3`` branch the eval envs are now ManiSkill3-registered
gymnasium environments (``mani_skill.envs.tasks.digital_twins.*``), so
this adapter shares the obs / state extraction helpers with
:mod:`openral_sim.backends.maniskill3`.

Task ID convention
------------------
``"simpler_env/<friendly_name>"`` e.g.
``"simpler_env/widowx_carrot_on_plate"``. The friendly name is
translated via ``simpler_env.ENVIRONMENT_MAP`` to the underlying MS3
env id + kwargs, so configs stay readable. Friendly names that are
not in the map fall through to ``gym.make(<friendly_name>)`` so users
can also pass a raw MS3 env id directly when needed.

Only the WidowX bridge tasks are registered end-to-end in MS3 v3.0.x. The
``google_robot_*`` friendly names exist in ``ENVIRONMENT_MAP`` but their MS3
envs (``GraspSingleOpenedCokeCanInScene``, ``MoveNearGoogleBakedTexInScene``,
…) are not registered upstream yet, so they raise ``NameNotFound`` — the
``simpler_env_google_robot`` benchmark suite was removed for that reason.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.backends.maniskill3 import (
    _extract_rgb,
    _extract_state,
    _sapien_sim_time_ns,
    _unbatch,
    _unbatch_info,
    _unbatch_obs,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


# Set to "1" by :meth:`openral_sim.SimRunner.activate` for the duration
# of the scene build when ``openral sim run --view`` is on. See the same
# constant in :mod:`openral_sim.sim_runner`. We read it here to decide
# whether to advertise ``render_mode='human'`` to ``gym.make`` — that
# is the toggle that tells SAPIEN to build a live window. Reading an
# env var (rather than threading a kwarg through every SCENES factory)
# keeps the registry contract untouched.
_VIEW_ENV = "OPENRAL_SIM_VIEW"


_SIMPLER_ENV_SCENE_ID = "simpler_env"
# SimplerEnv bridge envs ported into ManiSkill3 v3.0.x advertise a single
# ``rgb+segmentation`` obs mode (see SUPPORTED_OBS_MODES on
# BridgeDatasetEvalBase). Earlier branches accepted ``rgbd`` — the
# upstream ``simpler_env.make()`` still passes that, which is why
# calling it directly errors on a fresh MS3 install. Override via
# ``scene.backend_options.obs_mode`` if a future upstream relaxes it.
_DEFAULT_OBS_MODE = "rgb+segmentation"


def _parse_task_id(task_id: str) -> str:
    """Parse ``"simpler_env/<env_id>"`` and return ``<env_id>``."""
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0] != _SIMPLER_ENV_SCENE_ID:
        raise ROSConfigError(f"simpler_env task id must be 'simpler_env/<env_id>', got {task_id!r}")
    return parts[1]


def _bump_version_if_deprecated(env_id: str) -> str:
    """Round a ``-v0`` env id up to the highest registered version.

    Upstream ``simpler_env.ENVIRONMENT_MAP`` still ships ``-v0`` suffixes,
    but ManiSkill3 v3.0.x registers the same tasks as ``-v1``. Gymnasium's
    versioning resolver raises ``DeprecatedEnv`` instead of silently
    bumping, so do it here.
    """
    import gymnasium as gym

    if env_id in gym.envs.registry:
        return env_id
    if "-v" not in env_id:
        return env_id
    base, _, _ = env_id.rpartition("-v")
    candidates = sorted(
        (e for e in gym.envs.registry if e.startswith(base + "-v")),
        key=lambda e: int(e.rsplit("-v", 1)[1]),
        reverse=True,
    )
    return candidates[0] if candidates else env_id


def _resolve_friendly_name(task_name: str) -> tuple[str, dict[str, Any]]:
    """Translate a SimplerEnv friendly task name into ``(ms3_env_id, kwargs)``.

    Falls back to ``(task_name, {})`` when the name is already an MS3
    env id, so users can author configs against either naming style.
    """
    try:
        from simpler_env import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
            ENVIRONMENT_MAP,
        )
    except ImportError:
        return _bump_version_if_deprecated(task_name), {}
    if task_name in ENVIRONMENT_MAP:
        env_id, kwargs = ENVIRONMENT_MAP[task_name]
        return _bump_version_if_deprecated(env_id), dict(kwargs)
    return _bump_version_if_deprecated(task_name), {}


# SAPIEN exposes the WidowX TCP as ``ee_gripper_link`` (between the
# fingertips, matches what Bridge-Data's real-robot eef_pos was recorded
# from). Google's everyday-robot rig uses ``link_ee``.
_TCP_LINK_CANDIDATES: tuple[str, ...] = (
    "ee_gripper_link",  # WidowX (Bridge-data digital twin)
    "link_ee",  # Google everyday robot
    "tcp",  # generic fallback
)


def _compute_eef_pos(env: Any, qpos_last: float) -> NDArray[np.float32] | None:
    """Approximate the legacy MS2 SimplerEnv ``agent.eef_pos`` 8-vector.

    The upstream ``rldx/eval/sim/SimplerEnv/simpler_env.py`` consumes
    ``obs['agent']['eef_pos']`` — an 8-D ``[x, y, z, qw, qx, qy, qz,
    gripper]`` vector — but MS3 v3.0.x no longer exposes that field on
    the Bridge digital-twin envs. We rebuild it from the agent
    controller's ``ee_pose_at_base`` (the EEF pose expressed in the
    robot's root frame; SAPIEN quaternion order is ``[w, x, y, z]``)
    and the last qpos channel (mirrors what the legacy MS2 env wrapped
    into ``eef_pos[7]``).

    **Frame matters**: bridge_data_v2 recorded `end_effector_position`
    in the WidowX base frame, not world frame. The MS3 bridge env
    plants the WidowX at ``[0.147, 0.028, 0.870]`` in world (see
    ``mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/
    base_env.py:409``), so reading ``link.pose`` (world frame) would
    feed the RLDX-1-FT-SIMPLER-WIDOWX checkpoint a position offset by
    that ~85 cm vertical lift and the policy would emit garbage
    actions trying to return the EEF to its training workspace.
    """
    try:
        agent = env.unwrapped.agent
    except AttributeError:
        return None
    # Prefer the controller's ``ee_pose_at_base`` when available — it
    # already does ``root_link.pose.inv() * ee_link.pose`` (see
    # ``mani_skill/agents/controllers/pd_ee_pose.py:71``) and so
    # returns the base-frame pose the policy expects.
    controller = getattr(agent, "controller", None)
    pose_at_base = _resolve_pose_at_base(controller)
    if pose_at_base is None:
        # Fallback path for non-EE controllers: walk the link map and
        # transform the candidate link pose into base frame manually.
        pose_at_base = _resolve_pose_at_base_from_links(agent.robot)
    if pose_at_base is None:
        return None
    pos_t, quat_t = pose_at_base
    pos = _to_numpy(pos_t).reshape(-1)[:3].astype(np.float32)
    quat = _to_numpy(quat_t).reshape(-1)[:4].astype(np.float32)
    return np.concatenate([pos, quat, np.asarray([qpos_last], dtype=np.float32)])


def _resolve_pose_at_base(controller: Any) -> tuple[Any, Any] | None:
    """Return ``(pos, quat)`` from the controller's ``ee_pose_at_base``, if any.

    Most MS3 PD-EE controllers expose ``ee_pose_at_base`` as a
    property; for compound controllers the ``arm`` sub-controller is
    typically the one that owns it.
    """
    candidates = [controller]
    arm = getattr(controller, "controllers", None)
    if isinstance(arm, dict):
        candidates.append(arm.get("arm"))
    for cand in candidates:
        if cand is None:
            continue
        pose = getattr(cand, "ee_pose_at_base", None)
        if pose is None:
            continue
        return pose.p, pose.q
    return None


def _resolve_pose_at_base_from_links(robot: Any) -> tuple[Any, Any] | None:
    """Compute (ee_link.pose in root_link frame) by hand.

    Used when the agent's controller does not expose ``ee_pose_at_base``
    directly. We look up the first matching TCP link and the robot's
    root link, then return ``root.pose.inv() * ee.pose`` decomposed
    into (p, q).
    """
    links_map = getattr(robot, "links_map", None)
    if not isinstance(links_map, dict):
        return None
    root_link = getattr(robot, "root", None)
    if root_link is None:
        return None
    for name in _TCP_LINK_CANDIDATES:
        ee_link = links_map.get(name)
        if ee_link is None:
            continue
        relative = root_link.pose.inv() * ee_link.pose
        return relative.p, relative.q
    return None


def _to_numpy(value: Any) -> NDArray[np.float32]:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _resolve_gripper_qpos(flat: Any) -> float:
    """Return the last qpos channel — the gripper opening MS2 stored at index 7."""
    if not isinstance(flat, dict):
        return 0.0
    agent = flat.get("agent")
    if not isinstance(agent, dict):
        return 0.0
    qpos = agent.get("qpos")
    if qpos is None:
        return 0.0
    arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
    return float(arr[-1]) if arr.size else 0.0


@dataclass
class _SimplerEnvSim:
    """Thin :class:`SimRollout` wrapper around a SimplerEnv-via-MS3 gym env."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # ManiSkill3-registered gym env (lazy-imported)
    # When True, the env was constructed with ``render_mode=None`` and
    # we need to promote it to ``"human"`` (and lazily open the SAPIEN
    # viewer) on the first :meth:`viewer_render` call. This is the
    # deferred-window mode used by ``openral sim run --view`` so the live
    # SAPIEN window doesn't open during the slow rldx sidecar boot
    # (during which the runner's main thread is blocked and the WM
    # would mark the empty window "Not Responding"). Set by
    # :func:`_build_simpler_env_scene` when ``OPENRAL_SIM_VIEW=1`` and
    # cleared on first promotion.
    _view_pending: bool = False
    _last_image: NDArray[np.uint8] | None = None

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        # SimplerEnv envs run with num_envs=1 under the hood; reshape so
        # the upstream batched API accepts the single-env vector.
        action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=float(_unbatch(reward)),
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
        MS3's ``render_human`` (``sapien_env.py:1355``) then creates
        the SAPIEN viewer + runs ``_setup_viewer`` against the
        already-populated scene (env was reset during ``gym.make``'s
        ``__init__``, so the carrot/plate/robot are present from
        construction). Doing this lazily here, rather than at
        ``gym.make`` time, lets the SAPIEN window open *after* the
        slow rldx sidecar boot — the runner's first viewer_render()
        call fires from inside ``_step_tick``, post-policy-build, so
        the WM never sees an empty unresponsive window.
        """
        if self._view_pending:
            self._env.unwrapped.render_mode = "human"
            self._view_pending = False
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def _wrap_obs(self, obs: Any) -> Observation:
        """Translate the underlying ManiSkill3 obs into the eval Observation."""
        flat = _unbatch_obs(obs)
        image = _extract_rgb(flat)
        state = _extract_state(flat)
        # Rebuild the legacy SimplerEnv ``agent.eef_pos`` vector so the
        # rldx ``simpler_widowx`` / ``simpler_google`` wire-shapers can
        # consume the same 8-D ``[x, y, z, qw, qx, qy, qz, gripper]``
        # contract the upstream reference implementation reads.
        eef_pos = _compute_eef_pos(self._env, _resolve_gripper_qpos(flat))
        if eef_pos is not None and isinstance(flat, dict):
            agent = flat.setdefault("agent", {})
            if isinstance(agent, dict):
                agent["eef_pos"] = eef_pos
        if image is not None:
            self._last_image = image
        h = self.scene.observation_height
        w = self.scene.observation_width
        if image is None:
            image = np.zeros((h, w, 3), dtype=np.uint8)
        return {
            "images": {"camera1": image},
            "state": state,
            "task": self.task.instruction,
            "raw": flat,
        }


@SCENES.register(_SIMPLER_ENV_SCENE_ID)
def _build_simpler_env_scene(env_cfg: SimEnvironment) -> _SimplerEnvSim:
    """Lazily import ``simpler_env`` and build a :class:`_SimplerEnvSim`.

    Delegates the install chain to :func:`openral_sim._deps.ensure_backend_deps`
    (`backend_id="simpler_env"`) so the first-use UX mirrors LIBERO /
    RoboCasa: one Rich banner + ``typer.confirm`` (bypassed by
    ``OPENRAL_AUTO_INSTALL_DEPS=1``) and the upstream git package gets
    pinned via ``uv pip install`` -- not ``uv run pip install``, which
    silently no-ops on this venv layout.
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("simpler_env")

    try:
        import gymnasium as gym
        import simpler_env  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: registers gym envs as a side effect of import
    except ImportError as exc:  # pragma: no cover — tested via runtime error path
        # ensure_backend_deps re-probes after running its plan; this
        # branch only fires when the install ran but `import simpler_env`
        # still refuses to resolve. Surface the manual hint as a typed
        # error so users have somewhere to look.
        raise ROSConfigError(
            "SimplerEnv backend installed but `import simpler_env` still fails. "
            "Inspect the auto-install output above and re-run: "
            "uv sync --all-packages --group simpler-env --inexact && "
            "uv pip install "
            '"simpler-env @ git+https://github.com/simpler-env/SimplerEnv.git@maniskill3"'
        ) from exc

    task_name = _parse_task_id(env_cfg.task.id)
    env_id, env_kwargs = _resolve_friendly_name(task_name)

    # Upstream ``simpler_env.make`` injects ``prepackaged_config=True`` and
    # ``obs_mode='rgbd'`` — both reject on MS3 v3.0.x (the bridge envs
    # only accept ``rgb+segmentation``). Calling ``gym.make`` directly
    # with the resolved kwargs keeps the adapter forward-compatible with
    # whatever the registered env actually supports.
    obs_mode = env_cfg.scene.backend_options.get("obs_mode", _DEFAULT_OBS_MODE)
    # SAPIEN/ManiSkill3 opens its viewer eagerly inside the env's
    # ``__init__`` (line ~327 of ``sapien_env.py``: ``self.reset(...)``
    # is called during construction, which calls ``_reconfigure``,
    # which then ``create_viewer`` whenever ``render_mode == "human"``).
    # That means passing ``render_mode='human'`` to ``gym.make`` here
    # would surface the SAPIEN window during the (slow, blocking)
    # rldx sidecar boot — the WM marks the unresponsive window "Not
    # Responding" and may force-close it.
    #
    # Deferred-window mode: when ``OPENRAL_SIM_VIEW=1`` we construct
    # with ``render_mode=None`` (no viewer) and stash a
    # ``_view_pending`` flag on the adapter. The first
    # ``viewer_render()`` call — which fires only after the runner has
    # built the policy and started ticking — promotes
    # ``env.unwrapped.render_mode`` to ``"human"`` and triggers
    # ``render_human()``, which lazily opens the SAPIEN viewer against
    # the already-populated scene.
    view_pending = os.environ.get(_VIEW_ENV) == "1"
    # MS3 registers each bridge / fractal env with a canonical
    # ``max_episode_steps`` (60 for the carrot/spoon/cube tasks, 120 for
    # the eggplant-in-basket task). Pass the YAML's ``task.max_steps``
    # through so the env's TimeLimitWrapper honours the user's budget —
    # otherwise long-horizon interactive runs get silently truncated
    # back to 60 steps. MS3's ``TimeLimitWrapper`` introspects the
    # ``gym.make`` frame and prefers the caller-supplied value when
    # non-None (see ``mani_skill.utils.registration.TimeLimitWrapper``).
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode=None,
        max_episode_steps=env_cfg.task.max_steps,
        **env_kwargs,
    )
    return _SimplerEnvSim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _env=env,
        _view_pending=view_pending,
    )
