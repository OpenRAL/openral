"""RoboCasa scene adapter -- MuJoCo kitchen tasks via robosuite + robocasa.

ADR-0015. The RoboCasa backend is opt-in via the ``robocasa`` dependency
group (`just sync --all-packages --group robocasa` + a manual ``uv pip install
"robocasa @ git+https://github.com/robocasa/robocasa.git"`` per
``pyproject.toml`` comments). Without it this module still imports
cleanly but the scene factories raise a typed :class:`ROSConfigError`
with the install hint.

Scene-id conventions
--------------------
* ``"robocasa/<task>"`` for the ~100 atomic prebuilt tasks (e.g.
  ``"robocasa/PnPCounterToCab"``). The ``<task>`` slug is passed through
  to ``robosuite.make(env_name=<task>, robots=[...])`` so the full
  RoboCasa catalogue is reachable without per-task adapter edits.
* ``"robocasa"`` (no slash) for the **procedural** scenario surface:
  the user authors a kitchen via :class:`RoboCasaBackendOptions` inside
  ``SceneSpec.backend_options`` (mode='procedural'), and this adapter
  resolves the (style x layout x fixtures x objects x task_verb) tuple
  into the matching robosuite env.

In both cases the env's CC-BY-4.0 kitchen assets are fetched lazily on
first use via :func:`openral_sim._assets.ensure_robocasa_assets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import RoboCasaBackendOptions
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

from openral_sim._assets import ensure_robocasa_assets
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_PREBUILT_SCENE_PREFIX = "robocasa"
_PROCEDURAL_SCENE_ID = "robocasa"


def _canonicalize_quat_xyzw_np(q: NDArray[np.float32]) -> NDArray[np.float32]:
    """Force ``w >= 0`` hemisphere on a 4-vec ``(x, y, z, w)`` quaternion.

    Mirror of ``openral_state_adapter.layouts.human300_16d.
    _canonicalize_quat_xyzw`` but accepts / returns numpy float32 so
    it fits straight into the obs-assembly path here. Apply at every
    site that materialises a quaternion into the state vector the
    policy sees (sim_run via this module AND deploy_sim via the
    ADR-0027 assembler) so both paths emit byte-identical bytes for the
    same physical rotation -- ``q`` and ``-q`` represent the same
    rotation but encode differently, and the two paths were landing
    on opposite hemispheres for ~half of the per-step quats.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    if w > 0.0:
        return q
    if w < 0.0:
        return np.asarray((-x, -y, -z, -w), dtype=np.float32)
    # w == 0 (180° rotation) — pick first non-zero of (x, y, z) positive.
    if x > 0.0 or (x == 0.0 and (y > 0.0 or (y == 0.0 and z >= 0.0))):
        return q
    return np.asarray((-x, -y, -z, -w), dtype=np.float32)


def _arm_part_config(controller_name: str) -> dict[str, Any]:
    """Load the default robosuite per-part config for a single arm controller.

    robosuite ships the per-part defaults as JSON under
    ``robosuite/controllers/config/default/parts/<controller>.json``
    (joint_position, osc_pose, etc.). We load the JSON shipped by the
    installed robosuite at runtime so a robosuite upgrade picks up the
    new defaults automatically.
    """
    import json as _json
    from pathlib import Path as _Path

    import robosuite as _rs

    parts_dir = _Path(_rs.__file__).parent / "controllers" / "config" / "default" / "parts"
    fname = parts_dir / f"{controller_name.lower()}.json"
    if not fname.is_file():
        raise ROSConfigError(
            f"robosuite ships no default config for arm controller {controller_name!r}; "
            f"expected {fname}"
        )
    return _json.loads(fname.read_text())  # type: ignore[no-any-return]


# A small curated set of prebuilt RoboCasa tasks we register up-front.
# Keeping this list short keeps `openral sim list` legible; users who need a
# different task can still pass `--scene robocasa/<task>` directly --
# the adapter resolves any robosuite-registered env_name. Promote
# additional entries here when they become benchmark targets. Sourced
# from robocasa/environments/kitchen/atomic/*.py at robocasa 1.0.1.
_CURATED_PREBUILT_TASKS: tuple[str, ...] = (
    "PickPlaceCounterToCabinet",
    "PickPlaceCabinetToCounter",
    "PickPlaceCounterToSink",
    "PickPlaceSinkToCounter",
    "PickPlaceCounterToMicrowave",
    "PickPlaceMicrowaveToCounter",
    "PickPlaceStoveToCounter",
    "PickPlaceCounterToBlender",
    "OpenDoor",
    "CloseDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    "TurnOnStove",
    "TurnOffStove",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "NavigateKitchen",
)
"""Tasks registered with ``<prefix>/<task>`` scene ids at import time.

Names match the keys robosuite uses for ``robosuite.make(env_name=...)``
(robocasa registers them via ``@register_env`` -- see
``robocasa/environments/kitchen/atomic/*.py``). The ``PickPlace*`` envs
are the atomic pick-and-place tasks; the upstream names use the
``Pick`` / ``Place`` prefix today, not the shorter ``PnP`` we used in
the issue-#88 draft.
"""


# The 24 GR1 tabletop tasks shipped by the upstream fork
# https://github.com/robocasa/robocasa-gr1-tabletop-tasks. Each name is
# the **robosuite env class** (defined under
# robocasa/environments/tabletop/*.py in the fork), NOT the gymnasium
# id used by the GR00T inference service. We register one scene id per
# task as ``robocasa/gr1/<TaskName>`` and pair it with the
# `gr1` RobotDescription. The GR1 fork only exposes the
# tabletop catalogue (no kitchen envs) so a host installs **either**
# the upstream `robocasa` kitchen package **or** the GR1 fork --
# never both -- and the two task families coexist in this curated
# tuple purely for ergonomics: the unavailable one raises a typed
# robosuite "unknown env_name" at adapter build time.
_GR1_TABLETOP_TASKS: tuple[str, ...] = (
    "PnPCupToDrawerClose",
    "PnPPotatoToMicrowaveClose",
    "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose",
    "PnPWineToCabinetClose",
    "PnPCanToDrawerClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)


_GR1_SCENE_PREFIX = "robocasa/gr1"
"""Scene-id prefix for GR1 tabletop tasks (registered with `gr1`)."""


_GR1_BASIC_DIM = 29
"""Fourier GR-1 BASIC composite action width: waist 3 + right_arm 7 +
left_arm 7 + right_hand 6 + left_hand 6 = 29.

Kept in sync with ``openral_sim.policies.rldx._GR1_BASIC_DIM`` (same
value, duplicated to keep the backend / policy layers
import-independent per CLAUDE.md §6.1)."""


# Gymnasium >= 0.26 reset returns ``(obs, info)``; older robosuite envs
# return just ``obs``. Named for clarity in the reset() dispatch.
_GYMNASIUM_RESET_TUPLE_LEN = 2


@dataclass
class _RoboCasaSim:
    """SimRollout wrapper around a robosuite/robocasa kitchen env.

    robosuite's env returns a flat dict observation keyed by topic
    (e.g. ``robot0_agentview_left_image``); we map the first camera to
    ``camera1`` and concatenate robot proprioception into ``state`` so
    the eval-layer contract matches the other adapters.
    """

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # robosuite env OR gymnasium-wrapped env (lazy-imported)
    _camera_keys: tuple[str, ...]
    _state_layout: str = "human300_16d"
    _last_image: NDArray[np.uint8] | None = None
    _debug_step: int = 0
    # True iff `_env` is a `gymnasium.Env` (returns (obs, info) on reset
    # and (obs, reward, terminated, truncated, info) on step, AND takes
    # a dict action with `action.<part>` keys). Set for GR1 envs that
    # we build via `gym.make('gr1_unified/...')`. Kitchen / Panda envs
    # stay on the raw `robosuite.make(...)` path with `_is_gymnasium_wrapped=False`.
    _is_gymnasium_wrapped: bool = False
    # Names of the robosuite robot compositions loaded into this env.
    # Used to gate panda_mobile-specific obs-dict additions (base
    # velocity, synthetic 2D laser) without affecting GR1 / Panda /
    # other compositions. See `_emit_panda_mobile_extras`.
    _robots: tuple[str, ...] = ()

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None and hasattr(self._env, "rng"):
            # robocasa's own gym_wrapper.reset uses this exact pattern;
            # robosuite envs don't expose set_random_state, so reseeding
            # the env.rng is the supported way to make placement
            # sampling deterministic.
            self._env.rng = np.random.default_rng(int(seed))
        result = self._env.reset() if not self._is_gymnasium_wrapped else self._env.reset(seed=seed)
        # Gymnasium >= 0.26 reset returns (obs, info); raw robosuite
        # returns just obs.
        raw = (
            result[0]
            if isinstance(result, tuple) and len(result) == _GYMNASIUM_RESET_TUPLE_LEN
            else result
        )
        self._debug_step = 0
        self._log_eef_distance(raw, step=0, action=None)
        return self._wrap_obs(raw)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_arr: NDArray[np.float32] = np.asarray(action, dtype=np.float32).reshape(-1)
        # GR1 path: robosuite GR1 envs (BASIC composite) take a
        # per-part DICT action with `robot0_left / robot0_right /
        # robot0_torso / robot0_left_gripper / robot0_right_gripper`
        # keys (the upstream GR1ArmsAndWaistKeyConverter.unmap_action
        # contract). Our adapter ships a flat 29-D vector laid out as
        # ``[waist(3) | right_arm(7) | left_arm(7) | right_hand(6) |
        # left_hand(6)]`` (the openral state ordering); unflatten back
        # into the dict and let the env step it. Skip the 11-vs-12
        # PandaMobile torso-skew logic entirely.
        if self._is_gymnasium_wrapped:
            # gymnasium GR1 env takes `action.{waist,right_arm,left_arm,
            # right_hand,left_hand}` keys directly; the wrapper
            # un-mapps them to robosuite per-controller slices.
            action_env: Any = self._to_gr1_action_dict_gym(action_arr)
        elif self._state_layout == "gr1":
            # Legacy code path retained for testing — raw robosuite
            # GR1 envs accept the per-controller dict via
            # GR1ArmsAndWaistKeyConverter.unmap_action keys.
            action_env = self._to_gr1_action_dict(action_arr)
        else:
            # Reconcile a benign ±1 action-width skew: the lerobot pi0.5 /
            # pi0 RoboCasa checkpoints were trained on a 12-D PandaMobile
            # action (arm 7 + gripper 1 + base 3 + torso 1), but the
            # robosuite master we install exposes 11-D (torso folded out
            # under the BASIC composite). Slice or pad so the downstream
            # length assert holds.
            env_dim = int(getattr(self._env, "action_dim", action_arr.shape[-1]))
            if action_arr.shape[-1] != env_dim:
                if action_arr.shape[-1] == env_dim + 1:
                    # 12-D dataset, 11-D env: drop the trailing torso slot
                    # (the dataset always recorded it as -1; verified from
                    # the pi0.5 RoboCasa-MG_300 unnormalizer per-dim stats).
                    action_arr = np.ascontiguousarray(action_arr[:env_dim])
                elif action_arr.shape[-1] == env_dim - 1:
                    # Inverse skew: re-append the torso slot at -1 (lowest).
                    pad: NDArray[np.float32] = -np.ones(1, dtype=np.float32)
                    action_arr = np.concatenate([action_arr, pad])
            action_env = action_arr
        # robosuite's `step` returns (obs, reward, done, info); the
        # gymnasium 5-tuple is only on robosuite>=1.5 envs that wrap
        # `GymWrapper`. Handle both shapes.
        result = self._env.step(action_env)
        gym_legacy_tuple_len = 4
        if len(result) == gym_legacy_tuple_len:
            raw, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raw, reward, terminated_raw, truncated_raw, info = result
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)

        success = self._check_success_fallback(terminated)
        self._debug_step += 1
        self._log_eef_distance(raw, step=self._debug_step, action=action_arr)
        step_info: dict[str, object] = dict(info)
        if self.task.success_key is not None:
            step_info[self.task.success_key] = success
        return StepResult(
            observation=self._wrap_obs(raw),
            reward=float(reward) if reward is not None else 0.0,
            terminated=terminated or success,
            truncated=truncated and not success,
            info=step_info,
        )

    def _check_success_fallback(self, terminated: bool) -> bool:
        """Read task success from raw RoboCasa envs and gymnasium-wrapped GR1 envs."""
        check = getattr(self._env, "_check_success", None)
        if callable(check):
            return bool(check())
        inner = getattr(getattr(self._env, "unwrapped", self._env), "env", None)
        check = getattr(inner, "_check_success", None)
        if callable(check):
            return bool(check())
        return bool(terminated)

    def _split_gr1_action(
        self, action_arr: NDArray[np.float32]
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Validate + slice a 29-D GR1 action vector into the five Fourier groups.

        Returns ``(waist, right_arm, left_arm, right_hand, left_hand)``
        as float32 slices in the openral GR1 state ordering — matches the
        upstream GR1ArmsAndWaistKeyConverter.map_obs output and the rldx
        adapter's ``_GR1_ACTION_KEYS`` ordering in
        ``openral_sim.policies.rldx._assemble_gr1_chunk``.
        """
        if action_arr.shape[-1] != _GR1_BASIC_DIM:
            raise ROSConfigError(
                f"GR1 action vector must be {_GR1_BASIC_DIM}-D "
                "(waist 3 + right_arm 7 + left_arm 7 + right_hand 6 + left_hand 6); "
                f"got {action_arr.shape[-1]}-D. Confirm the rskill is rldx1-ft-gr1-nf4 "
                "or any other 29-D GR1-Fourier policy."
            )
        a = action_arr.astype(np.float32)
        return a[0:3], a[3:10], a[10:17], a[17:23], a[23:29]

    def _to_gr1_action_dict_gym(
        self, action_arr: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        """Unflatten a 29-D GR1 action vector → gymnasium wrapper dict.

        Output keys match the action space of the upstream
        ``gr1_unified/<task>_GR1ArmsAndWaistFourierHands_Env``:
        ``action.{waist,right_arm,left_arm,right_hand,left_hand}``.
        """
        waist, right_arm, left_arm, right_hand, left_hand = self._split_gr1_action(action_arr)
        return {
            "action.waist": waist,
            "action.right_arm": right_arm,
            "action.left_arm": left_arm,
            "action.right_hand": right_hand,
            "action.left_hand": left_hand,
        }

    def _to_gr1_action_dict(
        self, action_arr: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        """Unflatten a 29-D GR1 action vector → robosuite per-part dict.

        Output dict matches what
        ``GR1ArmsAndWaistKeyConverter.unmap_action`` would return given
        ``action.{waist,right_arm,left_arm,right_hand,left_hand}`` —
        i.e. robosuite's BASIC composite per-part keys.
        """
        waist, right_arm, left_arm, right_hand, left_hand = self._split_gr1_action(action_arr)
        return {
            "robot0_torso": waist,
            "robot0_right": right_arm,
            "robot0_left": left_arm,
            "robot0_right_gripper": right_hand,
            "robot0_left_gripper": left_hand,
        }

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def refresh_obs(self) -> Observation | None:
        """Drive a zero-action env.step so observations + cameras refresh.

        Used by :class:`openral_hal.sim_attached.SimAttachedHAL` after a
        BODY_TWIST qpos write. The ``sim run`` path works correctly
        because every iteration calls ``env.step(action)`` which
        re-renders cameras through robosuite's standard pipeline; the
        ``deploy sim`` BODY_TWIST path was bypassing ``env.step``
        entirely (because robocasa's BASIC composite controller does
        NOT interpret the first 3 action slots as planar velocities
        on OmronMobileBase — non-zero base ctrl is a no-op) so
        cameras + state slots never refreshed.

        Fix: keep the direct base-qpos write (it's how the base
        actually moves), then drive ``env.step`` with a ZERO action
        purely as the "tick the env" call. Zero action means:

        * arm OSC delta == 0    → arm holds its current pose
        * base velocity == 0    → no controller effort on the base
                                  (and qvel is 0 because we never wrote
                                   it — physics step keeps qpos at the
                                   value we just wrote)
        * gripper == 0          → no gripper actuation

        This re-renders every camera (same path ``sim run`` uses), so
        the dashboard PERCEPTION cards now update on every twist
        application instead of staying on the connect-time frame.

        Returns ``None`` for backends without ``env.step`` / a
        readable ``action_dim`` — those paths keep the
        cached-from-last-step images.
        """
        env = self._env
        action_dim = getattr(env, "action_dim", None)
        step = getattr(env, "step", None)
        if action_dim is None or step is None:
            return None
        zero_action = np.zeros(int(action_dim), dtype=np.float32)
        try:
            result = step(zero_action)
        except Exception:  # reason: defensive — env step failure must not crash the HAL
            return None
        raw = result[0] if isinstance(result, tuple) and result else None
        if not isinstance(raw, dict):
            return None
        wrapped = self._wrap_obs(raw)
        # ``Observation`` is ``dict[str, Any]`` (openral_sim.rollout) —
        # use mapping access, not attribute access.
        wrapped_images = wrapped.get("images") or {}
        last_image = wrapped_images.get("agentview")
        if last_image is not None:
            self._last_image = np.asarray(last_image, dtype=np.uint8)
        return wrapped

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def _log_eef_distance(
        self,
        raw: dict[str, Any],
        *,
        step: int,
        action: NDArray[np.float32] | None,
    ) -> None:
        # Log only at episode boundaries: t=0 (after reset) and at the
        # task horizon (step==600). That's enough to see whether the eef
        # made progress without flooding a multi-seed sweep with traces.
        horizon = int(getattr(self._env, "horizon", 600) or 600)
        if step not in (0, horizon):
            return
        import structlog

        eef = raw.get("robot0_eef_pos")
        obj = raw.get("obj_to_robot0_eef_pos")
        grip = raw.get("robot0_gripper_qpos")
        if eef is None or obj is None:
            return
        eef_np = np.asarray(eef, dtype=np.float32)
        obj_np = np.asarray(obj, dtype=np.float32)
        dist = float(np.linalg.norm(obj_np))
        fields: dict[str, Any] = {
            "step": step,
            "eef_pos": [round(float(x), 3) for x in eef_np.tolist()],
            "obj_to_eef": [round(float(x), 3) for x in obj_np.tolist()],
            "obj_eef_dist_m": round(dist, 3),
        }
        if grip is not None:
            fields["gripper_qpos"] = [round(float(x), 3) for x in np.asarray(grip).tolist()]
        if action is not None:
            fields["action_eef_pos_delta"] = [round(float(x), 3) for x in action[0:3].tolist()]
            fields["action_eef_rot_delta"] = [round(float(x), 3) for x in action[3:6].tolist()]
            fields["action_gripper"] = round(float(action[6]), 3)
        structlog.get_logger(__name__).info("robocasa_eef_trace", **fields)

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Expose underlying MuJoCo model/data so `openral sim run --view` can attach.

        Two wrapping paths:
        * Raw robosuite envs (`_is_gymnasium_wrapped=False`, e.g. kitchen /
          PandaMobile tasks) hold `sim` directly on `self._env`.
        * The GR1 path goes through `gym.make("gr1_unified/...")` → a
          ``gymnasium.Env`` whose ``unwrapped`` is the upstream
          ``RoboCasaEnv`` (gymnasium_basic.py); that wrapper stores the
          robosuite env at ``.env``, which then exposes ``.sim``.

        We try both before giving up so `openral sim run --view` works on
        both the kitchen and GR1 backends.
        """
        env = self._env
        sim = getattr(env, "sim", None)
        if sim is None:
            # Gymnasium-wrapped path: env.unwrapped → RoboCasaEnv → .env → robosuite env.
            inner = getattr(getattr(env, "unwrapped", env), "env", None)
            sim = getattr(inner, "sim", None)
        if sim is None:
            return None
        model = getattr(getattr(sim, "model", None), "_model", None)
        data = getattr(getattr(sim, "data", None), "_data", None)
        if model is None or data is None:
            return None
        return model, data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1), or None.

        Reads ``MjData.time`` off :meth:`mujoco_handles`. RoboCasa rewinds the
        clock to 0 on ``reset``, so the value is monotonic only within an
        episode — :class:`~openral_hal.sim_attached.SimAttachedHAL.sim_time_ns`
        adds the cross-reset offset.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    def _wrap_obs(self, raw: dict[str, Any]) -> Observation:  # noqa: PLR0915  # reason: assembles the full observation (joints + eef + base + cameras) in one pass
        # GR1 path: reuse the upstream GR1ArmsAndWaistKeyConverter +
        # GrootRoboCasaEnv.process_img helpers so our wire matches what
        # the FT-GR1 checkpoint was trained on (canonical
        # `gr1_unified/...` gymnasium env). Without this, our state
        # blob is concatenated raw `robot0_joint_pos` / `*_gripper_qpos`
        # keys in MJCF order — which does NOT match the model's
        # in-distribution state.
        if self._state_layout == "gr1":
            return self._wrap_obs_gr1(raw)

        # Expose every raw camera stream the env produced -- both under
        # its native robosuite name (so a lerobot pi0.5 / pi0 checkpoint
        # that consumes ``observation.images.robot0_agentview_left_image``
        # works without an alias map) and under the openral ergonomic
        # alias `camera1` / `camera2` / `camera3` (so the smolvla / xvla /
        # mock policy adapters keep working unchanged).
        images: dict[str, NDArray[np.uint8]] = {}
        h = self.scene.observation_height
        w = self.scene.observation_width
        # MuJoCo's offscreen renderer emits images bottom-row-first (OpenGL
        # convention). RoboCasa's upstream training pipeline
        # (``gymnasium_basic.RoboCasaEnv.get_basic_observation``) flips
        # them with ``np.copy(img[::-1, :, :])`` before passing to the
        # policy AND before any downstream sensor consumer, so the
        # checkpoints + the standard ROS image conventions
        # (``sensor_msgs/Image`` rows top-down) all expect the flipped
        # orientation. Mirror it here so EVERY consumer
        # (dashboard, /openral/cameras/<name>/image publisher, rldx /
        # pi05 policy adapters via ``observation.images.<name>``) gets a
        # right-side-up frame. The per-rskill manifest
        # ``image_preprocessing.flip_vertical`` flag thus describes
        # "does the policy need ADDITIONAL flipping vs the standard
        # orientation?" and stays ``false`` for any rskill trained on
        # the standard robocasa pipeline.
        for i, key in enumerate(self._camera_keys):
            value = raw.get(key)
            if value is not None:
                arr = np.ascontiguousarray(np.asarray(value, dtype=np.uint8)[::-1, :, :])
                images[key] = arr
                images[f"camera{i + 1}"] = arr
                if self._last_image is None or i == 0:
                    self._last_image = arr
        if not images:
            images["camera1"] = np.zeros((h, w, 3), dtype=np.uint8)

        # 9-D smolvla layout: eef_pos(3) + eef_quat(4) + gripper_qpos(2).
        # The older pre-mg_300 single-arm shape; also the graceful
        # fallback when base-to-eef keys are absent (non-mobile bases).
        state_keys_smolvla = (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        # 16-D human300 layout — robocasa-benchmark/openpi
        # `pi05_pretrain_human300`, verified verbatim from
        # examples/robocasa/main.py:
        #   base_to_eef_pos(3) + base_to_eef_quat(4) + base_pos(3)
        #   + base_quat(4) + gripper_qpos(2) = 16.
        # Order matters: the gripper finger angles occupy the LAST two
        # dims, not the middle — a different concatenation order would
        # silently feed a quat component into the gripper slot.
        state_keys_human300 = (
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_gripper_qpos",
        )
        # GR1 ArmsAndWaistFourierHands proprioception (39-D). robosuite
        # exposes the controller-part joints as a single 17-D
        # `robot0_joint_pos` (waist 3 + right arm 7 + left arm 7 in
        # MJCF order; the GR00T fork's gymnasium_basic builds per-part
        # views off `robot._ref_joints_indexes_dict`, but that helper
        # is not on the public obs dict). Adding the two 11-D
        # Fourier-hand qpos vectors gives a contract-friendly 39-D
        # state that downstream rSkills can index into without needing
        # access to the upstream fork's helper.
        state_keys_gr1 = (
            "robot0_joint_pos",
            "robot0_right_gripper_qpos",
            "robot0_left_gripper_qpos",
        )
        keys: tuple[str, ...]
        if self._state_layout == "smolvla_9d":
            keys = state_keys_smolvla
        elif self._state_layout == "gr1":
            keys = state_keys_gr1
        else:  # "human300_16d" (default) — fall back to the 9-D smolvla
            # shape when base-to-eef keys are absent (non-mobile bases).
            keys = state_keys_human300 if "robot0_base_to_eef_pos" in raw else state_keys_smolvla
        # Quaternion keys whose sign we canonicalise so both this
        # path AND the deploy_sim ADR-0027 state assembler (which
        # canonicalises in ``human300_16d._quat_to_layout``) feed the
        # policy the same hemisphere. ``q`` and ``-q`` are the same
        # rotation but different bytes; without canonicalisation the
        # dump-diff regression test surfaces a phantom 2.0 max-diff
        # on the quat slots whenever the two paths happen to land on
        # opposite hemispheres for the same physical pose.
        _quat_keys_state = {
            "robot0_base_to_eef_quat",
            "robot0_base_quat",
            "robot0_eef_quat",
        }
        state_parts: list[NDArray[np.float32]] = []
        for key in keys:
            value = raw.get(key)
            if value is None:
                continue
            state_arr: NDArray[np.float32] = np.asarray(value, dtype=np.float32).reshape(-1)
            if key in _quat_keys_state and state_arr.shape[0] == 4:  # noqa: PLR2004
                state_arr = _canonicalize_quat_xyzw_np(state_arr)
            state_parts.append(state_arr)
        state = np.concatenate(state_parts) if state_parts else np.zeros(0, dtype=np.float32)

        # RoboCasa's task language is episode-specific: the env samples a
        # particular object (e.g. "hot dog") and ``env.get_ep_meta()["lang"]``
        # returns the canonical sentence with that name interpolated, which
        # is exactly what the policy was trained on (per
        # ``gymnasium_basic.get_basic_observation``:
        # ``raw_obs["language"] = self.env.get_ep_meta().get("lang", "")``).
        # If we forward the static ``task.instruction`` from the YAML (e.g.
        # "pick the object from the counter ...") the VLA never sees the
        # actual target object name and degenerates to spinning the base
        # while it searches. Prefer the env's lang when available; fall
        # back to the YAML instruction otherwise.
        task_lang = self.task.instruction
        if hasattr(self._env, "get_ep_meta"):
            try:
                em = self._env.get_ep_meta()
                if isinstance(em, dict):
                    env_lang = em.get("lang")
                    if isinstance(env_lang, str) and env_lang.strip():
                        task_lang = env_lang
            except Exception:  # reason: defensive — never crash obs assembly on lang lookup
                pass
        obs: Observation = {
            "images": images,
            "state": state,
            "task": task_lang,
        }
        # Preserve the raw RoboCasa proprio keys for downstream
        # consumers that need authoritative base / eef pose (e.g.
        # ``openral_hal.SimAttachedHAL.base_pose_6dof`` reads
        # ``raw_proprio["robot0_base_pos"]`` to publish a faithful
        # ``odom → base_link`` TF that mirrors what the policy was
        # trained on, rather than the planar (x, y, yaw) projection
        # the URDF joint chain produces — see the dump-diff regression
        # in ``openral deploy sim`` that found ~0.70 m of base-z drop
        # before this slot existed). Stored as a sub-dict (not at top
        # level) so it doesn't pollute the Observation Protocol's
        # documented surface.
        proprio_keys = (
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        raw_proprio: dict[str, NDArray[np.float64]] = {}
        for key in proprio_keys:
            value = raw.get(key)
            if value is not None:
                raw_proprio[key] = np.asarray(value, dtype=np.float64).reshape(-1)
        if raw_proprio:
            obs["raw_proprio"] = raw_proprio
        self._emit_panda_mobile_extras(obs)
        return obs

    def _emit_panda_mobile_extras(self, obs: Observation) -> None:
        """Attach ``robot0_base_vel`` + ``robot0_scan`` for panda_mobile envs.

        Gated on ``"PandaMobile" in self._robots`` so other compositions
        (Panda fixed-base, GR1 humanoid, etc.) are untouched. Silently
        no-ops if MuJoCo handles aren't accessible (e.g. the env was
        built but not yet reset) so the obs contract stays the same
        shape for every consumer that doesn't look at the extras.

        Surface contract:

        * ``obs["robot0_base_vel"]`` — ``(3,)`` float32, body-frame
          ``(vx, vy, wz)`` in m/s and rad/s. Matches the
          ``ControlMode.BODY_TWIST`` convention's first / second / sixth
          channels.
        * ``obs["robot0_scan"]`` — ``(_LASER_DEFAULT_N_BEAMS,)`` float32
          range array in metres, ``-π → +π`` body frame. Beams that
          didn't hit anything within ``_LASER_DEFAULT_MAX_RANGE_M`` are
          clamped to the max (never NaN, never inf).
        """
        if not self._has_mobile_base_robot():
            return
        handles = self.mujoco_handles()
        if handles is None:
            return
        model, data = handles
        # Source MJCF joint names from the active robot's
        # `RobotDescription.base_joints` declaration. The lookup goes
        # through the generic `extract_base_sim_joint_names` helper so
        # nothing in this adapter hardcodes panda_mobile or robosuite's
        # `mobilebase0_` namespace — any robot.yaml carrying
        # `base_joints: [...]` plus the per-joint `sim_joint_name`
        # overrides works. The helpers fall back to module-level
        # constants when `base_names is None`.
        base_names: tuple[str, str, str] | None = self._resolve_base_joint_names()
        obs["robot0_base_vel"] = read_panda_mobile_base_velocity(
            model, data, base_joint_names=base_names
        )
        obs["robot0_scan"] = synthesize_laser_scan_2d(
            model=model, data=data, base_joint_names=base_names
        )

    def _has_mobile_base_robot(self) -> bool:
        """True iff one of the loaded robosuite robots is a mobile base.

        Detection is structural (composition-name suffix `"Mobile"`)
        rather than a literal `"PandaMobile"` check — so a future
        ``GR1Mobile`` / ``SpotMobile`` etc. light up the extras path
        automatically. The historical `"PandaMobile"` literal stays as
        an OR-arm so existing fixtures and policies that key on the
        exact name continue to work.
        """
        return any(name.endswith("Mobile") for name in self._robots) or (
            "PandaMobile" in self._robots
        )

    def _resolve_base_joint_names(self) -> tuple[str, str, str] | None:
        """Resolve `(forward, side, yaw)` MJCF joint names for the loaded robot.

        Maps the robosuite composition name → the on-disk
        `robots/<robot_id>/robot.yaml` → `extract_base_sim_joint_names`.
        Returns ``None`` when the robot yaml is unreachable from the
        sim runner's cwd (the ray-cast helpers then fall back to
        their module-level defaults).
        """
        from openral_core import extract_base_sim_joint_names  # reason: scoped

        for robosuite_name in self._robots:
            robot_id = _robot_id_for_robosuite_name(robosuite_name)
            if robot_id is None:
                continue
            description = _load_robot_description_by_id(robot_id)
            if description is None:
                continue
            triple = extract_base_sim_joint_names(description)
            if triple is not None:
                return triple
        return None

    def _wrap_obs_gr1(self, raw: dict[str, Any]) -> Observation:
        """GR1 path: re-pack the gymnasium-wrapped GR1 obs into Observation.

        The upstream ``gr1_unified/...`` gymnasium env already returns the
        canonical Fourier GR-1 contract (per
        ``GrootRoboCasaEnv.get_groot_observation``):

        * Five state keys with per-key dims matching the FT-GR1
          ``general_embodiment`` modality config — `state.right_arm` 7,
          `state.left_arm` 7, `state.waist` 3, `state.right_hand` 6,
          `state.left_hand` 6.
        * Padded-and-resized camera at `video.ego_view_pad_res256_freq20`
          (256×256 uint8, vertical flip already applied upstream).
        * `annotation.human.coarse_action` text already prefixed with
          `"unlocked_waist: "` for ArmsAndWaist embodiments.

        We pluck those, concatenate the five state arrays into the
        openral 29-D order, and expose the camera under `camera1`
        (openral alias) plus `video.ego_view` (the short canonical key
        the rldx adapter sends to the FT-GR1 sidecar).
        """
        state = np.concatenate(
            [
                np.asarray(raw["state.waist"], dtype=np.float32).reshape(-1),
                np.asarray(raw["state.right_arm"], dtype=np.float32).reshape(-1),
                np.asarray(raw["state.left_arm"], dtype=np.float32).reshape(-1),
                np.asarray(raw["state.right_hand"], dtype=np.float32).reshape(-1),
                np.asarray(raw["state.left_hand"], dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

        ego = np.asarray(raw["video.ego_view_pad_res256_freq20"], dtype=np.uint8)
        self._last_image = ego

        # `annotation.human.coarse_action` already carries the
        # `"unlocked_waist: "` prefix for ArmsAndWaist tasks; fall back
        # to the openral task instruction with the same prefix if the
        # env didn't populate the language slot.
        language = raw.get("annotation.human.coarse_action") or (
            f"unlocked_waist: {self.task.instruction}"
        )

        return {
            "images": {
                "camera1": ego,
                "video.ego_view": ego,
                "video.ego_view_pad_res256_freq20": ego,
            },
            "state": state,
            "task": str(language),
        }


# ── Robosuite composition → openral robot_id lookup ─────────────────────────
#
# Each entry maps a robosuite composition class name (e.g. `PandaMobile`)
# to the canonical ``robots/<robot_id>/robot.yaml`` directory the OpenRAL
# registry ships. The sim adapter consults this when emitting
# extras (`robot0_base_vel`, `robot0_scan`) so the helpers can pull
# MJCF joint names from the per-robot description rather than hardcoding
# them at this layer. Extending support to a new mobile-base composition
# means (a) shipping a ``robots/<id>/robot.yaml`` with ``base_joints`` +
# ``sim_joint_name`` populated, and (b) adding the mapping row here.
_ROBOSUITE_NAME_TO_ROBOT_ID: dict[str, str] = {
    "PandaMobile": "panda_mobile",
    # Future: `"SpotArm": "spot_arm"`, `"StretchRE1": "stretch_re1"`, etc.
}


def _robot_id_for_robosuite_name(robosuite_name: str) -> str | None:
    """Return the OpenRAL robot_id for a robosuite composition name."""
    return _ROBOSUITE_NAME_TO_ROBOT_ID.get(robosuite_name)


def _load_robot_description_by_id(robot_id: str) -> Any:
    """Load ``robots/<robot_id>/robot.yaml`` from the workspace root.

    Walks parents of this source file looking for a ``robots/<id>/robot.yaml``
    fixture. Returns ``None`` when the file isn't reachable (the sim
    adapter is being exercised in a hermetic test fixture without the
    full workspace tree on disk). A malformed YAML raises the
    underlying :class:`~openral_core.exceptions.ROSConfigError` — fail
    loud rather than silently fall back to module-level joint-name
    defaults, because a typo in a robot.yaml is the kind of bug that
    "best-effort" used to hide.
    """
    from pathlib import Path  # reason: stdlib defer

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "robots" / robot_id / "robot.yaml"
        if candidate.is_file():
            from openral_core import RobotDescription  # reason: scoped

            return RobotDescription.from_yaml(str(candidate))
    return None


# ── PandaMobile base-velocity + synthetic 2D LaserScan helpers ─────────────
#
# ADR-0025.
#
# The robosuite ``OmronMobileBase`` MJCF declares three planar joints
# named below — robosuite tracks their qpos / qvel automatically. The
# adapter does not surface them via the `obs` dict by default, so a
# downstream consumer that needs Nav2 / SLAM-grade base velocity or a
# 2D laser scan would either have to (a) finite-difference qpos itself
# or (b) bypass the adapter and reach into ``env.sim``. Both are
# brittle. The two helpers below + the adapter's
# ``_emit_panda_mobile_extras`` surface the data cleanly under
# ``robot0_base_vel`` (3-vec, body-frame) and ``robot0_scan`` (range
# array, body-frame), gated on ``"PandaMobile" in self._robots``.

# Joint names in the robosuite ``OmronMobileBase`` MJCF
# (``robosuite/models/assets/bases/omron_mobile_base.xml``). Order matches
# ``PandaMobileHAL`` qpos: forward (x), side (y), yaw (rotation about z).
_OMRON_BASE_JOINT_NAMES: tuple[str, str, str] = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)
"""robosuite MJCF body / joint prefix for the OmronMobileBase. The
prefix ``mobilebase0_`` is robosuite's own auto-prefix when the base is
the first mobile-base actor in the env; second / third would be
``mobilebase1_`` / ``mobilebase2_``. Single-robot kitchen envs always
use index 0."""

# Fallback joint names (without the robosuite auto-prefix). Used when a
# host robosuite version uses raw MJCF names — historically true for
# some MuJoCo XMLs imported via ``mjcf_utils.find_elements`` rather than
# the composite-controller pipeline. Resolution order:
# prefixed → unprefixed → raise.
_OMRON_BASE_JOINT_NAMES_FALLBACK: tuple[str, str, str] = (
    "joint_mobile_forward",
    "joint_mobile_side",
    "joint_mobile_yaw",
)


def _resolve_base_joint_qvel_addrs(
    model: Any,
    *,
    base_joint_names: tuple[str, str, str] | None = None,
) -> tuple[int, int, int] | None:
    """Return the qvel addresses of the panda_mobile base joints, or None.

    ``None`` when the OmronMobileBase isn't present (e.g. a fixed-base
    Franka env was loaded but the caller still asked for extras).

    Args:
        model: Live ``mujoco.MjModel``.
        base_joint_names: Optional ``(forward, side, yaw)`` MJCF joint
            names. Supersedes the module-level
            :data:`_OMRON_BASE_JOINT_NAMES` defaults — callers that
            have access to a :class:`~openral_core.RobotDescription`
            should read names from the per-joint
            :attr:`~openral_core.JointSpec.sim_joint_name` field and
            pass them here so the helper never depends on hardcoded
            robosuite / robocasa naming conventions. ADR-0025.
    """
    name_to_addr: dict[str, int] = {}
    import mujoco  # reason: defer optional dep

    if base_joint_names is None:
        primary = _OMRON_BASE_JOINT_NAMES
        fallback = _OMRON_BASE_JOINT_NAMES_FALLBACK
    else:
        primary = base_joint_names
        # Caller-supplied names are authoritative; no separate fallback.
        fallback = base_joint_names

    for prefixed, unprefixed in zip(primary, fallback, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
        if jid < 0 and unprefixed != prefixed:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, unprefixed)
        if jid < 0:
            return None
        name_to_addr[prefixed] = int(model.jnt_dofadr[jid])
    return (
        name_to_addr[primary[0]],
        name_to_addr[primary[1]],
        name_to_addr[primary[2]],
    )


def read_panda_mobile_base_velocity(
    model: Any,
    data: Any,
    *,
    base_joint_names: tuple[str, str, str] | None = None,
) -> NDArray[np.float32]:
    """Return ``[vx_body, vy_body, wz]`` in m/s and rad/s, body frame.

    Reads the three OmronMobileBase joint velocities from ``data.qvel``,
    then de-rotates the world-frame (vx, vy) into body frame using the
    current base yaw from ``data.qpos``. Returns a length-3 float32
    array.

    Returns a zero vector when the base joints aren't in this model —
    i.e. this isn't a panda_mobile env. The caller is responsible for
    deciding whether that's acceptable (the adapter only calls this
    when ``"PandaMobile" in self._robots``).

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData``.
        base_joint_names: Optional MJCF names override — see
            :func:`_resolve_base_joint_qvel_addrs`. ADR-0025.

    Raises:
        ROSConfigError: when the MJCF declares the base joints but one
            of them isn't a 1-DoF slide / hinge (would indicate a model
            upstream change we should fail loudly on).
    """
    addrs = _resolve_base_joint_qvel_addrs(model, base_joint_names=base_joint_names)
    if addrs is None:
        return np.zeros(3, dtype=np.float32)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    vx_world = float(qvel[addrs[0]])
    vy_world = float(qvel[addrs[1]])
    wz = float(qvel[addrs[2]])

    import math  # reason: stdlib defer

    import mujoco  # reason: defer optional dep

    primary = base_joint_names if base_joint_names is not None else _OMRON_BASE_JOINT_NAMES
    fallback = (
        base_joint_names if base_joint_names is not None else _OMRON_BASE_JOINT_NAMES_FALLBACK
    )
    yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, primary[2])
    if yaw_jid < 0 and fallback[2] != primary[2]:
        yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fallback[2])
    yaw = float(np.asarray(data.qpos, dtype=np.float64)[int(model.jnt_qposadr[yaw_jid])])
    c, s = math.cos(-yaw), math.sin(-yaw)
    vx_body = c * vx_world - s * vy_world
    vy_body = s * vx_world + c * vy_world
    return np.array([vx_body, vy_body, wz], dtype=np.float32)


# Default 2D LaserScan parameters. Chosen to mirror a typical RPLIDAR
# A1 / A2 unit: 360 beams, 12 m max, ±π sweep. The 2D Hokuyo we'd use on
# a real OmronMobileBase has the same envelope. Operators can override
# per-call but the defaults are calibrated to "Nav2-default-costmap-friendly".
_LASER_DEFAULT_N_BEAMS = 360
_LASER_DEFAULT_MAX_RANGE_M = 12.0
_LASER_DEFAULT_MIN_RANGE_M = 0.05
_LASER_DEFAULT_HEIGHT_M = 0.30  # body-frame z at which the beams are cast
# Self-occlusion handling for the ray-cast: a beam may terminate on the
# robot's own bodies (chassis / wheels / arm). We re-cast past each such
# self-hit; this caps how many self-layers one beam steps through before
# giving up (returning max_range). 8 covers chassis + arm links with margin.
_LASER_MAX_SELF_SKIPS = 8
# Nudge (m) added past a self-hit point before re-casting, so the next
# ray doesn't re-detect the same surface it just exited.
_LASER_SELF_SKIP_EPS_M = 1e-3


def synthesize_laser_scan_2d(  # noqa: PLR0915  # reason: the body-name + joint-name + fallback resolution + ray-cast + clamp form one logical operation; splitting just hides the per-step structure
    *,
    model: Any,
    data: Any,
    base_body_id: int | None = None,
    base_joint_names: tuple[str, str, str] | None = None,
    n_beams: int = _LASER_DEFAULT_N_BEAMS,
    max_range_m: float = _LASER_DEFAULT_MAX_RANGE_M,
    laser_height_m: float = _LASER_DEFAULT_HEIGHT_M,
) -> NDArray[np.float32]:
    """Cast ``n_beams`` rays in a planar fan from the panda_mobile base.

    Uses MuJoCo's ``mj_multiRay`` for a single-origin batched ray-cast.
    Returns a ``(n_beams,)`` float32 array of ranges in metres, clamped
    to ``max_range_m`` for "no hit" beams so downstream consumers
    (sensor_msgs/LaserScan, Nav2 costmap) don't NaN-poison their grids.

    Beam ordering: ``angle = -pi + 2*pi * i / n_beams`` for ``i in [0,
    n_beams)``, body-frame. The caller adds yaw if the wire format
    expects world-frame beams.

    Args:
        model: Live ``mujoco.MjModel``.
        data: Live ``mujoco.MjData``.
        base_body_id: Optional body id to exclude from ray hits (so the
            robot doesn't see its own chassis). When ``None`` the helper
            looks up ``"base"`` (OmronMobileBase root); when that's also
            absent every beam is a no-op-exclude.
        base_joint_names: Optional MJCF joint-name triple override — see
            :func:`_resolve_base_joint_qvel_addrs`. ADR-0025.
        n_beams: Number of rays. 360 ≈ 1 deg resolution.
        max_range_m: Max sensor range. Beams with no hit return this
            value (NOT NaN, NOT inf).
        laser_height_m: Body-frame z at which to cast beams. RPLIDAR-A1
            mount height on the OmronMobileBase.

    Returns:
        ``(n_beams,)`` float32 ranges in metres.
    """
    import math  # reason: stdlib defer

    import mujoco  # reason: defer optional dep

    primary = base_joint_names if base_joint_names is not None else _OMRON_BASE_JOINT_NAMES
    fallback = (
        base_joint_names if base_joint_names is not None else _OMRON_BASE_JOINT_NAMES_FALLBACK
    )

    yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, primary[2])
    if yaw_jid < 0 and fallback[2] != primary[2]:
        yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fallback[2])

    # Origin: world-frame XY position of the base body. We do NOT
    # read from joint qpos: under composed scenes (robocasa kitchens)
    # the OmronMobileBase is parented under a wheeled-base body whose
    # spawn pose offsets the joint coordinate frame, so slide-joint
    # qpos is the offset from the spawn pose — not the world
    # position. The body's `data.xpos` is mujoco-canonical world
    # frame for any composition.
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel_addrs = _resolve_base_joint_qvel_addrs(model, base_joint_names=base_joint_names)
    origin = np.zeros(3, dtype=np.float64)
    yaw_world = 0.0
    if qvel_addrs is not None:
        # Prefer the body's world position. The body name follows the
        # same prefix convention as the joint names — strip the
        # `_joint_mobile_forward` tail from the first base joint name
        # and append `_base` to get the canonical body name
        # (`mobilebase0_base` for the canonical OmronMobileBase).
        forward_jname = primary[0]
        # `mobilebase0_joint_mobile_forward` → `mobilebase0_base`
        prefix = forward_jname.split("_joint_")[0] if "_joint_" in forward_jname else ""
        candidate_body = f"{prefix}_base" if prefix else "base"
        body_id_for_origin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate_body)
        if body_id_for_origin < 0:
            # Fall back to qpos semantics (synthetic test MJCFs without
            # the composed-scene prefix).
            body_id_for_origin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if body_id_for_origin >= 0:
            origin[0] = float(data.xpos[body_id_for_origin][0])
            origin[1] = float(data.xpos[body_id_for_origin][1])
            # Heading from the body's WORLD rotation, NOT the yaw-joint
            # qpos. Same rationale as the world-frame origin above: under a
            # composed kitchen scene the base body's spawn pose carries the
            # robot's facing while the yaw joint reads only its offset from
            # that spawn — so qpos[yaw] omits the spawn rotation and the
            # scan fan comes out rotated relative to the published
            # odom→base_link TF (which uses the body's true world
            # orientation via robot0_base_quat). That constant rotation is
            # what made the dashboard occupancy map appear turned relative
            # to the simulated kitchen. atan2 of the world rotation
            # matrix's first column recovers the planar yaw.
            xmat = np.asarray(data.xmat[body_id_for_origin], dtype=np.float64).reshape(3, 3)
            yaw_world = float(math.atan2(xmat[1, 0], xmat[0, 0]))
        else:
            # Last-ditch: joint qpos (matches synthetic test MJCFs).
            x_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, primary[0])
            if x_jid < 0 and fallback[0] != primary[0]:
                x_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fallback[0])
            y_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, primary[1])
            if y_jid < 0 and fallback[1] != primary[1]:
                y_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fallback[1])
            origin[0] = qpos[int(model.jnt_qposadr[x_jid])]
            origin[1] = qpos[int(model.jnt_qposadr[y_jid])]
            yaw_world = float(qpos[int(model.jnt_qposadr[yaw_jid])])
    origin[2] = float(laser_height_m)

    # Beam directions: world-frame unit vectors at angles
    # (yaw + body_angle) in the xy plane.
    angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False, dtype=np.float64)
    dirs = np.empty((n_beams, 3), dtype=np.float64)
    dirs[:, 0] = np.cos(yaw_world + angles)
    dirs[:, 1] = np.sin(yaw_world + angles)
    dirs[:, 2] = 0.0

    # Self-exclusion: look up the robot's chassis body so beams cast
    # outward don't immediately hit the chassis at radius ≈ 0. Same
    # prefix logic as the origin lookup: try `<prefix>_base` first
    # (e.g. `mobilebase0_base` under a composed kitchen scene), then
    # the bare `"base"` for synthetic-MJCF test paths.
    if base_body_id is None:
        prefix = primary[0].split("_joint_")[0] if "_joint_" in primary[0] else ""
        candidate = f"{prefix}_base" if prefix else "base"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid < 0:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        base_body_id = int(bid) if bid >= 0 else -1

    geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)

    # Self-exclusion must cover the ENTIRE robot kinematic tree, not just
    # one body. ``mj_ray``/``mj_multiRay`` accept a single ``bodyexclude``,
    # but on the OmronMobileBase the collision geometry lives on
    # ``mobilebase0_wheeled_base`` — a child of the (geomless)
    # ``mobilebase0_base`` root that the prefix lookup resolves. A lone
    # exclude therefore let every beam terminate on the wheeled base at
    # ~0.13-0.54 m (below ``range_min``), starving slam_toolbox so it only
    # ever published an empty 0x0 ``/map`` (ADR-0025 live-sim finding).
    #
    # Fix: cast per-beam and skip any hit whose body shares the base
    # body's kinematic-tree root (``model.body_rootid``), re-casting from
    # just past the self-hit so the beam reports the real obstacle behind
    # the robot. ``robot_rootid < 0`` (base unresolved) disables the skip
    # and falls back to nearest-hit, matching the prior behaviour.
    robot_rootid = (
        int(model.body_rootid[base_body_id])
        if base_body_id is not None and base_body_id >= 0
        else -1
    )
    ranges = np.empty(n_beams, dtype=np.float32)
    geomid_out = np.zeros(1, dtype=np.int32)
    for i in range(n_beams):
        vec = dirs[i]
        travelled = 0.0
        # "No hit" / fully-self-occluded → max_range_m (NOT NaN/inf) so
        # sensor_msgs/LaserScan + Nav2 costmap consumers stay clean.
        beam_range = float(max_range_m)
        for _ in range(_LASER_MAX_SELF_SKIPS):
            pnt = origin + vec * travelled
            geomid_out[0] = -1
            # mj_ray returns the forward distance to the nearest geom
            # (or <0 for no hit) and writes the geom id into geomid_out.
            hit = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, geomid_out)
            if hit < 0.0:
                break
            body_id = int(model.geom_bodyid[int(geomid_out[0])])
            is_self = robot_rootid >= 0 and int(model.body_rootid[body_id]) == robot_rootid
            if is_self:
                travelled += float(hit) + _LASER_SELF_SKIP_EPS_M
                if travelled >= max_range_m:
                    break
                continue
            beam_range = min(travelled + float(hit), float(max_range_m))
            break
        ranges[i] = beam_range
    return ranges


def _validate_backend_options(scene: SceneSpec) -> RoboCasaBackendOptions:
    """Validate ``scene.backend_options`` through the typed RoboCasaBackendOptions."""
    try:
        return RoboCasaBackendOptions.model_validate(scene.backend_options)
    except ValidationError as exc:
        raise ROSConfigError(
            f"scene.backend_options failed RoboCasaBackendOptions validation: {exc}"
        ) from exc


def _resolve_env_name(opts: RoboCasaBackendOptions, scene_id: str) -> str:
    """Map (scene_id, opts) to the robosuite env_name string."""
    if scene_id == _PROCEDURAL_SCENE_ID:
        if opts.mode != "procedural":
            raise ROSConfigError(
                f"scene id 'robocasa' (procedural) requires mode='procedural'; "
                f"got mode={opts.mode!r}. Pass scene id 'robocasa/<TaskName>' "
                "for prebuilt tasks."
            )
        # Procedural authoring picks the matching atomic Kitchen env for
        # the requested task_verb. The user composes layout / fixtures /
        # objects via robosuite's `Kitchen` (parent class); we map the
        # high-level verb to one of robocasa's existing atomic envs.
        verb_to_env = {
            "pnp": "PickPlaceCounterToCabinet",
            "open": "OpenDoor",
            "close": "CloseDoor",
            "press": "TurnOnMicrowave",
            "navigate": "NavigateKitchen",
        }
        env_name = verb_to_env.get(opts.task_verb or "")
        if env_name is None:
            raise ROSConfigError(
                f"RoboCasaBackendOptions.task_verb={opts.task_verb!r} has no "
                "registered atomic env; supported: " + ", ".join(verb_to_env)
            )
        return env_name

    # Prebuilt scene id: either
    #   - "robocasa/<task>"            (panda_mobile kitchen tasks)
    #   - "robocasa/gr1/<task>"        (GR1 tabletop tasks from the
    #                                   robocasa-gr1-tabletop-tasks fork)
    # The "gr1/" namespace flips the default robots/state-layout choices
    # and routes through the BASIC composite controller (the GR1 envs do
    # not accept the OSC_POSE composite the kitchen Panda uses).
    if scene_id.startswith(_GR1_SCENE_PREFIX + "/"):
        name = scene_id[len(_GR1_SCENE_PREFIX) + 1 :]
        if not name:
            raise ROSConfigError(
                f"unexpected RoboCasa GR1 scene id {scene_id!r}; "
                "expected 'robocasa/gr1/<TaskName>'."
            )
        if opts.mode != "prebuilt":
            raise ROSConfigError(
                f"GR1 scene id {scene_id!r} requires mode='prebuilt'; "
                f"got mode={opts.mode!r} (procedural authoring is kitchen-only)."
            )
        if opts.prebuilt_task and opts.prebuilt_task != name:
            raise ROSConfigError(
                f"GR1 scene id {scene_id!r} (task {name!r}) disagrees with "
                f"backend_options.prebuilt_task={opts.prebuilt_task!r}."
            )
        return name

    prefix, _, name = scene_id.partition("/")
    if prefix != _PREBUILT_SCENE_PREFIX or not name:
        raise ROSConfigError(
            f"unexpected RoboCasa scene id {scene_id!r}; "
            "expected 'robocasa' (procedural), 'robocasa/<TaskName>' "
            "(kitchen), or 'robocasa/gr1/<TaskName>' (tabletop)."
        )
    if opts.mode != "prebuilt":
        raise ROSConfigError(
            f"scene id {scene_id!r} (prebuilt) requires mode='prebuilt'; "
            f"got mode={opts.mode!r}. Pass scene id 'robocasa' (no slash) "
            "for procedural authoring."
        )
    if opts.prebuilt_task and opts.prebuilt_task != name:
        raise ROSConfigError(
            f"scene id {scene_id!r} (prebuilt task {name!r}) disagrees with "
            f"backend_options.prebuilt_task={opts.prebuilt_task!r}. "
            "Drop the prebuilt_task from backend_options or pick the "
            "scene id that matches."
        )
    return name


def _is_gr1_robot(robot_name: str) -> bool:
    """True if the robosuite robot name belongs to the GR1 family.

    The GR1 fork ships GR1ArmsOnly{,Inspire,Fourier}Hands,
    GR1ArmsAndWaistFourierHands, GR1FixedLowerBody{,Inspire,Fourier}Hands.
    Detecting by class-name prefix is what robocasa's own gymnasium
    wrapper does (see robocasa/utils/gym_utils/gymnasium_basic.py).
    """
    return robot_name.startswith("GR1")


_VERSION_SPOOF_LOGGED: list[bool] = [False]
"""One-element list used as a mutable "logged?" flag.

Wrapping the boolean in a list lets the helper mutate it without a
``global`` statement, which Pylint flags (PLW0603). The flag is
process-local; concurrency is irrelevant here because the adapter is
only invoked from a single sim runner thread.
"""


def _robosuite_conflict_hint(exc: ImportError) -> ROSConfigError | None:
    """Map a robocasa ImportError to an actionable resync error, or None.

    robocasa's ``__init__`` does ``from robosuite.models.robots import PandaOmron``
    (and ``PandaMobile``). Those symbols exist only in the robocasa-pinned
    robosuite fork; the LIBERO dependency group installs an older robosuite
    WITHOUT them. So the cryptic ``cannot import name 'PandaOmron'`` almost always
    means a robosuite VERSION conflict (libero's robosuite shadowing robocasa's),
    not a genuinely missing package — return the actionable resync hint. A
    plain ``No module named 'robocasa'`` (real absence) returns None so the
    caller re-raises and ``ensure_backend_deps`` handles install.
    """
    msg = str(exc)
    if "PandaOmron" in msg or "PandaMobile" in msg:
        return ROSConfigError(
            "robocasa import failed: the installed robosuite is missing "
            f"PandaOmron/PandaMobile ({exc}). This is the libero<->robocasa "
            "robosuite version conflict — the LIBERO dependency group pins an "
            "older robosuite that shadows the robocasa fork. Re-sync the robocasa "
            "group before running a robocasa scene: "
            "`just sync --all-packages --group robocasa --inexact` "
            "(then `uv pip install -e _external/robocasa --no-deps`)."
        )
    return None


def _spoof_robocasa_version_pins() -> None:
    """Spoof ``mujoco`` / ``numpy`` / ``robosuite`` __version__ before robocasa import.

    Two upstreams hard-assert exact micro versions at import:

    * Upstream ``robocasa/__init__.py`` (1.0.1 main / robocasa365_release
      branches) asserts ``mujoco==3.3.1`` and ``numpy==2.2.5``.
    * The fork ``robocasa-gr1-tabletop-tasks`` asserts
      ``mujoco==3.2.6``, ``numpy in {1.23.x, 1.26.4}`` and
      ``robosuite in {1.5.0, 1.5.1}``.

    The workspace's `dev` group depends on newer mujoco / numpy that
    every other sim backend (LIBERO, MetaWorld, ALOHA, PushT) relies on;
    we cannot pin everything to robocasa's exact micro versions without
    breaking those backends. Empirically both robocasa variants import
    + run with mujoco 3.8.x / numpy 2.2.x / robosuite 1.5.2 once the
    assertions are bypassed.

    Spoofing is local to the process and only applied at the very
    moment of importing robocasa; we restore the originals immediately
    afterwards so any other consumer of these __version__ attributes
    (e.g. the LIBERO adapter's `robosuite==1.4` compatibility check) is
    unaffected. Logged once per process so the workaround is visible
    in traces.
    """
    # Warm numba before we spoof numpy.__version__. numba 0.65 reads
    # numpy.__version__ at first import to pick a numpy<2 vs numpy>=2
    # internal types layout; if we spoof to "1.26.4" *before* numba is
    # imported, numba picks the numpy<2 path against an actual numpy
    # 2.x install and crashes with
    # `AttributeError: module 'numba.core.types' has no attribute 'bool'`
    # the next time anything (e.g. robocasa.utils.placement_samplers,
    # which lazily imports numba via scipy) touches the affected code
    # path. Importing numba here is harmless if numba is already loaded;
    # the GR1 fork pulls it transitively via robosuite -> mujoco-mjx so
    # we can rely on it being installed alongside the `robocasa` group.
    import contextlib

    import mujoco
    import numpy
    import robosuite

    with contextlib.suppress(ImportError):
        import numba  # type: ignore[import-untyped,import-not-found,unused-ignore]  # noqa: F401  # reason: force numba's numpy-version detection before the spoof (no stubs shipped upstream; the unused-ignore suppresses mypy when numba is absent and the import-untyped / import-not-found codes never fire)

    orig_mj = mujoco.__version__
    orig_np = numpy.__version__
    orig_rs = robosuite.__version__
    # mypy types `__version__` as Final on these modules; the spoof is
    # a deliberate runtime-only override so we silence the type errors.
    # The two robocasa variants have INCOMPATIBLE micro-version pins
    # (`assert mujoco.__version__ == "3.3.1"` upstream vs `== "3.2.6"`
    # on the GR1 fork — strict equality, not a range), so we have to
    # pick the right string based on which one is actually installed.
    # Detect by the variant-specific asset script: the fork ships
    # `download_tabletop_assets.py`, upstream ships
    # `download_kitchen_assets.py`. Same heuristic as
    # `_has_robocasa_gr1` / `_has_robocasa_kitchen` in `_deps.py`.
    import importlib.util as _il_util
    from pathlib import Path as _Path

    _spec = _il_util.find_spec("robocasa")
    _is_gr1_fork = False
    if _spec is not None and _spec.origin is not None:
        _scripts = _Path(_spec.origin).parent / "scripts"
        _is_gr1_fork = (_scripts / "download_tabletop_assets.py").is_file()
    if _is_gr1_fork:
        # robocasa-gr1-tabletop-tasks fork pins
        mujoco.__version__ = "3.2.6"
        numpy.__version__ = "1.26.4"  # type: ignore[misc]  # reason: GR1 fork accepts {1.23.x, 1.26.4}; runtime numpy is still 2.x.
        robosuite.__version__ = "1.5.1"  # fork accepts {1.5.0, 1.5.1}
    else:
        # Upstream robocasa 1.0.1 (kitchen) pins
        mujoco.__version__ = "3.3.1"
        numpy.__version__ = "2.2.5"  # type: ignore[misc]  # reason: upstream pins numpy==2.2.5 exactly; runtime numpy is still 2.2.x.
        robosuite.__version__ = "1.5.2"  # upstream asserts >=1.5.2
    try:
        import robocasa  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: registers robocasa env classes; module is optional and lacks stubs
    except ImportError as exc:
        hint = _robosuite_conflict_hint(exc)
        if hint is not None:
            raise hint from exc
        raise
    finally:
        mujoco.__version__ = orig_mj
        numpy.__version__ = orig_np  # type: ignore[misc]  # reason: restore real version
        robosuite.__version__ = orig_rs

    if not _VERSION_SPOOF_LOGGED[0]:
        import structlog

        structlog.get_logger(__name__).info(
            "robocasa_version_spoof",
            mujoco=orig_mj,
            numpy=orig_np,
            robosuite=orig_rs,
            note=(
                "robocasa 1.0.1 / robocasa-gr1-tabletop-tasks 0.2.0 "
                "hard-assert exact mujoco/numpy/robosuite micro versions "
                "at import; spoofed at import-time. Track upstream PRs "
                "that relax the pin."
            ),
        )
        _VERSION_SPOOF_LOGGED[0] = True


def _build_robocasa_sim(  # noqa: PLR0915  # reason: the controller-config / camera / state-layout branching is intrinsic to the GR1 vs kitchen split; factoring out would just add indirection
    env_cfg: SimEnvironment, *, scene_id: str | None = None
) -> _RoboCasaSim:
    """Resolve a SimEnvironment to a running RoboCasa env."""
    # Pick the install plan from the scene id BEFORE we touch
    # _spoof_robocasa_version_pins() -- the spoof needs robocasa
    # already on disk, which is exactly what ensure_backend_deps
    # makes true on first use. The scene id alone disambiguates the
    # two variants (kitchen `robocasa/<task>` vs GR1
    # `robocasa/gr1/<task>`); falling back to the kitchen plan for
    # the procedural `robocasa` scene id is the historical default.
    from openral_sim._deps import ensure_backend_deps

    sid = scene_id or env_cfg.scene.id
    backend_id = "robocasa_gr1" if sid.startswith(_GR1_SCENE_PREFIX + "/") else "robocasa_kitchen"
    ensure_backend_deps(backend_id)

    _spoof_robocasa_version_pins()
    import robosuite  # reason: probe -- ensure_backend_deps above guarantees this resolves

    _ = robosuite  # silence unused-import; the import is the actual probe
    ensure_robocasa_assets()

    opts = _validate_backend_options(env_cfg.scene)
    sid = scene_id or env_cfg.scene.id
    env_name = _resolve_env_name(opts, sid)
    is_gr1 = _is_gr1_robot(opts.robots[0])

    # robosuite's `controller_configs` shape changed in 1.5; the
    # documented helper accepts EITHER `controller=<name>` (a composite
    # name like BASIC / HYBRID_MOBILE_BASE / WHOLE_BODY_*) OR
    # `robot=<name>` (composite default for the robot). For RoboCasa we
    # need the BASIC composite (so base + torso + arm + gripper are all
    # wired up), then patch the arm's per-part controller for cases like
    # the lerobot pi0.5 RoboCasa-MG_300 dataset which was recorded with
    # JOINT_POSITION arm control (7-D arm) and gives a 12-D total
    # action -- vs the OSC_POSE default (6-D arm, 11-D total).
    from robosuite import load_composite_controller_config

    if is_gr1:
        # The GR1 fork's RoboCasaEnv (gymnasium_basic.RoboCasaEnv.__init__)
        # patches the loaded composite controller config to BASIC with
        # absolute (non-delta) joint targets. We mirror that exactly so
        # the action contract matches the GR00T datasets recorded in the
        # upstream sim.
        controller_config = load_composite_controller_config(controller=None, robot=opts.robots[0])
        if controller_config is not None:
            controller_config["type"] = "BASIC"
            controller_config["composite_controller_specific_configs"] = {}
            controller_config["control_delta"] = False
            # default_gr1.json declares part-controllers for `head`,
            # `base`, and `legs`. The GR1ArmsAndWaistFourierHands
            # composition removes those actuators (see GR1ArmsAndWaist
            # in the fork's gr1_robot.py: `_remove_joint_actuation`
            # for leg/head, `_remove_free_joint` for the base).
            # robosuite's `Robot._load_controller` then logs WARN for
            # every part defined in the config but missing from the
            # robot. Strip them ourselves so the config matches the
            # robot composition cleanly -- this is the real fix the
            # WARNs are asking for.
            body_parts = controller_config.get("body_parts")
            if isinstance(body_parts, dict):
                for absent_part in ("head", "base", "legs"):
                    body_parts.pop(absent_part, None)
    elif opts.controller == "BASIC":
        controller_config = load_composite_controller_config(robot=opts.robots[0])
    elif opts.controller in {
        "JOINT_POSITION",
        "JOINT_VELOCITY",
        "JOINT_TORQUE",
        "OSC_POSITION",
        "OSC_POSE",
    }:
        # Override the right-arm controller while keeping the rest of
        # the BASIC composite (base + torso + gripper) intact.
        controller_config = load_composite_controller_config(robot=opts.robots[0])
        if controller_config is not None and "body_parts" in controller_config:
            arm = controller_config["body_parts"].get("right", {})
            gripper = arm.pop("gripper", None) if isinstance(arm, dict) else None
            controller_config["body_parts"]["right"] = _arm_part_config(opts.controller)
            if gripper is not None:
                controller_config["body_parts"]["right"]["gripper"] = gripper
    else:
        controller_config = load_composite_controller_config(controller=opts.controller)

    if is_gr1:
        # GR1 humanoid uses a single head-mounted "egoview" camera.
        # robocasa's GR1*KeyConverter.get_camera_config() pins exactly
        # this one; the bot-harness scene contract still exposes
        # camera1 / camera2 / camera3 aliases (the second/third are
        # empty for GR1).
        camera_names: list[str] = ["egoview"]
        camera_keys: tuple[str, ...] = ("egoview_image",)
    else:
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        camera_keys = (
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        )

    # Scene-pool restrictors mirroring `robocasa/utils/eval_utils.py` and
    # `robocasa/scripts/collect_demos.py`. When the YAML leaves these as
    # None, the upstream Kitchen base class falls back to its "all 60
    # layouts x 60 styles" sampling. Pinning them here is how we match
    # either the eval split (`obj_instance_split="B"`,
    # `layout_and_style_ids=[[1,1],[2,2],[4,4],[6,9],[7,10]]`) or the
    # training split (`obj_instance_split="pretrain"`,
    # `layout_ids=[-2]`, `style_ids=[-2]`) without forking the env
    # factory.
    extra_env_kwargs: dict[str, Any] = {}
    if opts.obj_instance_split is not None:
        extra_env_kwargs["obj_instance_split"] = opts.obj_instance_split
    if opts.layout_and_style_ids is not None:
        # Pydantic stores list[list[int]] but robosuite/robocasa accept
        # the equivalent tuple-of-tuples; the Kitchen ctor reads it as
        # an iterable of (layout, style) pairs either way. Pass through
        # verbatim (including the "5x5" / "5x1" shorthand strings the
        # Kitchen ctor explicitly accepts).
        extra_env_kwargs["layout_and_style_ids"] = opts.layout_and_style_ids
    if opts.layout_ids is not None:
        extra_env_kwargs["layout_ids"] = opts.layout_ids
    if opts.style_ids is not None:
        extra_env_kwargs["style_ids"] = opts.style_ids
    if opts.obj_groups is not None:
        # Pin the PnP target object (e.g. "baguette") instead of the task's
        # default "all" random draw. Valid only for PickPlace tasks; the env
        # ctor raises for non-PnP tasks, which is the right loud failure.
        extra_env_kwargs["obj_groups"] = opts.obj_groups

    env: Any
    is_gymnasium_wrapped = False
    if is_gr1:
        # GR1 path uses the upstream `gr1_unified/<task>_<robot>_Env`
        # gymnasium wrapper, NOT raw `robosuite.make`. The wrapper:
        #   * runs `gather_robot_observations(env)` and `KeyConverter.map_obs`
        #     in `get_groot_observation` so the obs comes out with the
        #     canonical `state.right_arm` / `state.left_arm` / `state.waist`
        #     / `state.right_hand` / `state.left_hand` keys (matching
        #     RLDX-1-FT-GR1's `general_embodiment` modality config);
        #   * applies the upstream vertical-flip + pad-to-square +
        #     256×256 resize via `RoboCasaEnv.get_basic_observation` →
        #     `GrootRoboCasaEnv.process_img`, so the egoview frame
        #     reaches the policy in its training-distribution
        #     orientation;
        #   * accepts a DICT action keyed `action.{waist,right_arm,
        #     left_arm,right_hand,left_hand}` and translates back to
        #     the per-controller flat 29-D robosuite expects.
        # Bypassing the wrapper (the first attempt) silently fed the
        # policy raw `robot0_joint_pos` slices and the env raw flat
        # vectors — neither matched the training distribution.
        import gymnasium as gym
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  reason: triggers `register(id='gr1_unified/...')`

        gym_env_id = f"gr1_unified/{env_name}_{opts.robots[0]}_Env"
        env = gym.make(gym_env_id, enable_render=True, seed=int(env_cfg.seed))
        is_gymnasium_wrapped = True
    else:
        env = robosuite.make(
            env_name=env_name,
            robots=opts.robots,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_widths=env_cfg.scene.observation_width,
            camera_heights=env_cfg.scene.observation_height,
            control_freq=20,
            horizon=opts.horizon,
            ignore_done=opts.ignore_done,
            **extra_env_kwargs,
        )

    return _RoboCasaSim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _env=env,
        _camera_keys=camera_keys,
        _state_layout=opts.state_layout,
        _is_gymnasium_wrapped=is_gymnasium_wrapped,
        _robots=tuple(opts.robots),
    )


def _make_prebuilt_factory(scene_id: str) -> Any:
    """Bind ``scene_id`` into the factory closure for the registry decorator."""

    def _factory(env_cfg: SimEnvironment) -> _RoboCasaSim:
        return _build_robocasa_sim(env_cfg, scene_id=scene_id)

    _factory.__name__ = f"_build_robocasa_{scene_id.replace('/', '_')}"
    _factory.__qualname__ = _factory.__name__
    _factory.__module__ = __name__
    return _factory


# Register the curated prebuilt tasks. We do NOT enumerate every
# robocasa env (the upstream catalogue includes hundreds when you count
# composite tasks); the curated tuple covers the atomic benchmarks that
# would appear on a roadmap leaderboard. Users authoring a different
# `robocasa/<TaskName>` via `--scene` get a clean adapter-level error
# from `_resolve_env_name` (and can patch the tuple in their fork).
for _task in _CURATED_PREBUILT_TASKS:
    _scene = f"{_PREBUILT_SCENE_PREFIX}/{_task}"
    SCENES.register(_scene, fixed_robot="panda_mobile")(_make_prebuilt_factory(_scene))


# Register the GR1 tabletop tasks against the `gr1` robot.
# These envs come from the `robocasa-gr1-tabletop-tasks` fork; the
# upstream kitchen `robocasa` package does NOT ship them. The two
# python packages share the name `robocasa` so a host installs one or
# the other -- our adapter exposes both task families regardless and
# the missing one fails at `robosuite.make()` with a clean
# "unknown env_name" rather than at import time.
for _task in _GR1_TABLETOP_TASKS:
    _scene = f"{_GR1_SCENE_PREFIX}/{_task}"
    SCENES.register(_scene, fixed_robot="gr1")(_make_prebuilt_factory(_scene))


@SCENES.register(_PROCEDURAL_SCENE_ID, fixed_robot="panda_mobile")
def _build_robocasa_procedural(env_cfg: SimEnvironment) -> _RoboCasaSim:
    """Procedural scenario surface -- (style x layout x fixtures x objects x verb)."""
    return _build_robocasa_sim(env_cfg)
