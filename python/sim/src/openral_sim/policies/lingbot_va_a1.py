"""A1 Runtime LingBot-VA EEF policy adapter for real OpenRAL deployment.

The external A1 Runtime owns model serving and the proven EEF/IK contract.
This adapter owns only the policy boundary: OpenRAL observations go to the
LingBot server, predicted EEF steps are validated and lowered to six absolute
A1 joint targets plus one normalized gripper target, and that typed vector
continues through OpenRAL's safety kernel and HAL.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_sim.registry import POLICIES

_RUNTIME_ROOT_ENV = "OPENRAL_GALAXEA_A1_RUNTIME_ROOT"
_DEPLOYMENT_CONFIG = "configs/deployments/lingbot/fruit_placement_eef.toml"
_COLOR_NDIM = 3
_COLOR_CHANNELS = 3


def _bounded_joint_substep(
    current: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    max_step_rad: float,
) -> tuple[NDArray[np.float64], bool]:
    """Return a target within the declared feedback-relative joint-step limit."""
    if (
        current.shape != target.shape
        or current.ndim != 1
        or not np.isfinite(current).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("bounded joint substep requires matching finite vectors")
    if not np.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("max_step_rad must be finite and positive")
    delta = target - current
    largest = float(np.max(np.abs(delta)))
    if largest <= max_step_rad:
        return target.copy(), True
    float32_bound = float(
        np.nextafter(
            np.float32(max_step_rad),
            np.float32(0.0),
        )
    )
    return current + delta * (float32_bound / largest), False


def _validated_joint_substep_limit(
    *,
    policy_substep_rad: float,
    max_target_step_rad: float,
    initial_alignment_tolerance_rad: float,
) -> float:
    """Validate the policy bound against both HAL command phases."""
    policy_limit = float(policy_substep_rad)
    limits = (policy_limit, float(max_target_step_rad), float(initial_alignment_tolerance_rad))
    if any(not np.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("A1 joint substep limits must be finite and positive")
    if policy_limit > min(limits[1:]):
        raise ValueError(
            "policy joint substep must not exceed the HAL target-step or initial-alignment limit"
        )
    return policy_limit


def _joint_limits_from_description(
    robot_description: Any,
    *,
    expected_joint_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ordered command limits from the active OpenRAL robot contract."""
    joints = getattr(robot_description, "joints", ())
    by_name = {getattr(joint, "name", None): joint for joint in joints}
    if len(by_name) != len(joints) or set(by_name) != set(expected_joint_names):
        raise ValueError("A1 Runtime IK joint names must exactly match the OpenRAL description")
    lower: list[float] = []
    upper: list[float] = []
    for name in expected_joint_names:
        limits = getattr(by_name[name], "position_limits", None)
        if limits is None:
            raise ValueError(f"OpenRAL joint {name!r} requires finite position limits")
        try:
            lo_raw, hi_raw = limits
            lo, hi = float(lo_raw), float(hi_raw)
        except (TypeError, ValueError):
            raise ValueError(f"OpenRAL joint {name!r} requires finite position limits") from None
        if not np.isfinite((lo, hi)).all() or lo >= hi:
            raise ValueError(f"OpenRAL joint {name!r} requires finite position limits")
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _build_manifest_bounded_ik_solver(
    *,
    ik_module: Any,
    system: Any,
    robot_description: Any,
) -> Any:
    """Construct Runtime IK with the active OpenRAL command envelope."""
    joint_names = tuple(system.joint_safety.names)
    lower_limits, upper_limits = _joint_limits_from_description(
        robot_description,
        expected_joint_names=joint_names,
    )
    config = system.eef_ik
    return ik_module.A1EefIkSolver(
        urdf_path=config.urdf,
        joint_names=joint_names,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_iterations=config.max_iterations,
        damping=config.damping,
        orientation_weight=config.orientation_weight,
        max_iteration_step_rad=config.max_iteration_step_rad,
        position_tolerance_m=config.position_tolerance_m,
        orientation_tolerance_rad=config.orientation_tolerance_rad,
        max_solution_delta_rad=config.max_solution_delta_rad,
    )


def _runtime_root() -> Path:
    raw = os.environ.get(_RUNTIME_ROOT_ENV)
    if raw is None:
        raise ROSConfigError(f"{_RUNTIME_ROOT_ENV} is required for the lingbot_va policy adapter")
    root = Path(raw).expanduser().resolve()
    config = root / _DEPLOYMENT_CONFIG
    if not config.is_file():
        raise ROSConfigError(f"A1 Runtime LingBot config not found: {config}")
    for import_root in (root / "external" / "embodied-ops" / "src", root):
        value = str(import_root)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


class _LingBotVaA1Adapter:
    """Replay LingBot EEF chunks as safe A1 joint/gripper action steps."""

    device = "external:a1-runtime"

    def __init__(self, spec: VLASpec, *, robot_description: Any) -> None:
        root = _runtime_root()
        client_module, config_module, protocol_module, ik_module, action_module = (
            import_module("galaxea_a1_runtime.apps.lingbot.client"),
            import_module("galaxea_a1_runtime.apps.lingbot.config"),
            import_module("galaxea_a1_runtime.apps.lingbot.protocol"),
            import_module("galaxea_a1_runtime.hardware.eef_ik"),
            import_module("galaxea_a1_runtime.policies.eef_actions"),
        )
        self.spec = spec
        self._config = config_module.load_lingbot_config(
            root / _DEPLOYMENT_CONFIG,
            repo_root=root,
        )
        self._action_config = action_module.build_action_transform_config(
            system=self._config.system
        )
        if robot_description is None or robot_description.name != "galaxea_a1":
            raise ROSConfigError("lingbot_va requires the Galaxea A1 RobotDescription")
        system = self._config.system
        try:
            self._solver = _build_manifest_bounded_ik_solver(
                ik_module=ik_module,
                system=system,
                robot_description=robot_description,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ROSConfigError(
                "lingbot_va could not construct A1 Runtime IK with the "
                "OpenRAL RobotDescription joint contract"
            ) from exc
        try:
            hal_defaults = robot_description.hal.parameters.defaults
            policy_substep_raw = spec.extra["max_joint_substep_rad"]
            if isinstance(policy_substep_raw, bool) or not isinstance(
                policy_substep_raw, (int, float)
            ):
                raise TypeError("max_joint_substep_rad must be numeric")
            policy_substep_rad = float(policy_substep_raw)
            self._max_joint_step_rad = _validated_joint_substep_limit(
                policy_substep_rad=policy_substep_rad,
                max_target_step_rad=hal_defaults["max_target_step_rad"],
                initial_alignment_tolerance_rad=hal_defaults["initial_alignment_tolerance_rad"],
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ROSConfigError(
                "lingbot_va requires a finite positive policy "
                "max_joint_substep_rad plus the Galaxea A1 HAL target-step "
                "and initial-alignment limits"
            ) from exc
        server = self._config.server
        self._client = client_module.LingBotClient(
            server.host,
            server.port,
            connect_timeout_s=server.connect_timeout_s,
            close_timeout_s=server.close_timeout_s,
            expected_metadata=protocol_module.server_metadata(self._config),
        )
        latency_budget_ms = spec.extra.get("latency_budget_ms")
        if isinstance(latency_budget_ms, bool) or not isinstance(latency_budget_ms, (int, float)):
            raise ROSConfigError("lingbot_va requires its manifest latency budget in VLASpec.extra")
        self._inference_timeout_s = float(latency_budget_ms) / 1000.0
        if not np.isfinite(self._inference_timeout_s) or self._inference_timeout_s <= 0.0:
            raise ROSConfigError("lingbot_va requires a positive per-chunk latency budget")
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openral-lingbot-a1",
        )
        self._boundary_future: Future[dict[str, Any]] | None = None
        self._boundary_deadline = 0.0
        self._instruction: str | None = None
        self._origin_pose7: NDArray[np.float64] | None = None
        self._chunk: Any | None = None
        self._steps: list[tuple[int, int, int, NDArray[np.float32]]] = []
        self._step_index = 0
        self._pending_observation = False
        self._key_frames: list[dict[str, NDArray[np.uint8]]] = []
        self._first = True
        self._model_calls = 0
        self._active_joint_target: NDArray[np.float64] | None = None
        self._active_gripper: float | None = None
        self._active_cache_frame_index: int | None = None
        self._active_action_index: int | None = None
        self._last_joint_target: NDArray[np.float64] | None = None
        self._last_gripper_target: float | None = None

    def reset(self) -> None:
        """Clear episode coordinates and chunk/cache state."""
        if self._boundary_future is not None:
            raise ROSRuntimeError("cannot reset LingBot while cache inference is active")
        self._instruction = None
        self._origin_pose7 = None
        self._chunk = None
        self._steps.clear()
        self._step_index = 0
        self._pending_observation = False
        self._key_frames.clear()
        self._first = True
        self._model_calls = 0
        self._clear_active_action()
        self._last_joint_target = None
        self._last_gripper_target = None

    def close(self) -> None:
        """Close only the inference client; the managed server remains external."""
        self._client.close()
        self._inference_executor.shutdown(wait=True, cancel_futures=True)

    def step(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> NDArray[np.float32]:
        """Return one absolute joint/gripper action through the OpenRAL contract."""
        joints = self._joint_state(observation)
        model_observation = self._model_observation(observation)
        if self._instruction != instruction:
            self.reset()
            self._client.reset(instruction)
            self._instruction = instruction
        if self._origin_pose7 is None:
            xyz, quat = self._solver.forward(joints)
            self._origin_pose7 = np.concatenate([xyz, quat])

        if self._pending_observation:
            self._key_frames.append(model_observation)
            self._pending_observation = False
        boundary_hold = self._poll_chunk_boundary()
        if boundary_hold is not None:
            return boundary_hold
        if self._chunk is not None and self._step_index >= len(self._steps):
            self._start_chunk_boundary(model_observation, instruction)
            return self._hold_action()
        if self._chunk is None:
            self._infer_chunk(model_observation, instruction)

        assert self._chunk is not None
        if self._active_joint_target is None:
            _frame, action_index, cache_frame_index, raw_action = self._steps[self._step_index]
            self._begin_action(
                raw_action,
                joints=joints,
                cache_frame_index=cache_frame_index,
                action_index=action_index,
            )
        return self._step_active_action(joints)

    def _infer_chunk(
        self,
        model_observation: dict[str, NDArray[np.uint8]],
        instruction: str,
    ) -> None:
        limit = self._config.execution.max_model_calls
        if limit > 0 and self._model_calls >= limit:
            raise ROSRuntimeError(f"LingBot rollout reached configured max_model_calls={limit}")
        response = self._client.infer({"obs": [model_observation], "prompt": instruction})
        self._install_chunk(response, first=self._first)

    def _install_chunk(self, response: dict[str, Any], *, first: bool) -> None:
        """Validate and install one completed LingBot response on the runner thread."""
        rollout_module = import_module("galaxea_a1_runtime.apps.lingbot.rollout")
        try:
            raw = response["action"]
        except KeyError as exc:
            raise ROSRuntimeError("LingBot response did not contain 'action'") from exc
        policy = self._config.policy_server
        chunk = rollout_module.LingBotActionChunk.from_response(
            raw,
            expected_shape=(
                len(policy.action_channel_ids),
                policy.frame_chunk_size,
                policy.action_per_frame,
            ),
            first=first,
            execute_frames=self._config.execution.execute_frames,
            observations_per_frame=self._config.execution.kv_observations_per_frame,
        )
        self._chunk = chunk
        self._steps = list(chunk.steps())
        self._step_index = 0
        self._key_frames.clear()
        self._model_calls += 1
        self._first = first

    def _start_chunk_boundary(
        self,
        model_observation: dict[str, NDArray[np.uint8]],
        instruction: str,
    ) -> None:
        """Run cache synchronization and the next inference without starving actuation."""
        if self._chunk is None or self._boundary_future is not None:
            raise ROSRuntimeError("invalid LingBot chunk-boundary state")
        if not self._key_frames:
            raise ROSRuntimeError("LingBot chunk ended without KV-cache observations")
        limit = self._config.execution.max_model_calls
        if limit > 0 and self._model_calls >= limit:
            raise ROSRuntimeError(f"LingBot rollout reached configured max_model_calls={limit}")
        cache_request = {
            "obs": [
                {key: np.array(value, copy=True) for key, value in frame.items()}
                for frame in self._key_frames
            ],
            "compute_kv_cache": True,
            "imagine": False,
            "state": np.array(self._chunk.cache_state, copy=True),
        }
        next_request = {
            "obs": [{key: np.array(value, copy=True) for key, value in model_observation.items()}],
            "prompt": instruction,
        }

        def infer_next() -> dict[str, Any]:
            self._client.infer(cache_request)
            response = self._client.infer(next_request)
            if not isinstance(response, dict):
                raise ROSRuntimeError("LingBot inference response must be a mapping")
            return cast(dict[str, Any], response)

        self._boundary_future = self._inference_executor.submit(infer_next)
        self._boundary_deadline = time.monotonic() + self._inference_timeout_s
        self._first = False
        self._chunk = None
        self._steps.clear()
        self._step_index = 0
        self._key_frames.clear()

    def _poll_chunk_boundary(self) -> NDArray[np.float32] | None:
        """Return a hold while boundary inference is pending; install it when ready."""
        if self._boundary_future is None:
            return None
        if not self._boundary_future.done():
            if time.monotonic() > self._boundary_deadline:
                raise ROSRuntimeError(
                    "LingBot cache/inference exceeded the rSkill "
                    f"{self._inference_timeout_s:.3f} s per-chunk latency budget"
                )
            return self._hold_action()
        try:
            response = self._boundary_future.result()
        except Exception as exc:
            raise ROSRuntimeError(f"LingBot cache/inference failed: {exc}") from exc
        self._boundary_future = None
        self._install_chunk(response, first=False)
        return None

    def _hold_action(self) -> NDArray[np.float32]:
        """Repeat the last absolute target while bounded cache inference runs."""
        if self._last_joint_target is None or self._last_gripper_target is None:
            raise ROSRuntimeError("LingBot cannot hold before its first validated action")
        return np.concatenate([self._last_joint_target, [self._last_gripper_target]]).astype(
            np.float32
        )

    def _begin_action(
        self,
        raw_action: NDArray[np.float32],
        *,
        joints: NDArray[np.float64],
        cache_frame_index: int,
        action_index: int,
    ) -> None:
        action_module = import_module("galaxea_a1_runtime.policies.eef_actions")
        assert self._origin_pose7 is not None
        if self._config.action.pose_mode == "absolute":
            absolute = np.asarray(raw_action, dtype=np.float64)
        else:
            absolute = action_module.relative_action_to_absolute(
                raw_action,
                self._origin_pose7,
                min_quat_norm=self._action_config.min_quat_norm,
            )
        validated = action_module.validate_policy_action(absolute, self._action_config)
        try:
            solution = self._solver.solve(joints, validated[:3], validated[3:7])
        except Exception as exc:
            raise ROSRuntimeError(f"LingBot A1 EEF target was rejected: {exc}") from exc
        solved_joints = np.asarray(solution.joint_positions, dtype=np.float64)
        self._active_joint_target = solved_joints
        self._active_gripper = float(validated[7])
        self._active_cache_frame_index = cache_frame_index
        self._active_action_index = action_index

    def _step_active_action(self, joints: NDArray[np.float64]) -> NDArray[np.float32]:
        """Advance one model action without truncating its solved joint target."""
        assert self._active_joint_target is not None
        assert self._active_gripper is not None
        target, reaches_solution = _bounded_joint_substep(
            joints,
            self._active_joint_target,
            max_step_rad=self._max_joint_step_rad,
        )

        action = np.concatenate([target, [self._active_gripper]]).astype(np.float32)
        self._last_joint_target = target.copy()
        self._last_gripper_target = self._active_gripper
        if not reaches_solution:
            return action

        assert self._chunk is not None
        assert self._active_cache_frame_index is not None
        assert self._active_action_index is not None
        assert self._origin_pose7 is not None
        executed_xyz, executed_quat = self._solver.forward(target)
        executed = np.concatenate([executed_xyz, executed_quat, [self._active_gripper]])
        if self._config.action.pose_mode == "absolute":
            cache_action = executed
        else:
            action_module = import_module("galaxea_a1_runtime.policies.eef_actions")
            cache_action = action_module.absolute_action_to_relative(
                executed,
                self._origin_pose7,
                min_quat_norm=self._action_config.min_quat_norm,
            )
        self._chunk.cache_state[
            :,
            self._active_cache_frame_index,
            self._active_action_index,
        ] = cache_action
        self._pending_observation = self._chunk.needs_observation_after(self._active_action_index)
        self._step_index += 1
        self._clear_active_action()
        return action

    def _clear_active_action(self) -> None:
        self._active_joint_target = None
        self._active_gripper = None
        self._active_cache_frame_index = None
        self._active_action_index = None

    @staticmethod
    def _joint_state(observation: dict[str, Any]) -> NDArray[np.float64]:
        state = np.asarray(observation.get("state"), dtype=np.float64)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ROSRuntimeError(
                f"lingbot_va requires six finite A1 joint positions, got {state.shape}"
            )
        return state

    def _model_observation(self, observation: dict[str, Any]) -> dict[str, NDArray[np.uint8]]:
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ROSRuntimeError("lingbot_va requires OpenRAL image observations")
        result: dict[str, NDArray[np.uint8]] = {}
        for source, target in (
            ("front", self._config.observations.front_key),
            ("wrist", self._config.observations.wrist_key),
        ):
            image = np.asarray(images.get(source))
            if (
                image.ndim != _COLOR_NDIM
                or image.shape[2] != _COLOR_CHANNELS
                or image.dtype != np.uint8
            ):
                raise ROSRuntimeError(
                    f"lingbot_va image {source!r} must be HWC uint8 RGB, "
                    f"got shape={image.shape} dtype={image.dtype}"
                )
            result[target] = image
        return result


@POLICIES.register("lingbot_va")
def _build_lingbot_va(env_cfg: Any) -> _LingBotVaA1Adapter:
    """Build the external A1 Runtime-backed LingBot adapter."""
    return _LingBotVaA1Adapter(
        env_cfg.vla,
        robot_description=getattr(env_cfg, "robot_description", None),
    )
