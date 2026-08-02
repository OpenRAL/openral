"""BEHAVIOR-1K / OmniGibson scene adapter via the official evaluator."""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode

from openral_sim import _behavior_wire
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

_SCENE_ID = "behavior"
_ROBOT_ID = "r1pro"
_SIDECAR_PYTHON_ENV = "OPENRAL_BEHAVIOR_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_BEHAVIOR_SIDECAR_SCRIPT"
_AUTO_SPAWN_ENV = "OPENRAL_BEHAVIOR_AUTO_SPAWN"
_HOST_ENV = "OPENRAL_BEHAVIOR_HOST"
_PORT_ENV = "OPENRAL_BEHAVIOR_PORT"

_DEFAULT_HOST = "127.0.0.1"
_PORT_MIN = 23_000
_PORT_MAX = 23_999
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 1_200.0
_ACTION_DIM = 23
_ACTION_GROUP_SIZE = 6
_JOINT_DIM = 22

_STATE_KEY = _behavior_wire.STATE_KEY
_CAMERA_KEYS = _behavior_wire.CAMERA_RGB_KEYS
_explicit_port = _behavior_wire.explicit_port


def _scene_default_port(
    task: str, instance_index: int, mode: str, max_steps: int, env_wrapper: str
) -> int:
    # Every option that changes the running environment's behavior is part of
    # the port key, so a config change can never silently adopt a still-running
    # sidecar booted with the old config (the ping-identity check is lenient
    # about absent keys; the port hash is the primary stale-reuse guard).
    key = f"behavior|{task}|{instance_index}|{mode}|{max_steps}|{env_wrapper}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _opt_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _opt_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sidecar_python() -> Path:
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    candidates = (
        [Path(override).expanduser()]
        if override
        else [
            Path.home() / ".cache" / "openral" / "behavior" / ".venv" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "behavior" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "behavior" / "bin" / "python",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ROSConfigError(
        "BEHAVIOR sidecar Python not found. Install BEHAVIOR-1K v3.9.0 with "
        "`./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval`, "
        f"then set {_SIDECAR_PYTHON_ENV} to that environment's python."
    )


def _locate_sidecar_script() -> Path:
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "behavior_scene_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/behavior_scene_sidecar.py upwards from {here}. "
        f"Set {_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


def _joint_state_from_policy_state(
    state: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    positions = np.concatenate(
        [
            state[53:57],
            state[3:10],
            state[24:26],
            state[28:35],
            state[49:51],
        ]
    ).astype(np.float32)
    velocities = np.concatenate(
        [
            state[57:61],
            state[10:17],
            state[26:28],
            state[35:42],
            state[51:53],
        ]
    ).astype(np.float32)
    return positions, velocities


def _compose_action_group(actions: list[Action]) -> NDArray[np.float32]:
    if len(actions) != _ACTION_GROUP_SIZE:
        raise ROSRuntimeError(
            f"BEHAVIOR action group requires {_ACTION_GROUP_SIZE} slots, got {len(actions)}."
        )
    expected = (
        ControlMode.BODY_TWIST,
        ControlMode.JOINT_POSITION,
        ControlMode.JOINT_POSITION,
        ControlMode.GRIPPER_POSITION,
        ControlMode.JOINT_POSITION,
        ControlMode.GRIPPER_POSITION,
    )
    modes = tuple(action.control_mode for action in actions)
    if modes != expected:
        raise ROSRuntimeError(
            f"BEHAVIOR action slot order mismatch: got {[mode.value for mode in modes]}."
        )

    base, torso, left_arm, left_gripper, right_arm, right_gripper = actions
    if not base.body_twist:
        raise ROSRuntimeError("BEHAVIOR base slot has no body_twist payload.")
    vx, vy, _vz, _wx, _wy, wz = base.body_twist[0]

    def _joint_row(action: Action) -> NDArray[np.float32]:
        if not action.joint_targets:
            raise ROSRuntimeError("BEHAVIOR joint slot has no joint_targets payload.")
        row = np.asarray(action.joint_targets[0], dtype=np.float32).reshape(-1)
        if row.shape != (_JOINT_DIM,):
            raise ROSRuntimeError(
                f"BEHAVIOR joint slot must be padded to {_JOINT_DIM}, got {row.shape[0]}."
            )
        return row

    torso_row = _joint_row(torso)
    left_row = _joint_row(left_arm)
    right_row = _joint_row(right_arm)
    if not left_gripper.gripper or not right_gripper.gripper:
        raise ROSRuntimeError("BEHAVIOR gripper slots must carry scalar gripper payloads.")

    return np.asarray(
        [
            vx,
            vy,
            wz,
            *torso_row[0:4],
            *left_row[4:11],
            float(left_gripper.gripper[0]),
            *right_row[13:20],
            float(right_gripper.gripper[0]),
        ],
        dtype=np.float32,
    )


@dataclass
class _BehaviorSidecar:
    scene: SceneSpec
    task: TaskSpec
    _client: SidecarClient
    _last_image: NDArray[np.uint8] | None = None
    _last_state: NDArray[np.float32] | None = None
    _sim_time_ns: int | None = None

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def action_group_size(self) -> int:
        return _ACTION_GROUP_SIZE

    def reset(self, seed: int | None = None) -> Observation:
        reply = self._client.call("reset", {"seed": seed})
        self._sim_time_ns = int(reply.get("sim_time_ns", 0))
        return self._wrap_obs(self._client.require(reply, "observation"))

    def step(self, action: NDArray[np.float32]) -> StepResult:
        return self._step_vector(np.asarray(action, dtype=np.float32).reshape(-1))

    def idle_action(self) -> NDArray[np.float32]:
        """Hold the current articulated pose while advancing scene physics."""
        if self._last_state is None:
            return np.zeros(_ACTION_DIM, dtype=np.float32)
        state = self._last_state
        left_gripper = 2.0 * float(np.mean(state[24:26]) / 0.05) - 1.0
        right_gripper = 2.0 * float(np.mean(state[49:51]) / 0.05) - 1.0
        return np.asarray(
            [
                0.0,
                0.0,
                0.0,
                *state[53:57],
                *state[3:10],
                np.clip(left_gripper, -1.0, 1.0),
                *state[28:35],
                np.clip(right_gripper, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def step_action_group(self, actions: list[Action]) -> StepResult:
        ticks = {action.tick_index for action in actions}
        if len(ticks) != 1 or 0 in ticks:
            raise ROSRuntimeError(
                f"BEHAVIOR action group has invalid tick indices: {sorted(ticks)}"
            )
        return self._step_vector(_compose_action_group(actions))

    def _step_vector(self, action: NDArray[np.float32]) -> StepResult:
        if action.shape != (_ACTION_DIM,):
            raise ROSRuntimeError(
                f"BEHAVIOR environment expects {_ACTION_DIM} actions, got {action.shape[0]}."
            )
        reply = self._client.call("step", {"action": action})
        self._sim_time_ns = int(reply.get("sim_time_ns", 0))
        return StepResult(
            observation=self._wrap_obs(self._client.require(reply, "observation")),
            reward=float(self._client.require(reply, "reward")),
            terminated=bool(self._client.require(reply, "terminated")),
            truncated=bool(self._client.require(reply, "truncated")),
            info=dict(reply.get("info", {})),
        )

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def sim_time_ns(self) -> int | None:
        return self._sim_time_ns

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()

    def _wrap_obs(self, raw: dict[str, Any]) -> Observation:
        # ``base_pose`` is an OpenRAL-side addition the sidecar injects next to
        # the official evaluator keys (the robot's world (x, y, yaw), read from
        # OmniGibson — the 61-D proprio carries only base *velocity*). Pop it
        # out so ``behavior_raw`` — the payload forwarded verbatim to the
        # official GR00T policy wire — stays byte-identical to the evaluator's
        # own observation.
        raw = dict(raw)
        base_pose_raw = raw.pop("base_pose", None)
        state = np.asarray(raw.get(_STATE_KEY, []), dtype=np.float32).reshape(-1)
        if state.shape != (61,):
            raise ROSRuntimeError(
                f"BEHAVIOR sidecar observation state must have 61 values, got {state.shape[0]}."
            )
        self._last_state = state.copy()
        images: dict[str, NDArray[np.uint8]] = {}
        for role, raw_key in _CAMERA_KEYS.items():
            image = raw.get(raw_key)
            if image is not None:
                images[role] = np.asarray(image, dtype=np.uint8)
        if images:
            self._last_image = images.get("head", next(iter(images.values())))
        positions, velocities = _joint_state_from_policy_state(state)
        obs: Observation = {
            "images": images,
            "state": state,
            "policy_state": state,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "task": self.task.instruction,
            "behavior_raw": raw,
        }
        if base_pose_raw is not None:
            base_pose = np.asarray(base_pose_raw, dtype=np.float32).reshape(-1)
            if base_pose.shape[0] >= 3:  # noqa: PLR2004  # reason: x/y/yaw triple
                # SimAttachedHAL.base_pose reads this for its non-MuJoCo /odom path.
                obs["base_pose"] = base_pose[:3]
        return obs


@SCENES.register(_SCENE_ID, fixed_robot=_ROBOT_ID)
def _build_behavior_scene(env_cfg: SimEnvironment) -> _BehaviorSidecar:
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("behavior_groot_client")
    opts = env_cfg.scene.backend_options
    task = str(opts.get("task") or env_cfg.task.id.removeprefix("behavior/"))
    instance_index = _opt_int(opts.get("instance_index"), 0)
    mode = str(opts.get("mode", "public_test"))
    env_wrapper = str(opts.get("env_wrapper", "omnigibson.eval.wrappers.DefaultWrapper"))
    host = os.environ.get(_HOST_ENV, str(opts.get("host", _DEFAULT_HOST)))
    max_steps = int(env_cfg.task.max_steps or 500)
    default_port = _scene_default_port(task, instance_index, mode, max_steps, env_wrapper)
    port = _explicit_port(
        os.environ.get(_PORT_ENV), opts.get("port"), default_port, env_var=_PORT_ENV
    )
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    launch_argv: list[str] = []
    if auto_spawn:
        launch_argv = [
            str(_sidecar_python()),
            str(_locate_sidecar_script()),
            "--task",
            task,
            "--instance-index",
            str(instance_index),
            "--mode",
            mode,
            "--env-wrapper",
            env_wrapper,
            "--max-steps",
            str(max_steps),
            "--host",
            host,
            "--port",
            str(port),
            "--headless",
        ]

    client = SidecarClient(
        name="behavior",
        host=host,
        port=port,
        timeout_ms=_opt_int(opts.get("timeout_ms"), _DEFAULT_TIMEOUT_MS),
        boot_timeout_s=_opt_float(opts.get("boot_timeout_s"), _DEFAULT_BOOT_TIMEOUT_S),
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={
            "scene": "behavior",
            "task": task,
            "instance_index": instance_index,
            "mode": mode,
            "max_steps": max_steps,
            "env_wrapper": env_wrapper,
        },
    )
    client.connect()
    return _BehaviorSidecar(scene=env_cfg.scene, task=env_cfg.task, _client=client)
