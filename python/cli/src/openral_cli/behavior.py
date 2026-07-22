"""BEHAVIOR Challenge WebSocket policy server."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from numpy.typing import NDArray
from openral_core.exceptions import ROSRuntimeError

if TYPE_CHECKING:
    from openral_core import VLASpec
    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import Observation

_R1PRO_CAMERA_SENSORS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
}
_R1PRO_STATE_KEY = "robot_r1::proprio"
_R1PRO_STATE_DIM = 61
_R1PRO_ACTION_DIM = 23
_RGB_RANK = 3
_RGB_CHANNELS = 3


def _pack_data(value: object) -> object:
    """Encode NumPy values with BEHAVIOR's msgpack wire format."""
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_data(value: dict[object, object]) -> object:
    """Decode NumPy values from BEHAVIOR's msgpack wire format."""
    if b"__ndarray__" in value:
        data = value.get(b"data")
        dtype = value.get(b"dtype")
        shape = value.get(b"shape")
        if (
            isinstance(data, bytes)
            and isinstance(dtype, str)
            and isinstance(shape, (list, tuple))
            and all(isinstance(size, int) for size in shape)
        ):
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(tuple(shape))
    if b"__npgeneric__" in value:
        data = value.get(b"data")
        dtype = value.get(b"dtype")
        if isinstance(dtype, str):
            return np.asarray(data, dtype=np.dtype(dtype)).reshape(()).item()
    return value


_packb = functools.partial(msgpack.packb, default=_pack_data)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_data)


def _as_message(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ROSRuntimeError("BEHAVIOR policy request must be a string-keyed mapping.")
    return payload


def _normalize_behavior_observation(
    raw: dict[str, object],
    *,
    instruction: str,
    state_dim: int,
) -> Observation:
    """Map the official R1Pro wrapper output into OpenRAL's policy observation."""
    state_raw = raw.get(_R1PRO_STATE_KEY)
    if state_raw is None:
        raise ROSRuntimeError(
            f"BEHAVIOR observation is missing R1Pro proprioception key {_R1PRO_STATE_KEY!r}."
        )
    state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
    if state.shape != (state_dim,):
        raise ROSRuntimeError(
            f"BEHAVIOR R1Pro proprioception must have {state_dim} values, got {state.shape[0]}."
        )

    images: dict[str, NDArray[np.uint8]] = {}
    depths: dict[str, NDArray[np.float32]] = {}
    for role, sensor_key in _R1PRO_CAMERA_SENSORS.items():
        rgb = raw.get(f"{sensor_key}::rgb")
        if rgb is not None:
            image = np.asarray(rgb, dtype=np.uint8)
            if image.ndim != _RGB_RANK or image.shape[-1] != _RGB_CHANNELS:
                raise ROSRuntimeError(
                    f"BEHAVIOR camera {role!r} RGB must be HWC with 3 channels, got {image.shape}."
                )
            images[role] = image

        depth = raw.get(f"{sensor_key}::depth_linear")
        if depth is not None:
            depths[role] = np.asarray(depth, dtype=np.float32)

    if not images:
        raise ROSRuntimeError("BEHAVIOR observation contains no configured R1Pro RGB cameras.")

    return {
        "images": images,
        "depths": depths,
        "state": state,
        "task": instruction,
        "behavior_raw": raw,
    }


def _policy_action(
    policy: PolicyAdapter,
    observation: Observation,
    instruction: str,
    action_dim: int,
) -> NDArray[np.float32]:
    action = np.asarray(policy.step(observation, instruction), dtype=np.float32).reshape(-1)
    if action.shape != (action_dim,):
        raise ROSRuntimeError(
            f"BEHAVIOR R1Pro action must have {action_dim} values, got {action.shape[0]}."
        )
    if not np.isfinite(action).all():
        raise ROSRuntimeError("BEHAVIOR policy emitted a non-finite action.")
    return action


def _create_behavior_app(
    policy: PolicyAdapter,
    *,
    task: str,
    instruction: str,
    state_dim: int,
    action_dim: int,
) -> FastAPI:
    app = FastAPI(title="OpenRAL BEHAVIOR policy server", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def _healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/")
    async def _policy_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_bytes(
            _packb(
                {
                    "policy": policy.spec.weights_uri,
                    "task": task,
                    "action_dim": action_dim,
                }
            )
        )
        try:
            while True:
                message = _as_message(_unpackb(await websocket.receive_bytes()))
                if message.get("reset") is True:
                    await asyncio.to_thread(policy.reset)
                    continue
                observation = _normalize_behavior_observation(
                    message,
                    instruction=instruction,
                    state_dim=state_dim,
                )
                action = await asyncio.to_thread(
                    _policy_action,
                    policy,
                    observation,
                    instruction,
                    action_dim,
                )
                await websocket.send_bytes(_packb({"action": action}))
        except WebSocketDisconnect:
            return

    return app


def _serve_behavior_policy(
    vla_spec: VLASpec,
    *,
    task: str,
    instruction: str,
    host: str,
    port: int,
    state_dim: int = _R1PRO_STATE_DIM,
    action_dim: int = _R1PRO_ACTION_DIM,
) -> None:
    """Serve an OpenRAL rSkill through the official BEHAVIOR policy protocol."""
    from openral_core import (  # noqa: PLC0415  # reason: keep `openral --help` light
        PhysicsBackend,
        SceneSpec,
        SimEnvironment,
        TaskSpec,
    )
    from openral_sim import make_policy  # noqa: PLC0415  # reason: load registry on serve only

    env = SimEnvironment(
        robot_id="r1pro",
        scene=SceneSpec(
            id="behavior_challenge",
            backend=PhysicsBackend.ISAACSIM,
            cameras=list(_R1PRO_CAMERA_SENSORS),
        ),
        task=TaskSpec(
            id=f"behavior/{task}",
            scene_id="behavior_challenge",
            instruction=instruction,
        ),
        vla=vla_spec,
    )
    policy = make_policy(env)
    app = _create_behavior_app(
        policy,
        task=task,
        instruction=instruction,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    print(f"OpenRAL BEHAVIOR policy: ws://{host}:{port} task={task}", flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        policy.close()
