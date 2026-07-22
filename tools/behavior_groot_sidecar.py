#!/usr/bin/env python
"""Official Isaac-GR00T BEHAVIOR-1K policy sidecar."""

from __future__ import annotations

import argparse
import importlib
import io
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _encode_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": True, "npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    if "__ndarray__" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenRAL BEHAVIOR GR00T policy sidecar")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--instruction", default="turn on the radio")
    parser.add_argument(
        "--control-mode",
        choices=("temporal_ensemble", "receeding_temporal", "receeding_horizon"),
        default="temporal_ensemble",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22000)
    return parser.parse_args(argv)


def _register_r1pro_modality() -> dict[str, object]:
    configs = importlib.import_module("gr00t.configs.data.embodiment_configs")
    tags = importlib.import_module("gr00t.data.embodiment_tags")
    types = importlib.import_module("gr00t.data.types")
    register_modality_config = configs.register_modality_config
    EmbodimentTag = tags.EmbodimentTag
    ActionConfig = types.ActionConfig
    ActionFormat = types.ActionFormat
    ActionRepresentation = types.ActionRepresentation
    ActionType = types.ActionType
    ModalityConfig = types.ModalityConfig

    config = {
        "name": "robot_r1",
        "observation": {
            "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
            "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
        },
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["head", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "base_qvel",
                "torso",
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "base",
                "torso",
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="torso",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="left_arm",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    is_gripper=True,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="right_arm",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    is_gripper=True,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
    register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
    return {
        "state": {
            "base_qvel": {"start": 0, "end": 3},
            "torso": {"start": 53, "end": 57},
            "left_arm": {"start": 3, "end": 10},
            "left_gripper": {"start": 24, "end": 26},
            "right_arm": {"start": 28, "end": 35},
            "right_gripper": {"start": 49, "end": 51},
        },
        "action": {
            "base": {"start": 0, "end": 3},
            "torso": {"start": 3, "end": 7},
            "left_arm": {"start": 7, "end": 14},
            "left_gripper": {"start": 14, "end": 15},
            "right_arm": {"start": 15, "end": 22},
            "right_gripper": {"start": 22, "end": 23},
        },
    }


class _BehaviorGrootPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        tags = importlib.import_module("gr00t.data.embodiment_tags")
        wrapper = importlib.import_module("gr00t.eval.eval_b1k_wrapper")
        policy_module = importlib.import_module("gr00t.policy.gr00t_policy")
        EmbodimentTag = tags.EmbodimentTag
        B1KPolicyWrapper = wrapper.B1KPolicyWrapper
        Gr00tPolicy = policy_module.Gr00tPolicy

        modality = _register_r1pro_modality()
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=args.device,
            strict=True,
        )
        self._policy = B1KPolicyWrapper(
            policy=policy,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=modality,
            text_prompt=args.instruction,
            control_mode=args.control_mode,
        )

    def reset(self) -> None:
        self._policy.reset()

    def get_action(self, observation: dict[str, object]) -> NDArray[np.float32]:
        instruction = observation.pop("openral_instruction", None)
        if isinstance(instruction, str):
            self._policy.text_prompt = instruction
        action = self._policy.act(observation)
        return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def _serve(policy: _BehaviorGrootPolicy, *, task: str, host: str, port: int) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    print(f"[behavior_groot_sidecar] serving on tcp://{host}:{port}", flush=True)
    running = True
    while running:
        request = msgpack.unpackb(socket.recv(), object_hook=_decode_ndarray, raw=False)
        endpoint = request.get("endpoint")
        data = request.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, object] = {
                    "ok": True,
                    "model": "behavior_groot",
                    "task": task,
                    "action_dim": 23,
                }
            elif endpoint == "reset":
                policy.reset()
                reply = {"ok": True}
            elif endpoint == "get_action":
                reply = {"action": policy.get_action(dict(data["observation"]))}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        socket.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))
    socket.close(linger=0)
    ctx.term()
    return 0


def main(argv: list[str]) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args(argv)
    return _serve(
        _BehaviorGrootPolicy(args),
        task=args.task,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
