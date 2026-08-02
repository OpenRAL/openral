#!/usr/bin/env python
"""BEHAVIOR-1K OmniGibson environment sidecar."""

from __future__ import annotations

import argparse
import importlib
import io
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
    parser = argparse.ArgumentParser(description="OpenRAL BEHAVIOR scene sidecar")
    parser.add_argument("--task", required=True)
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("train", "public_test", "hidden_test"), default="public_test"
    )
    parser.add_argument(
        "--env-wrapper",
        # DefaultWrapper (rgb-only 224x224) is the only wrapper that boots on the
        # pinned OmniGibson build: RGBDFullResWrapper reloads the full observation
        # space in __init__, reading joint state before og.sim.update_handles().
        default="omnigibson.eval.wrappers.DefaultWrapper",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _to_numpy(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return value


def _to_numpy_dict(value: object) -> dict[str, object]:
    converted = _to_numpy(value)
    if not isinstance(converted, dict) or not all(isinstance(key, str) for key in converted):
        raise TypeError(
            f"expected string-keyed observation mapping, got {type(converted).__name__}"
        )
    return converted


class _BehaviorEnv:
    def __init__(self, args: argparse.Namespace) -> None:
        omega = importlib.import_module("omegaconf")
        evaluator_module = importlib.import_module("omnigibson.eval.evaluator")
        self._torch = importlib.import_module("torch")
        self._task = args.task
        self._instance_index = args.instance_index
        self._mode = args.mode
        instance_id = evaluator_module.resolve_instance_ids(
            args.task,
            [args.instance_index],
            mode=args.mode,
        )[0]
        cfg = omega.OmegaConf.create(
            {
                "env_wrapper": {"_target_": args.env_wrapper},
                "policy_name": "local",
                "model": {
                    "_target_": "omnigibson.eval.policies.LocalPolicy",
                    "action_dim": None,
                },
                "headless": args.headless,
                "partial_scene_load": True,
                "max_steps": args.max_steps,
                "write_video": False,
                "mode": args.mode,
                "seed": 0,
                "task": {"name": args.task},
                "robot": None,
            }
        )
        self._evaluator = evaluator_module.Evaluator(cfg)
        self._evaluator.reset()
        self._evaluator.load_task_instance(instance_id)
        self._steps = 0
        self.reset()

    @property
    def action_dim(self) -> int:
        return int(self._evaluator.robot.action_dim)

    def reset(self, seed: int | None = None) -> dict[str, object]:
        if seed is not None:
            # Apply the caller's per-episode seed the same way the official
            # eval entrypoint seeds (Python + NumPy + Torch via
            # ``seed_everything``) so multi-seed benchmark sweeps genuinely
            # vary and the recorded seed is the one that was applied.
            eval_utils = importlib.import_module("omnigibson.eval.utils.eval_utils")
            eval_utils.seed_everything(int(seed))
        self._evaluator.reset()
        self._steps = 0
        return self._obs_payload()

    def _obs_payload(self) -> dict[str, object]:
        """The evaluator observation plus the OpenRAL-side ``base_pose`` key.

        The 61-D proprio carries only base *velocity*; the robot's world
        ``(x, y, yaw)`` is read from OmniGibson directly so the HAL's /odom
        publisher can report real base motion. The backend pops the key back
        out before forwarding the official policy wire payload.
        """
        payload = _to_numpy_dict(self._evaluator.obs)
        robot = self._evaluator.robot
        try:
            transforms = importlib.import_module("omnigibson.utils.transform_utils")
            pos, quat = robot.get_position_orientation()
            yaw = float(transforms.quat2euler(quat)[2])
            payload["base_pose"] = np.asarray([float(pos[0]), float(pos[1]), yaw], dtype=np.float32)
        except Exception as exc:  # reason: pose is best-effort telemetry; never fail an obs over it
            print(f"[behavior_scene_sidecar] base_pose read failed: {exc}", flush=True)
        return payload

    def step(self, action: NDArray[np.float32]) -> dict[str, object]:
        evaluator = self._evaluator
        action_tensor = self._torch.from_numpy(np.asarray(action, dtype=np.float32))
        evaluator.robot_action = action_tensor
        obs, reward, terminated, truncated, info = evaluator.env.step(
            action_tensor,
            n_render_iterations=1,
        )
        obs = evaluator._sync_lights_and_get_obs(obs)
        evaluator.obs = evaluator._preprocess_obs(obs)
        for metric in evaluator.metrics:
            metric.step(evaluator.env, action_tensor, obs, reward, terminated, truncated, info)
        self._steps += 1
        metrics: dict[str, object] = {}
        if terminated or truncated:
            for metric in evaluator.metrics:
                metrics.update(metric.aggregate(evaluator.env))
        success = bool(info.get("done", {}).get("success", False))
        metric_info = _to_numpy_dict(metrics)
        return {
            "observation": self._obs_payload(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": {"is_success": success, **metric_info},
            "sim_time_ns": round(self._steps * (1.0 / 30.0) * 1e9),
        }

    def close(self) -> None:
        self._evaluator.__exit__(None, None, None)


def _serve(env: _BehaviorEnv, *, args: argparse.Namespace) -> int:
    msgpack = importlib.import_module("msgpack")
    zmq = importlib.import_module("zmq")
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"[behavior_scene_sidecar] serving on tcp://{args.host}:{args.port}", flush=True)
    running = True
    while running:
        request = msgpack.unpackb(socket.recv(), object_hook=_decode_ndarray, raw=False)
        endpoint = request.get("endpoint")
        data = request.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, object] = {
                    "ok": True,
                    "scene": "behavior",
                    "task": args.task,
                    "instance_index": args.instance_index,
                    "mode": args.mode,
                    "max_steps": args.max_steps,
                    "env_wrapper": args.env_wrapper,
                    "action_dim": env.action_dim,
                }
            elif endpoint == "reset":
                reply = {
                    "observation": env.reset(seed=data.get("seed")),
                    "sim_time_ns": 0,
                }
            elif endpoint == "step":
                reply = env.step(np.asarray(data["action"], dtype=np.float32))
            elif endpoint == "close":
                env.close()
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
    args = _parse_args(argv)
    return _serve(_BehaviorEnv(args), args=args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
