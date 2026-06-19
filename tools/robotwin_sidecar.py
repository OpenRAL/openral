#!/usr/bin/env python
"""RoboTwin 2.0 scene sidecar — runs the SAPIEN dual-arm env in a py3.10 venv (ADR-0061).

This is the **RoboTwin side** of the RoboTwin benchmark backend. It is launched
(auto-spawned) by :mod:`openral_sim.backends.robotwin` running under the openral
py3.12 venv, and it runs under the separate RoboTwin py3.10 venv whose interpreter is
named by ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON``.

It constructs LeRobot's native ``robotwin`` gym env (``lerobot-eval
--env.type=robotwin``) for the requested task — the authoritative way to drive the
SAPIEN tasks — and serves a ZMQ REP loop speaking the same msgpack + ndarray framing
the openral side uses (``openral_sim.sidecar``):

    ping  -> {"ok": True, "action_dim": int, "task": str, "env": "robotwin"}
    reset -> {"observation": {...}, "sim_time_ns": int|None}
    step  -> {"observation": {...}, "reward", "terminated", "truncated", "info"}
    render-> {"frame": <uint8 HWC>|None}
    close -> {"ok": True}

Observation dict shape (eval-layer contract):
    {"images": {"head_camera": <H,W,3 uint8>, "left_camera": ..., "right_camera": ...},
     "state": <14-D float32>, "task": str}

RoboTwin / SAPIEN / LeRobot are imported lazily inside :func:`main` so a syntax-only
import of this module (or `--help`) does not require the heavy venv.

Licensing: RoboTwin (MIT), SAPIEN (MIT), LeRobot (Apache-2.0) — all permissive. The
stack is large + CUDA-12.1-pinned, so it is an externally-provisioned sidecar venv,
never vendored into the repo (CLAUDE.md §1.9).
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any

import numpy as np

# ── msgpack ndarray codec (matches openral_sim.sidecar) ───────────────────────


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
    p = argparse.ArgumentParser(description="OpenRAL RoboTwin 2.0 scene sidecar")
    p.add_argument("--task", required=True, help="upstream RoboTwin task, e.g. lift_pot")
    p.add_argument("--instruction", default="")
    p.add_argument("--obs-height", type=int, default=240)
    p.add_argument("--obs-width", type=int, default=320)
    p.add_argument(
        "--cameras",
        default="head_camera,left_camera,right_camera",
        help="comma-separated RoboTwin camera names",
    )
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--success-key", default="is_success")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5757)
    return p.parse_args(argv)


# ── env construction + observation adaptation ─────────────────────────────────


# LeRobot's `robotwin` env exposes its three cameras under these native keys
# (head + per-wrist). The openral scene refers to them generically as
# camera1/camera2/camera3 (matching the `lerobot/smolvla_robotwin` checkpoint's
# config.json input keys); we re-key in this fixed head→left→right order.
_ENV_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


class _RoboTwinEnv:
    """Thin wrapper over LeRobot's ``robotwin`` gym env in the eval-layer shape.

    Builds a single (non-vectorised) gym env from a ``RoboTwinEnvConfig`` and adapts
    its observations to ``{"images", "state", "task"}``. LeRobot's robotwin obs is a
    dict with ``pixels`` (per-camera HWC uint8) + ``agent_pos``; we re-key the env's
    native cameras (:data:`_ENV_CAMERA_NAMES`) to the openral scene camera names
    (camera1/camera2/camera3) in order and expose ``agent_pos`` as ``state``.
    """

    def __init__(
        self,
        *,
        task: str,
        instruction: str,
        cameras: list[str],
        obs_height: int,
        obs_width: int,
        episode_length: int,
        success_key: str,
    ) -> None:
        self._task = task
        self._instruction = instruction
        # Map env-native camera keys -> requested openral scene keys, in order.
        self._cam_remap = dict(zip(_ENV_CAMERA_NAMES, cameras, strict=False))
        self._success_key = success_key
        self._env = self._make_env(
            obs_height=obs_height,
            obs_width=obs_width,
            episode_length=episode_length,
        )
        self._action_dim = int(np.prod(self._env.action_space.shape))

    def _make_env(
        self,
        *,
        obs_height: int,
        obs_width: int,
        episode_length: int,
    ) -> Any:
        """Construct LeRobot's robotwin gym env for ``self._task`` (single env)."""
        # Importing the module registers the gym ids and the env config.
        from lerobot.envs.configs import RoboTwinEnvConfig  # type: ignore[import-not-found]
        from lerobot.envs.factory import make_env  # type: ignore[import-not-found]

        cfg = RoboTwinEnvConfig(
            task=self._task,
            obs_type="pixels_agent_pos",
            camera_names=",".join(_ENV_CAMERA_NAMES),
            observation_height=obs_height,
            observation_width=obs_width,
            episode_length=episode_length,
        )
        # make_env returns a (vectorised) env; ask for a single, synchronous env so
        # reset/step deal in a 1-element batch we squeeze in `_unwrap_obs`.
        return make_env(cfg, n_envs=1, use_async_envs=False)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _unwrap_obs(self, obs: Any) -> dict[str, Any]:
        # Vectorised envs return batched obs (leading dim 1); squeeze it.
        def _first(x: Any) -> Any:
            arr = np.asarray(x)
            return arr[0] if arr.ndim and arr.shape[0] == 1 else arr

        images: dict[str, np.ndarray] = {}
        pixels = obs.get("pixels") if isinstance(obs, dict) else None
        if isinstance(pixels, dict):
            for cam, frame in pixels.items():
                out_key = self._cam_remap.get(str(cam), str(cam))
                images[out_key] = np.asarray(_first(frame), dtype=np.uint8)
        agent_pos = obs.get("agent_pos") if isinstance(obs, dict) else None
        state = (
            np.asarray(_first(agent_pos), dtype=np.float32).reshape(-1)
            if agent_pos is not None
            else np.zeros((self._action_dim,), dtype=np.float32)
        )
        return {"images": images, "state": state, "task": self._instruction}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        obs, _info = self._env.reset(seed=seed)
        return self._unwrap_obs(obs)

    def step(self, action: np.ndarray) -> dict[str, Any]:
        batched = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(batched)
        success = bool(np.asarray(info.get(self._success_key, False)).any())
        return {
            "observation": self._unwrap_obs(obs),
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
            "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
            "info": {self._success_key: success},
        }

    def render(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        self._env.close()


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    env = _RoboTwinEnv(
        task=args.task,
        instruction=args.instruction,
        cameras=cameras,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        episode_length=args.episode_length,
        success_key=args.success_key,
    )
    try:
        return _serve(env, host=args.host, port=args.port, task=args.task)
    finally:
        env.close()


def _serve(env: _RoboTwinEnv, *, host: str, port: int, task: str) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[robotwin_sidecar] serving task={task!r} on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                # Identity lets the client reject a stale sidecar serving a different
                # task on this port (SidecarClient.expected_identity).
                reply: dict[str, Any] = {
                    "ok": True,
                    "action_dim": env.action_dim,
                    "task": task,
                    "env": "robotwin",
                }
            elif endpoint == "reset":
                reply = {"observation": env.reset(seed=data.get("seed")), "sim_time_ns": None}
            elif endpoint == "step":
                reply = env.step(np.asarray(data["action"], dtype=np.float32))
            elif endpoint == "render":
                reply = {"frame": env.render()}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))

    sock.close(linger=0)
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
