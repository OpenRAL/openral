#!/usr/bin/env python
"""RoboTwin 2.0 scene sidecar — runs the SAPIEN dual-arm env in a separate venv (ADR-0061).

This is the **RoboTwin side** of the RoboTwin benchmark backend. It is launched
(auto-spawned) by :mod:`openral_sim.backends.robotwin` running under the openral
py3.12 venv, and it runs under the separate RoboTwin venv whose interpreter is
named by ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON``.

It constructs LeRobot's native ``robotwin`` gym env (``lerobot-eval
--env.type=robotwin``) for the requested task — the authoritative way to drive the
SAPIEN tasks — and serves a ZMQ REP loop speaking the same msgpack + ndarray framing
the openral side uses (``openral_sim.sidecar``):

    ping  -> {"ok": True, "action_dim": int, "task": str, "env": "robotwin"}
    reset -> {"observation": {...}, "sim_time_ns": int|None}
    step  -> {"observation": {...}, "reward", "terminated", "truncated", "info", "sim_time_ns": int|None}
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
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

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


def _scalar_float(value: object) -> float | None:
    if value is None or callable(value):
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
        scalar = _scalar_float(getattr(obj, name, None))
        if scalar is not None:
            return scalar
    return None


def _sapien_sim_time_ns(env: object) -> int | None:
    """Best-effort elapsed SAPIEN time from the wrapped RoboTwin env."""
    candidates = tuple(_sim_time_candidates(env))
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


def _sim_time_candidates(env: object) -> list[object]:
    """Return plausible wrapped/vectorized env objects that may expose time attrs."""
    seen: set[int] = set()
    out: list[object] = []

    def add(obj: object | None) -> None:
        if obj is None:
            return
        ident = id(obj)
        if ident in seen:
            return
        seen.add(ident)
        out.append(obj)

    add(env)
    for name in ("unwrapped", "env", "_env", "gym_env", "base_env", "scene", "_scene", "sim", "_sim"):
        add(getattr(env, name, None))
    for name in ("envs", "_envs", "venv"):
        value = getattr(env, name, None)
        if isinstance(value, (list, tuple)) and value:
            add(value[0])
    for obj in list(out):
        for name in ("unwrapped", "env", "_env", "gym_env", "base_env", "scene", "_scene", "sim", "_sim"):
            add(getattr(obj, name, None))
    return out


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
    p.add_argument(
        "--robotwin-root",
        default=os.environ.get("OPENRAL_ROBOTWIN_ROOT", ""),
        help="path to the RoboTwin checkout; used as cwd so assets/... resolves",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5757)
    p.add_argument(
        "--fallback-dt-s",
        type=float,
        default=1.0 / 30.0,
        help="deterministic sim-time step used only when RoboTwin hides elapsed-time attrs",
    )
    return p.parse_args(argv)


# ── env construction + observation adaptation ─────────────────────────────────


# LeRobot's `robotwin` env exposes its three cameras under these native keys
# (head + per-wrist). The openral scene refers to them generically as
# camera1/camera2/camera3 (matching the `lerobot/smolvla_robotwin` checkpoint's
# config.json input keys); we re-key in this fixed head→left→right order.
_ENV_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


class _RoboTwinEnv:
    """Thin wrapper over LeRobot's ``robotwin`` gym env in the eval-layer shape.

    Builds a single gym env and adapts its observations to ``{"images", "state",
    "task"}``. LeRobot's robotwin obs is a dict with ``pixels`` (per-camera HWC
    uint8) + ``agent_pos``; we re-key the env's native cameras
    (:data:`_ENV_CAMERA_NAMES`) to the openral scene camera names
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
        fallback_dt_s: float,
    ) -> None:
        self._task = task
        self._instruction = instruction
        # Map env-native camera keys -> requested openral scene keys, in order.
        self._cam_remap = dict(zip(_ENV_CAMERA_NAMES, cameras, strict=False))
        self._success_key = success_key
        self._obs_height = obs_height
        self._obs_width = obs_width
        self._env = self._make_env(
            obs_height=obs_height,
            obs_width=obs_width,
            episode_length=episode_length,
        )
        self._is_vector_env = hasattr(self._env, "single_action_space")
        self._action_dim = int(np.prod(self._env.action_space.shape))
        self._fallback_dt_ns = round(fallback_dt_s * 1_000_000_000)
        self._fallback_time_ns = 0

    def _make_env(
        self,
        *,
        obs_height: int,
        obs_width: int,
        episode_length: int,
    ) -> Any:
        """Construct LeRobot's robotwin gym env for ``self._task`` (single env)."""
        try:
            from lerobot.envs.robotwin import (
                RoboTwinEnv,  # type: ignore[import-not-found,import-untyped,unused-ignore]
            )

            return RoboTwinEnv(
                self._task,
                camera_names=_ENV_CAMERA_NAMES,
                observation_height=obs_height,
                observation_width=obs_width,
                episode_length=episode_length,
            )
        except ImportError:
            pass

        # Older LeRobot builds expose RoboTwin only through EnvConfig + make_env.
        from lerobot.envs.configs import (
            RoboTwinEnvConfig,  # type: ignore[import-not-found,import-untyped,unused-ignore]
        )
        from lerobot.envs.factory import (
            make_env,  # type: ignore[import-not-found,import-untyped,unused-ignore]
        )

        cfg = RoboTwinEnvConfig(
            task=self._task,
            obs_type="pixels_agent_pos",
            camera_names=",".join(_ENV_CAMERA_NAMES),
            observation_height=obs_height,
            observation_width=obs_width,
            episode_length=episode_length,
        )
        # Current LeRobot normalizes env factories to {suite: {task_id: vec_env}};
        # older builds returned the vector env directly. Accept both so the sidecar
        # stays pinned to the runtime env contract instead of a specific LeRobot SHA.
        envs = make_env(cfg, n_envs=1, use_async_envs=False)
        if isinstance(envs, dict):
            suite_envs = envs.get("robotwin") or next(iter(envs.values()))
            if isinstance(suite_envs, dict):
                return suite_envs.get(0) or next(iter(suite_envs.values()))
            return suite_envs
        return envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _unwrap_obs(self, obs: Any) -> dict[str, Any]:
        # Vectorised envs return batched obs (leading dim 1); squeeze it.
        def _first(x: Any) -> Any:
            arr = np.asarray(x)
            return arr[0] if arr.ndim and arr.shape[0] == 1 else arr

        images: dict[str, npt.NDArray[np.uint8]] = {}
        pixels = obs.get("pixels") if isinstance(obs, dict) else None
        if isinstance(pixels, dict):
            for cam, frame in pixels.items():
                out_key = self._cam_remap.get(str(cam), str(cam))
                frame_np = np.asarray(_first(frame), dtype=np.uint8)
                images[out_key] = self._resize_frame(frame_np)
        agent_pos = obs.get("agent_pos") if isinstance(obs, dict) else None
        state = (
            np.asarray(_first(agent_pos), dtype=np.float32).reshape(-1)
            if agent_pos is not None
            else np.zeros((self._action_dim,), dtype=np.float32)
        )
        return {"images": images, "state": state, "task": self._instruction}

    def _resize_frame(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if frame.shape[:2] == (self._obs_height, self._obs_width):
            return frame
        from PIL import Image

        img = Image.fromarray(frame)
        resized = img.resize((self._obs_width, self._obs_height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        obs, _info = self._env.reset(seed=seed)
        self._fallback_time_ns = 0
        return self._unwrap_obs(obs)

    def step(self, action: npt.NDArray[np.float32]) -> dict[str, Any]:
        action_arr: npt.NDArray[np.float32] = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._is_vector_env:
            action_arr = action_arr.reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action_arr)
        if _sapien_sim_time_ns(self._env) is None:
            self._fallback_time_ns += self._fallback_dt_ns
        success = bool(np.asarray(info.get(self._success_key, False)).any())
        return {
            "observation": self._unwrap_obs(obs),
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
            "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
            "info": {self._success_key: success},
            "sim_time_ns": self.sim_time_ns(),
        }

    def sim_time_ns(self) -> int | None:
        """Elapsed SAPIEN time from the wrapped RoboTwin env, when exposed."""
        return _sapien_sim_time_ns(self._env) or self._fallback_time_ns

    def render(self) -> npt.NDArray[np.uint8] | None:
        return None

    def close(self) -> None:
        self._env.close()


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.robotwin_root:
        root = Path(args.robotwin_root).expanduser().resolve()
        assets = root / "assets" / "objects" / "objaverse" / "list.json"
        if not assets.is_file():
            raise FileNotFoundError(f"RoboTwin assets not found at {assets}")
        os.chdir(root)
        sys.path.insert(0, str(root))
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    env = _RoboTwinEnv(
        task=args.task,
        instruction=args.instruction,
        cameras=cameras,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        episode_length=args.episode_length,
        success_key=args.success_key,
        fallback_dt_s=args.fallback_dt_s,
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
                reply = {
                    "observation": env.reset(seed=data.get("seed")),
                    "sim_time_ns": env.sim_time_ns(),
                }
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
