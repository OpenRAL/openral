#!/usr/bin/env python
"""RoboTwin 2.0 scene sidecar — runs the SAPIEN dual-arm env in a separate venv.

RoboTwin side of the backend in ``openral_sim.backends.robotwin`` (py3.12),
auto-spawned under the venv named by ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON``.
Constructs LeRobot's native ``robotwin`` gym env (``lerobot-eval
--env.type=robotwin``) and serves ZMQ REP + msgpack/ndarray framing
(``openral_sim.sidecar``):

    ping  -> {"ok", "action_dim", "task", "env": "robotwin"}
    reset/step -> {"observation", ["reward","terminated","truncated","info",]"sim_time_ns"}
    render -> {"frame": uint8 HWC|None}   close -> {"ok"}
    observation = {"images": {"head_camera",...}, "state": 14-D float32, "task": str,
                   "joint_positions"/"joint_velocities": measured 14-D (see _measured_joint_state)}

RoboTwin/SAPIEN/LeRobot import lazily inside ``main`` so `--help` doesn't
need the heavy venv.

Licensing: RoboTwin (MIT), SAPIEN (MIT), LeRobot (Apache-2.0). Large and
CUDA-bound, so this is an externally-provisioned sidecar venv, never vendored (CLAUDE.md §1.9).
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import io
import os
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Attribute names a gym/SAPIEN wrapper may use to expose the env it wraps. The
# nesting walk in ``_sim_time_candidates`` follows these one level deep, twice.
_NESTED_ENV_ATTRS = (
    "unwrapped",
    "env",
    "_env",
    "gym_env",
    "base_env",
    "scene",
    "_scene",
    "sim",
    "_sim",
)

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
    for name in _NESTED_ENV_ATTRS:
        add(getattr(env, name, None))
    for name in ("envs", "_envs", "venv"):
        value = getattr(env, name, None)
        if isinstance(value, (list, tuple)) and value:
            add(value[0])
    for obj in list(out):
        for name in _NESTED_ENV_ATTRS:
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
        # 25 Hz: lerobot 0.7's robotwin integration + robots/aloha_agilex control rate.
        default=1.0 / 25.0,
        help="deterministic sim-time step used only when RoboTwin hides elapsed-time attrs",
    )
    return p.parse_args(argv)


# ── RoboTwin cuRobo fallback ──────────────────────────────────────────────────


class _CuroboPlannerUnavailable:
    """Stand-in for RoboTwin's optional ``CuroboPlanner`` when cuRobo is absent.

    RoboTwin's ``envs/robot/planner.py`` defines ``CuroboPlanner`` only inside a
    ``try`` that imports cuRobo (not in the lock: CUDA-built, unneeded for the
    mplib planner every benchmark task uses), yet ``envs/robot/robot.py`` imports
    the name unconditionally and ``isinstance``-checks it. Gripper interpolation
    matches upstream's; anything else raises.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
        num_step = 200
        return {
            "num_step": num_step,
            "per_step": (target_val - now_val) / num_step,
            "result": np.linspace(now_val, target_val, num_step),
        }

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("CuroboPlanner is unavailable: cuRobo is not installed")


class _PlannerFallbackFinder(importlib.abc.MetaPathFinder):
    """Give ``envs.robot.planner`` a ``CuroboPlanner`` after it executes, if it lacks one.

    In-process, so neither the managed nor an operator's RoboTwin checkout is edited.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "envs.robot.planner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_with_fallback(module: types.ModuleType) -> None:
            exec_module(module)
            if not hasattr(module, "CuroboPlanner"):
                module.CuroboPlanner = _CuroboPlannerUnavailable  # type: ignore[attr-defined]  # reason: adding the name the module failed to define

        loader.exec_module = exec_with_fallback  # type: ignore[method-assign]  # reason: per-spec wrap of this one module's loader
        return spec


def _install_curobo_planner_fallback() -> None:
    if not any(isinstance(f, _PlannerFallbackFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PlannerFallbackFinder())


# ── env construction + observation adaptation ─────────────────────────────────


# LeRobot's `robotwin` env exposes its three cameras under these native keys
# (head + per-wrist). The openral scene refers to them by the canonical sensor
# names declared in robots/aloha_agilex/robot.yaml (the canonical camera-slot
# names — typically ``top`` / ``wrist_left`` / ``wrist_right``); we re-key in fixed
# head→left→right order to whatever names the scene's ``cameras`` list
# provides.
_ENV_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def _arm_measured(
    entity: Any, arm_joints: Sequence[Any], gripper: Sequence[Any], scale: Sequence[float]
) -> tuple[list[float], list[float]]:
    """One arm's measured (qpos, qvel): its joints, then the gripper normalised to [0, 1].

    The gripper is RoboTwin's own encoding (``Robot.get_normal_real_gripper_val``:
    ``(q - scale[0]) / (scale[1] - scale[0])``, 0 = closed, 1 = open) on the measured
    base finger joint, where upstream reads the drive target.
    """
    qpos = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    qvel = np.asarray(entity.get_qvel(), dtype=np.float64).reshape(-1)
    active = list(entity.get_active_joints())
    idx = [active.index(j) for j in arm_joints]
    g = active.index(gripper[0][0])
    span = float(scale[1]) - float(scale[0])
    pos = [float(qpos[i]) for i in idx]
    pos.append(float(np.clip((qpos[g] - float(scale[0])) / span, 0.0, 1.0)))
    vel = [float(qvel[i]) for i in idx]
    vel.append(float(qvel[g]) / span)
    return pos, vel


def _measured_joint_state(task: Any) -> tuple[list[float], list[float]] | None:
    """Measured dual-arm joint state ``[left arm, left gripper, right arm, right gripper]``.

    ``agent_pos`` (RoboTwin's ``joint_action.vector``) is the joints' drive *targets*
    and the commanded gripper value (``Robot.get_left_arm_jointState``), and LeRobot
    zero-fills it when absent, so it never feeds ``/joint_states``. This reads the
    articulation's qpos/qvel, as ``Robot.get_left_arm_real_jointState`` does. None when
    the task has no robot yet (before the first reset) or no gripper.
    """
    robot = getattr(task, "robot", None)
    if robot is None or any(
        not g or g[0][0] is None for g in (robot.left_gripper, robot.right_gripper)
    ):
        return None
    lp, lv = _arm_measured(
        robot.left_entity, robot.left_arm_joints, robot.left_gripper, robot.left_gripper_scale
    )
    rp, rv = _arm_measured(
        robot.right_entity, robot.right_arm_joints, robot.right_gripper, robot.right_gripper_scale
    )
    return lp + rp, lv + rv


class _RoboTwinEnv:
    """Thin wrapper over LeRobot's ``robotwin`` gym env in the eval-layer shape.

    Builds a single gym env and adapts its observations to ``{"images", "state",
    "task"}``. LeRobot's robotwin obs is a dict with ``pixels`` (per-camera HWC
    uint8) + ``agent_pos``; we re-key the env's native cameras
    (``_ENV_CAMERA_NAMES``) to the openral scene camera names (the canonical
    camera-slot names) in order and expose ``agent_pos`` as ``state``.
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
        # lerobot 0.7 moved the env from ``lerobot.envs.robotwin``; the venv pins 0.7.0.
        try:
            from lerobot.sim.integrations.robotwin import (  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: lives only in the sidecar venv, untyped upstream
                RoboTwinEnv,
            )
        except ModuleNotFoundError as exc:
            # Only a missing lerobot module means a wrong lerobot; any other missing
            # module (torch, gymnasium, ...) propagates as itself.
            if not (exc.name or "").startswith("lerobot"):
                raise
            from importlib.metadata import PackageNotFoundError, version

            try:
                found = version("lerobot")
            except PackageNotFoundError:
                found = "not installed"
            raise ModuleNotFoundError(
                f"RoboTwin sidecar needs lerobot 0.7 (lerobot.sim.integrations.robotwin); "
                f"this venv ({sys.executable}) has lerobot {found}. Sync it to the lock "
                "(uv pip sync --require-hashes tools/sidecar_requirements/robotwin.lock), "
                "or unset OPENRAL_ROBOTWIN_SIDECAR_PYTHON and set "
                "OPENRAL_ROBOTWIN_AUTO_PROVISION=1 to use the managed venv."
            ) from exc

        return RoboTwinEnv(
            self._task,
            camera_names=_ENV_CAMERA_NAMES,
            observation_height=obs_height,
            observation_width=obs_width,
            episode_length=episode_length,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _unwrap_obs(self, obs: Any) -> dict[str, Any]:
        images: dict[str, npt.NDArray[np.uint8]] = {}
        pixels = obs.get("pixels") if isinstance(obs, dict) else None
        if isinstance(pixels, dict):
            for cam, frame in pixels.items():
                out_key = self._cam_remap.get(str(cam), str(cam))
                frame_np = np.asarray(frame, dtype=np.uint8)
                images[out_key] = self._resize_frame(frame_np)
        agent_pos = obs.get("agent_pos") if isinstance(obs, dict) else None
        state = (
            np.asarray(agent_pos, dtype=np.float32).reshape(-1)
            if agent_pos is not None
            else np.zeros((self._action_dim,), dtype=np.float32)
        )
        out: dict[str, Any] = {"images": images, "state": state, "task": self._instruction}
        measured = _measured_joint_state(getattr(self._env, "_env", None))
        if measured is not None:
            out["joint_positions"], out["joint_velocities"] = measured
        return out

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
        obs, reward, terminated, truncated, info = self._env.step(action_arr)
        sim_ns = _sapien_sim_time_ns(self._env)
        if sim_ns is None:
            self._fallback_time_ns += self._fallback_dt_ns
        success = bool(np.asarray(info.get(self._success_key, False)).any())
        return {
            "observation": self._unwrap_obs(obs),
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
            "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
            "info": {self._success_key: success},
            "sim_time_ns": sim_ns or self._fallback_time_ns,
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
        sys.path.insert(0, str(root / "envs"))
        _install_curobo_planner_fallback()
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
