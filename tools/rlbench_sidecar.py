r"""RLBench scene sidecar — drives a CoppeliaSim/PyRep RLBench task over ZMQ.

ADR-0062. RLBench (James et al., 2020, arXiv:1909.12271) runs on CoppeliaSim +
PyRep, a heavy, externally-provisioned, ~py3.10 stack that cannot be loaded into
the openral py3.12 workspace (PyRep builds a Cython extension against a specific
CoppeliaSim 4.1.0 install; the released 3D keyframe policies pin the
``MohitShridhar/RLBench@peract`` fork). So — exactly like the Isaac Sim scene
backend (:mod:`openral_sim.backends.isaac_sim`) — we run RLBench in its own venv
and talk to it over ZMQ REQ/REP framed by msgpack.

This file is the **sidecar side** (no openral import — it runs under the
externally-provisioned ``rlbench`` venv). It owns CoppeliaSim, the RLBench task,
and the keyframe motion executor. The openral side is
:mod:`openral_sim.backends.rlbench`.

Wire protocol (mirrors ``tools/isaac_sidecar.py``)::

    ping  -> {"ok": True, "action_dim": 8, "task": <task_str>, "layout": "rlbench"}
    reset -> {"observation": {...}}
    step  -> {"observation": {...}, "reward", "terminated", "truncated", "info"}

Observation dict (eval-layer shape + RLBench multi-view extras the 3D keyframe
policies consume)::

    images:        {<cam>: HWC uint8 RGB}            # left_shoulder/right_shoulder/wrist/front
    point_clouds:  {<cam>: HWC float32 world-frame}  # one per camera
    gripper_pose:  (7,) float32  [x y z qx qy qz qw]
    gripper_open:  float                              # 1.0 open, 0.0 closed
    state:         (8,) float32  [gripper_pose(7), gripper_open(1)]
    task:          str instruction

Action: an 8-D keyframe ``[x y z qx qy qz qw gripper_open]`` (world frame,
``wxyz``→ executed as the RLBench convention). The sidecar appends the peract
fork's ``ignore_collisions`` channel and executes via a retry-mover that re-tries
the planned motion until the end-effector reaches the target pose (<5 mm) — the
same closed-loop execution the upstream 3D-policy evaluators use.

License (CLAUDE.md §1.9): CoppeliaSim is proprietary (free EDU license) and is
NEVER vendored — it is an externally-provisioned dependency the user installs
(see :mod:`openral_sim.backends.rlbench` for the provisioning hint). RLBench and
PyRep are open source.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Camera set the PerAct/3D-policy checkpoints were trained on (NOT overhead).
_CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")
_RENDER_CAMERA = "front"
# peract-fork action layout: [pose(7), gripper(1), ignore_collisions(1)] = 9-D.
# The openral side sends 8-D (pose+gripper); we append ignore_collisions=1.
_POSE_REACH_TOL_M = 5e-3

# RLBench / PyRep motion-planner path-finding failures. EndEffectorPoseViaPlanning
# is sampling-based and raises one of these when a predicted keypose is
# unreachable (no path / IK could not solve / no collision-free config path). The
# reference 3D Diffuser Actor evaluator counts such a step as a *failed episode*
# and continues, so a single stochastic miss does not abort a whole multi-episode
# benchmark run. Matched by class name (incl. base classes) so the openral side
# never needs to import the externally-provisioned rlbench/pyrep venv.
_PLANNER_FAILURE_EXC_NAMES = frozenset(
    {
        "InvalidActionError",  # rlbench.backend.exceptions — planner found no path
        "IKError",  # pyrep.errors — IK could not solve for the target pose
        "ConfigurationPathError",  # pyrep.errors — no collision-free config path
    }
)


def _is_planner_path_failure(exc: BaseException) -> bool:
    """True when *exc* is a RLBench/PyRep motion-planner path-finding failure.

    See :data:`_PLANNER_FAILURE_EXC_NAMES`. Walks the exception's MRO so
    subclasses of the known planner errors are matched too.
    """
    return any(t.__name__ in _PLANNER_FAILURE_EXC_NAMES for t in type(exc).__mro__)


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
    p = argparse.ArgumentParser(description="OpenRAL RLBench scene sidecar")
    p.add_argument("--task", required=True, help="task id, e.g. rlbench/open_drawer")
    p.add_argument(
        "--rlbench-task",
        required=True,
        help="RLBench task file stem, e.g. open_drawer / meat_off_grill / close_jar",
    )
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--instruction", default="")
    p.add_argument("--obs-height", type=int, default=256)
    p.add_argument("--obs-width", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--success-key", default="is_success")
    p.add_argument("--max-tries", type=int, default=10, help="retry-mover motion attempts")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=21100)
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


class _RLBenchScene:
    """Holds the RLBench env + keyframe executor; speaks the openral obs shape."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._cameras = _CAMERAS
        self._env: Any = None
        self._task: Any = None
        self._success_key = args.success_key
        self._arm_action_mode: Any = None
        self._motion_frames: list[NDArray[np.uint8]] = []
        # Last wrapped observation (from reset / a successful step). Returned
        # verbatim when a planner path-failure ends an episode — there is no
        # fresh obs on a failed mover step.
        self._last_wrapped_obs: dict[str, Any] = {}

    @property
    def action_dim(self) -> int:
        # openral-facing keyframe action: pose(7) + gripper(1).
        return 8

    def build(self) -> None:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig

        obs_config = ObservationConfig()
        obs_config.set_all(False)
        # Enable exactly the four PerAct cameras' RGB + point clouds.
        for cam in self._cameras:
            cam_cfg = getattr(obs_config, f"{cam}_camera")
            cam_cfg.rgb = True
            cam_cfg.point_cloud = True
            cam_cfg.depth = False
            cam_cfg.mask = False
            cam_cfg.image_size = (self._args.obs_height, self._args.obs_width)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.joint_positions = True

        self._arm_action_mode = EndEffectorPoseViaPlanning(collision_checking=False)
        action_mode = MoveArmThenGripper(
            arm_action_mode=self._arm_action_mode,
            gripper_action_mode=Discrete(),
        )
        # An empty dataset root is fine: we only run live tasks (no recorded demos).
        self._env = Environment(
            action_mode, "/tmp/openral_rlbench_empty", obs_config, headless=True
        )
        self._env.launch()
        self._task = self._env.get_task(_task_class(self._args.rlbench_task))
        self._task.set_variation(self._args.variation)
        print(
            f"[rlbench_sidecar] launched {self._args.rlbench_task} v{self._args.variation}",
            flush=True,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            np.random.seed(int(seed))
        self._task.set_variation(self._args.variation)
        descriptions, obs = self._task.reset()
        self._instruction = descriptions[0] if descriptions else self._args.instruction
        self._last_wrapped_obs = self._wrap_obs(obs)
        return self._last_wrapped_obs

    def step(self, action: NDArray[np.float32], *, record_video: bool = False) -> dict[str, Any]:
        target = np.asarray(action, dtype=np.float64).reshape(-1)[:8]
        act9 = np.ones(9, dtype=np.float64)  # peract fork: pose7 + gripper1 + ignore_collisions1
        act9[:8] = target
        act9[8] = 1.0  # ignore collisions during planning (eval default)
        obs = None
        reward = 0.0
        terminate = False
        self._motion_frames = []
        if self._arm_action_mode is not None:
            self._arm_action_mode.set_callable_each_step(
                self._record_motion_observation if record_video else None
            )
        planner_failed = False
        try:
            for _ in range(int(self._args.max_tries)):
                obs, reward, terminate = self._task.step(act9)
                reached = float(np.linalg.norm(target[:3] - obs.gripper_pose[:3]))
                if reached < _POSE_REACH_TOL_M or reward == 1:
                    break
        except BaseException as exc:
            # A genuine fault must still surface; only the sampling-based
            # planner's path-finding misses are downgraded to a failed episode.
            if not _is_planner_path_failure(exc):
                raise
            planner_failed = True
            print(
                f"[rlbench_sidecar] planner path-failure ({type(exc).__name__}: {exc}) "
                "-> ending episode as a failure (not crashing the run)",
                flush=True,
            )
        finally:
            if self._arm_action_mode is not None:
                self._arm_action_mode.set_callable_each_step(None)

        if planner_failed:
            # EndEffectorPoseViaPlanning could not reach the predicted keypose
            # (unreachable target / collision). End the episode as a failure —
            # matching the reference 3DDA evaluator — so a single stochastic
            # planner miss can't abort a whole multi-episode benchmark run
            # (ADR-0062). No fresh obs exists; return the last wrapped one.
            reply = {
                "observation": self._last_wrapped_obs,
                "reward": 0.0,
                "terminated": True,
                "truncated": False,
                "info": {self._success_key: False},
            }
            if record_video:
                reply["video_frames"] = self._motion_frames
            return reply

        success = bool(reward == 1)
        final_frame = self._frame_from_obs(obs)
        if record_video and final_frame is not None:
            self._motion_frames.append(final_frame)
        self._last_wrapped_obs = self._wrap_obs(obs)
        reply = {
            "observation": self._last_wrapped_obs,
            "reward": float(reward),
            "terminated": bool(terminate or success),
            "truncated": False,
            "info": {self._success_key: success},
        }
        if record_video:
            reply["video_frames"] = self._motion_frames
        return reply

    def _record_motion_observation(self, obs: Any) -> None:
        frame = self._frame_from_obs(obs)
        if frame is not None:
            self._motion_frames.append(frame)

    def _wrap_obs(self, obs: Any) -> dict[str, Any]:
        images: dict[str, NDArray[np.uint8]] = {}
        clouds: dict[str, NDArray[np.float32]] = {}
        for cam in self._cameras:
            rgb = getattr(obs, f"{cam}_rgb", None)
            if rgb is not None:
                images[cam] = np.asarray(rgb, dtype=np.uint8)
            pcd = getattr(obs, f"{cam}_point_cloud", None)
            if pcd is not None:
                clouds[cam] = np.asarray(pcd, dtype=np.float32)
        gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
        gripper_open = float(obs.gripper_open)
        state = np.concatenate([gripper_pose, [gripper_open]]).astype(np.float32)
        frame = self._frame_from_images(images)
        if frame is not None:
            self._last_image = frame
        return {
            "images": images,
            "point_clouds": clouds,
            "gripper_pose": gripper_pose,
            "gripper_open": gripper_open,
            "state": state,
            "task": getattr(self, "_instruction", self._args.instruction),
        }

    def _frame_from_obs(self, obs: Any) -> NDArray[np.uint8] | None:
        if obs is None:
            return None
        preferred = getattr(obs, f"{_RENDER_CAMERA}_rgb", None)
        if preferred is not None:
            return np.asarray(preferred, dtype=np.uint8)
        for cam in self._cameras:
            rgb = getattr(obs, f"{cam}_rgb", None)
            if rgb is not None:
                return np.asarray(rgb, dtype=np.uint8)
        return None

    def _frame_from_images(self, images: dict[str, NDArray[np.uint8]]) -> NDArray[np.uint8] | None:
        if _RENDER_CAMERA in images:
            return np.asarray(images[_RENDER_CAMERA], dtype=np.uint8)
        for frame in images.values():
            return np.asarray(frame, dtype=np.uint8)
        return None

    def render(self) -> NDArray[np.uint8] | None:
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.shutdown()


def _task_class(task_file: str) -> Any:
    """Resolve an RLBench task file stem (``open_drawer``) to its task class."""
    import importlib

    mod = importlib.import_module(f"rlbench.tasks.{task_file}")
    cls_name = "".join(w.capitalize() for w in task_file.split("_"))
    return getattr(mod, cls_name)


def _serve(scene: _RLBenchScene, *, host: str, port: int, task: str) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[rlbench_sidecar] serving on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {
                    "ok": True,
                    "action_dim": scene.action_dim,
                    "task": task,
                    "layout": "rlbench",
                }
            elif endpoint == "reset":
                reply = {"observation": scene.reset(seed=data.get("seed"))}
            elif endpoint == "step":
                reply = scene.step(
                    np.asarray(data["action"], dtype=np.float32),
                    record_video=bool(data.get("record_video")),
                )
            elif endpoint == "render":
                reply = {"frame": scene.render()}
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
    scene.close()
    return 0


def main(argv: list[str]) -> int:
    import os

    os.makedirs("/tmp/openral_rlbench_empty", exist_ok=True)
    args = _parse_args(argv)
    scene = _RLBenchScene(args)
    scene.build()
    return _serve(scene, host=args.host, port=args.port, task=args.task)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
