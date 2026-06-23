r"""RLBench scene adapter — drives a CoppeliaSim/PyRep task through a sidecar.

ADR-0062. RLBench (James et al., 2020, arXiv:1909.12271) is the standard
benchmark for 3D/keyframe manipulation policies: 100 Franka tasks on CoppeliaSim
via PyRep. CoppeliaSim is the heaviest sim dependency in the tree — a proprietary
(free-EDU) ~py3.10 stack whose PyRep Cython extension links a specific
CoppeliaSim 4.1.0 install, and the released 3D policies pin the
``MohitShridhar/RLBench@peract`` fork. None of that can live in the openral
py3.12 workspace, so — exactly like the Isaac Sim backend
(:mod:`openral_sim.backends.isaac_sim`) — we run RLBench in its own venv and talk
to it over ZMQ REQ/REP framed by msgpack.

This module is the **openral side**: a thin :class:`SimRollout` that marshals
``reset`` / ``step`` / ``render`` / ``close`` to the sidecar
(``tools/rlbench_sidecar.py``) and unwraps the responses. The sidecar owns
CoppeliaSim, the RLBench task, and the keyframe motion executor.

Scene category: **single-robot (fixed)** — RLBench tasks are baked onto the
Franka Panda, so ``fixed_robot="franka_panda"`` and the CLI rejects ``--robot``.

The ``step`` action is an 8-D keyframe ``[x y z qx qy qz qw gripper_open]`` (the
contract the released 3D keyframe policies emit). The sidecar executes it via a
plan-and-retry mover and returns multi-view RGB-D point clouds + gripper pose +
instruction — the observation those policies consume.

Licensing (CLAUDE.md §1.9): CoppeliaSim is proprietary and NEVER vendored — it is
an externally-provisioned dependency the user installs (free EDU license). RLBench
and PyRep are open source. The sidecar venv is provisioned out of band (there is
no auto-install plan for a multi-GB proprietary simulator); ``_sidecar_python``
raises a typed :class:`ROSConfigError` carrying the exact provisioning commands
when it is absent.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

_RLBENCH_SCENE_ID = "rlbench"
_SIDECAR_PYTHON_ENV = "OPENRAL_RLBENCH_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_RLBENCH_SIDECAR_SCRIPT"
_COPPELIASIM_ROOT_ENV = "COPPELIASIM_ROOT"
_AUTO_SPAWN_ENV = "OPENRAL_RLBENCH_AUTO_SPAWN"

# Externally-provisioned defaults (see the ROSConfigError hint for the recipe).
_RLBENCH_VENV = Path.home() / ".cache" / "openral" / "rlbench-policy" / ".venv"
_COPPELIASIM_DEFAULT = (
    Path.home() / ".cache" / "openral" / "coppeliasim" / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
)
# Per-scene ZMQ port band (clear of well-known + ephemeral ranges), so two
# DIFFERENT rlbench tasks never collide on a shared default port.
_PORT_MIN = 21_000
_PORT_MAX = 21_999
# RLBench launch + first reset is fast (~7 s); a keyframe step runs a motion plan
# (seconds). Generous timeouts so a slow plan is not read as a dead sidecar.
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 300.0
_VIDEO_FRAMES_INFO_KEY = "_openral_video_frames"
_RENDER_CAMERA = "front"


def _scene_default_port(rlbench_task: str, variation: int) -> int:
    """Deterministic per-task ZMQ port, stable across processes (SHA-256, not ``hash``)."""
    key = f"rlbench|{rlbench_task}|{variation}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _opt_int(value: object, default: int) -> int:
    """Coerce a ``backend_options`` value (typed ``object``) to int, else default.

    Rejects ``bool`` (an ``int`` subclass we do not want silently accepted) and
    non-scalar / unparsable values — never raises.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@dataclass
class _RLBenchSidecar:
    """:class:`SimRollout` proxying an RLBench task over the sidecar."""

    scene: SceneSpec
    task: TaskSpec
    _client: SidecarClient
    _record_video: bool = False
    _last_image: NDArray[np.uint8] | None = None
    _sim_time_ns: int | None = None

    @property
    def action_dim(self) -> int:
        reply = self._client.call("ping")
        return int(self._client.require(reply, "action_dim"))

    def reset(self, seed: int | None = None) -> Observation:
        reply = self._client.call("reset", {"seed": seed})
        self._sim_time_ns = _wire_sim_time_ns(reply.get("sim_time_ns"))
        return self._wrap_obs(self._client.require(reply, "observation"))

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        reply = self._client.call(
            "step",
            {"action": action_np, "record_video": self._record_video},
        )
        info = dict(reply.get("info", {}))
        if self._record_video and "video_frames" in reply:
            info[_VIDEO_FRAMES_INFO_KEY] = [
                np.asarray(frame, dtype=np.uint8) for frame in reply["video_frames"]
            ]
        self._sim_time_ns = _wire_sim_time_ns(reply.get("sim_time_ns"))
        return StepResult(
            observation=self._wrap_obs(self._client.require(reply, "observation")),
            reward=float(self._client.require(reply, "reward")),
            terminated=bool(self._client.require(reply, "terminated")),
            truncated=bool(self._client.require(reply, "truncated")),
            info=info,
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
        images = {k: np.asarray(v, dtype=np.uint8) for k, v in raw.get("images", {}).items()}
        frame = self._render_frame(images)
        if frame is not None:
            self._last_image = frame
        clouds = {
            k: np.asarray(v, dtype=np.float32) for k, v in raw.get("point_clouds", {}).items()
        }
        obs: Observation = {
            "images": images,
            "point_clouds": clouds,
            "state": np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1),
            "task": raw.get("task", self.task.instruction),
        }
        gp = raw.get("gripper_pose")
        if gp is not None:
            obs["gripper_pose"] = np.asarray(gp, dtype=np.float32).reshape(-1)
        if "gripper_open" in raw:
            obs["gripper_open"] = float(raw["gripper_open"])
        return obs

    def _render_frame(self, images: dict[str, NDArray[np.uint8]]) -> NDArray[np.uint8] | None:
        if _RENDER_CAMERA in images:
            return images[_RENDER_CAMERA]
        for frame in images.values():
            return frame
        return None


def _wire_sim_time_ns(value: object) -> int | None:
    """Coerce the optional sidecar sim-time wire value into nanoseconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float)):
        return None
    return int(value)


def _sidecar_python() -> Path:
    """Resolve the rlbench sidecar venv interpreter, or raise with the install hint."""
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    default = _RLBENCH_VENV / "bin" / "python"
    if default.is_file():
        return default
    raise ROSConfigError(
        "RLBench sidecar venv not found. RLBench needs CoppeliaSim (proprietary, "
        "free EDU license; NEVER vendored — ADR-0062) + PyRep + the peract RLBench "
        "fork in a py3.10 venv. Provision it (one-time) and point "
        f"{_SIDECAR_PYTHON_ENV} at its python:\n"
        "  # 1) CoppeliaSim 4.1.0 (Ubuntu20_04 build) -> set COPPELIASIM_ROOT\n"
        "  # 2) uv venv --python 3.10 ~/.cache/openral/rlbench-policy/.venv\n"
        "  # 3) uv pip install (with COPPELIASIM_ROOT set): PyRep (stepjam) +\n"
        "  #    RLBench (MohitShridhar@peract, editable) + gymnasium==1.0.0a2\n"
        "See docs/adr/0062-rlbench-benchmark-backend.md for the full recipe."
    )


def _coppeliasim_root() -> Path:
    override = os.environ.get(_COPPELIASIM_ROOT_ENV)
    root = Path(override).expanduser() if override else _COPPELIASIM_DEFAULT
    if not root.is_dir():
        raise ROSConfigError(
            f"CoppeliaSim root not found at {root}. Set {_COPPELIASIM_ROOT_ENV} to the "
            "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 directory (ADR-0062)."
        )
    return root


def _locate_sidecar_script() -> Path:
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "rlbench_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/rlbench_sidecar.py upwards from {here}. Set "
        f"{_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


@SCENES.register(_RLBENCH_SCENE_ID, fixed_robot="franka_panda")
def _build_rlbench_scene(env_cfg: SimEnvironment) -> _RLBenchSidecar:
    """Build an RLBench task behind the out-of-process CoppeliaSim sidecar.

    ``scene.backend_options`` keys: ``rlbench_task`` (required task file stem,
    e.g. ``open_drawer``), ``variation`` (default 0), ``port`` (default a
    per-task hash), ``max_tries`` (mover retries, default 10).
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("rlbench_client")
    try:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in rlbench group
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in rlbench group
    except ImportError as exc:  # pragma: no cover — runtime-error path
        raise ROSConfigError(
            "rlbench backend needs pyzmq + msgpack on the openral venv: "
            "uv sync --all-packages --group rlbench --inexact"
        ) from exc

    opts = env_cfg.scene.backend_options
    rlbench_task = str(opts.get("rlbench_task") or "")
    if not rlbench_task:
        raise ROSConfigError(
            "rlbench scene requires backend_options.rlbench_task (e.g. 'open_drawer')."
        )
    variation = _opt_int(opts.get("variation"), 0)
    host = str(opts.get("host", "127.0.0.1"))
    port = _opt_int(opts.get("port"), _scene_default_port(rlbench_task, variation))
    max_tries = _opt_int(opts.get("max_tries"), 10)
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    root = _coppeliasim_root()
    ld = f"{root}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    display = os.environ.get("DISPLAY", ":1")
    # Wrap with `env` so the sidecar's child process gets the CoppeliaSim vars
    # (SidecarClient._spawn inherits os.environ minus PYTHONPATH; an explicit
    # `env` prefix is clearer than mutating the parent's global environment).
    launch_argv = [
        "env",
        f"{_COPPELIASIM_ROOT_ENV}={root}",
        f"LD_LIBRARY_PATH={ld}",
        f"QT_QPA_PLATFORM_PLUGIN_PATH={root}",
        f"DISPLAY={display}",
        str(_sidecar_python()),
        str(_locate_sidecar_script()),
        "--task",
        env_cfg.task.id,
        "--rlbench-task",
        rlbench_task,
        "--variation",
        str(variation),
        "--instruction",
        env_cfg.task.instruction,
        "--obs-height",
        str(env_cfg.scene.observation_height),
        "--obs-width",
        str(env_cfg.scene.observation_width),
        "--max-steps",
        str(env_cfg.task.max_steps if env_cfg.task.max_steps is not None else 25),
        "--success-key",
        env_cfg.task.success_key or "is_success",
        "--max-tries",
        str(max_tries),
        "--host",
        host,
        "--port",
        str(port),
        "--headless",
    ]

    client = SidecarClient(
        name="rlbench",
        host=host,
        port=port,
        timeout_ms=_DEFAULT_TIMEOUT_MS,
        boot_timeout_s=_DEFAULT_BOOT_TIMEOUT_S,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={"task": env_cfg.task.id, "layout": "rlbench"},
    )
    client.connect()
    return _RLBenchSidecar(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _client=client,
        _record_video=env_cfg.record_video,
    )
