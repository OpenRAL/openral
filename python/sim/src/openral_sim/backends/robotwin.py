r"""RoboTwin 2.0 scene adapter — drives a SAPIEN dual-arm env through a sidecar venv.

ADR-0061. RoboTwin 2.0 (Chen et al., arXiv 2506.18088, MIT) is a large-scale
**bimanual** benchmark: 50 dual-arm tasks on the SAPIEN physics engine, evaluated
on the aloha-agilex embodiment (14-DoF, 7 per arm; action 14-D joint-space).

RoboTwin's stack (SAPIEN, CuRobo, mplib, pytorch3d) pins **Python 3.10 + CUDA 12.1**
and an incompatible torch build — it cannot share the openral ``>=3.12`` venv. So,
exactly like the Isaac Sim scene backend (:mod:`openral_sim.backends.isaac_sim`) and
the RLDX-1 policy sidecar (:mod:`openral_sim.policies.rldx`), we run it in its own
sidecar venv and talk to it over ZMQ REQ/REP framed by msgpack
(:class:`openral_sim.sidecar.SidecarClient`).

This module is the **openral side**: a thin :class:`SimRollout` that marshals
``reset`` / ``step`` / ``render`` / ``close`` to the sidecar
(``tools/robotwin_sidecar.py``) and unwraps the responses. The sidecar owns the
SAPIEN simulation, the aloha-agilex robot, and the three RoboTwin cameras — it wraps
LeRobot's native ``robotwin`` gym env (``lerobot-eval --env.type=robotwin``), the
authoritative way to drive the SAPIEN tasks.

Scene category: **single-robot (fixed)** — registered with
``fixed_robot="aloha_agilex"``. The SAPIEN env bakes in the aloha-agilex bimanual
robot; the CLI rejects a mismatched ``--robot``. ``robots/aloha_agilex/robot.yaml``
carries the 14-D action/state contract for the eval-layer compatibility gate (the
manifest ships no URDF/MJCF — the sidecar's SAPIEN model is authoritative).

Sidecar python resolution
-------------------------
The launcher runs under the **robotwin** venv, not this one. We resolve its
interpreter from ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` (absolute path to the
sidecar venv's ``python``), else a cache default, else (opt-in
``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) we provision it. The provisioning installs
LeRobot + the RoboTwin SAPIEN stack + downloads assets (a multi-GB, ~20-minute,
CUDA-12.1 / Linux-only job); without opt-in we raise a typed :class:`ROSConfigError`
carrying the exact manual recipe.

Licensing (CLAUDE.md §1.9): RoboTwin (MIT), SAPIEN (MIT), LeRobot (Apache-2.0) are
all permissive but the SAPIEN+RoboTwin stack is large and CUDA-pinned, so it is an
externally-provisioned sidecar venv — never vendored into the repo. The openral-side
wire is just pyzmq + msgpack (the ``robotwin`` dependency-group).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_ROBOTWIN_SCENE_ID = "robotwin"
_ROBOTWIN_ROBOT_ID = "aloha_agilex"

_AUTO_SPAWN_ENV = "OPENRAL_ROBOTWIN_AUTO_SPAWN"
_SIDECAR_PYTHON_ENV = "OPENRAL_ROBOTWIN_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_ROBOTWIN_SIDECAR_SCRIPT"
_ROBOTWIN_ROOT_ENV = "OPENRAL_ROBOTWIN_ROOT"
_AUTO_PROVISION_ENV = "OPENRAL_ROBOTWIN_AUTO_PROVISION"

# Default sidecar venv location + pinned install. RoboTwin's SAPIEN stack is large,
# CUDA-12.1 / Linux-only and pulls CuRobo / mplib / pytorch3d + multi-GB assets — so
# (like the Isaac sidecar) we do NOT auto-provision by default: provisioning runs
# only when the operator opts in with OPENRAL_ROBOTWIN_AUTO_PROVISION=1, and
# OPENRAL_ROBOTWIN_SIDECAR_PYTHON always overrides. Pins mirror the manual recipe in
# the ROSConfigError hint below; bump both together.
_ROBOTWIN_SIDECAR_HOME = Path.home() / ".cache" / "openral" / "robotwin-sidecar"
_ROBOTWIN_PYTHON = "3.10"
# LeRobot's `robotwin` env lives on `main` (NOT the PyPI 0.5.1 release — that has no
# `robotwin` extra), so auto-provision installs lerobot from git + SAPIEN + the
# openral-side wire. This is the pip-installable core ONLY: the RoboTwin checkout
# (RoboTwin-Platform/RoboTwin, passed via OPENRAL_ROBOTWIN_ROOT) and its multi-GB
# **assets** (`script/_download_assets.sh`) are a separate manual step — see the
# ROSConfigError recipe in `_sidecar_python` and ADR-0061. Auto-provision is a
# best-effort head start, not a complete install.
_ROBOTWIN_DEPS = (
    "lerobot @ git+https://github.com/huggingface/lerobot.git",
    "sapien>=3.0.0b1",
    "pyzmq",
    "msgpack",
)

_DEFAULT_HOST = "127.0.0.1"
# Per-scene default ports in 20000–39999 (clear of well-known + ephemeral ranges),
# the SAME band the Isaac / RLDX sidecars use. One sidecar serves one scene, so two
# DIFFERENT scenes must NOT share a port (a lingering sidecar from scene A would be
# silently adopted by scene B). ``_scene_default_port`` derives a stable per-scene
# port; an explicit ``backend_options.port`` still wins.
_SIDECAR_PORT_MIN = 20_000
_SIDECAR_PORT_MAX = 40_000

# REQ recv timeout for a steady-state step. SAPIEN + ray-traced rendering of one
# frame is slower than MuJoCo (closer to Isaac), so keep it generous.
_DEFAULT_TIMEOUT_MS = 120_000
# First boot pays the SAPIEN engine + scene + asset load.
_DEFAULT_BOOT_TIMEOUT_S = 600.0
# Truncation cap when the scene comes from a taskless DeployScene (deploy sim).
_DEFAULT_MAX_STEPS = 1_000_000


def _scene_default_port(task_id: str, robot_id: str) -> int:
    """Deterministic per-scene ZMQ port, stable across processes.

    Mirrors ``isaac_sim._scene_default_port``. Uses a ``hashlib`` digest (NOT the
    builtin ``hash``, which is salted per process via ``PYTHONHASHSEED``) so the port
    the sidecar binds in its spawn process matches the one a later client process
    probes for the same scene. Distinct scenes map to distinct ports with
    overwhelming probability; any residual collision is caught loudly by the
    identity-checked ping handshake, never served as wrong data. SHA-256 is used only
    to spread identities, never as a security boundary.
    """
    import hashlib

    key = f"{task_id}|{robot_id}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _SIDECAR_PORT_MIN + (digest % (_SIDECAR_PORT_MAX - _SIDECAR_PORT_MIN))


def _robotwin_task_name(task_id: str) -> str:
    """Extract the upstream RoboTwin task name from ``robotwin/<task>``.

    Task ids are namespaced ``robotwin/<snake_case>`` (e.g. ``robotwin/lift_pot``);
    the sidecar's LeRobot env wants the bare upstream name (``lift_pot``). A bare id
    with no ``/`` is passed through unchanged.
    """
    return task_id.split("/", 1)[1] if "/" in task_id else task_id


# ── SimRollout adapter ────────────────────────────────────────────────────────


def _coerce_sim_time_ns(value: object) -> int | None:
    """Coerce an optional wire ``sim_time_ns`` (int / float / None) to ``int | None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


@dataclass
class _RoboTwinSimSidecar:
    """:class:`SimRollout` that proxies a RoboTwin SAPIEN env over the sidecar.

    Observations come back from the sidecar already in the eval-layer shape
    (``images`` dict of HWC uint8 keyed by the RoboTwin camera names, ``state`` 1-D
    float32 of the 14 joint positions, ``task`` str); we re-wrap into a plain dict
    and cache the last RGB frame for ``render``.
    """

    scene: SceneSpec
    task: TaskSpec
    _client: SidecarClient
    _last_image: NDArray[np.uint8] | None = None
    _action_dim: int | None = None
    _last_sim_time_ns: int | None = None

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts — queried from the sidecar ping (14)."""
        if self._action_dim is None:
            reply = self._client.call("ping")
            self._action_dim = int(self._client.require(reply, "action_dim"))
        return self._action_dim

    def reset(self, seed: int | None = None) -> Observation:
        reply = self._client.call("reset", {"seed": seed})
        self._last_sim_time_ns = _coerce_sim_time_ns(reply.get("sim_time_ns"))
        return self._wrap_obs(self._client.require(reply, "observation"))

    def step(self, action: NDArray[np.float32]) -> StepResult:
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        reply = self._client.call("step", {"action": action_np})
        self._last_sim_time_ns = _coerce_sim_time_ns(reply.get("sim_time_ns"))
        return StepResult(
            observation=self._wrap_obs(self._client.require(reply, "observation")),
            reward=float(self._client.require(reply, "reward")),
            terminated=bool(self._client.require(reply, "terminated")),
            truncated=bool(self._client.require(reply, "truncated")),
            info=dict(reply.get("info", {})),
        )

    def sim_time_ns(self) -> int | None:
        """Elapsed simulation time in ns from the last sidecar reply, or ``None``."""
        return self._last_sim_time_ns

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()

    def _wrap_obs(self, raw: dict[str, Any]) -> Observation:
        images_raw = raw.get("images", {})
        images: dict[str, NDArray[np.uint8]] = {
            k: np.asarray(v, dtype=np.uint8) for k, v in images_raw.items()
        }
        if images:
            # Prefer the head camera as the cached render frame when present.
            head = images.get("head_camera")
            self._last_image = head if head is not None else next(iter(images.values()))
        else:
            h = self.scene.observation_height
            w = self.scene.observation_width
            images = {"head_camera": np.zeros((h, w, 3), dtype=np.uint8)}
        state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1)
        obs: Observation = {
            "images": images,
            "state": state,
            "task": raw.get("task", self.task.instruction),
        }
        return obs


# ── factory ───────────────────────────────────────────────────────────────────


_Num = TypeVar("_Num", int, float)


def _opt_num(
    opts: dict[str, object], key: str, default: _Num, cast: Callable[[int | float | str], _Num]
) -> _Num:
    """Coerce a ``backend_options`` value (typed ``object``) via ``cast``, else default."""
    value = opts.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def _provision_robotwin_venv() -> Path:
    """Create the robotwin sidecar venv from the pinned LeRobot + SAPIEN install.

    Opt-in (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) because it is a multi-GB,
    CUDA-12.1 / Linux-only download. Uses the shared :func:`ensure_pip_venv`
    provisioning order so it reuses an existing venv + sentinel. Returns the venv
    python (``<home>/.venv/bin/python``).

    NOTE: this installs only the pip-installable core — LeRobot from git ``main``
    (the PyPI 0.5.1 release has no ``robotwin`` env) + SAPIEN + the wire. The
    RoboTwin **task package** (``RoboTwin-Platform/RoboTwin`` on PYTHONPATH) and its
    multi-GB **assets** (``script/_download_assets.sh``) are NOT installed here; the
    ROSConfigError hint in :func:`_sidecar_python` documents the full manual recipe.
    On a host where this auto path is insufficient, provision manually and point
    ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` at the resulting py3.10 interpreter.
    """

    def _install(uv: str, py: Path) -> None:
        env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "900")}
        run_cmd(
            "robotwin-sidecar",
            [uv, "pip", "install", "--python", str(py), *_ROBOTWIN_DEPS],
            env=env,
        )

    return ensure_pip_venv(
        label="robotwin-sidecar",
        home=_ROBOTWIN_SIDECAR_HOME,
        python=_ROBOTWIN_PYTHON,
        install=_install,
    )


def _sidecar_python() -> Path:
    """Resolve the robotwin sidecar venv interpreter, or raise with the install hint.

    Resolution order: ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` override → an existing
    default venv → opt-in auto-provision (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) → a
    typed error carrying the exact manual commands.
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    default = _ROBOTWIN_SIDECAR_HOME / ".venv" / "bin" / "python"
    if default.is_file():
        return default
    if os.environ.get(_AUTO_PROVISION_ENV, "").strip() not in ("", "0", "false", "False"):
        return _provision_robotwin_venv()
    raise ROSConfigError(
        "RoboTwin sidecar venv not found. It is an externally-provisioned dependency "
        "(SAPIEN + RoboTwin 2.0, Python 3.10, CUDA 12.1, Linux-only, multi-GB; "
        "ADR-0061). Set "
        f"{_AUTO_PROVISION_ENV}=1 to auto-provision the LeRobot+SAPIEN venv (a "
        "multi-GB download), or provision it manually and point "
        f"{_SIDECAR_PYTHON_ENV} at its py3.10 python:\n"
        "  conda create -n robotwin python=3.10 -y && conda activate robotwin\n"
        "  git clone https://github.com/huggingface/lerobot.git && pip install -e ./lerobot\n"
        "  git clone https://github.com/RoboTwin-Platform/RoboTwin.git\n"
        "  cd RoboTwin && bash script/_install.sh && bash script/_download_assets.sh\n"
        f"  export {_ROBOTWIN_ROOT_ENV}=$(pwd)  # checkout + assets path for the sidecar\n"
        "  export OPENRAL_ROBOTWIN_SIDECAR_PYTHON=$(which python)"
    )


def _locate_sidecar_script() -> Path:
    """Find ``tools/robotwin_sidecar.py`` (env override, else walk up from here)."""
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "robotwin_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/robotwin_sidecar.py upwards from {here}. Set "
        f"{_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


def _robotwin_root() -> Path:
    """Resolve the RoboTwin checkout root the sidecar must run from.

    RoboTwin imports use process-relative ``assets/...`` paths, and
    :class:`SidecarClient` deliberately strips parent ``PYTHONPATH`` for ABI safety.
    Pass the checkout root explicitly so the sidecar can chdir and add it to
    ``sys.path`` before LeRobot imports the task package.
    """
    override = os.environ.get(_ROBOTWIN_ROOT_ENV)
    root = Path(override).expanduser() if override else _ROBOTWIN_SIDECAR_HOME / "RoboTwin"
    if not root.is_dir():
        raise ROSConfigError(
            "RoboTwin checkout not found. Clone https://github.com/RoboTwin-Platform/RoboTwin "
            f"and set {_ROBOTWIN_ROOT_ENV} to its path."
        )
    assets = root / "assets" / "objects" / "objaverse" / "list.json"
    if not assets.is_file():
        raise ROSConfigError(
            f"RoboTwin assets not found at {assets}. Run script/_download_assets.sh in "
            f"the RoboTwin checkout, then set {_ROBOTWIN_ROOT_ENV}={root}."
        )
    return root.resolve()


@SCENES.register(_ROBOTWIN_SCENE_ID, fixed_robot=_ROBOTWIN_ROBOT_ID)
def _build_robotwin_scene(env_cfg: SimEnvironment) -> _RoboTwinSimSidecar:
    """Build a RoboTwin 2.0 SAPIEN scene behind the out-of-process sidecar.

    Lazy-imports pyzmq/msgpack (the openral-side wire) via the ``robotwin_client``
    install plan, resolves the sidecar interpreter + script, and connects (auto-
    spawning the SAPIEN process on first use).
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("robotwin_client")
    try:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in robotwin group
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in robotwin group
    except ImportError as exc:  # pragma: no cover — runtime-error path
        raise ROSConfigError(
            "robotwin backend needs pyzmq + msgpack on the openral venv: "
            "uv sync --all-packages --group robotwin --inexact"
        ) from exc

    opts = env_cfg.scene.backend_options
    host = str(opts.get("host", _DEFAULT_HOST))
    timeout_ms = _opt_num(opts, "timeout_ms", _DEFAULT_TIMEOUT_MS, int)
    boot_timeout_s = _opt_num(opts, "boot_timeout_s", _DEFAULT_BOOT_TIMEOUT_S, float)
    task_name = _robotwin_task_name(env_cfg.task.id)
    port = _opt_num(opts, "port", _scene_default_port(env_cfg.task.id, _ROBOTWIN_ROBOT_ID), int)
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    cameras = env_cfg.scene.cameras or ["head_camera", "left_camera", "right_camera"]

    launch_argv = [
        str(_sidecar_python()),
        str(_locate_sidecar_script()),
        "--task",
        task_name,
        "--instruction",
        env_cfg.task.instruction,
        "--obs-height",
        str(env_cfg.scene.observation_height),
        "--obs-width",
        str(env_cfg.scene.observation_width),
        "--cameras",
        ",".join(cameras),
        "--episode-length",
        # A DeployScene (openral deploy sim) has no task, so max_steps is None — fall
        # back to a large cap so the continuously-driven deploy env never truncates.
        str(env_cfg.task.max_steps if env_cfg.task.max_steps is not None else _DEFAULT_MAX_STEPS),
        "--success-key",
        env_cfg.task.success_key or "is_success",
        "--robotwin-root",
        str(_robotwin_root()),
        "--host",
        host,
        "--port",
        str(port),
    ]

    client = SidecarClient(
        name="robotwin",
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        boot_timeout_s=boot_timeout_s,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        # Reject (loudly) an already-running sidecar on this port that serves a
        # different scene, instead of silently adopting its wrong task.
        expected_identity={"env": "robotwin", "task": task_name},
    )
    client.connect()
    return _RoboTwinSimSidecar(scene=env_cfg.scene, task=env_cfg.task, _client=client)
