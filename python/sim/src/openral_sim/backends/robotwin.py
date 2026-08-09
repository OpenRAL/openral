r"""RoboTwin 2.0 scene adapter — drives a SAPIEN dual-arm env through a sidecar venv.

RoboTwin 2.0 (Chen et al., arXiv 2506.18088, MIT) is a large-scale
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
LeRobot 0.6 + the RoboTwin SAPIEN stack + downloads assets (a multi-GB,
Linux-only job); without opt-in we raise a typed :class:`ROSConfigError`
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

from openral_sim._sidecar_common import ensure_pip_venv, ensure_source, run_cmd
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_ROBOTWIN_SCENE_ID = "robotwin"
_ROBOTWIN_ROBOT_ID = "aloha_agilex"
_DEPLOY_NOOP_SUFFIX = "/_hal_deploy_noop"
_DEFAULT_DEPLOY_TASK = "lift_pot"

_AUTO_SPAWN_ENV = "OPENRAL_ROBOTWIN_AUTO_SPAWN"
_SIDECAR_PYTHON_ENV = "OPENRAL_ROBOTWIN_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_ROBOTWIN_SIDECAR_SCRIPT"
_ROBOTWIN_ROOT_ENV = "OPENRAL_ROBOTWIN_ROOT"
_AUTO_PROVISION_ENV = "OPENRAL_ROBOTWIN_AUTO_PROVISION"

# Default sidecar venv location + pinned install. RoboTwin's SAPIEN stack is large,
# Linux-only and pulls SAPIEN / mplib + multi-GB assets — so
# (like the Isaac sidecar) we do NOT auto-provision by default: provisioning runs
# only when the operator opts in with OPENRAL_ROBOTWIN_AUTO_PROVISION=1, and
# OPENRAL_ROBOTWIN_SIDECAR_PYTHON always overrides. Pins mirror the manual recipe in
# the ROSConfigError hint below; bump both together.
_ROBOTWIN_SIDECAR_HOME = Path.home() / ".cache" / "openral" / "robotwin-sidecar"
_ROBOTWIN_PYTHON = "3.12"
_ROBOTWIN_REPO = "https://github.com/RoboTwin-Platform/RoboTwin.git"
# LeRobot 0.6.0 ships `lerobot.envs.robotwin` and supports the workspace's Python
# 3.12. The RoboTwin checkout is still needed as task package + asset root.
_ROBOTWIN_BASE_DEPS = (
    "lerobot==0.6.0",
    "sapien==3.0.3",
    "cuda-bindings==12.8.0",
    "open3d",
    "h5py",
    "pyzmq",
    "msgpack",
    "requests==2.32.5",
    "urllib3==2.5.0",
)
_ROBOTWIN_MPLIB_DEPS = ("mplib==0.2.1",)

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
# Last-resort sim-time cadence the sidecar uses only when the live SAPIEN env
# exposes neither an elapsed-time scalar nor a control period (control_timestep/
# control_dt/control_freq); RoboTwin's default control rate is ~30 Hz.
_FALLBACK_SIM_DT_S = 1.0 / 30.0


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


def _task_name_for_env(env_cfg: SimEnvironment) -> str:
    """Resolve a RoboTwin upstream task name, including deploy-sim's no-op task."""
    task_id = env_cfg.task.id
    if task_id.endswith(_DEPLOY_NOOP_SUFFIX):
        override = env_cfg.scene.backend_options.get("deploy_task_id")
        return str(override or _DEFAULT_DEPLOY_TASK)
    return _robotwin_task_name(task_id)


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


def _patch_robotwin_checkout(root: Path) -> None:
    """Apply narrow compatibility repairs to the operator-local RoboTwin checkout."""
    planner = root / "envs" / "robot" / "planner.py"
    text = planner.read_text()
    marker = "class CuroboPlanner:\n        def __init__(self, *args, **kwargs):"
    if marker in text:
        return
    needle = (
        "    traceback.print_exc()\n\n\n"
        "# ********************** MplibPlanner **********************"
    )
    replacement = """    traceback.print_exc()
    class CuroboPlanner:
        def __init__(self, *args, **kwargs):
            pass

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def __getattr__(self, name):
            raise RuntimeError("CuroboPlanner is unavailable in this environment")


# ********************** MplibPlanner **********************"""
    if needle not in text:
        raise ROSConfigError(
            f"RoboTwin planner source at {planner} does not match the expected optional "
            "cuRobo import block; remove the checkout and rerun provisioning."
        )
    planner.write_text(text.replace(needle, replacement))


def _provision_robotwin_venv() -> Path:
    """Create the robotwin sidecar venv from the pinned LeRobot + SAPIEN install.

    Opt-in (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) because it is a multi-GB,
    Linux-only download. Uses the shared :func:`ensure_pip_venv`
    provisioning order so it reuses an existing venv + sentinel. Returns the venv
    python (``<home>/.venv/bin/python``).

    Also clones RoboTwin and downloads its object assets into the default root so
    the strict verifier can run the live env instead of skipping on a half-provision.
    """

    def _install(uv: str, py: Path) -> None:
        env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "900")}
        run_cmd(
            "robotwin-sidecar",
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--exclude-newer",
                "2026-06-01T00:00:00Z",
                "--exclude-newer-package",
                "lerobot=2026-07-07T00:00:00Z",
                *_ROBOTWIN_BASE_DEPS,
            ],
            env=env,
        )
        run_cmd(
            "robotwin-sidecar",
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--exclude-newer",
                "2026-06-01T00:00:00Z",
                *_ROBOTWIN_MPLIB_DEPS,
            ],
            env=env,
        )

    py = ensure_pip_venv(
        label="robotwin-sidecar",
        home=_ROBOTWIN_SIDECAR_HOME,
        python=_ROBOTWIN_PYTHON,
        install=_install,
        # Keyed on the pins so editing them repairs an already-provisioned venv.
        spec=(*_ROBOTWIN_BASE_DEPS, *_ROBOTWIN_MPLIB_DEPS),
    )
    root = _ROBOTWIN_SIDECAR_HOME / "RoboTwin"
    if not root.is_dir():
        source = ensure_source("robotwin-assets", _ROBOTWIN_SIDECAR_HOME, _ROBOTWIN_REPO)
        source.rename(root)
    _patch_robotwin_checkout(root)
    assets = root / "assets" / "objects" / "objaverse" / "list.json"
    if not assets.is_file():
        run_cmd("robotwin-assets", ["bash", "script/_download_assets.sh"], cwd=root)
    return py


def _sidecar_python() -> Path:
    """Resolve the robotwin sidecar venv interpreter, or raise with the install hint.

    Resolution order: ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` override → opt-in
    auto-provision (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) → an existing default
    venv → a typed error carrying the exact manual commands.

    Auto-provision precedes the existing-venv shortcut so a venv built from
    superseded pins is repaired rather than returned untouched (same ordering
    fix as the Isaac backend — see :func:`ensure_pip_venv`'s ``spec``).
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    if os.environ.get(_AUTO_PROVISION_ENV, "").strip() not in ("", "0", "false", "False"):
        return _provision_robotwin_venv()
    default = _ROBOTWIN_SIDECAR_HOME / ".venv" / "bin" / "python"
    if default.is_file():
        return default
    raise ROSConfigError(
        "RoboTwin sidecar venv not found. It is an externally-provisioned dependency "
        "(SAPIEN + RoboTwin 2.0, Python 3.12, Linux-only, multi-GB). "
        "Set "
        f"{_AUTO_PROVISION_ENV}=1 to auto-provision the LeRobot+SAPIEN venv (a "
        "multi-GB download), or provision it manually and point "
        f"{_SIDECAR_PYTHON_ENV} at its py3.12 python:\n"
        "  uv venv --python 3.12 ~/.cache/openral/robotwin-sidecar/.venv\n"
        "  uv pip install --python ~/.cache/openral/robotwin-sidecar/.venv/bin/python "
        "lerobot==0.6.0 sapien mplib pyzmq msgpack\n"
        "  git clone https://github.com/RoboTwin-Platform/RoboTwin.git\n"
        "  cd RoboTwin && bash script/_download_assets.sh\n"
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


def provision_robotwin() -> None:
    """Build the RoboTwin sidecar venv — the slow half of a first run.

    ``_sidecar_python`` auto-provisions the multi-GB LeRobot + SAPIEN venv
    when ``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``, and otherwise raises with the
    manual recipe. Running it in front of ``ros2 launch`` keeps that work out
    of the HAL's ``on_configure``, which ``tools/lifecycle_autostart.py``
    bounds at 300 s — shorter than both the provisioning download and this
    backend's own 600 s sidecar boot budget.

    Idempotent — an existing venv short-circuits on its sentinel.
    :func:`_build_robotwin_scene` resolves the same interpreter again.

    Raises:
        ROSConfigError: When the venv is absent and auto-provisioning is off,
            or when the install fails.
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("robotwin_client")
    _sidecar_python()


@SCENES.register(_ROBOTWIN_SCENE_ID, fixed_robot=_ROBOTWIN_ROBOT_ID, provision=provision_robotwin)
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
    task_name = _task_name_for_env(env_cfg)
    port = _opt_num(opts, "port", _scene_default_port(task_name, _ROBOTWIN_ROBOT_ID), int)
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
        "--fallback-dt-s",
        str(_FALLBACK_SIM_DT_S),
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
