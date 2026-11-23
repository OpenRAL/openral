r"""RoboTwin 2.0 scene adapter — drives a SAPIEN dual-arm env through a sidecar venv.

RoboTwin 2.0 (Chen et al., arXiv 2506.18088, MIT) is a large-scale
**bimanual** benchmark: 50 dual-arm tasks on the SAPIEN physics engine, evaluated
on the aloha-agilex embodiment (14-DoF, 7 per arm; action 14-D joint-space).

RoboTwin's stack (SAPIEN, mplib 0.2.1, open3d, its ``envs`` task package + multi-GB
assets) is kept out of the openral venv: mplib 0.2.1 conflicts with ManiSkill's
``mplib==0.1.1`` pin, and the checkout must sit on ``sys.path`` as a generic
top-level ``envs`` package. So,
exactly like the Isaac Sim scene backend (``openral_sim.backends.isaac_sim``) and
the RLDX-1 policy sidecar (``openral_sim.policies.rldx``), we run it in its own
sidecar venv and talk to it over ZMQ REQ/REP framed by msgpack
(``openral_sim.sidecar.SidecarClient``).

This module is the **openral side**: a thin ``SimRollout`` that marshals
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
LeRobot 0.7 + the RoboTwin SAPIEN stack + downloads assets (a multi-GB,
Linux-only job); without opt-in we raise a typed ``ROSConfigError``
carrying the exact manual recipe.

Licensing (CLAUDE.md §1.9): RoboTwin (MIT), SAPIEN (MIT), LeRobot (Apache-2.0) are
all permissive but the SAPIEN+RoboTwin stack is large and CUDA-pinned, so it is an
externally-provisioned sidecar venv — never vendored into the repo. The openral-side
wire is just pyzmq + msgpack (the ``robotwin`` dependency-group).
"""

from __future__ import annotations

import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_sim._sidecar_common import (
    ensure_pip_venv,
    ensure_source,
    require_pinned_head,
    run_cmd,
    sidecar_port_for_key,
    venv_matches_spec,
)
from openral_sim._sidecar_common import (
    opt_num as _opt_num,
)
from openral_sim.backends.isaac_sim import IsaacActionLayout, pack_isaac_action
from openral_sim.registry import ROBOTS, SCENES
from openral_sim.rollout import DEFAULT_SUCCESS_KEY
from openral_sim.sidecar import SidecarClient, SidecarSimRollout

if TYPE_CHECKING:
    from openral_core import Action, SimEnvironment

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
# OPENRAL_ROBOTWIN_SIDECAR_PYTHON always overrides.
_ROBOTWIN_SIDECAR_HOME = Path.home() / ".cache" / "openral" / "robotwin-sidecar"
_ROBOTWIN_PYTHON = "3.12"
_ROBOTWIN_REPO = "https://github.com/RoboTwin-Platform/RoboTwin.git"
# Pinned upstream commit of the provisioned checkout, fresh or reused (``ensure_source``
# + ``require_pinned_head``; its asset script runs on the host): "Merge pull request
# #502", 2026-09-22.
_ROBOTWIN_SHA = "ee3d076fb61c4542e9be6c16f3af5ecd67d4a6f7"
# The venv installs from the hash-locked ``tools/sidecar_requirements/robotwin.lock``
# (compiled from ``robotwin.in``, whose header has the regenerate command and the
# exclude-newer cutoffs; ``robotwin-override.txt`` pins numpy 1.26.4 for mplib 0.2.1
# at compile time). The lock is the full closure, installed ``--no-deps``. Same
# lerobot as the workspace (0.7.0, openral#716): the sidecar ships raw pixels +
# agent_pos and every policy preprocesses in-process, so one lerobot to track. The
# RoboTwin checkout is still needed as task package + asset root.
_ROBOTWIN_LOCK = "robotwin.lock"
# RoboTwin 2.0 asset archives (HF dataset, as upstream's assets/_download.py names them).
_ROBOTWIN_ASSETS_REPO = "TianxingChen/RoboTwin2.0"
_ROBOTWIN_ASSET_ZIPS = ("background_texture.zip", "embodiments.zip", "objects.zip")

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
# control_dt/control_freq) — RoboTwin exposes none. lerobot 0.7's robotwin
# integration declares 25 fps (``conventions()``, ``metadata["render_fps"]``).
_FALLBACK_SIM_DT_S = 1.0 / 25.0


def _scene_default_port(task_id: str, robot_id: str) -> int:
    """Deterministic per-scene ZMQ port, stable across processes.

    Mirrors ``isaac_sim._scene_default_port``. Any residual hash collision is
    caught loudly by the identity-checked ping handshake, never served as
    wrong data. See ``sidecar_port_for_key`` for the shared derivation.
    """
    return sidecar_port_for_key(
        f"{task_id}|{robot_id}", port_min=_SIDECAR_PORT_MIN, port_max=_SIDECAR_PORT_MAX
    )


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


@dataclass
class _RoboTwinSimSidecar(SidecarSimRollout):
    """``SimRollout`` that proxies a RoboTwin SAPIEN env over the sidecar.

    Observations come back from the sidecar already in the eval-layer shape
    (``images`` dict of HWC uint8 keyed by the RoboTwin camera names, ``state`` 1-D
    float32 of the 14 joint drive targets the policy reads, the measured 14-D
    ``joint_positions`` / ``joint_velocities``, ``task`` str); we re-wrap into a plain dict
    and cache the last RGB frame for ``render``.

    Fields, ``reset``/``step``/``sim_time_ns``/``render``/``close`` live on
    ``SidecarSimRollout`` (shared verbatim with the Isaac Sim adapter); only
    ``_wrap_obs``, ``action_dim`` and the HAL hooks ``pack_action`` / ``idle_action``
    are RoboTwin-specific.

    The action is RoboTwin's absolute joint vector ``[left arm, left gripper, right arm,
    right gripper]`` — the ``aloha_agilex`` manifest's arm + gripper joints in manifest
    order, which ``layout`` (the name-keyed ``IsaacActionLayout``) packs by name. A slot
    no command names holds the measured pose: an absolute-target env reads a zero as
    "drive to 0 rad", so neither the HAL's default packer (contiguous arm slots, one
    gripper last) nor its zero idle vector is a hold here.
    """

    layout: IsaacActionLayout | None = None
    # Measured joint positions (manifest order) of the last observation.
    _measured: NDArray[np.float32] | None = None

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts — queried from the sidecar ping (14)."""
        if self._action_dim is None:
            reply = self._client.call("ping")
            self._action_dim = int(self._client.require(reply, "action_dim"))
        return self._action_dim

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
        # Measured joint state (manifest order), for `openral deploy sim`'s
        # SimAttachedHAL.read_state; ``state`` is the policy input (drive targets).
        for key in ("joint_positions", "joint_velocities"):
            if raw.get(key) is not None:
                obs[key] = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        self._measured = obs.get("joint_positions")
        return obs

    def _hold(self) -> NDArray[np.float32]:
        """The measured pose in action-slot order: what an unnamed slot holds."""
        layout = self._require_layout()
        if self._measured is None or self._measured.shape != (len(layout.joints),):
            raise ROSRuntimeError(
                "RoboTwin: no measured joint state to hold the uncommanded slots at "
                f"(got {None if self._measured is None else self._measured.shape}, need "
                f"{len(layout.joints)} manifest joints); reset first."
            )
        return np.asarray(
            [self._measured[layout.joints.index(n)] for n in layout.slots], dtype=np.float32
        )

    def _require_layout(self) -> IsaacActionLayout:
        layout = self.layout
        if layout is None:
            raise ROSConfigError("RoboTwin rollout built without an action layout.")
        if layout.dim != self.action_dim:
            raise ROSConfigError(
                f"RoboTwin: the manifest has {layout.dim} arm + gripper slots "
                f"{list(layout.slots)}; the sidecar's action is {self.action_dim}-D."
            )
        return layout

    def pack_action(
        self,
        action: Action,
        prev: NDArray[np.float32] | None,
        *,
        carry_twist: bool = False,
    ) -> NDArray[np.float32]:
        """``SimAttachedHAL``'s packing hook: slots addressed by manifest joint name.

        ``pack_isaac_action`` places the command (an arm-only, non-base or full
        JOINT_POSITION row, a named slot row, or a GRIPPER_POSITION by ``ee_name``);
        every slot it leaves unset (NaN) takes the measured pose (``_hold``).
        """
        out = pack_isaac_action(action, self._require_layout(), prev, carry_twist=carry_twist)
        unset = np.isnan(out)
        if unset.any():
            out[unset] = self._hold()[unset]
        return out

    def idle_action(self) -> NDArray[np.float32]:
        """``SimAttachedHAL``'s idle hold: the measured pose (zeros would drive to 0 rad)."""
        return self._hold()


# ── factory ───────────────────────────────────────────────────────────────────


def _provision_robotwin_venv() -> Path:
    """Create the robotwin sidecar venv from the hash-locked LeRobot + SAPIEN stack.

    Opt-in (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) because it is a multi-GB,
    Linux-only download. Uses the shared ``ensure_pip_venv``
    provisioning order so it reuses an existing venv + sentinel. Returns the venv
    python (``<home>/.venv/bin/python``).

    Also fetches RoboTwin at ``_ROBOTWIN_SHA`` into the default root (verifying a
    reused root against the pin) and downloads its object assets, so the strict
    verifier can run the live env instead of skipping on a half-provision. An
    ``OPENRAL_ROBOTWIN_ROOT`` pointing elsewhere is the operator's own checkout and
    is left untouched.

    Raises:
        ROSConfigError: On a host other than x86_64 Linux, the only platform the
            lock (SAPIEN / open3d / mplib wheels) resolves for.
    """
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ROSConfigError(
            f"The RoboTwin sidecar runs on x86_64 Linux only (this host: {sys.platform}/"
            f"{platform.machine()}): SAPIEN 3.0.3, open3d and mplib 0.2.1 ship no wheels "
            "for it, so tools/sidecar_requirements/robotwin.lock cannot install."
        )
    lock = _robotwin_lock()

    def _install(uv: str, py: Path) -> None:
        env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "900")}
        run_cmd(
            "robotwin-sidecar",
            [
                uv,
                "pip",
                # sync, not install: the venv ends up exactly the lock (a repaired
                # venv sheds packages an older stack left behind), and no resolver
                # runs, so mplib's numpy<2 vs lerobot's numpy>=2 metadata — settled
                # at compile time by robotwin-override.txt — is not re-checked.
                "sync",
                # --no-config: from the repo root uv would apply the workspace's
                # [tool.uv] settings to this unrelated venv.
                "--no-config",
                "--python",
                str(py),
                "--require-hashes",
                str(lock),
            ],
            env=env,
        )

    py = ensure_pip_venv(
        label="robotwin-sidecar",
        home=_ROBOTWIN_SIDECAR_HOME,
        python=_ROBOTWIN_PYTHON,
        install=_install,
        # Keyed on the lock text so recompiling it repairs an already-provisioned venv.
        spec=_robotwin_venv_spec(),
    )
    if not _is_managed_root(_robotwin_root_path()):
        # The operator's own checkout: theirs to pin, never cloned or verified here.
        return py
    root = _ensure_robotwin_checkout(_ROBOTWIN_SIDECAR_HOME)
    if not (root / "assets" / "objects" / "objaverse" / "list.json").is_file():
        _fetch_robotwin_assets(root, py)
    return py


def _fetch_robotwin_assets(root: Path, py: Path) -> None:
    """Download + unpack RoboTwin's assets the way its ``_download_assets.sh`` does.

    Not by running that script: its ``assets/_download.py`` passes
    ``snapshot_download(resume_download=True)``, which the locked huggingface-hub
    (2.x) no longer accepts, and the script has no ``set -e`` so the failure would
    surface only later as missing assets. The download runs under the sidecar venv
    ``py`` (its huggingface_hub), the unzip in-process, and every step fails loudly.
    """
    assets = root / "assets"
    run_cmd(
        "robotwin-assets",
        [
            str(py),
            "-c",
            "from huggingface_hub import snapshot_download; snapshot_download("
            f"repo_id={_ROBOTWIN_ASSETS_REPO!r}, repo_type='dataset', local_dir='.', "
            f"allow_patterns={list(_ROBOTWIN_ASSET_ZIPS)!r})",
        ],
        cwd=assets,
    )
    for name in _ROBOTWIN_ASSET_ZIPS:
        archive = assets / name
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(assets)
        archive.unlink()
    run_cmd("robotwin-assets", [str(py), _embodiment_config_script(root)], cwd=root)


def _ensure_robotwin_checkout(
    home: Path, repo: str = _ROBOTWIN_REPO, sha: str = _ROBOTWIN_SHA
) -> Path:
    """Return ``<home>/RoboTwin`` at commit ``sha``, fetching it on first use.

    Fresh or reused, the checkout must be the pin (its asset script runs on this
    host): anything else raises ``ROSConfigError`` with the delete-to-refetch hint.
    """
    root = home / "RoboTwin"
    try:
        if not root.is_dir():
            ensure_source("robotwin-assets", home, repo, sha).rename(root)
        require_pinned_head("robotwin-assets", root, sha)
    except SystemExit as exc:
        raise ROSConfigError(str(exc)) from exc
    return root


def _robotwin_root_path() -> Path:
    """The configured RoboTwin root: ``OPENRAL_ROBOTWIN_ROOT``, else the managed default."""
    override = os.environ.get(_ROBOTWIN_ROOT_ENV)
    return Path(override).expanduser() if override else _ROBOTWIN_SIDECAR_HOME / "RoboTwin"


def _is_managed_root(root: Path) -> bool:
    """Whether ``root`` is the provisioned ``<sidecar home>/RoboTwin`` (however it is named)."""
    return root.resolve() == (_ROBOTWIN_SIDECAR_HOME / "RoboTwin").resolve()


def _embodiment_config_script(root: Path) -> str:
    """Path (relative to ``root``) of RoboTwin's ``update_embodiment_config_path.py``.

    Upstream renamed ``script/`` to ``scripts/`` in 2026; an operator checkout may be
    either vintage, so both are accepted.
    """
    for candidate in (
        "scripts/update_embodiment_config_path.py",
        "script/update_embodiment_config_path.py",
    ):
        if (root / candidate).is_file():
            return candidate
    raise ROSConfigError(
        f"RoboTwin checkout at {root} has no scripts/update_embodiment_config_path.py; "
        "remove the checkout and rerun provisioning."
    )


def _find_sidecar_python() -> Path | None:
    """A usable RoboTwin interpreter that already exists — never provisions anything.

    ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` (when it is a file), or the default venv when
    its sentinel matches the current lock; else ``None``. For availability checks
    (the sim-test gate) that must not trigger a multi-GB install.
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    venv = _ROBOTWIN_SIDECAR_HOME / ".venv"
    default = venv / "bin" / "python"
    if default.is_file() and venv_matches_spec(venv, _robotwin_venv_spec()):
        return default
    return None


def _sidecar_python() -> Path:
    """Resolve the robotwin sidecar venv interpreter, or raise with the install hint.

    Resolution order: ``OPENRAL_ROBOTWIN_SIDECAR_PYTHON`` override → opt-in
    auto-provision (``OPENRAL_ROBOTWIN_AUTO_PROVISION=1``) → an existing default
    venv whose sentinel matches the current lock (``_find_sidecar_python``) → a typed
    error: re-provision for a venv from another lock, else the manual commands.

    Auto-provision precedes the existing-venv shortcut so a venv built from
    superseded pins is repaired rather than returned untouched (same ordering
    fix as the Isaac backend — see ``ensure_pip_venv``'s ``spec``). Without the
    opt-in, a default venv from superseded pins (e.g. lerobot 0.6.0) raises a
    re-provision ``ROSConfigError`` instead of booting a sidecar that cannot import
    the env.
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    if os.environ.get(_AUTO_PROVISION_ENV, "").strip() not in ("", "0", "false", "False"):
        return _provision_robotwin_venv()
    found = _find_sidecar_python()
    if found is not None:
        return found
    venv = _ROBOTWIN_SIDECAR_HOME / ".venv"
    if (venv / "bin" / "python").is_file():
        raise ROSConfigError(
            f"RoboTwin sidecar venv at {venv} was built from another lock (or did not "
            f"finish provisioning). Set {_AUTO_PROVISION_ENV}=1 to repair it in place."
        )
    raise ROSConfigError(
        "RoboTwin sidecar venv not found. It is an externally-provisioned dependency "
        "(SAPIEN + RoboTwin 2.0, Python 3.12, x86_64 Linux only, multi-GB). "
        "Set "
        f"{_AUTO_PROVISION_ENV}=1 to auto-provision the LeRobot+SAPIEN venv (a "
        "multi-GB download), or provision it manually from the openral repo root and "
        f"point {_SIDECAR_PYTHON_ENV} at its py3.12 python:\n"
        "  VENV=~/robotwin-sidecar/.venv\n"
        "  uv venv --python 3.12 $VENV\n"
        "  uv pip sync --no-config --python $VENV/bin/python --require-hashes "
        f"tools/sidecar_requirements/{_ROBOTWIN_LOCK}\n"
        f"  git clone {_ROBOTWIN_REPO} RoboTwin\n"
        f"  git -C RoboTwin checkout {_ROBOTWIN_SHA}  # the tested pin\n"
        "  cd RoboTwin/assets\n"
        '  $VENV/bin/python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download(repo_id='{_ROBOTWIN_ASSETS_REPO}', repo_type='dataset', "
        f"local_dir='.', allow_patterns={list(_ROBOTWIN_ASSET_ZIPS)!r})\"\n"
        "  for z in *.zip; do unzip -q $z && rm $z; done; cd ..\n"
        "  $VENV/bin/python scripts/update_embodiment_config_path.py\n"
        f"  export {_ROBOTWIN_ROOT_ENV}=$(pwd)  # checkout + assets path for the sidecar\n"
        f"  export {_SIDECAR_PYTHON_ENV}=$VENV/bin/python"
    )


def _robotwin_lock() -> Path:
    """``tools/sidecar_requirements/robotwin.lock``, found by walking up from this module.

    Independent of ``OPENRAL_ROBOTWIN_SIDECAR_SCRIPT``: a relocated sidecar script does
    not move the lock the venv is checked against.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        lock = parent / "tools" / "sidecar_requirements" / _ROBOTWIN_LOCK
        if lock.is_file():
            return lock
    raise ROSConfigError(
        f"Could not locate tools/sidecar_requirements/{_ROBOTWIN_LOCK} upwards from {here}."
    )


def _robotwin_venv_spec() -> tuple[str]:
    """Sentinel spec of the sidecar venv: the text of the lock it was built from."""
    return (_robotwin_lock().read_text(encoding="utf-8"),)


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
    ``SidecarClient`` deliberately strips parent ``PYTHONPATH`` for ABI safety.
    Pass the checkout root explicitly so the sidecar can chdir and add it to
    ``sys.path`` before LeRobot imports the task package.
    """
    root = _robotwin_root_path()
    if not root.is_dir():
        raise ROSConfigError(
            "RoboTwin checkout not found. Clone https://github.com/RoboTwin-Platform/RoboTwin "
            f"and set {_ROBOTWIN_ROOT_ENV} to its path."
        )
    assets = root / "assets" / "objects" / "objaverse" / "list.json"
    if not assets.is_file():
        raise ROSConfigError(
            f"RoboTwin assets not found at {assets}. Fetch them as the "
            f"{_AUTO_PROVISION_ENV}=1 recipe does (see the sidecar-venv error), then set "
            f"{_ROBOTWIN_ROOT_ENV}={root}."
        )
    if _is_managed_root(root):
        # The sidecar imports this checkout's code: the provisioned one must be the pin
        # on every launch, not only when it was (re)provisioned.
        _ensure_robotwin_checkout(_ROBOTWIN_SIDECAR_HOME, _ROBOTWIN_REPO, _ROBOTWIN_SHA)
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
    ``_build_robotwin_scene`` resolves the same interpreter again.

    Raises:
        ROSConfigError: When the venv is absent and auto-provisioning is off,
            or when the install fails.
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("robotwin_client")
    _sidecar_python()


@SCENES.register(
    _ROBOTWIN_SCENE_ID,
    fixed_robot=_ROBOTWIN_ROBOT_ID,
    provision=provision_robotwin,
    sim_clock=True,
    # RoboTwin's set_gripper clips to [0, 1], 0 = closed.
    gripper_convention="normalized_open_unit",
)
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
        env_cfg.task.success_key or DEFAULT_SUCCESS_KEY,
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
    return _RoboTwinSimSidecar(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _client=client,
        layout=IsaacActionLayout.from_description(ROBOTS.get(_ROBOTWIN_ROBOT_ID)()),
    )
