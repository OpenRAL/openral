r"""Isaac Sim scene adapter — drives an Isaac Lab env through an out-of-process sidecar.

NVIDIA Isaac Sim (Omniverse Kit + PhysX + RTX) ships per-interpreter wheels:
4.x→py3.10, 5.x→py3.11, 6.x→py3.12. The openral workspace pins
``>=3.12,<3.13``, and Isaac Sim's stack (its own torch / CUDA build, the
rigid ``SimulationApp``-before-``omni.*`` import order, a libgomp/OpenMP
``LD_PRELOAD`` clash with the VLA torch stack) makes an in-process load
impractical inside the 3.12 venv. So — like the RLDX-1 policy sidecar
(``openral_sim.policies.rldx``) — Isaac Lab runs in its own py3.11 venv,
reached over ZMQ REQ/REP framed by msgpack.

This module is the **openral side**: a thin ``SimRollout`` that marshals
``reset`` / ``step`` / ``render`` / ``close`` to the sidecar
(``tools/isaac_sidecar.py``) and unwraps the responses. The sidecar owns the
Omniverse app, the Franka manipulation env, PhysX stepping, and RTX camera
rendering.

Lifecycle (mirrors the RLDX adapter's auto-spawn block)
-------------------------------------------------------
* On build we ping the sidecar at ``host:port``.
* If the ping fails and ``auto_spawn`` is on (default; ``vla``-free scene path,
  toggle via ``OPENRAL_ISAAC_AUTO_SPAWN=0``), we ``Popen`` the launcher with the
  resolved scene config (task id, layout, obs size, instruction) and poll
  ``ping`` until it answers or ``boot_timeout_s`` elapses. First boot pays the
  tens-of-seconds Omniverse Kit start; ``boot_timeout_s`` defaults large.
* ``close`` terminates only a child we spawned ourselves; a pre-existing
  operator-launched sidecar is left running.

Sidecar python resolution
-------------------------
The launcher runs under the **isaac** interpreter, not this one. We resolve it
from ``OPENRAL_ISAAC_SIDECAR_PYTHON`` (absolute path to a pip venv's ``python``
or a binary install's ``python.sh``), else the opt-in auto-provisioned pip venv
(``~/.cache/openral/isaac-sidecar/.venv``), else a host-wide **binary install**
(``/opt/isaac-sim/python.sh``, NVIDIA's standalone package — any Isaac Sim
release whose Python the sidecar supports), and raise a typed ``ROSConfigError``
carrying the exact provisioning commands if none exists. A binary install's
Python lacks the sidecar's two wire deps (pyzmq, msgpack); they go into a
per-install user-cache dir (``--site-dir``), never into the install itself.

Scene category: **free-axis** (``fixed_robot=None``): any manifest robot with an
``assets.urdf`` is imported from its URDF (the one sidecar layout, ``manifest``).

Environment + placement
-----------------------
A deploy scene loads any Isaac stage around any manifest robot with three
fields — ``scene.assets_uri`` (the environment USD), ``robot_id``, and
``base_pose`` (the spawn; planar: x, y, z + a yaw-only quaternion)::

    robot_id: panda_mobile
    base_pose: {xyz: [-4.8, 0.0, 0.0], quat_xyzw: [0, 0, 0.7071068, 0.7071068], frame_id: world}
    scene:
      id: isaac_sim
      backend: isaacsim
      assets_uri: isaac:Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd

``assets_uri`` accepts a local path (``file://`` optional; relative to
the working directory), an ``http(s)://`` / ``omniverse://`` URL, or
``isaac:<path>`` under the sidecar's own Isaac asset root. NVIDIA's Isaac
environments are licensed for use inside Isaac Sim — they are referenced here,
never converted or vendored.

Licensing (CLAUDE.md §1.9, §3): Isaac Sim's Omniverse Kit
components are proprietary and **never vendored** — the sidecar venv is an
externally-provisioned dependency the user installs (and, by running the
launcher, accepts the NVIDIA Omniverse EULA via ``OMNI_KIT_ACCEPT_EULA=YES``).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from pydantic import BaseModel, ConfigDict, Field

from openral_sim._sidecar_common import ensure_pip_venv, run_cmd, sidecar_port_for_key
from openral_sim.registry import SCENES
from openral_sim.sidecar import SidecarClient, SidecarSimRollout

if TYPE_CHECKING:
    from openral_core import Action, Pose6D, RobotDescription, SensorSpec, SimEnvironment

    from openral_sim.rollout import Observation


_ISAAC_SCENE_ID = "isaac_sim"
_AUTO_SPAWN_ENV = "OPENRAL_ISAAC_AUTO_SPAWN"
_SIDECAR_PYTHON_ENV = "OPENRAL_ISAAC_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_ISAAC_SIDECAR_SCRIPT"
_AUTO_PROVISION_ENV = "OPENRAL_ISAAC_AUTO_PROVISION"

# Default sidecar venv location + the (pinned) Isaac install. Isaac Sim / Isaac
# Lab are ~50 GB, RTX-only, license-gated, and pulled from NVIDIA's own index —
# they are never vendored (CLAUDE.md §1.9). So unlike the LocateAnything / Qwen
# sidecars we do NOT auto-provision by default: provisioning runs only when the
# operator opts in with OPENRAL_ISAAC_AUTO_PROVISION=1 (a multi-GB download), and
# OPENRAL_ISAAC_SIDECAR_PYTHON always overrides. Pins mirror the manual recipe in
# the ROSConfigError hint below; bump both together.
_ISAAC_SIDECAR_HOME = Path.home() / ".cache" / "openral" / "isaac-sidecar"
_ISAAC_PYTHON = "3.11"
_ISAAC_DEPS = (
    "isaacsim==5.1.0.0",
    "isaacsim-core==5.1.0.0",
    "isaacsim-robot==5.1.0.0",
    "isaacsim-sensor==5.1.0.0",
    "isaacsim-robot-motion==5.1.0.0",
    "isaacsim-asset==5.1.0.0",
    "pyzmq",
    "msgpack",
)
# CUDA runtime floors, forced on top of the Isaac install (``--upgrade --no-deps``).
# The Kit extension ``omni.isaac.ml_archive`` prebundles a CUDA-12.8 libcusparse
# but NOT libnvJitLink, so nvJitLink resolves against the venv's copy — and
# torch 2.7's own ``nvidia-nvjitlink-cu12==12.6.85`` pin is too old for it
# (``undefined symbol: __nvJitLinkCreate_12_8``). The mismatch does not raise;
# extension startup fails and Kit sits there, so the client burns its whole boot
# timeout with no useful error (issue #89). Floors are declared here as
# ``dist -> minimum`` because both the pip spec above and the sidecar's boot
# probe (``--require-min``) are derived from them — one source of truth.
_ISAAC_CUDA_FLOORS = {
    "nvidia-nvjitlink-cu12": "12.8",
    "nvidia-cusparse-cu12": "12.5",
}
_ISAAC_CUDA_DEPS = tuple(f"{dist}>={floor},<13" for dist, floor in _ISAAC_CUDA_FLOORS.items())
# Host-wide Isaac Sim binary installs (NVIDIA's standalone package), probed after
# the pip venv. Each root ships ``python.sh`` (its bundled interpreter + env).
_BINARY_INSTALL_ROOTS = (Path("/opt/isaac-sim"), Path.home() / "isaacsim")
# The sidecar's wire deps, installed beside (not into) a binary install.
_BINARY_SITE_DEPS = ("pyzmq", "msgpack")
# Default ZMQ endpoint. Distinct port from the RLDX sidecar (5555-ish) so the
# two can coexist on one host.
_DEFAULT_HOST = "127.0.0.1"
# Per-scene default ports live in 20000–39999 (clear of well-known ports and the
# usual ephemeral range) — the SAME band the RLDX sidecar uses for the same
# reason (``policies.rldx._derive_sidecar_port``). One sidecar serves one scene,
# so two DIFFERENT scenes must NOT share a port: a lingering sidecar from scene A
# would otherwise be silently adopted by scene B (same host:port) and serve it
# the wrong layout. ``_scene_default_port`` derives a stable per-scene port so
# that never happens; an explicit ``backend_options.port`` still wins.
_SIDECAR_PORT_MIN = 20_000
_SIDECAR_PORT_MAX = 40_000


def _scene_default_port(task_id: str, robot_id: str, layout: str, world: str = "") -> int:
    """Deterministic per-scene ZMQ port, stable across processes.

    Mirrors ``policies.rldx._derive_sidecar_port`` (policy identity) for the
    scene-identity case. ``world`` keys the environment USD + spawn pose, so
    two deploy scenes that differ only in where the robot stands never share
    a sidecar (empty → the pre-environment key, so existing ports are
    unchanged). Any residual hash collision is caught loudly by the
    identity-checked ping handshake (``SidecarClient.expected_identity``),
    never served as wrong data. See ``sidecar_port_for_key`` for the shared
    derivation.
    """
    key = f"{task_id}|{robot_id}|{layout}" + (f"|{world}" if world else "")
    return sidecar_port_for_key(key, port_min=_SIDECAR_PORT_MIN, port_max=_SIDECAR_PORT_MAX)


# REQ recv timeout for a steady-state step (Omniverse PhysX + RTX render of one
# frame is far slower than MuJoCo — generous so a slow frame is not read as a
# dead sidecar).
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 900.0
# Truncation cap when the scene comes from a taskless DeployScene (deploy sim).
_DEFAULT_MAX_STEPS = 1_000_000


# ── SimRollout adapter ────────────────────────────────────────────────────────


_PLANAR_TWIST_EPS = 1e-6


@dataclass(frozen=True)
class IsaacActionLayout:
    """The manifest scene's action vector, named.

    ``[one absolute target per non-base manifest joint, in manifest order
    (arm joints in rad, grippers in the manifest's own units), then vx, vy, wz
    when the robot has a planar base]``. A NaN target means HOLD the joint's
    current target — the scene never reads "0" as "leave it alone", since 0 is
    a legal joint target.

    Example:
        >>> layout = IsaacActionLayout(
        ...     joints=("j1", "grip"), roles=("arm", "gripper"), end_effectors=("hand",)
        ... )
        >>> layout.slots, layout.dim
        (('j1', 'grip'), 2)
    """

    joints: tuple[str, ...]
    roles: tuple[str, ...]
    end_effectors: tuple[str, ...] = ()

    @classmethod
    def from_description(cls, desc: RobotDescription) -> IsaacActionLayout:
        """The layout ``_build_robot_spec`` gives the sidecar, from the manifest alone.

        Same roles as the spec: ``base`` (a ``base_joints`` entry), ``gripper``
        (manifest role), else ``arm``; fixed joints carry no slot.
        """
        base = set(desc.base_joints or [])
        joints = [j for j in desc.joints if getattr(j.joint_type, "value", j.joint_type) != "fixed"]
        return cls(
            joints=tuple(j.name for j in joints),
            roles=tuple(
                "base" if j.name in base else "gripper" if j.role == "gripper" else "arm"
                for j in joints
            ),
            end_effectors=tuple(e.name for e in desc.end_effectors),
        )

    @property
    def slots(self) -> tuple[str, ...]:
        """Manifest joints that own an action slot (everything but the base)."""
        return tuple(n for n, r in zip(self.joints, self.roles, strict=True) if r != "base")

    @property
    def has_base(self) -> bool:
        """Whether the robot has a planar base (three twist slots at the end)."""
        return "base" in self.roles

    @property
    def dim(self) -> int:
        """Action vector width."""
        return len(self.slots) + (3 if self.has_base else 0)

    def gripper_for(self, ee_name: str | None) -> str:
        """The gripper joint a gripper action drives.

        ``ee_name`` naming a gripper joint selects it; with exactly one gripper,
        any end-effector name (or none) selects that one (franka_panda's
        end-effector is ``panda_hand``, its gripper joint ``panda_gripper``).
        """
        grippers = [n for n, r in zip(self.joints, self.roles, strict=True) if r == "gripper"]
        if ee_name in grippers:
            return str(ee_name)
        if len(grippers) == 1 and (ee_name is None or ee_name in self.end_effectors):
            return grippers[0]
        raise ROSConfigError(
            f"gripper action names end-effector {ee_name!r}; this robot's grippers are "
            f"{grippers} (end-effectors {list(self.end_effectors)})."
        )


def pack_isaac_action(
    action: Action, layout: IsaacActionLayout, prev: NDArray[np.float32] | None
) -> NDArray[np.float32]:
    """One typed ``Action`` → the manifest scene's action vector (``IsaacActionLayout``).

    Addressed by name, never by position (ADR-0102): a JOINT_POSITION row writes
    the joints its ``joint_names`` list (the row zero-padded to full dof in
    manifest order), or — without names — a whole-vector row in manifest order
    (all joints, the non-base joints, or the arm joints alone); a
    GRIPPER_POSITION writes the gripper its ``ee_name`` resolves to; a
    BODY_TWIST writes ``(vx, vy, wz)``. Every other slot keeps ``prev`` (NaN =
    HOLD when nothing was commanded yet), except the twist, which is zeroed by
    any non-twist command — a velocity must not outlive the command that set
    it. Base joints in a joint row are skipped: the kinematic base only takes a
    BODY_TWIST.

    Raises:
        ROSConfigError: an unsupported control mode, a missing payload, an
            unknown joint / end-effector, or a row whose width matches no form.

    Example:
        >>> from openral_core.schemas import Action, ControlMode
        >>> layout = IsaacActionLayout(
        ...     joints=("x", "y", "yaw", "j1", "j2", "grip"),
        ...     roles=("base", "base", "base", "arm", "arm", "gripper"),
        ... )
        >>> a = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.5, -1.0]])
        >>> pack_isaac_action(a, layout, None).tolist()
        [0.5, -1.0, nan, 0.0, 0.0, 0.0]
    """
    from openral_core.schemas import ControlMode

    slots = layout.slots
    index = {n: i for i, n in enumerate(slots)}
    out: NDArray[np.float32] = np.full(layout.dim, np.nan, dtype=np.float32)
    if prev is not None and prev.shape == out.shape:
        out = prev.astype(np.float32, copy=True)
    if layout.has_base:
        out[-3:] = 0.0
    mode = action.control_mode
    if mode is ControlMode.BODY_TWIST:
        if not layout.has_base:
            raise ROSConfigError("BODY_TWIST on a robot without a planar base (base_joints).")
        if not action.body_twist:
            raise ROSConfigError("BODY_TWIST action with an empty body_twist.")
        vx, vy, vz, wx, wy, wz = (float(v) for v in action.body_twist[0])
        if max(abs(vz), abs(wx), abs(wy)) > _PLANAR_TWIST_EPS:
            raise ROSConfigError("BODY_TWIST with non-zero vz / wx / wy on a planar base.")
        out[-3:] = (vx, vy, wz)
        return out
    if mode is ControlMode.GRIPPER_POSITION:
        if not action.gripper:
            raise ROSConfigError("GRIPPER_POSITION action with an empty gripper payload.")
        out[index[layout.gripper_for(action.ee_name)]] = float(action.gripper[0])
        return out
    if mode is not ControlMode.JOINT_POSITION:
        raise ROSConfigError(
            f"the Isaac manifest scene takes JOINT_POSITION / GRIPPER_POSITION / "
            f"BODY_TWIST, not {mode.value!r}."
        )
    if not action.joint_targets:
        raise ROSConfigError("JOINT_POSITION action with no joint_targets.")
    row = [float(v) for v in action.joint_targets[0]]
    if action.joint_names:
        names = list(action.joint_names)
        if len(row) != len(layout.joints):
            raise ROSConfigError(
                f"JOINT_POSITION slot row has {len(row)} values; a named slot row is "
                f"zero-padded to the robot's {len(layout.joints)} joints."
            )
        pairs = [(n, row[layout.joints.index(n)]) for n in names if n in layout.joints]
        unknown = [n for n in names if n not in layout.joints]
        if unknown:
            raise ROSConfigError(f"JOINT_POSITION names unknown joints {unknown}.")
    else:
        arm = [n for n, r in zip(layout.joints, layout.roles, strict=True) if r == "arm"]
        forms = {len(layout.joints): list(layout.joints), len(slots): list(slots)}
        forms.setdefault(len(arm), arm)
        if len(row) not in forms:
            raise ROSConfigError(
                f"JOINT_POSITION row has {len(row)} values; expected {sorted(forms)} "
                "(all joints / non-base joints / arm joints, manifest order)."
            )
        pairs = list(zip(forms[len(row)], row, strict=True))
    for name, value in pairs:
        if name in index:  # base joints carry no slot (velocity-commanded)
            out[index[name]] = value
    return out


@dataclass
class _IsaacSimSidecar(SidecarSimRollout):
    """``SimRollout`` that proxies an Isaac Lab env over the sidecar.

    Observations come back from the sidecar already in the eval-layer shape
    (``images`` dict of HWC uint8, ``state`` 1-D float32, ``task`` str); we only
    re-wrap into a plain dict and cache the last RGB frame for ``render``.

    Fields, ``reset``/``step``/``sim_time_ns``/``render``/``close`` live on
    ``SidecarSimRollout`` (shared verbatim with the RoboTwin adapter); only
    ``_wrap_obs`` and this docstring-carrying ``action_dim`` are Isaac-specific.
    """

    layout: IsaacActionLayout | None = None

    def pack_action(self, action: Action, prev: NDArray[np.float32] | None) -> NDArray[np.float32]:
        """``SimAttachedHAL``'s packing hook: this env addresses its slots by name.

        See ``pack_isaac_action``. The HAL's default packer assumes the robosuite
        slot order (base first, one gripper last) and per-step deltas; this scene
        takes absolute targets in manifest order, base twist last, one slot per
        gripper — so the env, which knows its own layout, packs.
        """
        if self.layout is None:
            raise ROSConfigError("Isaac rollout built without an action layout.")
        return pack_isaac_action(action, self.layout, prev)

    def idle_action(self) -> NDArray[np.float32]:
        """HOLD every joint target (NaN) and stop the base — never "drive to 0 rad"."""
        out = np.full(self.action_dim, np.nan, dtype=np.float32)
        if self.layout is not None and self.layout.has_base:
            out[-3:] = 0.0
        return out

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts — queried from the sidecar ping.

        ``openral deploy sim`` wraps this rollout in ``SimAttachedHAL``, whose
        ``_probe_env_action_dim`` reads ``env.action_dim`` to size the HAL's
        action packing. The sidecar's ``ping`` reply already carries
        the scene's action width (``[arm, one slot per gripper, base twist]`` —
        11 for panda_mobile, 16 for OpenArm); we cache it on first access.
        """
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
            self._last_image = next(iter(images.values()))
        else:
            h = self.scene.observation_height
            w = self.scene.observation_width
            cam0 = self.scene.cameras[0] if self.scene.cameras else "camera1"
            images = {cam0: np.zeros((h, w, 3), dtype=np.uint8)}
        state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1)
        obs: Observation = {
            "images": images,
            "state": state,
            "task": raw.get("task", self.task.instruction),
        }
        # Real robot joint angles (manifest order), when the sidecar provides
        # them — `openral deploy sim`'s SimAttachedHAL.read_state reads this for
        # a non-MuJoCo backend's /joint_states.
        joints = raw.get("joint_positions")
        if joints is not None:
            obs["joint_positions"] = np.asarray(joints, dtype=np.float32).reshape(-1)
        joint_vel = raw.get("joint_velocities")
        if joint_vel is not None:
            obs["joint_velocities"] = np.asarray(joint_vel, dtype=np.float32).reshape(-1)
        # Kinematic planar-base pose (x, y, yaw), when the manifest scene drives a
        # mobile base — the deploy-sim odom path reads this.
        base_pose = raw.get("base_pose")
        if base_pose is not None:
            obs["base_pose"] = np.asarray(base_pose, dtype=np.float32).reshape(-1)
        # Per-depth-sensor point clouds ((N,3) base_link), when the manifest scene
        # renders a depth camera — SimSensorBridge publishes them as PointCloud2.
        clouds = raw.get("depth_points")
        if isinstance(clouds, dict):
            obs["depth_points"] = {
                k: np.asarray(v, dtype=np.float32).reshape(-1, 3) for k, v in clouds.items()
            }
        # 2-D LaserScan range fan (base_link), when the manifest scene has a lidar
        # — SimSensorBridge publishes it as /scan.
        scan = raw.get("scan")
        if scan is not None:
            obs["scan"] = np.asarray(scan, dtype=np.float32).reshape(-1)
        return obs


# ── factory ───────────────────────────────────────────────────────────────────


def _provision_isaac_venv() -> Path:
    """Create the isaac sidecar venv from the pinned NVIDIA-index install.

    Opt-in (``OPENRAL_ISAAC_AUTO_PROVISION=1``) because it is a multi-GB,
    RTX-only, license-gated download from NVIDIA's index. Uses the shared
    ``ensure_pip_venv`` provisioning order so it reuses an existing venv +
    sentinel, matching the LocateAnything / Qwen sidecars. Returns the venv
    python (``<home>/.venv/bin/python``).
    """

    def _install(uv: str, py: Path) -> None:
        env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "900")}
        run_cmd(
            "isaac-sidecar",
            [uv, "pip", "install", "--python", str(py), "pip", "setuptools<81"],
        )
        run_cmd(
            "isaac-sidecar",
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                "https://pypi.nvidia.com",
                *_ISAAC_DEPS,
            ],
            env=env,
        )
        run_cmd(
            "isaac-sidecar",
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--no-deps",
                *_ISAAC_CUDA_DEPS,
            ],
            env=env,
        )

    return ensure_pip_venv(
        label="isaac-sidecar",
        home=_ISAAC_SIDECAR_HOME,
        python=_ISAAC_PYTHON,
        install=_install,
        # Keyed on the pins so raising a floor (e.g. the nvJitLink one) repairs
        # an already-provisioned venv instead of being ignored forever.
        spec=(*_ISAAC_DEPS, *_ISAAC_CUDA_DEPS),
    )


def _sidecar_python() -> Path:
    """Resolve the isaac sidecar venv interpreter, or raise with the install hint.

    Resolution order: ``OPENRAL_ISAAC_SIDECAR_PYTHON`` override → opt-in
    auto-provision (``OPENRAL_ISAAC_AUTO_PROVISION=1``) → an existing default
    venv → a typed error carrying the exact manual commands.

    Auto-provision is tried *before* the existing-venv shortcut so a venv built
    from superseded pins gets repaired: ``ensure_pip_venv`` reuses it when
    its sentinel still matches (cheap) and re-installs when it does not. The
    old order returned any existing venv untouched, which is why the stale
    nvJitLink of issue #89 survived every re-provision attempt.
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    if os.environ.get(_AUTO_PROVISION_ENV, "").strip() not in ("", "0", "false", "False"):
        return _provision_isaac_venv()
    default = _ISAAC_SIDECAR_HOME / ".venv" / "bin" / "python"
    if default.is_file():
        return default
    for root in _BINARY_INSTALL_ROOTS:
        if (root / "python.sh").is_file():
            print(f"[isaac-sidecar] using the Isaac Sim binary install at {root}", flush=True)
            return root / "python.sh"
    raise ROSConfigError(
        "Isaac Sim sidecar venv not found. It is an externally-provisioned "
        "dependency (NVIDIA Isaac Sim / Isaac Lab, separate license, RTX GPU). "
        "No binary install found either "
        f"({', '.join(str(r) for r in _BINARY_INSTALL_ROOTS)}). Set "
        f"{_AUTO_PROVISION_ENV}=1 to auto-provision it (a multi-GB download), or "
        f"provision it manually and point {_SIDECAR_PYTHON_ENV} at its py3.11 python:\n"
        "  uv venv --python 3.11 ~/.cache/openral/isaac-sidecar/.venv\n"
        "  ~/.cache/openral/isaac-sidecar/.venv/bin/python -m pip install \\\n"
        "    --extra-index-url https://pypi.nvidia.com \\\n"
        "    isaacsim==5.1.0.0 isaacsim-core==5.1.0.0 isaacsim-robot==5.1.0.0 \\\n"
        "    isaacsim-sensor==5.1.0.0 isaacsim-robot-motion==5.1.0.0 \\\n"
        "    isaacsim-asset==5.1.0.0 pyzmq msgpack 'nvidia-nvjitlink-cu12>=12.8,<13'"
    )


def _is_binary_install(py: Path) -> bool:
    """True when ``py`` is a binary install's ``python.sh``, not a pip venv's python.

    The two need different handling: a pip venv carries the CUDA ``nvidia-*``
    wheels the ``--require-min`` floors check, and the wire deps; a binary install
    bundles its own CUDA runtime and needs the wire deps beside it
    (``_binary_site_dir``).
    """
    return py.name == "python.sh"


def _binary_site_dir(py: Path) -> Path:
    """Install the sidecar's wire deps for a binary install; return the dir.

    ``pip install --target`` with the install's own interpreter (ABI-matched),
    into a user-cache dir keyed on the install path + its ``VERSION`` — never into
    the (often root-owned, shared) install itself. Idempotent via a sentinel.
    """
    import hashlib

    version_file = py.parent / "VERSION"
    version = version_file.read_text().strip() if version_file.is_file() else "unknown"
    key = hashlib.sha256(f"{py}|{version}".encode()).hexdigest()[:12]
    site = _ISAAC_SIDECAR_HOME / "binary-site" / key
    sentinel = site / ".deps-installed"
    if not sentinel.is_file():
        run_cmd(
            "isaac-sidecar",
            [str(py), "-m", "pip", "install", "--target", str(site), *_BINARY_SITE_DEPS],
        )
        sentinel.write_text(f"{py}\n{version}\n")
    return site


def _locate_sidecar_script() -> Path:
    """Find ``tools/isaac_sidecar.py`` (env override, else walk up from here)."""
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "isaac_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/isaac_sidecar.py upwards from {here}. Set "
        f"{_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


# ── backend options ──


class IsaacSceneObject(BaseModel):
    """One extra USD object placed in the ``isaac_sim`` manifest scene.

    ``usd`` takes the same forms as ``scene.assets_uri`` (local path, URL,
    ``isaac:<path>``). ``dynamic`` objects are graspable rigid bodies (a rigid
    body + convex-hull colliders are added when the asset has none); static ones
    are fixed props. ``yaw`` is radians about world z.

    Example:
        >>> IsaacSceneObject(
        ...     usd="isaac:Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        ...     name="cracker_box",
        ...     xyz=(-0.4, 5.0, 0.35),
        ... ).dynamic
        True
    """

    model_config = ConfigDict(extra="forbid")

    usd: str
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    xyz: tuple[float, float, float]
    yaw: float = 0.0
    dynamic: bool = True


class IsaacSimOptions(BaseModel):
    """``scene.backend_options`` for the ``isaac_sim`` scene (validated at load).

    Example:
        >>> IsaacSimOptions().headless
        True
    """

    model_config = ConfigDict(extra="forbid")

    # The one sidecar layout; kept as a field so existing YAMLs that spell it
    # out still validate.
    layout: str = Field(default="manifest", pattern=r"^manifest$")
    headless: bool = True
    host: str = _DEFAULT_HOST
    port: int | None = Field(default=None, ge=1, le=65535)
    boot_timeout_s: float = Field(default=_DEFAULT_BOOT_TIMEOUT_S, gt=0)
    timeout_ms: int = Field(default=_DEFAULT_TIMEOUT_MS, gt=0)
    objects: list[IsaacSceneObject] = Field(default_factory=list)


# ── environment + spawn (manifest layout) ──

_USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
# Forwarded verbatim: the sidecar (Kit's omni.client) resolves these itself.
_REMOTE_USD_PREFIXES = ("isaac:", "http://", "https://", "omniverse://")
# |sin(roll/2)|, |sin(pitch/2)| above this reads as a tilted placement.
_PLANAR_QUAT_TOL = 1e-4


def _resolve_environment_usd(assets_uri: str | None, field: str = "scene.assets_uri") -> str | None:
    """Validate ``scene.assets_uri`` (or an object's ``usd``) as a USD for the sidecar.

    Remote forms (``isaac:``, ``http(s)://``, ``omniverse://``) pass through; a
    local path (optionally ``file://``) is resolved against the working
    directory and must exist — a typo fails here, not minutes into a Kit boot.

    Example:
        >>> _resolve_environment_usd("isaac:Isaac/Environments/Simple_Warehouse/warehouse.usd")
        'isaac:Isaac/Environments/Simple_Warehouse/warehouse.usd'
        >>> _resolve_environment_usd(None) is None
        True
    """
    if not assets_uri:
        return None
    if not assets_uri.lower().endswith(_USD_SUFFIXES):
        raise ROSConfigError(
            f"{field} {assets_uri!r} is not a USD file ({', '.join(_USD_SUFFIXES)})."
        )
    if assets_uri.startswith(_REMOTE_USD_PREFIXES):
        return assets_uri
    path = Path(assets_uri.removeprefix("file://")).expanduser().resolve()
    if not path.is_file():
        raise ROSConfigError(f"{field} {assets_uri!r}: no such file ({path}).")
    return str(path)


def _spawn_pose(base_pose: Pose6D | None) -> tuple[float, float, float, float]:
    """``base_pose`` → the sidecar's planar ``(x, y, z, yaw)`` placement.

    The manifest scene places (and, for a mobile base, drives) the robot on the
    floor plane, so the orientation must be a pure yaw; a roll/pitch is rejected
    rather than silently dropped.

    Example:
        >>> from openral_core import Pose6D
        >>> p = Pose6D(xyz=(1.0, 2.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="world")
        >>> _spawn_pose(p)
        (1.0, 2.0, 0.0, 0.0)
    """
    if base_pose is None:
        return (0.0, 0.0, 0.0, 0.0)
    qx, qy, qz, qw = base_pose.quat_xyzw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ROSConfigError("base_pose.quat_xyzw is the zero quaternion.")
    if abs(qx) / norm > _PLANAR_QUAT_TOL or abs(qy) / norm > _PLANAR_QUAT_TOL:
        raise ROSConfigError(
            f"base_pose.quat_xyzw {base_pose.quat_xyzw} tilts the robot; the isaac_sim "
            "scene places robots upright on the floor, so only a yaw (rotation about z) "
            "is supported."
        )
    x, y, z = (float(v) for v in base_pose.xyz)
    return (x, y, z, 2.0 * math.atan2(qz, qw))


def _world_key(
    environment_usd: str | None, base_pose: Pose6D | None, objects_json: str = ""
) -> str:
    """The environment + spawn + objects part of a scene's sidecar identity.

    Empty when none is set, so a scene without them keeps its pre-existing
    port; otherwise two scenes differing only in stage, spawn or objects get
    distinct sidecars (``_scene_default_port``).
    """
    if environment_usd is None and base_pose is None and not objects_json:
        return ""
    key = f"{environment_usd or ''}|{list(_spawn_pose(base_pose))}"
    return key + (f"|{objects_json}" if objects_json else "")


def _objects_json(objects: list[IsaacSceneObject]) -> str:
    """The validated objects as the sidecar's ``--objects-json`` (USDs resolved)."""
    return json.dumps(
        [
            {
                **o.model_dump(),
                "usd": _resolve_environment_usd(o.usd, field=f"objects[{o.name}].usd"),
            }
            for o in objects
        ]
    )


# ── robot-spec marshalling (robot-agnostic manifest scene) ──


def _sensor_dict(sensor: SensorSpec) -> dict[str, Any]:
    """Serialize one ``SensorSpec`` to the plain-JSON shape the sidecar reads.

    The Isaac sidecar runs py3.11 and cannot import ``openral_core``, so each
    sensor crosses the venv boundary as a dict of primitives — only the fields
    the URDF-driven scene needs to attach a camera / depth / lidar.
    """
    intr = sensor.intrinsics
    return {
        "name": sensor.name,
        "modality": getattr(sensor.modality, "value", str(sensor.modality)),
        "frame_id": sensor.frame_id,
        "parent_frame": sensor.parent_frame,
        "vla_feature_key": sensor.vla_feature_key,
        "intrinsics": (
            None
            if intr is None
            else {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.cx,
                "cy": intr.cy,
            }
        ),
        "range_min_m": sensor.range_min_m,
        "range_max_m": sensor.range_max_m,
        "n_channels": sensor.n_channels,
    }


@dataclass(frozen=True)
class _UrdfJoint:
    """One ``<joint>`` of a URDF, as far as the Isaac spec needs it."""

    name: str
    joint_type: str
    parent: str
    child: str
    lower: float | None
    upper: float | None
    mimic: tuple[str, float, float] | None  # (leader, multiplier, offset)


def _parse_urdf_joints(urdf_path: Path) -> dict[str, _UrdfJoint]:
    """``{name: _UrdfJoint}`` for every joint in a URDF (stdlib XML; URDF is XML)."""
    import xml.etree.ElementTree as ET

    joints: dict[str, _UrdfJoint] = {}
    for el in ET.parse(urdf_path).getroot().iter("joint"):
        parent, child = el.find("parent"), el.find("child")
        if parent is None or child is None:
            continue  # a <transmission>'s <joint> reference, not a joint
        limit, mimic = el.find("limit"), el.find("mimic")
        joints[el.attrib["name"]] = _UrdfJoint(
            name=el.attrib["name"],
            joint_type=el.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            lower=None
            if limit is None or "lower" not in limit.attrib
            else float(limit.attrib["lower"]),
            upper=None
            if limit is None or "upper" not in limit.attrib
            else float(limit.attrib["upper"]),
            mimic=None
            if mimic is None
            else (
                mimic.attrib["joint"],
                float(mimic.attrib.get("multiplier", 1.0)),
                float(mimic.attrib.get("offset", 0.0)),
            ),
        )
    return joints


def _urdf_subtree_links(urdf: dict[str, _UrdfJoint], root: str) -> set[str]:
    """Every link below ``root`` in the URDF kinematic tree."""
    children: dict[str, list[str]] = {}
    for j in urdf.values():
        children.setdefault(j.parent, []).append(j.child)
    out: set[str] = set()
    stack = [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _match_urdf_joint(j: Any, urdf: dict[str, _UrdfJoint], claimed: set[str]) -> str:
    """The URDF joint a manifest joint drives, by the URDF's own structure.

    In order: the manifest name itself; the URDF joint with the same
    ``(parent_link, child_link)``; ``sim_joint_name`` (when it is a URDF joint —
    it usually names the MuJoCo joint, so it is not trusted first); else the one
    unclaimed, movable, non-mimic URDF joint below the manifest ``parent_link``
    (a gripper whose manifest child is a logical link, e.g. ``panda_finger_pair``).

    Raises:
        ROSConfigError: no match, or an ambiguous subtree.
    """
    if j.name in urdf:
        return str(j.name)
    for u in urdf.values():
        if (u.parent, u.child) == (j.parent_link, j.child_link):
            return u.name
    if j.sim_joint_name and j.sim_joint_name in urdf:
        return str(j.sim_joint_name)
    below = _urdf_subtree_links(urdf, str(j.parent_link))
    candidates = [
        u.name
        for u in urdf.values()
        if u.child in below
        and u.joint_type != "fixed"
        and u.mimic is None
        and u.name not in claimed
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ROSConfigError(
        f"manifest joint {j.name!r} matches no URDF joint (by name, "
        f"({j.parent_link!r}, {j.child_link!r}), or sim_joint_name {j.sim_joint_name!r}); "
        f"movable URDF joints below {j.parent_link!r}: {candidates or 'none'}."
    )


def _gripper_spec(j: Any, leader: _UrdfJoint, urdf: dict[str, _UrdfJoint]) -> dict[str, Any]:
    """Open/closed targets (URDF units) + mimic followers for one manifest gripper.

    ``closed`` is the URDF limit nearer zero and ``open`` lies toward the other
    limit — by at most the manifest's own travel when that is the smaller (the
    hardware range: OpenArm's jaw spans 0.785 rad of the URDF's 1.571). A
    manifest range in other units (a normalised ``[0, 1]`` width) only ever
    shrinks nothing — the URDF travel wins. The manifest's own closed/open ends
    (nearer/farther from zero) map ``/joint_states`` back into manifest units.
    """
    if leader.lower is None or leader.upper is None:
        raise ROSConfigError(f"URDF gripper joint {leader.name!r} has no <limit>.")
    closed, far = sorted((leader.lower, leader.upper), key=abs)
    m_lo, m_hi = j.position_limits if j.position_limits is not None else (0.0, 1.0)
    m_closed, m_open = sorted((float(m_lo), float(m_hi)), key=abs)
    travel = min(abs(far - closed), abs(m_open - m_closed))
    opened = closed + math.copysign(travel, far - closed)
    return {
        "name": j.name,
        "leader": leader.name,
        "closed": closed,
        "open": opened,
        "manifest_closed": m_closed,
        "manifest_open": m_open,
        "followers": [
            {"dof": u.name, "multiplier": u.mimic[1], "offset": u.mimic[2]}
            for u in urdf.values()
            if u.mimic is not None and u.mimic[0] == leader.name
        ],
    }


# Public description packages fetched on demand when no sourced workspace or
# ancestor directory provides them: ``{package: "module:function"}`` returning
# the package root.
_PUBLIC_ROS_PACKAGES: dict[str, str] = {
    "openarm_description": "openral_hal._openarm_description_assets:ensure_openarm_description",
}


def _fetch_public_package(pkg: str) -> Path | None:
    """The root of a known public ``pkg`` (``_PUBLIC_ROS_PACKAGES``), else ``None``."""
    import importlib

    target = _PUBLIC_ROS_PACKAGES.get(pkg)
    if target is None:
        return None
    module, _, func = target.partition(":")
    return Path(getattr(importlib.import_module(module), func)())


def _ros_package_paths(urdf_path: Path) -> list[dict[str, str]]:
    """Resolve every ``package://<pkg>/`` root a URDF references.

    The standard ROS 2 lookup first (``<AMENT_PREFIX_PATH entry>/share/<pkg>``),
    so a URDF whose meshes live in a sourced workspace imports without copying
    them; else an ancestor directory of the URDF named ``<pkg>`` (a standalone
    description repo, e.g. the ``robot_descriptions`` cache); else a known
    public package fetched into the openral cache (``_PUBLIC_ROS_PACKAGES`` —
    e.g. Enactic's ``openarm_description``).

    Raises:
        ROSConfigError: a referenced package is found neither way.
    """
    import re

    pkgs = sorted(set(re.findall(r"package://([^/\"']+)/", urdf_path.read_text())))
    prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    out: list[dict[str, str]] = []
    for pkg in pkgs:
        candidates = [Path(p) / "share" / pkg for p in prefixes]
        candidates += [a for a in urdf_path.parents if a.name == pkg]
        share = next((c for c in candidates if c.is_dir()), None) or _fetch_public_package(pkg)
        if share is None:
            raise ROSConfigError(
                f"URDF {urdf_path} references package://{pkg}/, found on no "
                f"AMENT_PREFIX_PATH entry (share/{pkg}), as an ancestor directory, nor "
                "among the known public packages; source the ROS workspace that provides it."
            )
        out.append({"name": pkg, "path": str(share)})
    return out


def _build_robot_spec(desc: RobotDescription, robot_id: str) -> dict[str, Any]:
    """Marshal a ``RobotDescription`` to the JSON isaac robot spec.

    Resolves the manifest ``assets.urdf.ref`` to an on-disk file, maps every
    actuated manifest joint onto its URDF joint (``_match_urdf_joint`` — manifest
    names, URDF names and MuJoCo ``sim_joint_name`` all differ across robots),
    and carries the action contract, one entry per gripper (open/closed targets
    + mimic followers, ``_gripper_spec``), the ``package://`` roots the URDF
    needs, and the sensors. The planar base joints (``base_joints``) are not URDF
    DOFs; the sidecar drives the base kinematically.

    Action layout (``IsaacActionLayout``): ``[one absolute target per non-base
    joint in manifest order — grippers in manifest units — then the base twist
    (vx, vy, wz) when mobile]``; NaN holds a joint.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    if desc.assets.urdf is None:
        raise ROSConfigError(
            f"robot {robot_id!r} has no assets.urdf; the Isaac manifest scene "
            "(--layout manifest) imports the robot from its URDF."
        )
    # ``file:`` URDFs (the vendored arms) resolve against the robot's manifest
    # dir; ``rd:`` refs ignore it. The repo-root fallback in resolve_asset covers
    # a relative run cwd, so deriving robots/<robot_id>/ is a safe primary root.
    manifest_dir = Path("robots") / robot_id
    try:
        urdf_path = resolve_asset(desc.assets.urdf.ref, "urdf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(
            f"could not resolve assets.urdf.ref {desc.assets.urdf.ref!r} for {robot_id!r}: {exc}"
        ) from exc
    if urdf_path is None or not urdf_path.is_file():
        raise ROSConfigError(
            f"assets.urdf.ref {desc.assets.urdf.ref!r} for {robot_id!r} did not resolve to a file."
        )
    urdf_path = urdf_path.resolve()
    urdf = _parse_urdf_joints(urdf_path)
    base_joint_names = set(desc.base_joints or [])

    def _joint_type(j: Any) -> str:
        return getattr(j.joint_type, "value", str(j.joint_type))

    def _eff_role(j: Any) -> str:
        """``base`` (kinematic, not a URDF DOF) / ``gripper`` / otherwise ``arm``.

        The manifest's own ``role`` is advisory and often left ``unknown`` on
        arm joints (e.g. ``franka_panda`` tags only the gripper).
        """
        if j.name in base_joint_names:
            return "base"
        if j.role == "gripper":
            return "gripper"
        return "arm"

    # Keep ALL non-fixed manifest joints in order (base joints stay so the sidecar
    # can fill /joint_states from the kinematic base pose).
    joints = [j for j in desc.joints if _joint_type(j) != "fixed"]
    claimed: set[str] = set()
    entries: list[dict[str, Any]] = []
    grippers: list[dict[str, Any]] = []
    for j in joints:
        role = _eff_role(j)
        urdf_name = None
        if role != "base":
            urdf_name = _match_urdf_joint(j, urdf, claimed)
            claimed.add(urdf_name)
            if role == "gripper":
                grippers.append(_gripper_spec(j, urdf[urdf_name], urdf))
        entries.append(
            {"name": j.name, "role": role, "joint_type": _joint_type(j), "urdf_name": urdf_name}
        )
    arm_n = sum(1 for e in entries if e["role"] == "arm")
    has_base = bool(desc.base_joints)

    return {
        "robot_id": robot_id,
        "urdf_path": str(urdf_path),
        "ros_package_paths": _ros_package_paths(urdf_path),
        "base_frame": desc.base_frame,
        # The arm is always pinned to its (possibly moving) root: a fixed arm is
        # pinned to the world; a mobile base teleports that pinned root each step
        # (kinematic base). Either way fix_base=True keeps the arm from falling.
        "fix_base": True,
        "joints": entries,
        "grippers": grippers,
        "base_joints": desc.base_joints,
        "base_kinematics": desc.base_kinematics,
        "action": {
            # IsaacActionLayout: absolute targets per non-base joint in manifest
            # order (NaN = hold), then the base twist.
            "dim": arm_n + len(grippers) + (3 if has_base else 0),
            "control_mode": "joint_position",
            "has_base": has_base,
        },
        "sensors": [_sensor_dict(s) for s in desc.sensors],
    }


def _write_robot_spec(env_cfg: SimEnvironment) -> tuple[str, RobotDescription]:
    """Build the robot spec for ``env_cfg.robot_id`` and write it to a temp JSON.

    Returns the temp file path passed to the sidecar via ``--robot-spec`` (the
    caller unlinks it after connecting) and the robot's description.
    """
    from openral_sim.registry import ROBOTS

    robot_id = env_cfg.robot_id or "franka_panda"
    try:
        desc = ROBOTS.get(robot_id)()
    except KeyError as exc:
        raise ROSConfigError(
            f"unknown robot_id {robot_id!r} for the Isaac manifest scene; "
            "expected a robots/<id>/robot.yaml manifest."
        ) from exc
    spec = _build_robot_spec(desc, robot_id)
    fd, path = tempfile.mkstemp(prefix=f"isaac_robot_spec_{robot_id}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return path, desc


def provision_isaac_sim() -> None:
    """Build the Isaac sidecar venv — the slow half of a first run.

    ``_sidecar_python`` auto-provisions a multi-GB, RTX-only, license-gated
    install from NVIDIA's index when ``OPENRAL_ISAAC_AUTO_PROVISION=1``, and
    otherwise raises with the manual recipe. Either way that belongs in front
    of ``ros2 launch`` rather than inside the HAL's ``on_configure``, which
    ``tools/lifecycle_autostart.py`` bounds at 300 s: the download cannot
    finish inside the deadline, and the refusal error is far more legible on
    a terminal than as a lifecycle timeout.

    Idempotent — an existing venv short-circuits on its sentinel, so a warm
    host pays one ``is_file()``. ``_build_isaac_sim_scene`` resolves the
    same interpreter again on the build path.

    Raises:
        ROSConfigError: When the venv is absent and auto-provisioning is off,
            or when the NVIDIA-index install fails.
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("isaac_client")
    py = _sidecar_python()
    if _is_binary_install(py):
        _binary_site_dir(py)


def _placement(
    env_cfg: SimEnvironment, opts: IsaacSimOptions
) -> tuple[str, str | None, tuple[float, float, float, float], str, str]:
    """``(layout, environment_usd, spawn, objects_json, world_key)`` for a scene."""
    environment_usd = _resolve_environment_usd(env_cfg.scene.assets_uri)
    spawn = _spawn_pose(env_cfg.base_pose)
    objects_json = _objects_json(opts.objects) if opts.objects else ""
    layout = opts.layout
    world = _world_key(environment_usd, env_cfg.base_pose, objects_json)
    return layout, environment_usd, spawn, objects_json, world


@SCENES.register(
    _ISAAC_SCENE_ID,
    fixed_robot=None,
    provision=provision_isaac_sim,
    sim_clock=True,
    base_pose=True,
    options_model=IsaacSimOptions,
)
def _build_isaac_sim_scene(env_cfg: SimEnvironment) -> _IsaacSimSidecar:
    """Build an Isaac Lab scene behind the out-of-process sidecar.

    Lazy-imports pyzmq/msgpack (the openral-side wire) via the ``isaac_client``
    install plan, resolves the sidecar interpreter + script, and connects (auto-
    spawning the Isaac Lab process on first use).
    """
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("isaac_client")
    try:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in isaacsim group
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in isaacsim group
    except ImportError as exc:  # pragma: no cover — runtime-error path
        raise ROSConfigError(
            "isaac_sim backend needs pyzmq + msgpack on the openral venv: "
            "uv sync --all-packages --group isaacsim --inexact"
        ) from exc

    opts = SCENES.validate_options(env_cfg.scene.id, env_cfg.scene.backend_options)
    if not isinstance(opts, IsaacSimOptions):  # pragma: no cover — registry invariant
        raise ROSConfigError("isaac_sim backend options did not validate to IsaacSimOptions")
    host = opts.host
    timeout_ms = opts.timeout_ms
    boot_timeout_s = opts.boot_timeout_s
    layout, environment_usd, spawn, objects_json, world = _placement(env_cfg, opts)
    # Default to a per-scene port (no cross-scene sidecar reuse); an explicit
    # ``port`` in backend_options still wins.
    robot_id = env_cfg.robot_id or "franka_panda"
    port = opts.port or _scene_default_port(env_cfg.task.id, robot_id, layout, world)
    headless = opts.headless
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    sidecar_python = _sidecar_python()
    launch_argv = [
        str(sidecar_python),
        str(_locate_sidecar_script()),
        "--task",
        env_cfg.task.id,
        "--robot",
        env_cfg.robot_id or "franka_panda",
        "--instruction",
        env_cfg.task.instruction,
        "--layout",
        layout,
        "--obs-height",
        str(env_cfg.scene.observation_height),
        "--obs-width",
        str(env_cfg.scene.observation_width),
        "--max-steps",
        # A DeployScene (openral deploy sim) has no task, so build_sim_env_from_yaml
        # synthesises a noop task whose max_steps is None — fall back to a large
        # cap so the continuously-driven deploy env never truncates mid-run.
        str(env_cfg.task.max_steps if env_cfg.task.max_steps is not None else _DEFAULT_MAX_STEPS),
        "--success-key",
        env_cfg.task.success_key or "is_success",
        "--host",
        host,
        "--port",
        str(port),
    ]
    # Boot probe: the sidecar verifies these floors before Kit starts, so a venv
    # provisioned before a pin correction (or by hand from the recipe in
    # _sidecar_python's error) reports the actual versions in a second instead of
    # loading a CUDA library too old for Isaac's own prebundled one and hanging
    # for the whole boot_timeout_s (issue #89).
    # The floors guard the pip venv's own nvidia-* wheels; a binary install
    # bundles its CUDA runtime instead and gets its wire deps via --site-dir.
    if _is_binary_install(sidecar_python):
        launch_argv += ["--site-dir", str(_binary_site_dir(sidecar_python))]
    else:
        for dist, floor in _ISAAC_CUDA_FLOORS.items():
            launch_argv += ["--require-min", f"{dist}>={floor}"]
    if headless:
        launch_argv.append("--headless")

    # The scene imports the manifest robot's URDF. Marshal the RobotDescription
    # to a temp JSON the sidecar reads (it cannot import openral_core) and pass
    # it via --robot-spec.
    robot_spec_path, desc = _write_robot_spec(env_cfg)
    launch_argv += ["--robot-spec", robot_spec_path]
    if environment_usd is not None:
        launch_argv += ["--environment-usd", environment_usd]
    if env_cfg.base_pose is not None:
        launch_argv += ["--spawn-pose", *(repr(v) for v in spawn)]
    if objects_json:
        launch_argv += ["--objects-json", objects_json]

    client = SidecarClient(
        name="isaac",
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        boot_timeout_s=boot_timeout_s,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        # Reject (loudly) an already-running sidecar on this port that serves a
        # different scene, instead of silently adopting its wrong layout.
        expected_identity={
            "task": env_cfg.task.id,
            "layout": layout,
            "environment": environment_usd or "",
            "spawn": list(spawn),
        },
    )
    try:
        client.connect()
    finally:
        # The sidecar reads the spec once at boot (before it answers ping), so by
        # the time connect() returns or fails the temp file is consumed — unlink
        # it here rather than leaking it past process exit.
        with contextlib.suppress(OSError):
            os.unlink(robot_spec_path)
    return _IsaacSimSidecar(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _client=client,
        layout=IsaacActionLayout.from_description(desc),
    )
