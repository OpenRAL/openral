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
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        """THE slot layout — ``_build_robot_spec`` derives the sidecar's from it too.

        Every manifest joint, in manifest order (a slot row is zero-padded to
        all of them), with a role: ``fixed`` (no slot), ``base`` (a
        ``base_joints`` entry — twist-commanded, no slot), ``gripper`` (manifest
        role), else ``arm``.
        """
        base = set(desc.base_joints or [])

        def role(j: Any) -> str:
            if getattr(j.joint_type, "value", j.joint_type) == "fixed":
                return "fixed"
            if j.name in base:
                return "base"
            return "gripper" if j.role == "gripper" else "arm"

        return cls(
            joints=tuple(j.name for j in desc.joints),
            roles=tuple(role(j) for j in desc.joints),
            end_effectors=tuple(e.name for e in desc.end_effectors),
        )

    @property
    def slots(self) -> tuple[str, ...]:
        """Manifest joints that own an action slot (arm + gripper joints)."""
        return tuple(
            n for n, r in zip(self.joints, self.roles, strict=True) if r in ("arm", "gripper")
        )

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
    action: Action,
    layout: IsaacActionLayout,
    prev: NDArray[np.float32] | None,
    *,
    carry_twist: bool = False,
) -> NDArray[np.float32]:
    """One typed ``Action`` → the manifest scene's action vector (``IsaacActionLayout``).

    Addressed by name, never by position (ADR-0102): a JOINT_POSITION row writes
    the joints its ``joint_names`` list (the row zero-padded to full dof in
    manifest order), or — without names — a whole-vector row in manifest order
    (all joints, the non-base joints, or the arm joints alone); a
    GRIPPER_POSITION writes the gripper its ``ee_name`` resolves to; a
    BODY_TWIST writes ``(vx, vy, wz)``. Every other slot keeps ``prev`` (NaN =
    HOLD when nothing was commanded yet), except the twist, which starts at
    zero for every committed step — a velocity must not outlive the command
    that set it. ``carry_twist=True`` keeps ``prev``'s twist instead: ``prev``
    is then an earlier slot of the SAME step (a mobile-manipulation tick packs
    twist, arm and gripper slots into one vector). Base joints in a joint row
    are skipped: the kinematic base only takes a BODY_TWIST.

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
    if layout.has_base and not carry_twist:
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

    def pack_action(
        self,
        action: Action,
        prev: NDArray[np.float32] | None,
        *,
        carry_twist: bool = False,
    ) -> NDArray[np.float32]:
        """``SimAttachedHAL``'s packing hook: this env addresses its slots by name.

        See ``pack_isaac_action``. The HAL's default packer assumes the robosuite
        slot order (base first, one gripper last) and per-step deltas; this scene
        takes absolute targets in manifest order, base twist last, one slot per
        gripper — so the env, which knows its own layout, packs.
        """
        if self.layout is None:
            raise ROSConfigError("Isaac rollout built without an action layout.")
        return pack_isaac_action(action, self.layout, prev, carry_twist=carry_twist)

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
        # Registered colour + metric depth per depth camera (sidecar ``_depth_frames``):
        # the RGB-D streams a real driver publishes, for the vision attachment leg.
        frames = raw.get("depth_frames")
        if isinstance(frames, dict):
            obs["depth_frames"] = {
                str(k): {
                    "depth": np.asarray(v["depth"], dtype=np.float32),
                    "rgb": np.asarray(v["rgb"], dtype=np.uint8),
                    "k": np.asarray(v["k"], dtype=np.float64).reshape(4),
                    "optical_in_base": np.asarray(v["optical_in_base"], dtype=np.float64).reshape(
                        4, 4
                    ),
                }
                for k, v in frames.items()
                if isinstance(v, dict)
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


def _find_sidecar_python() -> Path | None:
    """An Isaac interpreter that already exists — never provisions anything.

    ``OPENRAL_ISAAC_SIDECAR_PYTHON`` (when it is a file), the default pip venv,
    or a binary install's ``python.sh``; else ``None``. For availability checks
    (the sim-test gate) that must not trigger a multi-GB install.
    """
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    default = _ISAAC_SIDECAR_HOME / ".venv" / "bin" / "python"
    if default.is_file():
        return default
    return next(
        (r / "python.sh" for r in _BINARY_INSTALL_ROOTS if (r / "python.sh").is_file()), None
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
    are fixed props. ``roll``/``pitch``/``yaw`` are radians about world x/y/z in the
    URDF ``rpy`` order (``Rz·Ry·Rx``): an asset authored lying down stands upright
    with a quarter-turn roll or pitch.

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
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    dynamic: bool = True


class IsaacCameraMount(BaseModel):
    """Where one manifest camera sits on the robot in the ``isaac_sim`` scene.

    Without a mount a camera gets the scene's generic base-relative viewpoint; with
    one it is a child of the named URDF ``link`` (``None``: the robot root) at
    ``xyz``, oriented by ``quat_wxyz`` (in Isaac's ``axes`` convention: ``world`` =
    x forward / z up, ``usd`` = -z forward / y up, ``ros`` = z forward / y down) or
    aimed at ``look_at`` (a point in the same link frame). So it rides the link:
    a wrist camera follows the wrist. The pose is a fact about the cell being
    simulated, so it lives in the scene, never in the robot manifest.

    Example:
        >>> IsaacCameraMount(link="openarm_base", xyz=(0, 0, 0.2), look_at=(0.4, 0, -0.5)).axes
        'world'
    """

    model_config = ConfigDict(extra="forbid")

    link: str | None = None
    xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    axes: Literal["world", "usd", "ros"] = "world"
    # Horizontal field of view; None: from the sensor's manifest intrinsics.
    hfov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)

    @model_validator(mode="after")
    def _one_orientation(self) -> IsaacCameraMount:
        """Exactly one of ``quat_wxyz`` / ``look_at``; ``look_at`` aims world axes."""
        if (self.quat_wxyz is None) == (self.look_at is None):
            raise ValueError("camera mount: give exactly one of quat_wxyz or look_at")
        if self.look_at is not None and self.axes != "world":
            raise ValueError("camera mount: look_at aims the camera's x axis (axes: world)")
        return self


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
    # Per-sensor camera mounts (manifest sensor name -> mount); see IsaacCameraMount.
    camera_mounts: dict[str, IsaacCameraMount] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _object_names_are_free(self) -> IsaacSimOptions:
        """Reject duplicate or reserved object names at load, not mid-boot.

        Names key ``/World/objects/<name>`` and the World's registry, which
        already holds the scene's own ``robot`` / ``obstacle_<i>``.
        """
        names = [o.name for o in self.objects]
        dupes = sorted({n for n in names if names.count(n) > 1})
        reserved = [n for n in names if n == "robot" or re.fullmatch(r"obstacle_\d+", n)]
        if dupes or reserved:
            raise ValueError(
                f"objects: duplicate names {dupes} / reserved names {reserved} "
                "(the scene registers 'robot' and 'obstacle_<i>' itself)."
            )
        return self


# ── environment + spawn (manifest layout) ──

_USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
# Forwarded verbatim: the sidecar (Kit's omni.client) resolves these itself.
_REMOTE_USD_PREFIXES = ("isaac:", "http://", "https://", "omniverse://")
# |sin(roll/2)|, |sin(pitch/2)| above this reads as a tilted placement.
_PLANAR_QUAT_TOL = 1e-4


def _resolve_environment_usd(assets_uri: str | None, field: str = "scene.assets_uri") -> str | None:
    """Validate ``scene.assets_uri`` (or an object's ``usd``) as a USD for the sidecar.

    Remote forms (``isaac:``, ``http(s)://``, ``omniverse://``) pass through; a
    local path (optionally ``file://``) must exist — a typo fails here, not
    minutes into a Kit boot. A relative path resolves against the repository
    root first (the directory holding ``scenes/``, as the scene YAML itself is
    resolved — the HAL node's working directory is arbitrary), then the
    working directory.

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
    raw = Path(assets_uri.removeprefix("file://")).expanduser()
    candidates = [raw] if raw.is_absolute() else [*(_repo_roots()), Path.cwd()]
    tried = [(c / raw if not raw.is_absolute() else raw).resolve() for c in candidates]
    path = next((t for t in tried if t.is_file()), None)
    if path is None:
        raise ROSConfigError(
            f"{field} {assets_uri!r}: no such file (tried {', '.join(map(str, tried))})."
        )
    return str(path)


def _repo_roots() -> list[Path]:
    """The checkout this module lives in (the ancestor holding ``scenes/``), if any."""
    return [p for p in Path(__file__).resolve().parents if (p / "scenes").is_dir()][:1]


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


def _camera_mounts_json(mounts: dict[str, IsaacCameraMount]) -> str:
    """The mounts as stable JSON (``""`` when none): part of the sidecar identity."""
    return json.dumps({k: m.model_dump() for k, m in sorted(mounts.items())}) if mounts else ""


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


_LIMIT_EPS = 1e-6


# Command value at (closed, open) for each normalised ``GripperConvention``; the
# scene maps a command linearly between them onto the URDF closed/open targets.
_CONVENTION_ENDS: dict[str, tuple[float, float]] = {
    "normalized_open_unit": (0.0, 1.0),
    "normalized_open_symmetric": (-1.0, 1.0),
    "normalized_close_symmetric": (1.0, -1.0),
    "binary_close_one": (1.0, 0.0),
}


def _gripper_spec(
    j: Any,
    leader: _UrdfJoint,
    urdf: dict[str, _UrdfJoint],
    *,
    passthrough: bool,
    command_convention: str | None = None,
) -> dict[str, Any]:
    """Open/closed targets (URDF units) + mimic followers for one manifest gripper.

    The manifest's gripper unit is declared, not guessed
    (``SimDescription.grippers[].write_mode``):

    * ``passthrough`` — ``position_limits`` are physical joint values (OpenArm's
      jaw, 0..0.785 rad) on the URDF joint's own axis, written unchanged:
      closed/open are the manifest ends nearer/farther from zero. A manifest
      range outside the URDF limits is a ``ROSConfigError``: it means the two
      disagree on the axis, and negating to fit would drive the jaw the wrong
      way (an inverted OpenArm left jaw swings its fingers through each other).
    * ``normalised`` (the default, or no declaration) — ``[0 = closed, 1 =
      open]`` is a fraction of the URDF's full travel: closed is the zero pose
      (clamped into the limits), open the limit farther from it (SO-100's jaw:
      0 → 2.0 rad; the Panda's fingers: 0 → 0.04 m).

    The manifest's own closed/open ends map ``/joint_states`` back. A command is
    read in the driving end effector's ``command_convention``
    (``command_closed`` / ``command_open``): a normalised one by its own ends
    (``normalized_close_symmetric``: +1 closed, -1 open), ``raw_joint_rad`` (or
    no declaration) as the manifest joint value itself. Any other convention
    (``width_meters``) has no mapping onto the joint and is a ``ROSConfigError``.
    """
    if leader.lower is None or leader.upper is None:
        raise ROSConfigError(f"URDF gripper joint {leader.name!r} has no <limit>.")
    lo, hi = float(leader.lower), float(leader.upper)
    m_lo, m_hi = j.position_limits if j.position_limits is not None else (0.0, 1.0)
    m_closed, m_open = sorted((float(m_lo), float(m_hi)), key=abs)

    def inside(v: float) -> bool:
        return lo - _LIMIT_EPS <= v <= hi + _LIMIT_EPS

    if passthrough:
        if not (inside(m_open) and inside(m_closed)):
            raise ROSConfigError(
                f"passthrough gripper {j.name!r}: manifest range ({m_lo}, {m_hi}) lies "
                f"outside URDF joint {leader.name!r} limits ({lo}, {hi}); the manifest and "
                "the URDF disagree on the jaw's axis."
            )
        closed, opened = m_closed, m_open
    else:
        closed = min(max(0.0, lo), hi)
        opened = hi if abs(hi - closed) >= abs(lo - closed) else lo
    if command_convention in _CONVENTION_ENDS:
        cmd_closed, cmd_open = _CONVENTION_ENDS[command_convention]
    elif command_convention is None or (command_convention == "raw_joint_rad" and passthrough):
        cmd_closed, cmd_open = m_closed, m_open
    else:
        raise ROSConfigError(
            f"gripper {j.name!r}: command_convention {command_convention!r} has no mapping "
            f"onto its {'passthrough' if passthrough else 'normalised'} joint in the Isaac "
            f"scene (raw_joint_rad needs write_mode: passthrough)."
        )
    return {
        "name": j.name,
        "leader": leader.name,
        "closed": closed,
        "open": opened,
        "manifest_closed": m_closed,
        "manifest_open": m_open,
        "command_closed": cmd_closed,
        "command_open": cmd_open,
        "followers": [
            {"dof": u.name, "multiplier": u.mimic[1], "offset": u.mimic[2]}
            for u in urdf.values()
            if u.mimic is not None and u.mimic[0] == leader.name
        ],
    }


def _fetch_public_package(pkg: str) -> Path | None:
    """The root of a known public ``pkg`` (``openral_hal.ros_package_overlay``), else ``None``."""
    # Lazy: openral_hal's package import pulls torch/lerobot.
    from openral_hal.ros_package_overlay import PUBLIC_ROS_PACKAGES, fetch_public_package

    if pkg not in PUBLIC_ROS_PACKAGES:
        return None
    # A fallback, said out loud (CLAUDE.md §1.4): the first use clones a pinned
    # public repo into the openral cache.
    print(
        f"[isaac-sim] package://{pkg}/ is on no AMENT_PREFIX_PATH entry and not "
        "beside the URDF; using its pinned public clone.",
        flush=True,
    )
    return fetch_public_package(pkg)


def _ros_package_paths(urdf_path: Path) -> list[dict[str, str]]:
    """Resolve every ``package://<pkg>/`` root a URDF references.

    The standard ROS 2 lookup first (``<AMENT_PREFIX_PATH entry>/share/<pkg>``),
    so a URDF whose meshes live in a sourced workspace imports without copying
    them; else an ancestor directory of the URDF named ``<pkg>`` (a standalone
    description repo, e.g. the ``robot_descriptions`` cache); else a known
    public package fetched into the openral cache
    (``openral_hal.ros_package_overlay.PUBLIC_ROS_PACKAGES`` — e.g. Enactic's
    ``openarm_description``).

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


def _root_to_base(desc: RobotDescription) -> NDArray[np.float64]:
    """4x4 pose of the manifest ``base_frame`` in the URDF root frame.

    Identity unless the manifest's URDF declares ``base_to_root_xyz_rpy`` (the
    ``base_frame`` -> URDF-root transform), which is inverted here: OpenArm's
    ``openarm_base`` is no URDF link but sits 0.698 m above the URDF root ``world``.

    Example:
        >>> from openral_sim.registry import ROBOTS
        >>> _root_to_base(ROBOTS.get("openarm")())[:3, 3].round(3).tolist()
        [0.0, 0.0, 0.698]
    """
    urdf = getattr(desc.assets, "urdf", None) if desc.assets is not None else None
    xyzrpy = getattr(urdf, "base_to_root_xyz_rpy", None)
    if xyzrpy is None:
        return np.eye(4)
    x, y, z, roll, pitch, yaw = (float(v) for v in xyzrpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    base_to_root = np.eye(4)
    base_to_root[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    base_to_root[:3, 3] = (x, y, z)
    return np.asarray(np.linalg.inv(base_to_root), dtype=np.float64)


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
    joint in manifest order — grippers in their end effector's command_convention —
    then the base twist (vx, vy, wz) when mobile]``; NaN holds a joint.
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
    # One derivation of the slot layout (roles + order), shared with the HAL's
    # packer (IsaacActionLayout.from_description) so the two cannot drift.
    layout = IsaacActionLayout.from_description(desc)
    passthrough = {
        g.joint
        for g in ((desc.sim.grippers if desc.sim else None) or [])
        if getattr(g.write_mode, "value", g.write_mode) == "passthrough"
    }
    # The encoding each gripper joint's commands arrive in: its end effector's.
    conventions: dict[str, str] = {}
    for ee in desc.end_effectors:
        if ee.command_convention is not None:
            conventions[layout.gripper_for(ee.name)] = ee.command_convention
    by_name = {j.name: j for j in desc.joints}
    claimed: set[str] = set()
    entries: list[dict[str, Any]] = []
    grippers: list[dict[str, Any]] = []
    for name, role in zip(layout.joints, layout.roles, strict=True):
        j = by_name[name]
        urdf_name = None
        if role in ("arm", "gripper"):
            urdf_name = _match_urdf_joint(j, urdf, claimed)
            claimed.add(urdf_name)
            if role == "gripper":
                grippers.append(
                    _gripper_spec(
                        j,
                        urdf[urdf_name],
                        urdf,
                        passthrough=name in passthrough,
                        command_convention=conventions.get(name),
                    )
                )
        entries.append(
            {
                "name": name,
                "role": role,
                "joint_type": getattr(j.joint_type, "value", str(j.joint_type)),
                "urdf_name": urdf_name,
            }
        )
    has_base = layout.has_base

    return {
        "robot_id": robot_id,
        "urdf_path": str(urdf_path),
        "ros_package_paths": _ros_package_paths(urdf_path),
        "base_frame": desc.base_frame,
        "root_to_base": _root_to_base(desc).tolist(),
        # The arm is always pinned to its (possibly moving) root: a fixed arm is
        # pinned to the world; a mobile base teleports that pinned root each step
        # (kinematic base). Either way fix_base=True keeps the arm from falling.
        "fix_base": True,
        "joints": entries,
        "grippers": grippers,
        "base_joints": desc.base_joints,
        "base_kinematics": desc.base_kinematics,
        "action": {
            # IsaacActionLayout: absolute targets per arm/gripper joint in
            # manifest order (NaN = hold), then the base twist.
            "dim": layout.dim,
            "control_mode": "joint_position",
            "has_base": has_base,
        },
        "sensors": [_sensor_dict(s) for s in desc.sensors],
    }


def _write_robot_spec(
    env_cfg: SimEnvironment, camera_mounts: dict[str, IsaacCameraMount] | None = None
) -> tuple[str, RobotDescription]:
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
    cameras = {s.name for s in desc.sensors if s.modality in ("rgb", "depth")}
    # A robot-framed camera with no mount renders a generic base-relative viewpoint,
    # not its frame's view (on Spark an unmounted wrist_right came out all-black).
    # Say so (CLAUDE.md §1.4).
    unmounted = sorted(
        s.name
        for s in desc.sensors
        if s.name in cameras and s.frame_id != "world" and s.name not in (camera_mounts or {})
    )
    if unmounted:
        print(
            f"[isaac-sim] cameras {unmounted} have no backend_options.camera_mounts entry: "
            "each renders a generic base-relative viewpoint, not the view from its frame_id.",
            flush=True,
        )
    if camera_mounts:
        unknown = sorted(set(camera_mounts) - cameras)
        if unknown:
            raise ROSConfigError(
                f"backend_options.camera_mounts names {unknown}, which are not "
                f"{robot_id!r} camera sensors (have {sorted(cameras)})."
            )
        spec["camera_mounts"] = {k: m.model_dump() for k, m in camera_mounts.items()}
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
    mounts = _camera_mounts_json(opts.camera_mounts)
    world = _world_key(environment_usd, env_cfg.base_pose, objects_json + mounts)
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
    spawn_args = [f"{v:.9f}" for v in spawn]
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
    robot_spec_path, desc = _write_robot_spec(env_cfg, opts.camera_mounts)
    launch_argv += ["--robot-spec", robot_spec_path]
    if environment_usd is not None:
        launch_argv += ["--environment-usd", environment_usd]
    if env_cfg.base_pose is not None:
        # Fixed-point, never repr(): argparse reads "-1e-05" as an option flag
        # (Python <= 3.13), so a tiny negative coordinate or yaw broke the boot.
        launch_argv += ["--spawn-pose", *spawn_args]
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
            # What the sidecar parses back from --spawn-pose, exactly.
            "spawn": [float(v) for v in spawn_args],
            "robot": robot_id,
            "objects": objects_json,
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
