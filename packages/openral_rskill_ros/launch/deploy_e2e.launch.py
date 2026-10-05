r"""Generic end-to-end ROS graph for ``openral deploy sim``.

One launch file for every robot. Every robot-specific bit is a launch argument resolved inside
an ``OpaqueFunction`` so concrete strings reach ``LifecycleNode(package=, executable=, name=)``:

* ``robot_yaml`` — RobotDescription manifest path, loaded via Pydantic at launch time; the
  safety kernel envelope is synthesised from it
  (``openral_safety.envelope_loader.compute_intersection``) and forwarded as ROS parameters on
  the kernel node. No envelope YAML file is written or read.
* ``hal_package`` / ``hal_executable`` / ``hal_node_name`` — HAL spawn, derived by
  ``openral deploy sim|run``: always the one manifest-driven ``openral_hal_node``
  package, run under the node NAME ``openral_hal_<robot_id>``.
* ``hal_params_file`` — ephemeral ROS parameter YAML the CLI writes with the HAL's per-robot
  knobs (``/**`` wildcard).
* ``reset_to_pose_service``, ``dashboard_port``, ``reasoner_model``, ``reasoner_endpoint`` —
  shared knobs.

Spawned processes: dashboard + safety_kernel + the three independent E-stop sources
(deadman_watchdog + hardware_estop + human_estop forwarder) + runtime + reasoner +
prompt_router + HAL.
Lifecycle nodes auto-transition UNCONFIGURED → INACTIVE → ACTIVE.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import site
import subprocess
import sys
import tempfile
import uuid

# `ros2 launch` runs under the system Python by default; the launch's
# deferred imports (openral_core, openral_safety) live in the OpenRAL
# workspace venv. ``openral deploy sim`` exports OPENRAL_VENV_SITE pointing
# at that venv's site-packages — process its ``.pth`` files via
# ``site.addsitedir`` so editable installs become importable. Setting
# PYTHONPATH alone is not enough: ``.pth`` files are only processed by
# the ``site`` module on registered site-dirs.
_VENV_SITE = os.environ.get("OPENRAL_VENV_SITE")
if _VENV_SITE and os.path.isdir(_VENV_SITE):
    site.addsitedir(_VENV_SITE)

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.events.matchers import matches_node_name as _matches_node_name

if TYPE_CHECKING:
    # openral_core stays a DEFERRED import at runtime (see the note above):
    # this file must import on a host without the OpenRAL workspace sourced.
    # `from __future__ import annotations` keeps the annotation a string.
    from openral_core import RobotDescription, SensorSpec
from lifecycle_msgs.msg import Transition
from openral_core import (
    CameraTopicKind,
    DeployRuntime,
    apply_sensor_overlays,
    camera_topic,
    deploy_cloud_topic,
    merge_deploy_sensors,
    publishing_sensors,
    resolve_sensor_overlays,
)
from openral_foxglove_bringup.topics import (
    ASSET_URI_ALLOWLIST,
    BUCKET1_TOPIC_WHITELIST,
    READ_ONLY_CAPABILITIES,
)


def _resolve_repo_root() -> pathlib.Path:
    """Locate the repo root from wherever this launch file is running.

    ``parents[3]`` is correct only from the source tree
    (``<repo>/packages/openral_rskill_ros/launch/``). ``ros2 launch`` resolves
    the file through the ament index, so it normally runs from the *installed*
    copy at ``<repo>/install/share/openral_rskill_ros/launch/`` — where the same
    index lands on ``<repo>/install`` and every path built from it
    (``tools/lifecycle_autostart.py``, ``rskills/``, ``.venv/bin/openral``)
    points at a directory that does not exist. A `--symlink-install` layout
    resolves back through the symlink to the source tree and hides this, which
    is why it survived: it only bites on a copied install, i.e. after a clean
    build.

    The consequence was not a launch failure. The graph came up and the
    per-node autostart processes died individually with exit code 2 (python:
    no such file), leaving the safety kernel and the reasoner parked
    unconfigured while every other node reported healthy.

    Reuses ``openral_rskill.loader.find_repo_root_from`` — the repo already
    has this search (``pyproject.toml`` + ``rskills/``), and a second copy here
    would be a second thing to keep true. Falls back to the old arithmetic when
    no marked ancestor exists, which is the genuinely-installed-elsewhere case
    where none of these repo-relative paths are meaningful anyway.
    """
    here = pathlib.Path(__file__).resolve()
    try:
        from openral_rskill.loader import find_repo_root_from
    except ImportError:  # pragma: no cover  # reason: workspace not on the path
        return here.parents[3]
    return find_repo_root_from(here) or here.parents[3]


_REPO_ROOT = _resolve_repo_root()
_RSKILLS_DIR = str(_REPO_ROOT / "rskills")

_VENV_RAL = _REPO_ROOT / ".venv" / "bin" / "openral"
_RAL_EXECUTABLE = str(_VENV_RAL) if _VENV_RAL.exists() else "openral"


def _resolve_git_sha() -> str:
    """Short git SHA for the ``openral.run.git_sha`` resource attribute.

    Prefers the CI/env spellings the rest of OpenRAL honours
    (``OPENRAL_GIT_SHA`` / ``GIT_SHA`` / ``GITHUB_SHA``), falling back to
    ``git rev-parse`` against the repo. Returns ``"unknown"`` when nothing
    resolves so the dashboard shows a value rather than a blank cell.
    """
    for env in ("OPENRAL_GIT_SHA", "GIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(env)
        if value:
            return value[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _run_resource_attrs(hal_mode: str) -> str:
    """Build the ``OTEL_RESOURCE_ATTRIBUTES`` value for every launched node.

    The dashboard's Identity card reads ``openral.run.{id,mode,git_sha}`` from
    OTLP **resource** attributes (TelemetryStore.ingest_spans). Setting them
    here — shared by every node in the graph via ``otel_env`` — makes the OTel
    SDK's ``OTELResourceDetector`` fold them into each node's Resource
    automatically (``Resource.create`` merges ``OTEL_RESOURCE_ATTRIBUTES``).
    ``hal_mode=="real"`` is the ``deploy run`` hardware path; everything else
    (``deploy sim``) is a simulation run.
    """
    run_mode = "hardware" if hal_mode == "real" else "sim"
    attrs = {
        "openral.run.id": uuid.uuid4().hex,
        "openral.run.mode": run_mode,
        "openral.run.git_sha": _resolve_git_sha(),
    }
    return ",".join(f"{k}={v}" for k, v in attrs.items())


def _world_voxel_margin_m(hal_mode: str) -> float:
    """Return the calibrated world-voxel clearance for this boundary.

    The real value is owned by ``openral_core.depth_extrinsic``, whose extrinsic pass
    limits are derived from it: a margin change re-tightens the calibration gate.
    """
    from openral_core.depth_extrinsic import REAL_WORLD_VOXEL_MARGIN_M

    return 0.0 if hal_mode == "sim" else REAL_WORLD_VOXEL_MARGIN_M


def _cpuset_prefix(env: str) -> str:
    """A ``taskset`` launch prefix from a cpuset env var, or ``""`` for none.

    ``OPENRAL_PERCEPTION_CPUSET`` pins ``octomap_server`` and the voxel bridge;
    ``OPENRAL_RUNTIME_CPUSET`` pins the runtime node (inference and its 750 Hz
    joint-state ingest). Both are unset by default: nothing is pinned and the
    graph runs as before. The seam exists because a deploy host has no
    privilege to raise priorities (``ulimit -e`` 0, no ``sudo`` on Thor), while
    affinity needs none, and the octree stalls of 1.1-1.4 s measured on Thor
    (2026-09-24) came from CPU contention with the runtime, not from the
    camera. A value is a ``taskset -c`` list (``"12,13"``, ``"0-9"``); an
    unparseable one is refused loudly rather than pinning to a set nobody
    chose.
    """
    raw = os.environ.get(env, "").strip()
    if not raw:
        return ""
    if not re.fullmatch(r"[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*", raw):
        raise RuntimeError(f"{env}={raw!r} is not a taskset cpu list (e.g. '12,13' or '0-9')")
    return f"taskset -c {raw} "


def _collision_scale_params() -> dict[str, float]:
    """Kernel overrides for distance-graded velocity scaling (#188), if asked.

    Returns ``{}`` — the kernel's own defaults, which disable the band and
    reproduce the pre-#188 republish exactly — unless
    ``OPENRAL_COLLISION_SCALE_PROXIMITY_M`` is set. That env var is the seam
    the A/B five-round battery flips, so the band can be swept across rounds
    without a rebuild and without editing this file mid-battery.

    An unparseable value yields no override and says so. It must never be read
    as some other band width: silently arming a safety mechanism at a number
    nobody chose is worse than not arming it.
    """
    band = os.environ.get("OPENRAL_COLLISION_SCALE_PROXIMITY_M")
    if not band:
        return {}
    params: dict[str, float] = {}
    for env, param in (
        ("OPENRAL_COLLISION_SCALE_PROXIMITY_M", "collision_scale_proximity_m"),
        ("OPENRAL_COLLISION_SCALE_K", "collision_scale_k"),
        ("OPENRAL_COLLISION_SCALE_MIN", "collision_scale_min"),
    ):
        raw = os.environ.get(env)
        if not raw:
            continue
        try:
            params[param] = float(raw)
        except ValueError:
            print(f"openral: ignoring unparseable {env}={raw!r}")
    return params


def _octomap_occupancy_threshold(hal_mode: str) -> float:
    """Require repeated simulated hits while preserving real-map behavior."""
    return 0.8 if hal_mode == "sim" else 0.6


def _octomap_clamping_max(hal_mode: str) -> float:
    """Let one exact simulated clearing ray remove a confirmed hit."""
    return 0.85 if hal_mode == "sim" else 0.97


def _octomap_resolution(hal_mode: str) -> float:
    """Use manipulation-scale voxels in sim without changing real maps.

    ``OPENRAL_OCTOMAP_RESOLUTION_M`` overrides it, for the resolution battery in
    the programme note §5 (``docs/reference/collision-validation-evidence.md``).
    Same mechanism as the #188 graded band's ``OPENRAL_COLLISION_SCALE_*``: an
    env var the launch reads and ``validation_matrix.py`` records, so a round
    that changed it can never be mistaken afterwards for one that did not.

    **A finer grid is LESS conservative**, not more: the cell half-diagonal is
    the kernel's quantisation term, so shrinking it makes the kernel stop later
    and nearer. Sim ships **15 mm** as of #253's ruling; real hardware stays at
    50 mm.

    The sim evidence: replayed on the real kernel over identical payload poses,
    `PickPlaceCounterToSink` (the DOP-only, field-typical payload) goes from
    **154** false stops of 182 genuinely-clear poses at 25 mm with a
    box-lowered payload to **15** at 15 mm with #266's refinement -- 90% fewer,
    with all 7 real contacts still caught. Neither lever alone is worth much
    (12% and 10%); the terms ADD, so both have to move. See
    `docs/reference/collision-validation-evidence.md` and hazard-log Entry 027.

    **Real hardware ships 20 mm on that same sim evidence, and nothing else.**
    No real map has been measured at any resolution. Sim rasterises a kitchen's
    own solid geometry; a real map comes from a depth camera and is dilated by
    the octree->grid bridge, and `docs/reference/world-map-fidelity.md` is
    explicit that the live map stops MORE often than the rasterised one, never
    less. The quantisation term is also not the only thing between the payload
    and reality on hardware -- depth noise, extrinsics and the octree dilation
    sit beside it, and 50 -> 20 mm removes 25.98 mm of guaranteed margin while
    leaving those untouched. Recorded as a deliberate reduction in
    conservatism, gated on the safety-WG, in its own hazard-log entry. It
    should be re-measured on a real map before it is relied on.

    20 mm on hardware and 15 mm in sim, not one value, because the two maps
    have different error budgets and the sim one is the only one with numbers.

    Cell counts against `octree_to_grid.cpp`'s `kMaxCells` (4 000 000): sim
    15 mm is 141/axis = 2 803 221 (70% of the guard, and the finest value that
    ships without raising it -- 12.5 mm needs 4 826 809 and is refused); real
    20 mm is 106/axis = 1 191 016.
    """
    override = os.environ.get("OPENRAL_OCTOMAP_RESOLUTION_M", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        # A resolution the lattice cannot place is not an experiment, it is a
        # silently empty grid. Fall through to the shipped default.
        if 0.001 <= value <= 0.5:
            return value
    return 0.015 if hal_mode == "sim" else 0.02


def _octomap_frames(description: RobotDescription) -> tuple[str, str]:
    """The ``(fixed_frame, base_frame)`` the octomap leg should map in.

    ``octomap_server`` accumulates its octree in a frame that must not move under the robot, and
    ``octomap_voxel_bridge`` / ``WorldCloudBridge`` express the result in the robot's own base
    frame. The MOBILE-BASE convention (``"odom"`` / ``"base_link"``) holds only while something
    publishes odometry — a fixed-base arm has no odometry and no ``odom`` frame, so its
    world-fixed ``base_frame`` is the correct accumulation frame instead. Getting this wrong is
    silent: every node comes up healthy and each cloud is dropped on a TF lookup, leaving an
    empty map and an empty dashboard card.

    Returns the manifest's own frames, so a robot that names them differently (``pelvis``,
    ``panda_link0``, ``openarm_base``) is honoured rather than assumed.
    """
    base_frame = description.base_frame
    locomotion = getattr(description.capabilities, "locomotion", None) or ["none"]
    mobile = any(kind != "none" for kind in locomotion)
    return (description.odom_frame if mobile else base_frame), base_frame


def _world_voxel_max_cells(radius_m: float, resolution_m: float) -> int:
    """Cells the published coverage ball of ``radius_m`` needs at ``resolution_m``.

    This was written out by hand as ``614125`` with a comment explaining it was
    ``85^3``, the worst case for a 1.05 m ball at 25 mm cells including the one
    cell per axis the lattice snap can add. A derived constant kept by hand is
    exactly what goes wrong when the resolution moves: at 15 mm the ball needs
    141^3 = 2 803 221 cells, and a kernel still reserving 614 125 rejects every
    grid it is sent -- which reads as "no world" and is a fail-*open* on the
    world check.

    The bridge (``octree_to_grid.cpp::build_lattice``) snaps the ball's bounding
    box onto the octree's cells: per axis it needs ``floor(2r/res + 1/2) + 1``
    cells at most (the half cell is the snap). Now that the radius is per robot
    and arbitrary, ``int(2r/res) + 1`` would undercount by one layer whenever the
    fraction is at least a half, so the snap term is explicit (plus a hair of
    float slack: over-reserving costs memory, under-reserving is fail-open).
    """
    per_axis = math.floor(2.0 * radius_m / resolution_m + 0.5 + 1e-9) + 1
    return per_axis**3


# The per-rig ``DeployRuntime`` fields ``openral deploy`` forwards as launch args.
_RIG_LAUNCH_ARGS = (
    "world_voxel_deadline_s",
    "max_octree_age_s",
    "world_voxel_data_age_budget_s",
    "robot_self_filter_padding_m",
)


def _rig_from_launch_args(raw: dict[str, str]) -> DeployRuntime:
    """The per-rig ``DeployRuntime`` values from their launch args (empty = schema default).

    The kernel's voxel deadline is how long it trusts the last ``/openral/world_voxels``
    grid; the bridge's bound is how long it republishes the last octree
    (``octomap_server`` publishes only when it inserts a cloud, so a dead camera is a
    silent octree). Past both, the kernel drops with ``DROP_VOXEL_UNAVAILABLE`` (hazard
    log Entry 033). The data-age budget bounds the age of the world behind a grid, from
    its ``source_stamp`` (Entry 034). Validated through ``DeployRuntime``, which refuses
    an octree age above the deadline, so the kernel -- not the bridge -- fails closed
    whoever launches this file.

    Example:
        >>> _rig_from_launch_args({}).voxel_freshness_s
        (1.0, 1.0)
        >>> rig = _rig_from_launch_args({"world_voxel_deadline_s": "2.0", "max_octree_age_s": ""})
        >>> rig.voxel_freshness_s, rig.world_voxel_data_age_budget_s
        ((2.0, 2.0), 1.5)
    """
    return DeployRuntime.model_validate({k: float(v) for k, v in raw.items() if v.strip()})


# Where the filtered cloud is published for octomap_server. Not a camera topic
# (ADR-0108): it is not one camera's cloud, it is the world map's input.
_SELF_FILTERED_CLOUD_TOPIC = "/openral/world_cloud/self_filtered"


def _self_filter_params(
    collision_params: dict[str, object],
    description: object,
    joint_states_topic: str,
    padding_m: float,
) -> dict[str, object]:
    """Parameters for ``openral_octomap_bridge``'s ``robot_self_filter``.

    The filter poses the SAME collision model the kernel checks (every
    ``collision_*`` key the kernel receives), indexed by the manifest's joint
    names, with each joint's ``sim_joint_name`` as an alias so a vendor
    ``/joint_states`` that spells joints the upstream way still matches.
    ``joint_states_topic`` is the one the runtime nodes read
    (``hal_joint_states_topic``: the scene's override, else a real
    ros2_control HAL's rate-limited ``~/joint_states``, else ``/joint_states``),
    so the filter poses the robot from the same stream the runner acts on.
    ``padding_m`` is the rig's ``DeployRuntime.robot_self_filter_padding_m``.
    """
    joints = list(getattr(description, "joints", []))
    params: dict[str, object] = {
        k: v for k, v in collision_params.items() if k.startswith("collision_")
    }
    params["collision_joint_names"] = [j.name for j in joints]
    params["collision_joint_aliases"] = [j.sim_joint_name or "" for j in joints]
    params["joint_states_topic"] = joint_states_topic
    params["padding_m"] = padding_m
    return params


# The octomap bridge's allocation guard (``octree_to_grid.cpp``, ``kMaxCells``): a ball
# needing more cells is REFUSED by the bridge, i.e. published as an empty grid -- every
# obstacle dropped. ``_coverage_ball`` refuses such a ball at launch instead.
_BRIDGE_MAX_CELLS = 4_000_000

#: Margin of the coverage ball over the measured reach: the kernel's real
#: ``world_voxel_margin_m`` (20 mm) plus half a 20 mm cell (the window it scans around
#: each link), plus 20 mm for the measured maximum falling short of the true one
#: (``test_deploy_e2e_coverage_ball.py`` bounds that shortfall well inside it).
_COVERAGE_MARGIN_M = 0.05
#: Reach measurement (``_ReachModel.maximise``): random joint-space samples, then the best
#: few refined by coordinate ascent, one exhaustive 1-D grid per joint per round.
_COVERAGE_SAMPLES = 4096
_COVERAGE_REFINE_SEEDS = 8
_COVERAGE_REFINE_ROUNDS = 3
_COVERAGE_REFINE_GRID = 49
_COVERAGE_SEED = 0
#: The ball for a robot with no collision model, whose map feeds no kernel check (the
#: kernel's voxel check needs the model): the pre-per-robot default.
_UNCHECKED_COVERAGE = ((0.0, 0.0, 0.5), 1.05)


class _CoverageBall(NamedTuple):
    """The world map's coverage, in ``base_frame``: a ball, and the octomap input clip.

    ``centre`` / ``radius`` is the ball the voxel bridge publishes and the kernel checks
    against; ``clip_lo`` / ``clip_hi`` is the measured reach box plus the margin, which
    lies inside the ball's bounding box: a return outside it can never be in any link's
    check window, so octomap does not integrate it.
    """

    centre: tuple[float, float, float]
    radius: float
    clip_lo: tuple[float, float, float]
    clip_hi: tuple[float, float, float]


def _xyzrpy_to_matrices(xyzrpy: np.ndarray) -> np.ndarray:
    """``(K, 6)`` xyz + URDF rpy (``Rz Ry Rx``, as the kernel composes it) -> ``(K, 4, 4)``."""
    x = np.asarray(xyzrpy, dtype=np.float64).reshape(-1, 6)
    cr, sr = np.cos(x[:, 3]), np.sin(x[:, 3])
    cp, sp = np.cos(x[:, 4]), np.sin(x[:, 4])
    cy, sy = np.cos(x[:, 5]), np.sin(x[:, 5])
    out = np.zeros((len(x), 4, 4))
    out[:, 0] = np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x[:, 0]], 1)
    out[:, 1] = np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, x[:, 1]], 1)
    out[:, 2] = np.stack([-sp, cp * sr, cp * cr, x[:, 2]], 1)
    out[:, 3, 3] = 1.0
    return out


class _ReachModel:
    """The kernel's collision model, posed in batches: where its MOVING primitives can reach.

    The kernel's own forward kinematics over the collision params it is handed
    (``collision.cpp::forward_kinematics``: parent frame, joint origin, then the joint's
    motion), vectorised over configurations. A link is moving when a non-base joint lies
    between it and the root; a static link (pedestal, torso) cannot be driven into
    anything, and base dofs are held at zero as the kernel zeroes them (the grid rides
    the base). A capsule contributes its two segment ends with its radius, a box its
    eight corners with radius 0, so ``d . p + r`` / ``|p - c| + r`` are a primitive's
    exact farthest reach along ``d`` / from ``c`` in that configuration.
    """

    def __init__(
        self,
        params: dict[str, object],
        position_limits: list[tuple[float, float]],
        base_dofs: list[int],
    ) -> None:
        def ints(key: str) -> list[int]:
            return [int(v) for v in params.get(key, [])]

        def floats(key: str, width: int) -> np.ndarray:
            return np.asarray(params.get(key, []), dtype=np.float64).reshape(-1, width)

        self.parent, self.kind, self.dof = (
            ints("collision_parent"),
            ints("collision_joint_kind"),
            ints("collision_dof_index"),
        )
        self.origin = _xyzrpy_to_matrices(floats("collision_origin_xyzrpy", 6))
        self.axis = floats("collision_axis", 3)
        self.base = set(base_dofs)
        n = len(self.parent)
        moving = [False] * n
        for i in range(n):  # topological order: a parent precedes its children
            driven = self.dof[i] >= 0 and self.kind[i] != 0 and self.dof[i] not in self.base
            moving[i] = driven or (self.parent[i] >= 0 and moving[self.parent[i]])
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        # (link, local 4x4 of the primitive, local points (k, 3), radius) per moving primitive
        self.prims: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        cap_xf = _xyzrpy_to_matrices(floats("collision_capsule_origin_xyzrpy", 6))
        half = floats("collision_capsule_half_length", 1)[:, 0]
        rad = floats("collision_capsule_radius", 1)[:, 0]
        for c, li in enumerate(ints("collision_capsule_link")):
            ends = np.array([[0.0, 0.0, -half[c]], [0.0, 0.0, half[c]]])
            self.prims += [(li, cap_xf[c], ends, float(rad[c]))] if moving[li] else []
        box_xf = _xyzrpy_to_matrices(floats("collision_box_origin_xyzrpy", 6))
        ext = floats("collision_box_half_extents", 3)
        for b, li in enumerate(ints("collision_box_link")):
            self.prims += [(li, box_xf[b], corners * ext[b], 0.0)] if moving[li] else []
        # An unbounded (continuous) joint sweeps one turn.
        limits = np.asarray(position_limits, dtype=np.float64).reshape(-1, 2)
        self.limits = np.where(np.isfinite(limits), limits, np.sign(limits) * np.pi)
        self.free = [
            j for j in range(len(self.limits)) if j not in self.base and j in set(self.dof)
        ]

    def extremes(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(points (N, K, 3), radii (K,))``: every moving primitive's bounding points."""
        q = np.array(q, dtype=np.float64)
        q[:, sorted(self.base)] = 0.0
        count = len(q)
        world: list[np.ndarray] = []
        for i in range(len(self.parent)):
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            if self.dof[i] >= 0 and self.kind[i] in (1, 2):
                qi = q[:, self.dof[i]]
                a = self.axis[i] / np.linalg.norm(self.axis[i])
                if self.kind[i] == 1:  # revolute: Rodrigues about the joint axis
                    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
                    motion[:, :3, :3] = (
                        np.eye(3)
                        + np.sin(qi)[:, None, None] * k
                        + (1.0 - np.cos(qi))[:, None, None] * (k @ k)
                    )
                else:  # prismatic
                    motion[:, :3, 3] = qi[:, None] * a
            local = self.origin[i] @ motion
            world.append(local if self.parent[i] < 0 else world[self.parent[i]] @ local)
        points: list[np.ndarray] = []
        radii: list[float] = []
        for li, xf, local_pts, r in self.prims:
            frame = world[li] @ xf
            points.append(frame[:, None, :3, :3] @ local_pts[None, :, :, None])
            points[-1] = points[-1][..., 0] + frame[:, None, :3, 3]
            radii += [r] * len(local_pts)
        if not points:
            return np.zeros((count, 0, 3)), np.zeros(0)
        return np.concatenate(points, axis=1), np.asarray(radii)

    def maximise(
        self, score: Callable[[np.ndarray, np.ndarray], np.ndarray], *, seed: int
    ) -> float:
        """The largest ``score(points, radii)`` (per config) over the joint limits.

        Random samples (each joint at its lower limit, its upper limit or uniform between,
        a third each), then the best few refined by coordinate ascent with an exhaustive
        1-D grid per joint: sampling alone stayed 20 mm short on panda_mobile at 4096
        configurations and was still climbing at 262 144.
        """
        if not self.prims:
            return -math.inf
        rng = np.random.default_rng(seed)
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        pick = rng.integers(0, 3, size=(_COVERAGE_SAMPLES, len(lo)))
        q = np.where(pick == 0, lo, np.where(pick == 1, hi, rng.uniform(lo, hi, pick.shape)))
        values = score(*self.extremes(q))
        best = q[np.argsort(values)[-_COVERAGE_REFINE_SEEDS:]]
        for _ in range(_COVERAGE_REFINE_ROUNDS):
            for j in self.free:
                grid = np.linspace(lo[j], hi[j], _COVERAGE_REFINE_GRID)
                trial = np.repeat(best, len(grid), axis=0)
                trial[:, j] = np.tile(grid, len(best))
                v = score(*self.extremes(trial)).reshape(len(best), len(grid))
                keep = v.argmax(axis=1)
                best[:, j] = grid[keep]
        return float(max(values.max(), score(*self.extremes(best)).max()))


def _coverage_ball(
    collision_params: dict[str, object],
    position_limits: list[tuple[float, float]],
    base_dofs: list[int],
    resolution_m: float,
) -> _CoverageBall:
    """The world map's coverage for THIS robot: its moving links' reach, plus a margin.

    Sized by the robot, not by the cell cap. The fixed ball this replaced -- 1.05 m
    about ``(0, 0, 0.5)``, measured on panda_mobile -- left the OpenArm's arms, which
    hang from a torso-top ``base_frame``, reaching 108 mm below it: a bin the arm can
    reach was a kernel blind spot. Both the centre and the radius are now measured the
    way that radius was: over the joint limits, against the SAME collision model the
    kernel is handed (``_swept_surface_extremes``). The centre is the middle of the
    reach box, the radius the farthest reach from it plus ``_COVERAGE_MARGIN_M``.

    A radius, not a box, because the grid's lattice is the OctoMap's: its axes turn
    relative to ``base_frame`` as the robot does, and only a ball is invariant to that.
    Reach is a property of the arm, so this does not vary with ``hal_mode``.

    Raises:
        ROSConfigError: the ball needs more cells at ``resolution_m`` than the bridge
            will publish (``_BRIDGE_MAX_CELLS``), which it would turn into an empty grid.
    """
    from openral_core.exceptions import ROSConfigError

    if int(collision_params.get("collision_n_links", 0)) <= 0:
        (cx, cy, cz), r = _UNCHECKED_COVERAGE
        print(
            f"[deploy_e2e] no collision model: world map covers the default ball "
            f"(centre {(cx, cy, cz)}, radius {r} m); no kernel voxel check reads it.",
            flush=True,
        )
        ball = _CoverageBall((cx, cy, cz), r, (cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r))
    else:
        reach = _ReachModel(collision_params, position_limits, base_dofs)

        def along(d: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
            return lambda p, r: (p @ d + r).max(axis=1)

        axes = np.eye(3)
        hi = np.array([reach.maximise(along(d), seed=_COVERAGE_SEED) for d in axes])
        lo = -np.array([reach.maximise(along(-d), seed=_COVERAGE_SEED) for d in axes])
        centre = np.round((lo + hi) / 2.0, 3)
        farthest = reach.maximise(
            lambda p, r: (np.linalg.norm(p - centre, axis=2) + r).max(axis=1),
            seed=_COVERAGE_SEED,
        )
        radius = math.ceil((farthest + _COVERAGE_MARGIN_M) * 1000.0) / 1000.0
        m = _COVERAGE_MARGIN_M
        ball = _CoverageBall(
            (float(centre[0]), float(centre[1]), float(centre[2])),
            radius,
            (float(lo[0] - m), float(lo[1] - m), float(lo[2] - m)),
            (float(hi[0] + m), float(hi[1] + m), float(hi[2] + m)),
        )
    cells = _world_voxel_max_cells(ball.radius, resolution_m)
    if cells > _BRIDGE_MAX_CELLS:
        raise ROSConfigError(
            f"the world map's coverage ball (radius {ball.radius} m about {ball.centre}) needs "
            f"{cells} cells at {resolution_m} m, more than the voxel bridge publishes "
            f"({_BRIDGE_MAX_CELLS}); it would publish an empty grid. Coarsen "
            "OPENRAL_OCTOMAP_RESOLUTION_M or run without the octomap leg."
        )
    return ball


def _octomap_input_bounds(ball: _CoverageBall) -> dict[str, float]:
    """``octomap_server``'s input-cloud clip: the robot's reach box plus the margin.

    Returns the ``point_cloud_{min,max}_{x,y,z}`` parameters, in the map frame
    (``base_frame``). A return outside it is outside every link's check window, so
    integrating it is pure cost: on Thor the floor 0.7 m below the base and the wall
    3 m out were most of the ZED cloud.
    """
    return {
        f"point_cloud_{side}_{axis}": value
        for side, corner in (("min", ball.clip_lo), ("max", ball.clip_hi))
        for axis, value in zip("xyz", corner, strict=True)
    }


def _cloud_shows_the_robot(hal_mode: str, scene_backend: str | None) -> bool:
    """Whether the octomap input cloud contains the robot's own body, so it needs the self filter.

    A real depth camera sees the arm. In sim it depends on who makes the cloud: the MuJoCo
    sensor bridge ray-casts with the robot's bodies transparent, but an Isaac Sim scene
    renders its depth camera with the robot in view (``read_depth_clouds``). Unfiltered,
    those returns become occupied voxels on the arm and the kernel's world-voxel check stops
    every motion as a collision with the robot itself.

    Example:
        >>> _cloud_shows_the_robot("sim", "isaacsim"), _cloud_shows_the_robot("sim", "mujoco")
        (True, False)
    """
    return hal_mode == "real" or scene_backend == "isaacsim"


def _attached_collision_enabled(hal_mode: str, vision_attachment_enabled: bool) -> bool:
    """Whether the kernel checks attached payloads: where something publishes attachments.

    Sim: the sim attachment manager. Real: only with the vision attachment leg
    (``DeployRuntime.vision_attachment``), and then ALWAYS. The coupling rule is that the
    vision leg on real turns the kernel's attached check on with it, never the leg alone:
    the octomap bridge clears a published payload from the map and the robot self-filter
    removes it from the cloud regardless of this flag, so a leg without the kernel check
    would leave the payload checked by nothing (an invisible payload).
    """
    return hal_mode == "sim" or vision_attachment_enabled


def _attached_collision_deadline_ms(hal_mode: str) -> float:
    """How long the kernel trusts the last attachment state.

    Sim keeps 5000 ms. Real is 1000 ms: the HAL's attachment heartbeat runs at 5 Hz, and a
    real payload must not stay trusted for seconds after its producer went silent.
    """
    return 5000.0 if hal_mode == "sim" else 1000.0


def _hal_file_params(path: str, node_name: str) -> dict[str, object]:
    """The ROS params ``hal_params_file`` gives the HAL node (``/**``, then its own name).

    The launch's safety couplings judge the HAL's EFFECTIVE params, whatever wrote the file
    (``openral deploy``, a ``--hal`` override, or a hand-written file on a bare
    ``ros2 launch``). A missing or unreadable file reads as ``{}``: launch_ros refuses it
    when the HAL starts, and every check below treats an absent key as "off".
    """
    import yaml

    try:
        data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}
    params: dict[str, object] = {}
    if isinstance(data, dict):
        for key in ("/**", node_name, f"/{node_name}"):
            entry = data.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("ros__parameters"), dict):
                params.update(entry["ros__parameters"])
    return params


def _autostart_lifecycle(node: LifecycleNode, node_name: str) -> list:
    """Event handlers that drive ``node`` UNCONFIGURED → INACTIVE → ACTIVE once.

    The activate handler is scoped to the **configure** transition
    (``start_state="configuring"``) so it fires exactly once at boot, after
    ``on_configure`` lands the node in ``inactive``. A bare ``goal_state=
    "inactive"`` matcher would also re-fire on a *runtime* deactivate
    (``active → deactivating → inactive``), which fights VRAM eviction:
    the reasoner deactivates the object detector to free its VRAM before a VLA,
    and an auto-reactivate immediately reloads the model and OOMs an 8 GB card.
    Other autostarted nodes (safety kernel, reasoner, prompt_router) are never
    runtime-deactivated, so this is behaviour-preserving for them.
    """
    matcher = _matches_node_name(node_name)
    return [
        RegisterEventHandler(
            OnProcessStart(
                target_action=node,
                on_start=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matcher,
                            transition_id=Transition.TRANSITION_CONFIGURE,
                        ),
                    ),
                ],
            ),
        ),
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=node,
                start_state="configuring",
                goal_state="inactive",
                entities=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matcher,
                            transition_id=Transition.TRANSITION_ACTIVATE,
                        ),
                    ),
                ],
            ),
        ),
    ]


def _resolve_clock_origin(value: str) -> str:
    """Resolve the OpenRAL clock authority origin forwarded by the CLI.

    ``simulation`` means the HAL publishes simulator elapsed time on ROS
    ``/clock`` and the whole graph runs with ``use_sim_time=true``. ``host_wall``
    means no OpenRAL ``/clock`` publisher and every node stays on ROS system
    time. Operators should not toggle ROS ``use_sim_time`` directly.
    """
    origin = value.strip().lower().replace("-", "_")
    if origin not in ("host_wall", "simulation"):
        raise ValueError(
            f"clock_origin must be 'host_wall' or 'simulation', got {value!r}. "
            "It is resolved by `openral deploy`, not a ROS use_sim_time toggle."
        )
    return origin


def _stereo_camera_topics(names_csv: str) -> tuple[str, str, str, str] | None:
    """Map a ``"<left>,<right>"`` camera-name CSV to the four stereo topics.

    Returns ``(left_image, left_camera_info, right_image, right_camera_info)`` built by
    ``openral_core.camera_topic``, or ``None`` when unset or not exactly two names — in
    which case the stereo visual SLAM impl gets no camera topics and refuses to start
    (its launch arguments carry no default, ADR-0108).

    Example:
        >>> t = _stereo_camera_topics("l, r")
        >>> t[0], t[2]
        ('/openral/cameras/l/image', '/openral/cameras/r/image')
        >>> _stereo_camera_topics("") is None
        True
    """
    parts = [p.strip() for p in names_csv.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    left, right = parts
    return (
        camera_topic(left),
        camera_topic(left, CameraTopicKind.CAMERA_INFO),
        camera_topic(right),
        camera_topic(right, CameraTopicKind.CAMERA_INFO),
    )


def _primary_rgb_camera(sensors: list[SensorSpec]) -> str:
    """The RGB sensor the perception legs default to, or ``""`` when there is none.

    Prefers an optical-framed RGB camera (its intrinsics/extrinsics resolve directly), else
    the first RGB sensor. Shared by the object detector's ``locate_in_view`` camera and the
    reasoner's completion camera so both watch the same view — a hard-coded name is a dead
    topic on every robot that spells its cameras differently. Pass ``publishing_sensors``,
    not the raw manifest: on a real deploy only bound cameras have a topic.

    Example:
        >>> from openral_core import RobotDescription
        >>> mobile = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
        >>> _primary_rgb_camera(mobile.sensors)
        'shoulder_left'
        >>> so101 = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> _primary_rgb_camera(so101.sensors)
        'top'
    """
    rgb = [s for s in sensors if s.modality == "rgb"]
    return next(
        (s.name for s in rgb if s.frame_id.endswith("_optical_frame")),
        rgb[0].name if rgb else "",
    )


def _live_camera_info_topic(spec: SensorSpec, hal_mode: str) -> str:
    """The ``CameraInfo`` topic carrying ``spec``'s live calibration, or ``""`` when none does.

    Sim: the bridge's ``camera_topic(name, CAMERA_INFO)`` (K from the rendered camera). Real:
    the driver's own ``CameraInfo`` beside a ``ros2_image`` binding's topic — never
    ``/openral/cameras/<name>/camera_info``, which the sensor leg rebuilds from the manifest's
    (possibly sim stand-in) intrinsics and frame. Any other real backend has no driver
    calibration, so ``""``: consumers then fall back to the manifest, and log that they did.

    Example:
        >>> from openral_core import RobotDescription
        >>> arm = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> thor = resolve_sensor_overlays("robots/openarm/robot.yaml", "thor", required=True)
        >>> top = next(s for s in apply_sensor_overlays(arm.sensors, thor) if s.name == "top")
        >>> _live_camera_info_topic(top, "real")
        '/zed/zed_node/rgb/color/rect/camera_info'
        >>> _live_camera_info_topic(top, "sim")
        '/openral/cameras/top/camera_info'
    """
    from openral_core import SensorReaderBackend
    from openral_sensors.ros_publisher import camera_info_topic_for

    if hal_mode != "real":
        return camera_topic(spec.name, CameraTopicKind.CAMERA_INFO)
    binding = spec.deploy_binding
    if binding is None or binding.backend != SensorReaderBackend.ROS2_IMAGE:
        return ""
    topic = binding.backend_params.get("topic")
    return camera_info_topic_for(topic) if isinstance(topic, str) and topic else ""


# Sim runner joint-state window: the former node default. A sim bridge's
# /joint_states is not the rig the manifest's window was measured on, so sim
# keeps it rather than the (possibly much tighter) real value. It is also the
# ceiling of a real runner window: the value every real deploy shipped with.
_SIM_JOINT_STATE_STALENESS_S = 0.5


def _runner_joint_state_staleness_s(
    description: RobotDescription, hal_mode: str, *, republished: bool
) -> float:
    """The runner's joint-state freshness window for this deploy.

    Real, raw ``/joint_states``: the manifest's
    ``safety.joint_state_staleness_limit_s`` — the same number ``build_hal``
    hands the real HAL, so the two cannot disagree.

    Real, HAL republish (``republished``: the runner reads the HAL node's
    rate-limited ``~/joint_states``, see ``hal_joint_states_topic``): source
    staleness plus two republish periods, ``limit + 2 / control_freq_hz``. The
    manifest window was measured on the raw stream; the republish timer adds up
    to one period of age per sample, and a window of ~3 periods (OpenArm: 0.1 s
    at 30 Hz) trips on two late timer ticks under GIL load. The HAL itself still
    checks its source at the manifest value. Capped at
    ``_SIM_JOINT_STATE_STALENESS_S`` so no real runner is looser than it shipped.

    Sim: ``_SIM_JOINT_STATE_STALENESS_S``.

    Raises:
        ROSConfigError: ``hal_mode == "real"`` and the manifest declares no
            window, or ``republished`` and it declares no
            ``action_spec.control_freq_hz`` (the republish rate).
    """
    if hal_mode != "real":
        return _SIM_JOINT_STATE_STALENESS_S
    from openral_core.exceptions import ROSConfigError

    declared = description.safety.joint_state_staleness_limit_s
    if declared is None:
        raise ROSConfigError(
            f"robot {description.name!r} declares no safety.joint_state_staleness_limit_s; "
            "a real deploy needs the measured window (tools/joint_state_staleness_probe.py)."
        )
    if not republished:
        return float(declared)
    rate_hz = description.control_rate_hz
    if rate_hz is None:
        raise ROSConfigError(
            f"robot {description.name!r} declares no action_spec.control_freq_hz; the "
            "runner reads the HAL's ~/joint_states republish at that rate and needs it "
            "to size its staleness window."
        )
    return min(float(declared) + 2.0 / rate_hz, _SIM_JOINT_STATE_STALENESS_S)


def _depth_camera(description: RobotDescription) -> str:
    """Name of the manifest's first depth sensor with intrinsics, else ``""``.

    The sensors the sim bridge back-projects (``SensorSpec.is_depth_camera``) and
    publishes ``depth/image`` + ``depth/camera_info`` + ``points`` for.

    Example:
        >>> from openral_core import RobotDescription
        >>> _depth_camera(RobotDescription.from_yaml("robots/panda_mobile/robot.yaml"))
        'front_depth'
    """
    return next((s.name for s in description.sensors if s.is_depth_camera), "")


def _octomap_cloud_topic(pinned: str, description: RobotDescription, hal_mode: str) -> str:
    """The cloud ``octomap_server`` maps (``openral_core.deploy_cloud_topic``), never silence.

    A scene-pinned ``octomap_cloud_topic`` wins (a real depth driver's topic); otherwise, in
    sim, the cloud the sim sensor bridge back-projects for the manifest's one depth sensor
    with intrinsics. ``openral deploy`` refuses the same cases before launch; this repeats
    the check for a direct ``ros2 launch``.

    Raises:
        ROSConfigError: nothing publishes a cloud (a real deploy with nothing pinned, or a
            robot with no depth sensor) or several depth sensors and nothing pinned --
            never spawn octomap_server against a topic nothing publishes.

    Example:
        >>> from openral_core import RobotDescription
        >>> _octomap_cloud_topic("", RobotDescription.from_yaml("robots/openarm/robot.yaml"), "sim")
        '/openral/cameras/head_zed/points'
    """
    topic = deploy_cloud_topic(description.sensors, pinned=pinned, hal_mode=hal_mode)
    if not topic:
        from openral_core.exceptions import ROSConfigError

        raise ROSConfigError(
            f"octomap enabled but nothing publishes a cloud for robot {description.name!r} "
            f"(hal_mode={hal_mode}): pin octomap_cloud_topic to the depth driver's "
            "PointCloud2 topic, or declare a depth sensor with intrinsics for sim"
        )
    return topic


def _build_driver_includes(scene_drivers: list, deploy_config: str) -> list:  # type: ignore[type-arg]  # reason: openral_core.LaunchInclude, imported lazily
    """Include the vendor sensor drivers a deploy scene declares.

    A ``deploy_binding`` with a ``ros2_*`` backend subscribes to a topic somebody
    else publishes; these launches are that somebody. Until the scene could name
    them, an operator started the driver by hand in a second terminal — and the
    failure when they forgot was a silently empty camera panel, never an error,
    because subscribing to an unpublished topic is perfectly legal.

    Resolved eagerly with ``get_package_share_directory`` rather than a lazy
    ``FindPackageShare`` substitution, so an unresolvable package fails here with
    a message naming the scene, the driver and the likely cause. A vendor driver
    is usually built into *its own* colcon workspace (``zed_wrapper`` lives in a
    ``zed_ws``, not in the OpenRAL overlay), and a package is only findable if
    that workspace is sourced — the substitution's own error says just "package
    not found", which does not point at the overlay you forgot.
    """
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from openral_core.exceptions import ROSConfigError

    scene_dir = pathlib.Path(deploy_config).parent

    def _resolve(key: str, value: str) -> str:
        """Resolve a relative ``*_path`` argument against the scene's directory.

        A driver's config file belongs with the scene that needs it, not in an
        operator's home directory — a committed scene carrying
        ``/home/<someone>/...`` works on exactly one machine. Same rule the CLI
        already applies to ``calibration_dir``. Only ``*_path`` keys and only
        relative values, so an absolute path still wins.
        """
        if key.endswith("_path") and value and not pathlib.Path(value).is_absolute():
            return str((scene_dir / value).resolve())
        return value

    includes = []
    for d in scene_drivers:
        try:
            share = get_package_share_directory(d.package)
        except PackageNotFoundError as exc:
            raise ROSConfigError(
                f"{pathlib.Path(deploy_config).name} declares driver "
                f"{d.package!r}, which is not on the ament path. A vendor driver "
                f"is usually built into its own colcon workspace — source that "
                f"overlay before `openral deploy run` (e.g. "
                f"`source ~/<ws>/install/setup.bash`), or drop the driver from "
                f"the scene's `drivers:` if this cell does not have it. Refusing "
                f"rather than starting a graph whose sensors can never publish."
            ) from exc
        includes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(share, "launch", d.launch_file)),
                launch_arguments=tuple((k, _resolve(k, v)) for k, v in d.args.items()),
            )
        )
    return includes


def _write_foxglove_layout(cameras: list[str], robot_id: str, base_frame: str) -> str | None:
    """Generate a Foxglove layout for the cameras this deploy actually publishes.

    The shipped ``config/openral_layout.json`` is generated for
    ``layout.DEFAULT_CAMERAS``, which is a guess: camera slots are sensor names
    out of the robot manifest and the deploy scene, and they differ per robot
    (``top`` / ``context`` / ``wrist_left`` / ``camera1``…). A panel pointed at a
    slot this deploy has no publisher for renders "Image topic does not exist",
    which looks exactly like a dead camera. The 3D panels have the same
    problem one level up: they follow a TF frame, and Foxglove draws an empty
    scene — no robot, no point clouds, no markers — when that frame is not in
    the tree. The library default is the ROS-conventional ``base_link``, which
    OpenArm does not broadcast (its root is ``openarm_base``), so the robot's
    own ``base_frame`` is threaded through rather than guessed.

    A layout is imported client-side, so no launch argument can push one into
    the viewer — but the launch is the only place that knows the answer, so it
    writes the correct layout to a file and logs the path for the operator to
    import once. Returns that path, or ``None`` when there is nothing to
    generate or the write fails (a viz convenience must never take the graph
    down with it).
    """
    if not cameras:
        return None
    try:
        from openral_foxglove_bringup.layout import build_layout

        path = pathlib.Path(tempfile.gettempdir()) / f"openral_layout_{robot_id}.json"
        path.write_text(
            json.dumps(build_layout(cameras, follow_frame=base_frame), indent=2),
            encoding="utf-8",
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"[deploy_e2e] could not write a scene-matched Foxglove layout: {exc!r}", flush=True)
        return None
    return str(path)


def _foxglove_asset_env(robot_description_xml: str | None) -> dict[str, str]:
    """``foxglove_bridge``'s env, so a viewer can fetch the URDF's ``package://`` meshes.

    The bridge serves a 3D panel's ``fetchAsset`` through ``resource_retriever``,
    which resolves ``package://<pkg>`` only through the ament index. A public
    description package OpenRAL fetches into its cache (OpenArm's
    ``openarm_description``) is on no workspace's index, so on Spark the bridge
    refused every mesh (``Package [openarm_description] does not exist``) and the
    panel drew TF axes with no robot. Prepend an overlay prefix that indexes it.
    Empty when no overlay is needed or it cannot be built — a viz convenience
    must never take the graph down, so that failure is printed, not raised.
    """
    if robot_description_xml is None:
        return {}
    from openral_core.exceptions import ROSConfigError
    from openral_hal.ros_package_overlay import public_package_overlay

    try:
        overlay = public_package_overlay(robot_description_xml)
    except (ROSConfigError, OSError) as exc:
        print(f"[deploy_e2e] foxglove: URDF meshes will not load: {exc!r}", flush=True)
        return {}
    if overlay is None:
        return {}
    print(f"[deploy_e2e] foxglove: serving package:// URDF meshes via {overlay}", flush=True)
    return {
        "AMENT_PREFIX_PATH": os.pathsep.join(
            p for p in (str(overlay), os.environ.get("AMENT_PREFIX_PATH", "")) if p
        )
    }


def _build_real_bringup_include(real_bringup: str | None) -> object | None:
    """Include the robot's vendor ros2_control bringup, if its manifest declares one.

    ``real_bringup`` is the manifest's ``hal.real_bringup``
    (``"<pkg>:<file>.launch.py"``, e.g.
    ``"openral_hal_openarm:real_bringup.launch.py"``). A declared bringup that
    is not installed raises instead of silently running without controllers;
    ``None`` means the robot needs none.

    Every real-hardware HAL in this repo publishes to controllers it does not
    start: ``controller_manager`` is C++ at 400 Hz+ and belongs under a vendor
    bringup launch, not under a Python HAL (CLAUDE.md §1.5). Until this include
    existed, ``deploy run`` assumed that graph was already up, so an operator
    brought it up out-of-band from a second terminal — which put a *second*
    ``/joint_states`` publisher on the bus and tripped the shared-graph guard
    (``openral_cli._dds_scope``, #227) on the documented bring-up path. Starting
    it here keeps that guard meaningful: one graph, one publisher, and no
    escape hatch needed to deploy a real robot — the occupied-graph refusal is
    now unwaivable (``openral_cli._dds_scope``).

    Returns ``None`` when the manifest declares no bringup — the case for every
    HAL whose controller graph is started elsewhere (a vendor daemon, a
    robot-side controller) and for every sim-only HAL. Callers must only reach
    here on ``hal_mode == "real"``.

    No launch arguments are forwarded. The vendor launch's own defaults are
    required to agree with the HAL's constants already — that agreement is what
    ``tests/hil/test_openarm_bringup_agreement.py`` guards — so passing a second
    copy of the CAN interface names through here would create a way for the two
    to disagree without any test noticing.

    The include starts concurrently with the HAL lifecycle node, not before it.
    The HAL's ``connect()`` preflights the CAN links (up regardless of
    controllers) and does not block on ``/joint_states``, so for the few seconds
    the controllers take to spawn it logs ``read_state failed: Joint state is
    N s old`` and then recovers once the broadcaster is active. That is
    warn-only and self-healing; sequencing it behind a controller-active event
    handler would buy a quieter log and nothing else.

    A vendor bringup typically also spawns its own ``robot_state_publisher``,
    alongside the one this file derives from ``assets.urdf``. Both are kept: the
    manifest URDF is load-bearing (for OpenArm it carries ``openarm_base``, the
    ``world`` bridge and the sensor mounts, none of which the vendor xacro
    describes), so suppressing it loses frames something downstream reads.

    Their ``/tf`` output is additive only because both now spell joints and
    links the same way. It was not: the vendored OpenArm URDF used to strip the
    ``openarm_`` prefix, and two spellings of one robot on ``/robot_description``
    empty out ``joint_state_broadcaster`` (its ``use_urdf_to_filter`` publishes
    only joints the URDF also names) while freezing this node's tree at the rest
    pose, since none of its joint names match the arm's ``/joint_states``. Both
    failures are silent. The names are standardised upstream-side now, and the
    caller additionally keeps this node off ``/robot_description`` whenever a
    vendor bringup owns it — see the remapping at the node itself.
    """
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    if real_bringup is None:
        return None
    pkg, _, launch_file = real_bringup.partition(":")
    try:
        bringup_path = os.path.join(get_package_share_directory(pkg), "launch", launch_file)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"hal.real_bringup={real_bringup!r}: ROS package {pkg!r} is not installed."
        ) from exc
    if not os.path.isfile(bringup_path):
        raise RuntimeError(f"hal.real_bringup={real_bringup!r}: {bringup_path} does not exist.")
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(bringup_path))


def _build_nav2_include(
    robot_yaml: str, *, use_sim_time: bool, slam_backend: str = "lidar"
) -> object:
    """Construct the IncludeLaunchDescription for upstream Nav2.

    Pulled out of ``compose_runtime_graph`` for line-count hygiene. Nav2 is always-on
    (unlike slam_toolbox, which idles until activate): its in-stack
    ``lifecycle_manager_navigation`` brings the planner / controller / behavior / smoother /
    velocity_smoother sub-nodes to ACTIVE automatically. The Reasoner triggers Nav2 by
    dispatching the ``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose-none`` wrapped-action
    rSkill, not by lifecycle-transitioning the planner.

    ``use_sim_time`` is derived from the graph-wide clock authority (see
    ``_resolve_clock_origin``), never hardcoded: with no ``/clock`` on the bus it must be
    ``False`` so Nav2's controller loop and costmaps run on wall-clock, matching the HAL's
    wall-clock ``/scan`` + odom→base_link TF — ``true`` with no ``/clock`` pins every Nav2 node
    at t=0 ("loop rate inf Hz"), producing an empty costmap → collision.
    """
    from ament_index_python.packages import get_package_share_directory
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    nav2_launch_path = os.path.join(
        get_package_share_directory("openral_nav2_bringup"),
        "launch",
        "nav2.launch.py",
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_path),
        # ``robot_yaml`` lets nav2.launch.py rewrite the base params with
        # this robot's footprint_radius / base_kinematics —
        # generic across mobile bases, no hand-vendored per-robot file.
        launch_arguments={
            "use_sim_time": "true" if use_sim_time else "false",
            "robot_yaml": robot_yaml,
            # visual robots get the `/map`-consuming costmap profile
            # (nav2_visual.yaml); lidar robots keep the `/scan` base config.
            "slam_backend": slam_backend,
        }.items(),
    )


def _build_visual_slam_includes(
    slam_share: str,
    *,
    visual_impl: str,
    use_sim_time: bool,
    stereo_cameras_csv: str,
    enable_nav2: bool,
    robot_yaml: str,
    mono_camera: str = "",
    mono_depth_frame: str = "",
    depth_sidecar_autostart: bool = True,
    nav2_depth_camera: str = "",
) -> list[object]:
    """Build the cuVSLAM (+ optional nvblox) includes for the visual backend.

    Pulled out of ``compose_runtime_graph`` so the impl→launch-file selection and per-scene
    stereo-camera remaps are unit-testable (mirrors ``_build_nav2_include``).

    ``visual_impl`` picks the engine — ``"pycuvslam"`` composes the in-process PyCuVSLAM wheel
    node (``pycuvslam.launch.py``, rectified stereo, no Isaac ROS apt stack); anything else
    composes the composable ``isaac_ros_visual_slam`` C++ node (``cuvslam.launch.py``).
    ``stereo_cameras_csv`` (``"<left>,<right>"``) supplies the stereo camera topics, keyed to
    each impl's own arg names (the impls carry no default, ADR-0108). ``enable_nav2`` also
    composes nvblox (cuVSLAM gives pose, not an occupancy grid), fed ``nav2_depth_camera``'s
    depth stream (the manifest's depth sensor, ``_depth_camera``); empty leaves nvblox without
    an input, which its depth height filter refuses at startup.

    ``mono_camera`` (pycuvslam only) selects the mono RGBD path: one RGB camera + the DA3
    metric-depth provider. Auto-composes ``depth_provider_node`` (RGB → 32FC1 depth, framed at
    ``mono_depth_frame`` so nvblox can place it via the HAL's ``base → <camera>_optical_frame``
    TF) and nvblox — cuVSLAM gets depth for scale, nvblox for the occupancy grid + voxels. The
    DA3 sidecar (``tools/da3_depth_sidecar.py``) spawns alongside by default
    (``depth_sidecar_autostart``); the depth provider retries until it answers (first boot
    provisions the sidecar venv). ``False`` = operator-run/shared.
    """
    from launch.actions import ExecuteProcess, IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    sim_time_arg = "true" if use_sim_time else "false"
    mono_camera = mono_camera.strip()
    if visual_impl == "pycuvslam" and mono_camera:
        rgb_image = camera_topic(mono_camera)
        rgb_info = camera_topic(mono_camera, CameraTopicKind.CAMERA_INFO)
        depth_image = camera_topic(mono_camera, CameraTopicKind.DEPTH_IMAGE)
        depth_info = camera_topic(mono_camera, CameraTopicKind.DEPTH_CAMERA_INFO)
        depth_frame = mono_depth_frame or f"{mono_camera}_optical_frame"
        actions: list[object] = []
        if depth_sidecar_autostart:
            # DA3 metric-depth sidecar (ZMQ :5771). The launcher is stdlib-only
            # (any python3 runs it) and provisions its own venv on first boot —
            # which can take minutes; the depth provider retries until it
            # answers. If an operator-run sidecar already holds the port, this
            # one fails to bind and exits; ExecuteProcess death does not tear
            # down the launch, so the external sidecar keeps serving.
            actions.append(
                ExecuteProcess(
                    cmd=[
                        sys.executable,
                        str(_REPO_ROOT / "tools" / "da3_depth_sidecar.py"),
                        "--port",
                        "5771",
                    ],
                    name="da3_depth_sidecar",
                    output="screen",
                )
            )
        return [
            *actions,
            # cuVSLAM in RGBD mode: track the single RGB camera fused with DA3 depth.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_share, "launch", "pycuvslam.launch.py")
                ),
                launch_arguments={
                    "use_sim_time": sim_time_arg,
                    "left_image_topic": rgb_image,
                    "left_camera_info_topic": rgb_info,
                    "depth_image_topic": depth_image,
                }.items(),
            ),
            # DA3 metric-depth provider: RGB → 32FC1 depth for BOTH cuVSLAM (scale)
            # and nvblox (dense map). Depth is framed at the RGB camera's optical
            # frame so nvblox locates it via the HAL's live camera TF.
            Node(
                package="openral_perception_ros",
                executable="depth_provider_node.py",
                name="openral_depth_provider",
                namespace="",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "image_topic": rgb_image,
                        "depth_topic": depth_image,
                        "camera_info_topic": depth_info,
                        "depth_frame_id": depth_frame,
                    }
                ],
                output="screen",
            ),
            # nvblox: cuVSLAM pose + DA3 depth → /map occupancy grid + voxels.
            # Always composed in mono mode (the map IS the point), not gated on nav2.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_share, "launch", "nvblox.launch.py")
                ),
                launch_arguments={
                    "use_sim_time": sim_time_arg,
                    "robot_yaml": robot_yaml,
                    "depth_image_topic": depth_image,
                    "depth_camera_info_topic": depth_info,
                }.items(),
            ),
        ]

    stereo = _stereo_camera_topics(stereo_cameras_csv)
    # PyCuVSLAM gets robot_yaml so the node derives the rig frame from the
    # manifest base_frame → cuVSLAM multi-camera mode (per-camera extrinsics from
    # TF), which handles the sim's arbitrary (toed-in) camera rigs. The Isaac ROS
    # composable node has no such arg, so it is passed only for pycuvslam.
    if visual_impl == "pycuvslam":
        launch_file = "pycuvslam.launch.py"
        cam_arg_names = (
            "left_image_topic",
            "left_camera_info_topic",
            "right_image_topic",
            "right_camera_info_topic",
        )
        extra_args = {"robot_yaml": robot_yaml}
    else:
        launch_file = "cuvslam.launch.py"
        cam_arg_names = (
            "image_0_topic",
            "camera_info_0_topic",
            "image_1_topic",
            "camera_info_1_topic",
        )
        extra_args = {}
    cam_args = dict(zip(cam_arg_names, stereo, strict=True)) if stereo is not None else {}

    includes: list[object] = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(slam_share, "launch", launch_file)),
            launch_arguments={"use_sim_time": sim_time_arg, **cam_args, **extra_args}.items(),
        )
    ]
    # When navigating, fuse depth + cuVSLAM pose into the ESDF cost map Nav2
    # needs. The depth stream comes from the monocular metric-depth provider
    # (depth_provider_node + DA3 sidecar) or a real RGB-D sensor — operator-run
    # (the model sidecar provisions its own venv), so it is not auto-spawned.
    if enable_nav2:
        includes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_share, "launch", "nvblox.launch.py")
                ),
                launch_arguments={
                    "use_sim_time": sim_time_arg,
                    "robot_yaml": robot_yaml,
                    **(
                        {
                            "depth_image_topic": camera_topic(
                                nav2_depth_camera, CameraTopicKind.DEPTH_IMAGE
                            ),
                            "depth_camera_info_topic": camera_topic(
                                nav2_depth_camera, CameraTopicKind.DEPTH_CAMERA_INFO
                            ),
                        }
                        if nav2_depth_camera
                        else {}
                    ),
                }.items(),
            )
        )
    return includes


def _resolve_urdf_path(ref: str, manifest_dir: pathlib.Path) -> str | None:
    """Resolve a ``RobotDescription.assets.urdf.ref`` to a concrete URDF path.

    Thin wrapper over ``openral_core.assets.resolve_asset``. Returns
    ``None`` for the ``ros2://robot_description`` dynamic marker (the URDF is on
    the ``/robot_description`` topic at runtime — no file to read). ``file:`` refs
    resolve against the robot's manifest dir, then the repo root.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    try:
        path = resolve_asset(ref, "urdf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        print(f"[deploy_e2e] could not resolve urdf ref {ref!r}: {exc}", flush=True)
        return None
    return None if path is None else str(path)


def compose_runtime_graph(context: LaunchContext, *_args: object, **_kwargs: object) -> list:  # noqa: PLR0915  # reason: launch compose is naturally linear — arg resolution + node construction + autostart wiring in one place is the clearest expression of the boot order
    """Resolve every launch arg, load ``robot.yaml``, build the graph.

    Bound to an ``OpaqueFunction`` in
    ``generate_launch_description`` so the launch args resolve to
    concrete strings before they reach
    ``launch_ros.actions.LifecycleNode`` (which doesn't accept
    ``LaunchConfiguration`` in every field).
    The name mirrors ``openral_rskill_ros.compose_runtime`` — same
    "build the runtime in one place" semantics, scoped to the launch
    layer instead of the in-process composer.
    """
    # Deferred import: launch files are imported by `ros2 launch` even
    # without a sourced workspace, so keep openral_core / openral_safety
    # off the module top.
    from openral_core import DeployScene, RobotDescription
    from openral_safety.envelope_loader import (
        collision_params_from_description,
        compute_intersection,
        ee_link_index_from_collision_params,
        kernel_params_from_envelope,
        merge_extra_allowed_pairs,
    )

    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)
    hal_package = LaunchConfiguration("hal_package").perform(context)
    hal_executable = LaunchConfiguration("hal_executable").perform(context)
    hal_node_name = LaunchConfiguration("hal_node_name").perform(context)
    hal_params_file = LaunchConfiguration("hal_params_file").perform(context)
    reset_to_pose_service = LaunchConfiguration("reset_to_pose_service").perform(context)
    approach_skill_id = LaunchConfiguration("approach_skill_id").perform(context)
    preload_rskill_id = LaunchConfiguration("preload_rskill_id").perform(context)
    preload_rskill_revision = LaunchConfiguration("preload_rskill_revision").perform(context)
    preload_prompt = LaunchConfiguration("preload_prompt").perform(context)
    place_declaration_json = LaunchConfiguration("place_declaration_json").perform(context)
    grasp_declaration_json = LaunchConfiguration("grasp_declaration_json").perform(context)
    # Record the deploy session to a rosbag2 mcap.
    dataset_out = LaunchConfiguration("dataset_out").perform(context)
    dataset_repo_id = LaunchConfiguration("dataset_repo_id").perform(context)
    dataset_license = LaunchConfiguration("dataset_license").perform(context)
    # Real deploys — DeployScene YAML; the runtime node opens every
    # deploy-bound sensor (robot manifest ∪ scene `sensors:`) and
    # publishes each as /openral/cameras/<name>/image.
    deploy_config = LaunchConfiguration("deploy_config").perform(context)
    dashboard_port = LaunchConfiguration("dashboard_port").perform(context)
    reasoner_model = LaunchConfiguration("reasoner_model").perform(context)
    reasoner_endpoint = LaunchConfiguration("reasoner_endpoint").perform(context)
    spatial_memory_path = LaunchConfiguration("spatial_memory_path").perform(context)
    spatial_memory_ingest = LaunchConfiguration("spatial_memory_ingest").perform(
        context
    ).lower() in ("1", "true", "yes")
    # The deploy memory bundle. `memory_md_path` loads the
    # self-maintained MEMORY.md (+ enables the memory_write / memory_search tools);
    # `map_path` seeds a static 2D occupancy grid into nav2 map_server. Both are the
    # bundle's text/grid modalities alongside spatial_memory_path's scene graph.
    memory_md_path = LaunchConfiguration("memory_md_path").perform(context)
    map_path = LaunchConfiguration("map_path").perform(context)
    # Deploy-path selector for the reasoner's action-mode
    # palette gate. ``openral deploy sim`` shells this launch with
    # ``hal_mode:=sim`` (digital-twin path: the scene's robosuite OSC
    # controller synthesises cartesian/OSC action modes, so cartesian
    # skills are admissible); ``openral deploy run`` passes ``hal_mode:=real``
    # so the reasoner admits only skills whose action modes ∈ the robot's
    # ``supported_control_modes``. Default ``"sim"`` matches the launch's
    # digital-twin heritage.
    hal_mode = LaunchConfiguration("hal_mode").perform(context)
    hardware_estop_device = LaunchConfiguration("hardware_estop_device").perform(context)
    enable_slam = LaunchConfiguration("enable_slam").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    # Which SLAM backend to compose when enable_slam: "lidar"
    # (slam_toolbox), "visual" (cuVSLAM, camera-based, lidar-less robots),
    # or "none". Resolved upstream in deploy_sim.py from capabilities;
    # default "lidar" preserves the legacy lidar-only behaviour for any caller
    # that sets enable_slam without forwarding slam_backend.
    slam_backend = LaunchConfiguration("slam_backend").perform(context).strip().lower()
    # Which cuVSLAM engine the visual backend composes: "isaac_ros" (composable
    # C++ node) or "pycuvslam" (in-process wheel). Ignored unless slam_backend
    # is "visual". Optional "<left>,<right>" stereo camera names override the
    # impl's default left/right topics (workcell rig binding from the scene).
    slam_visual_impl = LaunchConfiguration("slam_visual_impl").perform(context).strip().lower()
    slam_stereo_cameras = LaunchConfiguration("slam_stereo_cameras").perform(context).strip()
    slam_mono_camera = LaunchConfiguration("slam_mono_camera").perform(context).strip()
    slam_depth_sidecar_autostart = LaunchConfiguration("slam_depth_sidecar_autostart").perform(
        context
    ).strip().lower() in ("1", "true")
    enable_nav2 = LaunchConfiguration("enable_nav2").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    enable_octomap = LaunchConfiguration("enable_octomap").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    # Decouple the octomap PERCEPTION leg (publishing
    # /openral/world_voxels for the world-state object-lift) from the SAFETY
    # KERNEL's capsule-vs-voxel check. Default True preserves the bundled
    # world-collision-check behaviour; set False to publish the voxel map for object-lift
    # while keeping the kernel voxel check OFF (its posture under
    # --no-enable-octomap: envelope + self-collision only). Lets perception use
    # the world map without the kitchen false-positive E-stop. Never weakens the
    # kernel below the --no-enable-octomap baseline.
    enable_octomap_kernel_check = LaunchConfiguration("enable_octomap_kernel_check").perform(
        context
    ).lower() in ("1", "true", "yes")
    octomap_cloud_topic = LaunchConfiguration("octomap_cloud_topic").perform(context)
    rig = _rig_from_launch_args(
        {name: LaunchConfiguration(name).perform(context) for name in _RIG_LAUNCH_ARGS}
    )
    world_voxel_deadline_s, max_octree_age_s = rig.voxel_freshness_s
    # Object-detection perception leg. Off by default; when on,
    # the ROS-Image detector node runs RT-DETR over the agentview RGB tee and
    # publishes ObjectsMetadata to /openral/perception/objects, which the
    # world-state node's object-lift (enabled by default) raises into voxels.
    enable_object_detector = LaunchConfiguration("enable_object_detector").perform(
        context
    ).lower() in (
        "1",
        "true",
        "yes",
    )
    object_detector_onnx = LaunchConfiguration("object_detector_onnx").perform(context)
    object_detector_manifest = LaunchConfiguration("object_detector_manifest").perform(context)
    object_detector_query = LaunchConfiguration("object_detector_query").perform(context)
    # Vision attachment leg (DeployRuntime.vision_attachment): the SAM 2.1 segmenter the
    # HAL's attachment-evidence bridge asks at grasp events. Off by default; on, it also
    # turns the kernel's attached check on (_attached_collision_enabled).
    vision_attachment_enabled = LaunchConfiguration("enable_vision_attachment").perform(
        context
    ).lower() in ("1", "true", "yes")
    # The scene's manifest-link -> TF-frame renames ("link=frame", comma-joined), the same
    # strings the HAL gets as `vision_attachment_tf_frames`. The octomap bridge needs them
    # to find a held payload's attach link on TF; empty = every link is its own frame.
    attach_link_tf_frames = [
        e
        for e in LaunchConfiguration("vision_attachment_tf_frames").perform(context).split(",")
        if e
    ]
    # Grasp-target exemption (DeployRuntime.grasp_allowance_enabled). Default off.
    grasp_allowance_enabled = LaunchConfiguration("grasp_allowance_enabled").perform(
        context
    ).lower() in ("1", "true", "yes")
    # Reward-monitor leg. Off by default; when on, a reward_monitor_node
    # runs PARALLEL to the VLA, buffering the agentview RGB stream, and the reasoner
    # is told task_progress_available=True so its LLM may poll
    # /openral/perception/query_task_progress (the query_task_progress tool) whenever
    # it sees fit. Advisory-only — never actuates.
    enable_reward_monitor = LaunchConfiguration("enable_reward_monitor").perform(
        context
    ).lower() in ("1", "true", "yes")
    reward_monitor_manifest = LaunchConfiguration("reward_monitor_manifest").perform(context)
    reward_monitor_task = LaunchConfiguration("reward_monitor_task").perform(context)
    # Scene-VLM leg. Off by default; when on, a scene_vlm_node caches every
    # manifest RGB camera's latest frame and serves
    # /openral/perception/query_scene, and the reasoner is told
    # scene_query_available=True so its LLM may ask open-ended questions about the
    # current view (the query_scene tool). Read-only — never actuates.
    enable_scene_vlm = LaunchConfiguration("enable_scene_vlm").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    scene_vlm_manifest = LaunchConfiguration("scene_vlm_manifest").perform(context)
    # Tier-C critic-producer leg. Off by default; when on, a
    # critic_producer_node watches the generic /openral/critic/score topic and
    # turns a critic stall into a Tier-C FailureTrigger on /openral/failure/critic
    # (the reasoner already subscribes it). Advisory-only — never actuates.
    enable_critic = LaunchConfiguration("enable_critic").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    critic_stall_patience = LaunchConfiguration("critic_stall_patience").perform(context)
    # Comma-separated on-demand locator manifest paths. Each becomes a
    # namespaced locate_in_view lifecycle node (/openral/perception/<alias>/...) so
    # the reasoner can choose a model via LocateInViewTool.detector. Alias/segment
    # derivation is the single source of truth in openral_reasoner.palette. The
    # yaml / palette imports stay local so the detector-off base graph keeps its
    # zero import-time cost (mirrors the detector node block below).
    object_detector_locators_raw = LaunchConfiguration("object_detector_locators").perform(context)
    locator_tokens = [p for p in object_detector_locators_raw.split(",") if p]
    locator_specs: list[dict[str, str]] = []
    if locator_tokens:
        import yaml
        from openral_reasoner.palette import detector_alias, detector_service_segment

        for _mpath in locator_tokens:
            with pathlib.Path(_mpath).open(encoding="utf-8") as _handle:
                _lman = yaml.safe_load(_handle) or {}
            _alias = detector_alias(str(_lman.get("name", _mpath)))
            _segment = detector_service_segment(_alias)
            locator_specs.append(
                {
                    "manifest": _mpath,
                    "alias": _alias,
                    "segment": _segment,
                    "node": f"openral_ros_image_detector_{_segment}",
                    "engine": str((_lman.get("detector") or {}).get("engine") or ""),
                }
            )
    enable_dashboard = LaunchConfiguration("enable_dashboard").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    enable_reasoner = LaunchConfiguration("enable_reasoner").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    # Read-only Foxglove live-scene bridge. Off by default;
    # ``openral deploy sim --foxglove`` opts in.
    enable_foxglove = LaunchConfiguration("enable_foxglove").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    foxglove_port = LaunchConfiguration("foxglove_port").perform(context)
    # Graph-wide clock domain (single source of truth). The CLI resolves the
    # OpenRAL ClockAuthority origin; this launch only maps it to ROS
    # use_sim_time. ``simulation`` is backed by the HAL's /clock publisher.
    # ``host_wall`` keeps every node on system time. There is no operator-facing
    # ROS time toggle to drift from the authority.
    clock_origin = _resolve_clock_origin(LaunchConfiguration("clock_origin").perform(context))
    use_sim_time = clock_origin == "simulation"
    # Startup operator prompt set by --initial-task or /openral/prompt. When
    # non-empty, prompt_router_node publishes it onto /openral/prompt at
    # on_activate so the reasoner's first tick sees the operator's goal without
    # a manual ``openral prompt`` call. Empty string = no startup prompt (idle).
    initial_task_prompt = LaunchConfiguration("initial_task_prompt").perform(context)
    workcell_json = LaunchConfiguration("workcell_json").perform(context)
    workcell = DeployScene.model_validate_json(workcell_json) if workcell_json else None

    # Synthesise the kernel envelope from the manifest. ``skill=None``
    # because ``openral deploy sim`` does not preselect an rSkill — the
    # reasoner picks dynamically. The robot ceiling is the right boot-
    # time envelope; future per-skill tightening will hot-swap via a
    # kernel reload, not by mounting a different envelope at boot.
    description = RobotDescription.from_yaml(robot_yaml)
    description.validate_for_e2e_pipeline()  # loud failure on missing fields
    # The deploy scene's sensors and drivers, loaded once: the reasoner's completion camera,
    # the detector's camera, WorldState's subscriptions and the vendor driver includes all
    # need to know which cameras exist (and, on a real deploy, which are bound).
    scene_sensors: list[SensorSpec] = []
    scene_drivers: list = []  # type: ignore[type-arg]  # reason: openral_core.LaunchInclude, deferred import
    joint_states_override: str | None = None
    scene_unit: str | None = None
    scene_backend: str | None = None
    if deploy_config:
        from openral_core import DeployScene

        _scene = DeployScene.from_yaml(deploy_config)
        scene_sensors = list(_scene.sensors)
        scene_drivers = list(_scene.drivers)
        scene_unit = _scene.robot_unit
        if _scene.runtime is not None:
            joint_states_override = _scene.runtime.joint_states_topic
        if _scene.scene is not None:
            scene_backend = str(getattr(_scene.scene.backend, "value", _scene.scene.backend))
    from openral_hal.resolver import hal_joint_states_topic

    # The Python nodes' JointState topic: the scene's override, else the HAL's
    # rate-limited `~/joint_states` on a real ros2_control arm (whose global
    # `/joint_states` is the broadcaster's full-rate stream), else "" = default.
    runtime_joint_states_topic = (
        hal_joint_states_topic(
            description,
            mode="real" if hal_mode == "real" else "sim",
            hal_node_name=hal_node_name,
            override=joint_states_override,
        )
        or ""
    )
    # This host's unit overlay (scene `robot_unit`, or $OPENRAL_ROBOT_UNIT which `openral
    # deploy` resolved identically before launching): per-host bindings and per-unit mount
    # calibration replace the manifest's nominal values for every consumer below.
    description = description.model_copy(
        update={
            "sensors": apply_sensor_overlays(
                description.sensors,
                resolve_sensor_overlays(robot_yaml, scene_unit, required=hal_mode == "real"),
            )
        }
    )
    publishing = publishing_sensors(description.sensors, scene_sensors, hal_mode)
    gripper_convention = LaunchConfiguration("gripper_convention").perform(context)
    envelope = compute_intersection(
        description,
        skill=None,
        deploy=workcell.safety if workcell is not None else None,
        gripper_convention=gripper_convention or None,
    )
    # Self-collision model. Prefer lowering from the robot's MJCF
    # (the full kinematic tree, incl. fixed mounts + floating base, that the
    # manifest's actuated-only ``joints`` can't express); fall back to the
    # manifest geometry otherwise. Returns ``{"self_collision_enabled": False}``
    # when no geometry is available, so the kernel runs the scalar envelope
    # check exactly as before. A lowering error falls back loudly so a geometry
    # hiccup never blocks the boot.
    collision_params: dict[str, object] = collision_params_from_description(description)
    if description.assets.mjcf:
        try:
            import mujoco
            from openral_core.assets import resolve_asset
            from openral_safety.mjcf_lowering import lower_collision_params

            _mjcf_path = resolve_asset(
                description.assets.mjcf, "mjcf", manifest_dir=pathlib.Path(robot_yaml).parent
            )
            model = mujoco.MjModel.from_xml_path(str(_mjcf_path))
            mjcf_params = lower_collision_params(model, [j.name for j in description.joints])
            # Only override the manifest model when the MJCF actually yields a
            # self-collision model. MJCFs whose collision geoms are meshes (e.g.
            # bimanual openarm) lower to {"self_collision_enabled": False}; using
            # that would silently DISABLE self-collision, so keep the manifest's
            # hand-authored capsules + ACM instead.
            if mjcf_params.get("self_collision_enabled"):
                collision_params = mjcf_params
            else:
                print(
                    "[deploy_e2e] MJCF has no primitive collision geometry; "
                    "keeping the manifest self-collision model.",
                    flush=True,
                )
        except Exception as exc:  # never let a geometry hiccup block the boot
            print(
                f"[deploy_e2e] MJCF self-collision lowering failed: {exc!r}; "
                "using manifest geometry",
                flush=True,
            )
    if workcell is not None and workcell.extra_allowed_collision_pairs:
        before = list(collision_params.get("collision_allowed_pairs", []))
        collision_params = merge_extra_allowed_pairs(
            collision_params, workcell.extra_allowed_collision_pairs
        )
        after = list(collision_params.get("collision_allowed_pairs", []))
        if len(after) > len(before):
            for a, b in workcell.extra_allowed_collision_pairs:
                print(f"[deploy_e2e] ACM +pair {a}<->{b} (deploy override)", flush=True)
    kernel_params = {**kernel_params_from_envelope(envelope), **collision_params}
    kernel_params["use_sim_time"] = use_sim_time
    # Actuated joint order (length n_dof) so the kernel maps /joint_states (named) into q_meas
    # in the action's dof index space — same order as the per-joint envelope arrays +
    # collision_dof_index. `collision_seed_dt_s` is the velocity-integration look-ahead step;
    # 0.0 keeps the conservative reactive (measured-config) check only. Deliberate: the only
    # JOINT_VELOCITY emitter in-tree is the robocasa BASE chunk, whose dofs are listed in
    # collision_base_dofs and zeroed before FK, so integrating them is a no-op. dt>0 would help
    # only a future fixed-base velocity arm, and requires validating the chunk's velocity units
    # match this dt (wrong dt mispredicts and could under-report) — stays off (fail-safe) until
    # validated.
    kernel_params["collision_joint_names"] = [j.name for j in description.joints]
    kernel_params["collision_seed_dt_s"] = 0.0
    # deploy-sim publishes /joint_states only as fast as the sim steps, which
    # slows to ~3 Hz under heavy VLA inference (the sim advances on the same host
    # the policy runs on). A 200 ms seed deadline would fail-closed on every
    # chunk; 1 s is a safe backstop here because when stepping is slow the arm
    # also moves slowly in sim-time, so a wall-stale seed is still spatially
    # accurate. Real hardware (30 Hz+ /joint_states) never approaches this bound.
    kernel_params["collision_state_deadline_ms"] = 1000.0
    # Dof indices of the planar mobile-base joints (manifest base_joints). The kernel zeroes
    # these before the base-relative collision FK so a mobile manipulator's arm is checked in
    # the base_link frame the world/voxel grid lives in. Empty for fixed-base arms.
    #
    # ``collision_base_dofs`` is omitted when empty: launch_ros's evaluate_parameter_dict
    # normalises a Python list to a typed array, and an EMPTY list collapses to ``()``, which
    # ensure_argument_type rejects ("got '()' of type tuple"). Empty for every fixed-base arm
    # (openarm, so101, franka_panda, ur5e, ur10e, …) — the majority of in-tree robots — so
    # passing it unconditionally crashed the whole launch before any node started. The kernel
    # declares its own ``[]`` default for this parameter, so omitting it means "no base dofs to
    # zero" (same semantics as the ``lifecycle_peer_node_ids`` guard 90 lines below).
    _base_joint_set = set(getattr(description, "base_joints", None) or [])
    _collision_base_dofs = [
        i for i, j in enumerate(description.joints) if j.name in _base_joint_set
    ]
    if _collision_base_dofs:
        kernel_params["collision_base_dofs"] = _collision_base_dofs
    # Predictive Cartesian: the EE control link for the
    # Jacobian look-ahead (deepest collision link = wrist/tip). -1 (no collision
    # model) leaves predictive Cartesian off; the reactive measured-config check
    # is the floor regardless. Base dofs above are blocked from the arm Jacobian.
    kernel_params["collision_ee_link_index"] = ee_link_index_from_collision_params(collision_params)
    # The world map's coverage, measured from the same collision model and base dofs the
    # kernel is handed (refused here when the bridge could not publish it).
    coverage = (
        _coverage_ball(
            collision_params,
            [j.position_limits for j in description.joints],
            _collision_base_dofs,
            _octomap_resolution(hal_mode),
        )
        if enable_octomap
        else None
    )
    if coverage is not None:
        print(
            f"[deploy_e2e] world map coverage: ball r={coverage.radius} m about "
            f"{coverage.centre} in {description.base_frame}; octomap input clip "
            f"{coverage.clip_lo}..{coverage.clip_hi}",
            flush=True,
        )
    # When octomap is enabled, turn on the kernel's allocation-free capsule-vs-voxel
    # world-collision check and subscribe /openral/world_voxels (published by the octomap
    # bridge below). max_cells covers the bridge's default 2×2×2 m @ 0.05 grid (64k cells) with
    # headroom; margin inflates obstacles conservatively. Fail-closed staleness/over-capacity
    # semantics are the kernel's.
    #
    # The check tests each robot link CAPSULE against the grid, so it needs a collision model
    # with links: the kernel hard-fails ``on_configure`` if a geometric check is enabled but
    # ``collision_n_links == 0``. Robots that declare a depth sensor but no collision geometry
    # (e.g. panda_mobile) still get the map produced (octomap_server + bridge launch below for
    # observability), but the kernel voxel check stays off so the kernel configures cleanly on
    # its scalar envelope.
    has_collision_capsules = int(collision_params.get("collision_n_links", 0)) > 0
    if vision_attachment_enabled and not has_collision_capsules:
        from openral_core.exceptions import ROSConfigError

        # Coupling rule: the vision leg never runs without the kernel's attached check, and
        # that check needs the robot's collision model.
        raise ROSConfigError(
            f"enable_vision_attachment is on but robot {description.name!r} declares no "
            "collision geometry, so the kernel could not check the payload the leg publishes."
        )
    if has_collision_capsules and _attached_collision_enabled(hal_mode, vision_attachment_enabled):
        kernel_params = {
            **kernel_params,
            "attached_collision_enabled": True,
            "attached_collision_margin_m": 0.0,
            "attached_collision_deadline_ms": _attached_collision_deadline_ms(hal_mode),
            # No tolerance override (HZ-0095-2). This used to be raised to the
            # octomap resolution because a legitimate support contact read as
            # ~one voxel of penetration and there was nothing else to absorb it.
            # The support-contact witness (ADR-0092 D6) accounts for that
            # discretisation geometrically against the attested support plane,
            # so the parameter goes back to being what its name says — physical
            # slack — and keeps the honest 1 mm kernel default.
            "attached_max_objects": 8,
            "attached_max_primitives": 16,
            "attached_max_touch_links": 32,
        }
    # Grasp-target exemption (real pick-and-place design §2.1; ADR draft in
    # docs/reference/real-pick-place-adr-drafts.md). The allowlist is ALWAYS the manifest's
    # gripper links, on or off, so the kernel resolves the same names either way and a
    # GraspDeclaration can never name a link the robot does not grip with. Omitted when
    # empty (an empty list has no ROS parameter type), and refused when the flag is on.
    grasp_contact_links = [j.child_link for j in description.joints if j.role == "gripper"]
    if grasp_allowance_enabled and not grasp_contact_links:
        from openral_core.exceptions import ROSConfigError

        raise ROSConfigError(
            f"grasp_allowance_enabled is on but robot {description.name!r} declares no "
            "role: gripper joint, so there is no contact link the exemption could apply to."
        )
    hal_file_params = _hal_file_params(hal_params_file, hal_node_name)
    # On real the HAL's vision target leg is the only grasp-region producer (in sim the HAL's
    # MuJoCo evidence tracker is), so the exemption must not arm without it — checked here as
    # well as in `openral deploy run`, so a bare `ros2 launch` cannot skip it.
    if (
        grasp_allowance_enabled
        and hal_mode != "sim"
        and not (
            vision_attachment_enabled
            and hal_file_params.get("vision_attachment_enabled") is True
            and hal_file_params.get("vision_attachment_grasp_target_enabled") is True
        )
    ):
        from openral_core.exceptions import ROSConfigError

        raise ROSConfigError(
            "grasp_allowance_enabled on the real path needs enable_vision_attachment:=true and "
            "the HAL's vision_attachment_enabled and vision_attachment_grasp_target_enabled "
            f"true in hal_params_file ({hal_params_file}): the kernel would arm the "
            "grasp-target exemption with no producer measuring its region."
        )
    kernel_params["grasp_allowance_enabled"] = grasp_allowance_enabled
    # How old a producer-measured grasp/place region may be and still exempt anything. The
    # regions are measured off the voxel map, so the bound is derived from that map's own
    # freshness deadline: twice it (the kernel's own default, passed explicitly so the
    # launch record shows the value in force). The kernel refuses above 2 x its voxel cap.
    kernel_params["grasp_region_max_age_s"] = 2.0 * world_voxel_deadline_s
    kernel_params["place_region_max_age_s"] = 2.0 * world_voxel_deadline_s
    if grasp_contact_links:
        kernel_params["grasp_contact_links"] = grasp_contact_links
    if enable_octomap and has_collision_capsules and enable_octomap_kernel_check:
        assert coverage is not None  # reason: computed whenever enable_octomap
        kernel_params = {
            **kernel_params,
            "world_voxel_enabled": True,
            # Sim uses exact digital-twin OBBs plus conservative occupied cubes;
            # No additional sim margin avoids vetoing trained close-contact
            # manipulation; exact OBB-vs-cube overlap still E-stops. Real
            # deploy keeps 2 cm.
            "world_voxel_margin_m": _world_voxel_margin_m(hal_mode),
            # Derived from the robot's coverage ball and the octree resolution rather
            # than pinned. See `_world_voxel_max_cells` for why a hand-kept derived
            # constant is the wrong shape here.
            "world_voxel_max_cells": _world_voxel_max_cells(
                coverage.radius, _octomap_resolution(hal_mode)
            ),
            "world_voxel_deadline_ms": world_voxel_deadline_s * 1000.0,
            # How old the world behind a grid may be at check time (Entry 034).
            "world_voxel_data_age_budget_ms": rig.world_voxel_data_age_budget_s * 1000.0,
        }

    kernel_params = {**kernel_params, **_collision_scale_params()}

    # Run identity for the dashboard's Identity card. These
    # ride as OTLP resource attributes on every node so run mode / id /
    # git sha populate regardless of which span family the operator is
    # looking at.
    otel_env: dict[str, str] = {
        "OTEL_RESOURCE_ATTRIBUTES": _run_resource_attrs(hal_mode),
    }
    # ``--no-dashboard`` is a true headless mode: don't forward an OTLP
    # endpoint nobody is listening on. ``openral_observability._sdk``
    # treats an absent ``OTEL_EXPORTER_OTLP_ENDPOINT`` as a no-op and
    # skips installing the BatchSpanProcessor / PeriodicExportingMetricReader
    # / BatchLogRecordProcessor — so SIGINT teardown is near-instant.
    # When the dashboard IS running (default), point every node at it on
    # ``dashboard_port`` over OTLP/HTTP-protobuf. Without this guard
    # ``--no-dashboard`` left every node blocked for ~30s on connection
    # retries to a port nothing was listening on, stalling every
    # headless caller (CI runs, audit tools, batch scripts).
    if enable_dashboard:
        otel_env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://127.0.0.1:{dashboard_port}"
        otel_env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    reasoner_env = {**otel_env, "OPENRAL_REASONER_MODEL": reasoner_model}
    max_tokens = os.environ.get("OPENRAL_REASONER_MAX_TOKENS")
    if max_tokens or reasoner_model in {"gpt-5.5", "gpt-5.6"}:
        reasoner_env["OPENRAL_REASONER_MAX_TOKENS"] = max_tokens or "16384"
    if reasoner_endpoint:
        reasoner_env["OPENRAL_REASONER_ENDPOINT"] = reasoner_endpoint

    dashboard = ExecuteProcess(
        cmd=[_RAL_EXECUTABLE, "dashboard", "--port", dashboard_port],
        name="openral_dashboard",
        output="screen",
    )

    safety_kernel = LifecycleNode(
        package="openral_safety_kernel",
        executable="safety_kernel_node",
        name="openral_safety_kernel",
        namespace="",
        parameters=[kernel_params],
        additional_env=otel_env,
        output="screen",
    )

    # -- Defense-in-depth E-stop sources (CLAUDE.md section 3 "Safety") -------
    # Three nodes, each its own process, each an independent producer of
    # /openral/estop, so a crash in the in-band path (openral_safety, the C++
    # kernel, the runner) cannot leave motors energised. Both hal_modes get all
    # three: a sim graph is where the wiring is exercised before a real arm.
    deadman_watchdog = LifecycleNode(
        package="openral_safety_watchdog",
        executable="deadman_watchdog_node.py",
        name="openral_deadman_watchdog",
        namespace="",
        parameters=[
            {
                # The runner's own action-status topic, NOT an advisory task
                # string: a dead runner cannot publish a terminal status, so
                # the window stays open and the silence still fires. The
                # advisory /openral/reward/active_task has two publishers, and
                # the reasoner's dispatch watchdog clears it exactly when the
                # runner dies — which would silence this node's headline case.
                "arm_status_topic": "/openral/execute_rskill/_action/status",
                # Gap between two consecutive chunks is one VLA inference, not
                # the node's 0.2 s standalone default. Reuses the horizon the
                # actuation path already applies to this stream
                # (action_applied_timeout_s). Conservative first value;
                # tightening it toward the chunk cadence is a safety-WG call
                # backed by measured per-policy gaps.
                "safe_action_deadline_s": 8.0 if hal_mode == "sim" else 5.0,
                # Bound on goal-accepted -> first chunk, covering a cold policy
                # load. Unbounded, a runner that dies before its first chunk
                # holds the window open and is never braked.
                "first_chunk_deadline_s": 120.0,
                # For the HZ-0096-1 staleness check on /openral/safety_status.
                # The deadlines themselves stay on wall time: a watchdog must
                # measure real elapsed time, and sim time can pause.
                "use_sim_time": use_sim_time,
            }
        ],
        additional_env=otel_env,
        output="screen",
    )
    # Pendant bridge. ``hardware_estop_device`` is empty by default; the node
    # then refuses to configure and stays UNCONFIGURED (see
    # HardwareEstopNode._resolve_read_source). It is still spawned so the
    # refusal is visible on every deploy and ``ros2 lifecycle get`` tells the
    # truth. This repo ships no pendant driver, so a declared device also needs
    # a vendor subclass.
    hardware_estop = LifecycleNode(
        package="openral_safety_watchdog",
        executable="hardware_estop_node.py",
        name="openral_hardware_estop",
        namespace="",
        parameters=[{"device": hardware_estop_device, "use_sim_time": use_sim_time}],
        additional_env=otel_env,
        output="screen",
    )
    # channel_label stays at the node's "unknown_human_channel" default: no
    # in-repo node publishes /openral/human_estop (the dashboard stop button
    # publishes /openral/estop directly), so naming a channel would assert a
    # producer that does not exist. Brought up anyway so an external adapter
    # can attach to a running graph.
    human_estop_forwarder = LifecycleNode(
        package="openral_human_estop",
        executable="forwarder_node.py",
        name="openral_human_estop_forwarder",
        namespace="",
        parameters=[{"use_sim_time": use_sim_time}],
        additional_env=otel_env,
        output="screen",
    )
    # Lifecycle peer node ids the Reasoner should surface to
    # the LLM via `LifecycleTransitionTool`. Today only slam_toolbox is
    # opt-in; future managed services (RTAB-Map, perception trees) will
    # append themselves here under their own `enable_<svc>` launch args.
    lifecycle_peer_node_ids: list[str] = []
    # GPU peers the reasoner AUTO-deactivates before a VLA dispatch
    # and reactivates after (distinct from the LLM-facing palette peers above).
    vram_lifecycle_peers: list[str] = []
    if enable_slam:
        lifecycle_peer_node_ids.append("openral_slam_toolbox")
    if enable_object_detector:
        # Expose the detector as a lifecycle peer so the reasoner can
        # DEACTIVATE it (freeing the detector's VRAM) before dispatching a
        # co-resident grab policy on a memory-constrained GPU.
        lifecycle_peer_node_ids.append("openral_ros_image_detector")
        # …and AUTO-free it: the reasoner deactivates the detector before each
        # execute_rskill and reactivates it on completion, so an 8 GB card does
        # not OOM with the detector (~1.3 GB) co-resident with the VLA (~4.5 GB).
        vram_lifecycle_peers.append("openral_ros_image_detector")
        # Each on-demand locator is its own lifecycle node, so it is an
        # independent LLM-facing peer (toggle) and VRAM peer (evict before a VLA;
        # LocateAnything is 5 GB so this matters on an 8 GB card).
        for _spec in locator_specs:
            lifecycle_peer_node_ids.append(_spec["node"])
            vram_lifecycle_peers.append(_spec["node"])
    # ``lifecycle_peer_node_ids`` is omitted when empty: launch_ros's
    # evaluate_parameter_dict normalises a Python list to a typed array and
    # an EMPTY list collapses to ``()``, which ensure_argument_type rejects
    # ("got '()' of type tuple"). The list is empty whenever no opt-in peer
    # service (slam_toolbox) is enabled — e.g. every panda_mobile boot — so
    # passing it unconditionally crashed the whole launch before any node
    # started. The reasoner declares its own ``[]`` default, so omitting the
    # param is equivalent to "no peers".
    reasoner_params: dict[str, object] = {
        "robot_yaml": robot_yaml,
        "rskill_search_paths": [_RSKILLS_DIR],
        # Tell the reasoner which deploy path it is on so its
        # action-mode palette gate matches the HAL this launch brings up.
        "hal_mode": hal_mode,
        # A grounded grasp object's / place surface's search box is padded by one cell of the
        # map the producer searches: this launch's octree resolution, the same value
        # octomap_server gets.
        "grasp_target_voxel_m": _octomap_resolution(hal_mode),
    }
    if lifecycle_peer_node_ids:
        reasoner_params["lifecycle_peer_node_ids"] = lifecycle_peer_node_ids
    # Same empty-list-omission rule as lifecycle_peer_node_ids
    # (launch_ros rejects an empty typed array); the reasoner defaults to [].
    if vram_lifecycle_peers:
        reasoner_params["vram_lifecycle_peers"] = vram_lifecycle_peers
    # Preload a persisted scene graph as the reasoner's read-only
    # spatial-memory query backend when a path is provided.
    if spatial_memory_path:
        reasoner_params["spatial_memory_path"] = spatial_memory_path
    # Load the self-maintained MEMORY.md (read path) and enable the
    # memory_write / memory_search tools when a bundle path is provided.
    if memory_md_path:
        reasoner_params["memory_md_path"] = memory_md_path
    # Accumulate the durable scene graph live from the object-detection
    # producer's WorldState.detected_objects (auto-creates an empty backend when
    # no path is preloaded).
    reasoner_params["spatial_memory_ingest"] = spatial_memory_ingest
    # Offer the read-only locate_in_view tool to the LLM only when an on-demand
    # locator (``--object-detector-locator``) is actually in the graph.
    # ``detector_node_wiring`` (detector_factory.py) makes the two detector
    # modes mutually exclusive at the node: a continuous detector
    # (``--object-detector``; the always-on RT-DETR/omdet background producer
    # feeding WorldState) runs with ``serve_on_demand=False`` and never
    # constructs the LocateInView service at all — it streams
    # ``/openral/perception/objects`` and nothing else, on any path, under any
    # alias. Only ``on_demand`` mode (an ``--object-detector-locator`` entry)
    # sets ``serve_on_demand=True`` and advertises
    # /openral/perception/<alias>/locate_in_view.
    #
    # This composite used to be `enable_object_detector or bool(locator_specs)`
    # on the false claim (this comment, pre-fix) that both modes served the
    # service. `--object-detector` alone surfaced the tool to the LLM with no
    # backing service and no `default_on_demand_detector` to route to:
    # reproduced live (openral deploy sim --config scenes/deploy/libero_pnp.yaml
    # --object-detector --initial-task "..."), the reasoner correctly called
    # locate_in_view per its own system prompt, got
    # "/openral/perception/default/locate_in_view not on graph; skipping" on
    # every tick, and the mission stalled — a phantom capability, not a live
    # one. A lean ``--no-object-detector`` deploy with a locator still grounds
    # fine; only the composite with the continuous leg was ever wrong.
    reasoner_params["detector_available"] = bool(locator_specs)
    # Offer the read-only query_task_progress tool only when a reward
    # monitor is co-active (otherwise the tool would dispatch to a dead service).
    reasoner_params["task_progress_available"] = enable_reward_monitor
    # Offer the read-only query_scene tool only when the scene VLM is co-active
    # (otherwise the tool would dispatch to a dead service). The reasoner's own
    # default is False, so a deploy without this leg never sees the tool.
    reasoner_params["scene_query_available"] = enable_scene_vlm
    # Give the reasoner the SAME reward-model manifest the
    # monitor loads (incl. the robometer default when the arg is empty — mirrors
    # the monitor's resolution below) so it reads the active RewardContract
    # calibration (three-tier band edges + default patience) instead of the
    # module-level system defaults.
    if enable_reward_monitor:
        reasoner_params["reward_manifest_path"] = reward_monitor_manifest or str(
            pathlib.Path(_RSKILLS_DIR) / "robometer-4b" / "rskill.yaml"
        )
    # The default on-demand locator the reasoner routes to when a
    # locate_in_view call leaves ``detector`` empty (the first locator brought up).
    if locator_specs:
        reasoner_params["default_on_demand_detector"] = locator_specs[0]["alias"]
    # The completion-camera topic is raw (bottom-up for LIBERO/MuJoCo);
    # mirror OPENRAL_DASHBOARD_FLIP_180 so the VLM judges an upright frame (the topic
    # itself is not flipped — sim_sensor_bridge flips only the dashboard thumbnail).
    # The node's own default names a ``top`` camera that most robots do not declare; derive
    # the view from the cameras that actually publish on this deploy (same rule as the
    # detector) so the VLM completion check gets frames on every robot — on a real cell that
    # means a bound camera. Empty disables the subscription when there is none.
    _completion_camera = _primary_rgb_camera(publishing)
    reasoner_params["completion_camera_topic"] = (
        camera_topic(_completion_camera) if _completion_camera else ""
    )
    reasoner_params["completion_camera_flip_180"] = os.environ.get(
        "OPENRAL_DASHBOARD_FLIP_180", ""
    ) not in ("", "0", "false", "False")
    reasoner = LifecycleNode(
        package="openral_reasoner_ros",
        executable="reasoner_node.py",
        name="openral_reasoner",
        namespace="",
        parameters=[reasoner_params],
        additional_env=reasoner_env,
        output="screen",
    )
    prompt_router = LifecycleNode(
        package="openral_prompt_router",
        executable="prompt_router_node.py",
        name="openral_prompt_router",
        namespace="",
        parameters=[{"startup_prompt": initial_task_prompt}],
        additional_env=otel_env,
        output="screen",
    )
    # The vision bridge reads the same voxel path the kernel checks, so its bounds derive from
    # this launch's single sources: a grid is usable no longer than the kernel's voxel
    # deadline, and a released payload is clear once it sits the kernel's world margin plus
    # one octree cell away. First in the list, so an explicit `--hal` value still wins.
    hal_derived_params: list[dict[str, float | str]] = (
        [
            {
                "vision_attachment_grid_max_age_s": world_voxel_deadline_s,
                "vision_attachment_release_clear_m": _world_voxel_margin_m(hal_mode)
                + _octomap_resolution(hal_mode),
                # The self-filter's output for the camera octomap maps: the grasp target
                # fit drops the robot's own points as the map does (empty: no filter).
                "vision_attachment_self_filtered_cloud_topic": _SELF_FILTERED_CLOUD_TOPIC
                if enable_octomap
                and _cloud_shows_the_robot(hal_mode, scene_backend)
                and has_collision_capsules
                else "",
            }
        ]
        if vision_attachment_enabled
        else []
    )
    # Whatever the source (derived above, or a params-file value that wins over it), the
    # vision leg may never trust a grid the kernel would refuse as stale, nor drop a released
    # payload's record nearer than the kernel's margin plus one cell.
    if vision_attachment_enabled or hal_file_params.get("vision_attachment_enabled") is True:
        effective = {**(hal_derived_params[0] if hal_derived_params else {}), **hal_file_params}
        clear_floor_m = _world_voxel_margin_m(hal_mode) + _octomap_resolution(hal_mode)
        grid_max_age_s = effective.get("vision_attachment_grid_max_age_s")
        release_clear_m = effective.get("vision_attachment_release_clear_m")
        if isinstance(grid_max_age_s, int | float) and grid_max_age_s > world_voxel_deadline_s:
            from openral_core.exceptions import ROSConfigError

            raise ROSConfigError(
                f"vision_attachment_grid_max_age_s={grid_max_age_s} exceeds the kernel's "
                f"world_voxel_deadline_s={world_voxel_deadline_s}: the vision leg would vouch "
                "for a region from a grid the kernel itself refuses as stale."
            )
        if isinstance(release_clear_m, int | float) and release_clear_m < clear_floor_m - 1e-9:
            from openral_core.exceptions import ROSConfigError

            raise ROSConfigError(
                f"vision_attachment_release_clear_m={release_clear_m} is below the kernel's "
                f"world_voxel_margin_m + octomap resolution ({clear_floor_m}): a released "
                "payload's record would drop while the kernel still measures it inside its margin."
            )
    # Sim twin: the HAL publishes joint states under the manifest's logical names; a URDF
    # that names its joints differently (the OpenArm's vendored one) gets a renamed copy
    # on `~/urdf_joint_states` for robot_state_publisher, or no moving link reaches /tf
    # (`openral_hal.resolver.urdf_joint_names`). Real: the vendor broadcaster's own names.
    rsp_urdf_joint_names: list[str] = []
    if hal_mode == "sim" and description.assets.urdf is not None:
        _rsp_urdf = _resolve_urdf_path(description.assets.urdf.ref, pathlib.Path(robot_yaml).parent)
        if _rsp_urdf is not None:
            from openral_hal.resolver import urdf_joint_names

            rsp_urdf_joint_names = urdf_joint_names(
                description, pathlib.Path(_rsp_urdf).read_text(encoding="utf-8")
            )
    hal = LifecycleNode(
        package=hal_package,
        executable=hal_executable,
        name=hal_node_name,
        namespace="",
        # Clock domain via the graph-wide flag. The HAL is the clock
        # authority — it stamps /scan, odom→base_link TF and joint_states.
        # Host-wall origin is unchanged; a simulation clock origin makes those
        # stamps sim-time, coherent with the HAL's /clock publisher.
        parameters=[
            *hal_derived_params,
            *([{"urdf_joint_names": rsp_urdf_joint_names}] if rsp_urdf_joint_names else []),
            hal_params_file,
            {"use_sim_time": use_sim_time},
        ],
        additional_env=otel_env,
        output="screen",
    )
    # Derive ``camera_names`` from the robot manifest's RGB sensors so the WorldState aggregator
    # subscribes to the topics the HAL actually publishes. Hard-coding
    # ``[top, left_wrist, right_wrist]`` broke panda_mobile / robocasa-kitchen: that robot
    # declares ``camera1/camera2/camera3`` (robocasa renders ``robot0_agentview_left_image``
    # etc., remapped to ``cameraN``), so WorldState subscribed to topics nothing publishes and
    # the rldx adapter's ``observation.images['camera1']`` lookup raised
    # ``ROSConfigError: rldx adapter expects observation.images['camera1']; got []``. A robot
    # declaring no RGB sensors (pure-base robots) subscribes to none — a guessed fallback name
    # would subscribe to a topic nothing publishes.
    rgb_camera_names = [s.name for s in description.sensors if s.modality == "rgb"]
    # Workcell-mounted cameras (DeployScene.sensors) publish on the same
    # `/openral/cameras/<name>/image` prefix via the real-deploy sensor
    # leg — WorldState must subscribe to them too, but only on `hal_mode:=real`:
    # in sim, SimSensorBridge renders the manifest's cameras only, so a
    # scene-only camera would be a subscription with no publisher (a stale
    # diagnostic forever).
    if deploy_config and hal_mode == "real":
        scene_rgb = [
            s.name for s in scene_sensors if s.modality == "rgb" and s.name not in rgb_camera_names
        ]
        rgb_camera_names = [*rgb_camera_names, *scene_rgb]

    # Cameras that will actually publish on a real deploy: a declared RGB sensor
    # only gets a reader (and therefore a topic) when it carries a
    # `deploy_binding`. A sim-only sensor (a MuJoCo render the manifest never
    # bound to hardware) has none — so on a real cell that slot has zero
    # publishers while the Foxglove bridge still advertises the channel (its
    # allowlist is the pattern `/openral/cameras/.*/image`), and the panel reads
    # "Image topic does not exist", indistinguishable from a broken camera. The
    # robot manifest fixes that by binding the slot to real hardware. The
    # Foxglove layout is generated from this list rather than a hardcoded
    # default, which cannot know the scene (see `_write_foxglove_layout`).
    # In sim the publishers are SimSensorBridge's renders of the manifest's RGB
    # sensors (deploy_binding or not) — exactly `rgb_camera_names`, which already
    # leaves out scene-only hardware cameras there.
    # On real, read the merged list: the manifest's bound cameras plus the
    # scene's bound workcell cameras (`merge_deploy_sensors` refuses a name clash).
    bound_rgb_camera_names = (
        [s.name for s in publishing if s.modality == "rgb"]
        if hal_mode == "real"
        else list(rgb_camera_names)
    )
    runtime = Node(
        package="openral_rskill_ros",
        executable="runtime_node",
        prefix=_cpuset_prefix("OPENRAL_RUNTIME_CPUSET"),
        parameters=[
            {
                "robot_yaml": robot_yaml,
                # `[""]` is the node's own "no cameras" default: launch_ros cannot
                # type an empty list and refuses it when the node starts.
                "camera_names": rgb_camera_names or [""],
                # World-state object-lift depth fallback: the same cloud octomap maps — the
                # scene-pinned driver cloud when there is one (a real ZED publishes on its
                # own topic, not the sim bridge's), else in sim the manifest depth
                # sensor's; "" (a real deploy with nothing pinned) disables the fallback.
                "object_depth_points_topic": deploy_cloud_topic(
                    description.sensors, pinned=octomap_cloud_topic, hal_mode=hal_mode
                ),
                # 512 px RoboCasa renders can arrive at ~0.6 Hz wall time while
                # idle. Keep joint/EE diagnostics at 0.5 s, but give simulated
                # cameras enough room for one slow frame without stale flapping.
                "image_staleness_limit_s": 5.0 if hal_mode == "sim" else 0.5,
                # Joint state older than this aborts the blocking wait as a
                # perception fault. The node has no default. Real: the manifest's
                # safety.joint_state_staleness_limit_s (the HAL's window too), plus
                # two republish periods when the runner reads the HAL's
                # ~/joint_states; sim: the former node default (audit C F3).
                "joint_state_staleness_limit_s": _runner_joint_state_staleness_s(
                    description,
                    hal_mode,
                    republished=runtime_joint_states_topic == f"/{hal_node_name}/joint_states",
                ),
                # One grouped action may synchronously attach a payload, then
                # wait for a transparent depth frame + the next OctoMap raster
                # before acknowledging application. Real HALs keep the 5 s
                # transport watchdog; sim gets a bounded 8 s transaction.
                "action_applied_timeout_s": 8.0 if hal_mode == "sim" else 5.0,
                "gripper_convention": gripper_convention,
                "rskill_search_paths": [_RSKILLS_DIR],
                "reset_to_pose_service": reset_to_pose_service,
                "approach_skill_id": approach_skill_id,
                # Load the scene's policy before any goal exists, so the
                # deadman watchdog's first-chunk window (armed on goal accept)
                # never has to cover a multi-minute cold load. Empty = the
                # first goal loads its own skill, as before.
                "preload_rskill_id": preload_rskill_id,
                "preload_rskill_revision": preload_rskill_revision,
                "preload_prompt": preload_prompt,
                # Applied by runtime_node to both composed nodes (world_state
                # ingest + the runner's joint-state cache); "" = /joint_states.
                "joint_states_topic": runtime_joint_states_topic,
                # ADR-0097 — the scene's committed place-phase declaration for a
                # direct dispatch. Empty (every scene today) = no declaration, so
                # no place witness can arm and payload contact mid-carry stops.
                "place_declaration_json": place_declaration_json,
                # Grasp-phase sibling (real pick-and-place design §2.1).
                "grasp_declaration_json": grasp_declaration_json,
                # Approach-armed grasp target (§2.2): a goal-scope declaration per
                # goal, only when the exemption is on AND the HAL's grasp-target
                # leg measures around an approaching hand.
                "grasp_approach_enabled": grasp_allowance_enabled
                and float(hal_file_params.get("vision_attachment_grasp_target_approach_m") or 0.0)
                > 0.0,
                # Place mirror (§2.3): a goal-scope place declaration per goal, only when
                # the HAL's place-target leg runs — it attaches its measured region to
                # that declaration and never declares on its own (no goal, no allowance).
                "place_approach_enabled": vision_attachment_enabled
                and hal_file_params.get("vision_attachment_enabled") is True
                and hal_file_params.get("vision_attachment_place_target_enabled") is True,
                # Attach the WorldCloudBridge → dashboard world.pointcloud when a
                # voxel cloud exists: octomap's centers, or (mono visual SLAM)
                # nvblox's ESDF cloud so the card shows the vision-built voxels.
                # Dashboard-only (PNG + `world.pointcloud` span per cloud): with the
                # dashboard off it still cost the runner's executor a Python
                # deserialize of every octomap cloud, so it is gated on it.
                "enable_world_cloud_bridge": (enable_octomap or bool(slam_mono_camera))
                and enable_dashboard,
                "world_cloud_topic": (
                    "/openral_nvblox/static_esdf_pointcloud"
                    if (slam_mono_camera and not enable_octomap)
                    else ""
                ),
                # Credit the dashboard SLAM card to the node that builds /map:
                # nvblox on the visual backend (composed for mono RGBD always,
                # and for stereo when nav2 needs a cost map); empty keeps the
                # slam_toolbox default of the lidar backend.
                "slam_source_node": (
                    "openral_nvblox"
                    if (slam_backend == "visual" and (slam_mono_camera or enable_nav2))
                    else ""
                ),
                # The runtime node (WorldState aggregator +
                # the GStreamer/runner sensor readers + skill_runner) must share
                # the graph-wide clock domain. Under a simulation clock origin the HAL
                # stamps camera/state data on sim time; a wall-clock runtime
                # would see it as ~1.78e9 s stale and drop every frame at the
                # WorldState staleness gate. Default false keeps it wall-clock.
                "use_sim_time": use_sim_time,
                # When set, compose_runtime attaches the
                # DatasetRecorderBridge and records the session to this mcap.
                "dataset_out": dataset_out,
                "dataset_repo_id": dataset_repo_id,
                "dataset_license": dataset_license,
                # Real deploys — when set, the runtime opens the deploy
                # config's camera readers (sensor_leg.py) and publishes
                # them onto the WorldState image topics. Gated on hal_mode,
                # not on the arg: `deploy sim` forwards deploy_config too (for
                # the scene's boot timeout + sensor mounts), but a sim's
                # cameras come from the HAL bridge, never physical readers.
                "deploy_config": deploy_config if hal_mode == "real" else "",
                # The RESOLVED consumer flags, not the scene's raw (tri-state)
                # ones. The scene YAML may leave enable_object_detector /
                # enable_slam as None ("auto") and the deploy CLI resolves
                # those to on/off at launch time — but the runtime node
                # re-reads the original YAML, so without this forward it
                # would treat an auto-enabled detector/SLAM leg as OFF and
                # rate-cap + downscale the very cameras those nodes consume
                # (sensor_leg.apply_launch_overrides). These launch values
                # gate the actual detector/SLAM nodes above, so they are the
                # ground truth of which subscribers exist in this graph.
                "resolved_enable_object_detector": "true" if enable_object_detector else "false",
                "resolved_enable_slam": "true" if enable_slam else "false",
                "resolved_slam_stereo_cameras": slam_stereo_cameras,
                "resolved_slam_mono_camera": slam_mono_camera,
            }
        ],
        additional_env=otel_env,
        output="screen",
    )

    autostart: list = []
    # The safety_kernel MUST reliably reach ACTIVE: if it stays INACTIVE it
    # publishes neither /openral/safe_action (so the HAL never steps the sim →
    # the runner feeds the policy a FROZEN observation.state → blind open-loop
    # arm fold) nor /openral/estop (so no E-stop fires). The launch_ros
    # ``OnStateTransition`` matcher hits the same Jazzy race documented for the
    # HAL below and intermittently drops the ACTIVATE on slow first-boots, so
    # route the kernel through the active-polling ``tools/lifecycle_autostart.py``
    # exactly like the HAL — a missed transition event can no longer leave the
    # safety kernel (and therefore the whole graph) running unprotected.
    _kernel_autostart_path = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
    autostart.append(
        ExecuteProcess(
            cmd=[
                sys.executable,
                _kernel_autostart_path,
                "--node",
                "/openral_safety_kernel",
                "--target",
                "active",
                "--service-timeout-s",
                "60.0",
                "--transition-timeout-s",
                "120.0",
            ],
            output="log",
        )
    )
    # The E-stop sources take the same poll-based autostart path as the kernel:
    # a dropped ACTIVATE would leave a brake source sitting in INACTIVE, i.e.
    # silently absent, which is the exact failure this wiring removes.
    #
    # The deadman is REQUIRED. --required makes an absent node a non-zero exit
    # (the default treats "service never appeared" as informational), and the
    # OnProcessExit below turns that into a graph shutdown. A deploy that
    # cannot arm its only independent watchdog must refuse to run rather than
    # run unprotected.
    deadman_autostart = ExecuteProcess(
        cmd=[
            sys.executable,
            _kernel_autostart_path,
            "--node",
            "/openral_deadman_watchdog",
            "--target",
            "active",
            "--required",
            "--service-timeout-s",
            "60.0",
            "--transition-timeout-s",
            "30.0",
        ],
        output="log",
    )
    autostart.append(deadman_autostart)

    def _shutdown_unless_clean(what: str) -> object:
        """OnProcessExit callback: a non-zero exit of ``what`` ends the graph."""

        def _on_exit(event: object, _context: object) -> list:
            returncode = getattr(event, "returncode", None)
            if returncode == 0:
                return []
            return [
                LogInfo(
                    msg=(
                        f"FATAL: {what} exited {returncode}. This graph has no "
                        "independent E-stop source, so it is shutting down instead "
                        "of running unprotected (CLAUDE.md section 3)."
                    )
                ),
                Shutdown(reason=f"{what} exited {returncode}"),
            ]

        return _on_exit

    # Gate the graph on the deadman ARMING (the autostart) and on it STAYING
    # ALIVE (the node itself). Without the second, a Python crash in the
    # watchdog at t+60 s left the graph running unprotected and silent.
    autostart.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=deadman_autostart,
                on_exit=_shutdown_unless_clean("the deadman watchdog autostart"),
            )
        )
    )
    autostart.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=deadman_watchdog,
                on_exit=_shutdown_unless_clean("the deadman watchdog node"),
            )
        )
    )
    # The human forwarder has no in-repo producer, so its absence is not a
    # safety regression: autostart it, but do not gate the graph on it.
    autostart.append(
        ExecuteProcess(
            cmd=[
                sys.executable,
                _kernel_autostart_path,
                "--node",
                "/openral_human_estop_forwarder",
                "--target",
                "active",
                "--service-timeout-s",
                "60.0",
                "--transition-timeout-s",
                "30.0",
            ],
            output="log",
        )
    )
    # The pendant is only autostarted when a device is actually declared.
    # Driving it otherwise produces an expected configure FAILURE on every
    # deploy, and that routine traceback is exactly the noise a REAL E-stop
    # source failure would hide behind.
    if hardware_estop_device:
        autostart.append(
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    _kernel_autostart_path,
                    "--node",
                    "/openral_hardware_estop",
                    "--target",
                    "active",
                    "--service-timeout-s",
                    "60.0",
                    "--transition-timeout-s",
                    "30.0",
                ],
                output="log",
            )
        )

    # Reasoner AND prompt_router both use the robust script-based autostart
    # (tools/lifecycle_autostart.py), not _autostart_lifecycle's launch_ros
    # event handlers — same Jazzy race as HAL/slam_toolbox below. Under a
    # heavy graph (reward monitor + critic loading concurrently with the
    # reasoner's configure) the OnStateTransition(configuring → inactive)
    # handler can miss the transition_event, silently dropping ACTIVATE so
    # the node sits in INACTIVE forever (launch_ros logs "Abandoning wait
    # for /<node>/change_state"; the deploy never reaches the tick loop /
    # the startup prompt never publishes). The script polls the node's
    # state and drives CONFIGURE→ACTIVATE with a generous timeout, immune
    # to the race. Neither node is ever runtime-deactivated, so a one-shot
    # drive to active is behaviour-preserving — same as the HAL block below.
    #
    # prompt_router was on the racy path until a live `--object-detector`
    # deploy (heavier graph than the reasoner-only case the race was first
    # caught on) reproduced it: the reasoner reached ACTIVE via its poll
    # script, but prompt_router's OnStateTransition handler silently missed
    # its own transition_event and stuck in INACTIVE, so `--initial-task`
    # was accepted by the CLI, threaded through `initial_task_prompt`, and
    # then never published — the reasoner ticked at 0.2 Hz with an empty
    # mission and no diagnostic anywhere named the stall.
    _reasoner_autostart_path = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
    if enable_reasoner:
        autostart.append(
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    _reasoner_autostart_path,
                    "--node",
                    "/openral_reasoner",
                    "--target",
                    "active",
                    "--service-timeout-s",
                    "60.0",
                    "--transition-timeout-s",
                    "300.0",
                ],
                output="log",
            )
        )
        autostart.append(
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    _reasoner_autostart_path,
                    "--node",
                    "/openral_prompt_router",
                    "--target",
                    "active",
                    "--service-timeout-s",
                    "60.0",
                    "--transition-timeout-s",
                    "120.0",
                ],
                output="log",
            )
        )
    # HAL autostart goes through ``tools/lifecycle_autostart.py`` rather than
    # ``_autostart_lifecycle`` because launch_ros's ``lifecycle_event_manager`` race on Jazzy
    # (same one as slam_toolbox below) silently swallows ACTIVATE on robocasa-kitchen
    # first-boots: HAL's ``on_configure`` takes ~6 s (MuJoCo + robosuite import + env.reset),
    # and by the time the FSM publishes ``transition_event(inactive)``, the
    # ``OnStateTransition(goal_state="inactive")`` handler's ``EmitEvent(ChangeState=ACTIVATE)``
    # is dropped. End-state: HAL stuck in INACTIVE, no ``on_activate``, no /joint_states, no
    # /odom, no /openral/cameras/*/image publishers — Nav2 + dashboard cameras can't come up.
    # Mirrors the slam_toolbox workaround.
    hal_autostart_path = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
    from openral_hal.sim_bringup import hal_transition_timeout_s

    autostart.append(
        ExecuteProcess(
            cmd=[
                sys.executable,
                hal_autostart_path,
                "--node",
                f"/{hal_node_name}",
                "--target",
                "active",
                "--service-timeout-s",
                "60.0",
                # Derived from the scene, not fixed: a sidecar backend boots
                # inside ``on_configure`` and a measured cold Isaac Sim boot
                # runs past the old 300 s literal. See
                # ``openral_hal.sim_bringup.hal_transition_timeout_s``.
                "--transition-timeout-s",
                hal_transition_timeout_s(deploy_config),
            ],
            output="log",
        )
    )

    # robot_state_publisher: when robot.yaml carries an ``assets.urdf`` ref, launch
    # ``robot_state_publisher`` so the per-link arm + sensor TF chain lands on ``/tf``
    # (consumed by the ``openral_state_adapter`` registry at step time; also by
    # Nav2 / MoveIt / RViz when present). The ref is either:
    #
    # * ``file:<relpath>`` (vendored URDF, resolved against the manifest dir then repo root) or
    #   ``rd:<module>`` (pulled from the ``robot_descriptions`` package, no large file checked
    #   in-tree);
    # * ``ros2://robot_description`` — declared by the detection assembler when the robot
    #   publishes its own URDF on ``/robot_description``. ``resolve_asset`` returns ``None`` for
    #   it, so RSP is skipped (the URDF is already on the bus).
    extra_nodes: list = []

    # Vendor ros2_control bringup, on the real path only — see
    # ``_build_real_bringup_include``. This is what keeps ``deploy run`` a
    # single graph with a single /joint_states publisher.
    vendor_owns_robot_description = False
    if hal_mode == "real":
        real_bringup = _build_real_bringup_include(description.hal.real_bringup)
        if real_bringup is not None:
            extra_nodes.append(real_bringup)
            vendor_owns_robot_description = True
        # Vendor sensor drivers the scene declares (a ZED wrapper, a RealSense
        # node…). Real path only: on the sim path cameras are rendered, not
        # driven. These publish the topics the scene's `ros2_*` sensor bindings
        # read, so they go up with the graph.
        if scene_drivers:
            extra_nodes.extend(_build_driver_includes(scene_drivers, deploy_config))

    robot_description_xml: str | None = None
    urdf_asset = description.assets.urdf
    if urdf_asset is not None:
        urdf_path = _resolve_urdf_path(urdf_asset.ref, pathlib.Path(robot_yaml).parent)
        if urdf_path is not None:
            with open(urdf_path, encoding="utf-8") as fh:
                robot_description_xml = fh.read()
            extra_nodes.append(
                Node(
                    package="robot_state_publisher",
                    executable="robot_state_publisher",
                    name="robot_state_publisher",
                    namespace="",
                    output="log",
                    # When a vendor bringup is in the graph it publishes its own
                    # `/robot_description`, and `controller_manager` reads that
                    # topic on Jazzy. Ours describes the same robot under the
                    # same names but declares `mock_components/GenericSystem`
                    # where the vendor declares the real hardware plugin — so a
                    # controller_manager that latched ours would come up with
                    # mock hardware: controllers active, `/joint_states`
                    # plausible, and the arm never moving. Step off the topic
                    # rather than race for it. `/tf` is unaffected (that is this
                    # node's actual job here) and the manifest URDF stays
                    # readable at the `/openral/` name.
                    remappings=[
                        *(
                            [("robot_description", "/openral/robot_description")]
                            if vendor_owns_robot_description
                            else []
                        ),
                        *(
                            [("joint_states", f"/{hal_node_name}/urdf_joint_states")]
                            if rsp_urdf_joint_names
                            else []
                        ),
                    ],
                    parameters=[
                        {
                            "robot_description": robot_description_xml,
                            # Graph-wide clock domain (see _resolve_clock_origin).
                            # Must match the HAL: with no /clock, sim-time would
                            # pin RSP's TF stamps at 0 while the HAL publishes
                            # odom→base_link on wall-clock — the split that
                            # broke Nav2's TF lookups into the costmap frame.
                            "use_sim_time": use_sim_time,
                            # publish_frequency at 30 Hz matches
                            # the runner's tick rate. Higher rates are
                            # wasted (TF buffer interpolates); lower rates
                            # add latency to the state-vector assembly.
                            "publish_frequency": 30.0,
                        }
                    ],
                    additional_env=otel_env,
                ),
            )
            # Some robots (mobile manipulators, multi-arm setups) need
            # a static transform between the HAL-published ``base_link``
            # and the URDF root (e.g. ``base_link → panda_link0`` when
            # the Franka URDF's root differs from the robot.yaml's
            # ``base_frame``). When ``assets.urdf`` declares
            # ``base_to_root_xyz_rpy`` + ``root_frame``, spawn a
            # ``static_transform_publisher`` to bridge.
            static_xform = urdf_asset.base_to_root_xyz_rpy
            static_root_frame = urdf_asset.root_frame
            if static_xform is not None and static_root_frame is not None:
                x, y, z, roll, pitch, yaw = static_xform
                extra_nodes.append(
                    Node(
                        package="tf2_ros",
                        executable="static_transform_publisher",
                        name=f"static_{description.base_frame}_to_{static_root_frame}",
                        arguments=[
                            "--x",
                            str(x),
                            "--y",
                            str(y),
                            "--z",
                            str(z),
                            "--roll",
                            str(roll),
                            "--pitch",
                            str(pitch),
                            "--yaw",
                            str(yaw),
                            "--frame-id",
                            description.base_frame,
                            "--child-frame-id",
                            static_root_frame,
                        ],
                        output="log",
                    ),
                )

    # Sensor mount poses. A sensor whose manifest entry declares both ``parent_frame`` and
    # ``static_transform_xyz_rpy`` gets that transform on /tf_static, so its readings are
    # located by TF rather than mislabelled into an existing frame. panda_mobile's
    # ``base_scan`` is why this exists: it declared ``frame_id: base_link`` and silently lost
    # its 0.40 m mount offset, handing Nav2 and slam_toolbox every return 0.40 m above where the
    # ray was cast. Same shape and reason as the URDF-root bridge above — the manifest owns the
    # geometry, not the launch file.
    #
    # Manifest sensors then DeployScene sensors (`merge_deploy_sensors`): a scene never names
    # a robot sensor, so a robot sensor's pose is always the manifest's; iterating only the
    # manifest would silently drop the mount publish for a workcell-mounted camera declared
    # entirely at scene level.
    for sensor in merge_deploy_sensors(description.sensors, scene_sensors):
        if sensor.parent_frame is None or sensor.static_transform_xyz_rpy is None:
            continue
        sx, sy, sz, sroll, spitch, syaw = sensor.static_transform_xyz_rpy
        extra_nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"static_{sensor.parent_frame}_to_{sensor.frame_id}",
                arguments=[
                    "--x",
                    str(sx),
                    "--y",
                    str(sy),
                    "--z",
                    str(sz),
                    "--roll",
                    str(sroll),
                    "--pitch",
                    str(spitch),
                    "--yaw",
                    str(syaw),
                    "--frame-id",
                    sensor.parent_frame,
                    "--child-frame-id",
                    sensor.frame_id,
                ],
                output="log",
            ),
        )

    # Opt-in SLAM. The backend is selected by
    # ``slam_backend`` (resolved from capabilities in deploy_sim.py):
    # ``visual`` composes cuVSLAM (camera-based, lidar-less robots);
    # anything else composes slam_toolbox (2D lidar). ``enable_slam`` is
    # the on/off gate; the two are kept consistent upstream.
    if enable_slam:
        # Deferred share-dir lookup so deployments without the
        # openral_slam_bringup package built still launch successfully
        # when enable_slam is left at its default (false).
        from ament_index_python.packages import get_package_share_directory

        if slam_backend == "visual":
            # cuVSLAM is the camera-based backend for lidar-less robots; it
            # fills the same ``map→odom`` TF edge slam_toolbox fills on lidar
            # robots. The engine impl (composable Isaac ROS C++ node vs the
            # in-process PyCuVSLAM wheel) is chosen by ``slam_visual_impl`` — a
            # host property, not a capability. Both single-source the node spec
            # from the openral_slam_bringup launch files; see
            # ``_build_visual_slam_includes``.
            slam_share = get_package_share_directory("openral_slam_bringup")
            # Mono RGBD frames its DA3 depth at the camera's TF frame so nvblox
            # can place it via the HAL's live camera TF; resolve that frame from
            # the already-loaded manifest sensor (fallback to the
            # <name>_optical_frame convention if the sensor is unlisted).
            mono_depth_frame = ""
            if slam_mono_camera:
                mono_depth_frame = next(
                    (
                        str(s.frame_id)
                        for s in description.sensors
                        if s.name == slam_mono_camera and s.frame_id
                    ),
                    f"{slam_mono_camera}_optical_frame",
                )
            extra_nodes.extend(
                _build_visual_slam_includes(
                    slam_share,
                    visual_impl=slam_visual_impl,
                    use_sim_time=use_sim_time,
                    stereo_cameras_csv=slam_stereo_cameras,
                    enable_nav2=enable_nav2,
                    robot_yaml=robot_yaml,
                    mono_camera=slam_mono_camera,
                    mono_depth_frame=mono_depth_frame,
                    depth_sidecar_autostart=slam_depth_sidecar_autostart,
                    nav2_depth_camera=_depth_camera(description),
                )
            )
        else:
            # slam_toolbox lidar backend, Reasoner-managed
            # background service. Auto-transitions UNCONFIGURED → INACTIVE
            # only; activation is the Reasoner's job (LifecycleTransitionTool).
            slam_params_path = os.path.join(
                get_package_share_directory("openral_slam_bringup"),
                "config",
                "slam_toolbox_2d.yaml",
            )
            slam_node = LifecycleNode(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="openral_slam_toolbox",
                namespace="",
                # Graph-wide clock domain (see _resolve_clock_origin); overrides the
                # yaml's use_sim_time so slam_toolbox shares the HAL's wall-clock
                # /scan + odom TF. Sim-time without a /clock pins its pose-graph at
                # 0 → empty map → Nav2 plans through obstacles.
                parameters=[slam_params_path, {"use_sim_time": use_sim_time}],
                additional_env=otel_env,
                output="screen",
            )
            extra_nodes.append(slam_node)
            # Auto-CONFIGURE + ACTIVATE slam_toolbox externally via a tiny in-process Python
            # helper (rclpy.lifecycle, with retries). Direct ``ros2 lifecycle set`` was racey on
            # robocasa-kitchen boots: the kitchen install subprocess prints ~60 lines to stdout
            # before slam_toolbox's service is fully advertised, so a fixed-delay TimerAction
            # fired ``ros2 lifecycle set`` while the node was still "Node not found", exiting 1
            # and surfacing as ``[ERROR] [ros2-9]: process has died, exit code 1``. The rclpy
            # helper waits for the service, retries, and never logs at ERROR on transient
            # absence.
            #
            # Also avoids launch_ros's ``lifecycle_event_manager``, which on Jazzy logs a
            # spurious ``[ERROR] Failed to make transition 'TRANSITION_CONFIGURE'`` even when
            # slam_toolbox's ``on_configure`` returns SUCCESS (the change_state response arrives
            # with ``success=false`` on the first call due to a service-responder race
            # upstream).
            lifecycle_autostart_path = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
            slam_autostart = ExecuteProcess(
                cmd=[
                    sys.executable,
                    lifecycle_autostart_path,
                    "--node",
                    "/openral_slam_toolbox",
                    "--target",
                    "active",
                    "--service-timeout-s",
                    "60.0",
                ],
                output="log",
            )
            extra_nodes.append(slam_autostart)

    if enable_nav2:
        # Nav2's local_costmap needs ``odom -> base_link`` TF to be
        # already on the bus when its ``lifecycle_manager_navigation``
        # transitions the costmap sub-nodes to ACTIVE — otherwise the
        # costmap throws "Timed out waiting for transform from
        # base_link to odom" and the lifecycle bond fails. The HAL
        # publishes that TF only after ``on_activate``, which on a
        # robocasa-kitchen boot lags Nav2's autostart by ~10–20 s.
        # Gate the Nav2 include on the HAL's transition to ACTIVE
        # so TF is already streaming when Nav2 sub-nodes wake up.
        extra_nodes.append(
            RegisterEventHandler(
                OnStateTransition(
                    target_lifecycle_node=hal,
                    goal_state="active",
                    entities=[
                        _build_nav2_include(
                            robot_yaml, use_sim_time=use_sim_time, slam_backend=slam_backend
                        )
                    ],
                ),
            ),
        )
        # reasoner_node seeds its rSkill palette at on_configure (~5 s after launch), long
        # before Nav2 finishes its 15-30 s lifecycle bringup. The graph-availability filter
        # drops the ``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose-none`` rSkill because
        # ``/navigate_to_pose`` isn't yet advertised, so the LLM never sees the Nav2 tool and
        # replies "I do not have a tool available to perform base movement". Spawn a small
        # helper that polls the ROS graph for ``/navigate_to_pose`` and fires Empty on
        # ``/openral/skill_registry_changed`` once it appears; the reasoner re-seeds the palette
        # and the Nav2 rSkill becomes dispatchable.
        palette_reseed_helper = str(_REPO_ROOT / "tools" / "wait_for_action_and_signal_palette.py")
        extra_nodes.append(
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    palette_reseed_helper,
                    "--action",
                    "/navigate_to_pose",
                    "--lifecycle-node",
                    "/bt_navigator",
                    "--timeout-s",
                    "120.0",
                ],
                output="log",
            ),
        )

    # Manifest-derived, never the mobile-base literals: see _octomap_frames.
    # Resolved unconditionally so the runtime node's world-cloud bridge gets the
    # same base frame whether or not the octomap leg itself is spawned.
    octomap_fixed_frame, octomap_base_frame = _octomap_frames(description)

    if enable_octomap:
        assert coverage is not None  # reason: computed whenever enable_octomap
        octomap_cloud_topic = _octomap_cloud_topic(octomap_cloud_topic, description, hal_mode)
        # The world-collision perception leg. octomap_server builds a 3-D OcTree from the
        # HAL's depth PointCloud2 (``synthesize_depth_image`` back-projected by
        # ``points_from_depth_grid`` → ``octomap_cloud_topic``), and openral_octomap_bridge
        # lowers that octree into the dense ``/openral/world_voxels`` grid the kernel rasterizes
        # capsules against — keeps the octomap dependency OUT of the real-time kernel.
        # ``frame_id`` is the fixed tree frame (odom, already on /tf via the HAL's
        # odom→base_link broadcast); ``cloud_in`` is remapped to the robot's depth topic.
        # Requires ros-${ROS_DISTRO}-octomap-server + the openral_octomap_bridge package built —
        # opt-in, default off, like slam/nav2.
        perception_prefix = _cpuset_prefix("OPENRAL_PERCEPTION_CPUSET")
        # A camera that sees the robot (real, or a rendered sim depth camera): remove the
        # robot's (and a held payload's) own returns before octomap inserts the cloud, as
        # the MuJoCo sensor bridge does by making those bodies transparent
        # (_cloud_shows_the_robot); a robot with no collision model has nothing to filter
        # against.
        octomap_input_topic = octomap_cloud_topic
        self_filter_nodes: list = []  # type: ignore[type-arg]  # reason: launch_ros.actions.Node, deferred import
        if _cloud_shows_the_robot(hal_mode, scene_backend) and has_collision_capsules:
            octomap_input_topic = _SELF_FILTERED_CLOUD_TOPIC
            self_filter_nodes.append(
                Node(
                    package="openral_octomap_bridge",
                    executable="robot_self_filter",
                    name="openral_robot_self_filter",
                    namespace="",
                    prefix=perception_prefix,
                    parameters=[
                        {
                            **_self_filter_params(
                                collision_params,
                                description,
                                runtime_joint_states_topic or "/joint_states",
                                rig.robot_self_filter_padding_m,
                            ),
                            "use_sim_time": use_sim_time,
                        }
                    ],
                    remappings=[
                        ("cloud_in", octomap_cloud_topic),
                        ("cloud_out", _SELF_FILTERED_CLOUD_TOPIC),
                    ],
                    additional_env=otel_env,
                    output="screen",
                )
            )
        octomap_server = Node(
            package="octomap_server",
            executable="octomap_server_node",
            name="openral_octomap_server",
            namespace="",
            prefix=perception_prefix,
            parameters=[
                {
                    "resolution": _octomap_resolution(hal_mode),
                    "frame_id": octomap_fixed_frame,
                    "base_frame_id": octomap_base_frame,
                    # Only the coverage ball reaches the kernel, so octomap only
                    # integrates returns a link could reach: the input cloud is clipped
                    # to the robot's reach box plus margin (inside the ball's bounding
                    # box, in ``frame_id`` == the ball's frame) and a ray
                    # from any camera inside the ball ends within one diameter. On Thor
                    # the unclipped ZED cloud (4 m rays, floor and far wall) held
                    # octomap_server at ~2 Hz with multi-second gaps, i.e. a stale map
                    # (2026-10-02).
                    "sensor_model.max_range": 2.0 * coverage.radius,
                    **_octomap_input_bounds(coverage),
                    # Keep the map fresh for manipulation: octomap ray-clears
                    # free space, so a grasped/moved object's old cells decay
                    # back to free once re-observed. A slightly higher
                    # occupancy threshold + speckle filter clears transient /
                    # isolated noise voxels faster so they don't linger as
                    # phantom obstacles in front of the arm.
                    # One default 0.7-probability hit exceeds a 0.6 threshold;
                    # sim uses 0.8 so a transient self/floating point cannot
                    # become a safety voxel until a second frame confirms it.
                    "occupancy_thres": _octomap_occupancy_threshold(hal_mode),
                    "sensor_model.miss": 0.4,
                    # Exact sim depth makes robot/attached bodies transparent,
                    # so a background return is a trustworthy miss through the
                    # old object cell. Real maps keep OctoMap's 0.97 saturation
                    # because sensor masking/completion remains uncertain.
                    "sensor_model.max": _octomap_clamping_max(hal_mode),
                    "filter_speckles": True,
                    # Graph-wide clock domain (see _resolve_clock_origin). With no
                    # /clock publisher this is wall-clock: use_sim_time=True
                    # would pin octomap_server's clock at 0 while the HAL stamps
                    # the depth cloud + base_link->optical TF on wall-clock — the
                    # cloud then looks "in the future", every insert is dropped,
                    # and the octree (hence /openral/world_voxels and the
                    # kernel's world-collision check) stays empty so the arm
                    # crashes into the table uncaught. Same flag drives Nav2.
                    "use_sim_time": use_sim_time,
                }
            ],
            remappings=[("cloud_in", octomap_input_topic)],
            additional_env=otel_env,
            output="screen",
        )
        octomap_bridge = Node(
            package="openral_octomap_bridge",
            executable="octomap_voxel_bridge",
            name="openral_octomap_voxel_bridge",
            namespace="",
            prefix=perception_prefix,
            parameters=[
                {
                    "base_frame": octomap_base_frame,
                    "octomap_topic": "/octomap_binary",
                    "output_topic": "/openral/world_voxels",
                    "resolution": _octomap_resolution(hal_mode),
                    "coverage_radius_m": coverage.radius,
                    "coverage_center_x": coverage.centre[0],
                    "coverage_center_y": coverage.centre[1],
                    "coverage_center_z": coverage.centre[2],
                    # Stop republishing an octree that stopped arriving, so the
                    # kernel's voxel deadline can fail closed (Entry 033).
                    "max_octree_age_s": max_octree_age_s,
                    # Graph-wide clock domain — matches octomap_server above
                    # (sim-time without a /clock pins its TF lookups at 0).
                    "use_sim_time": use_sim_time,
                    # A held payload is cleared by looking its (manifest) attach link up
                    # on TF; a cell that names that body differently needs the rename.
                    # Omitted when there is none (an empty list has no ROS param type).
                    **(
                        {"attach_link_tf_frames": attach_link_tf_frames}
                        if attach_link_tf_frames
                        else {}
                    ),
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.extend([*self_filter_nodes, octomap_server, octomap_bridge])

    if enable_object_detector or locator_specs:
        # The perception leg runs when EITHER the continuous detector is on OR an on-demand
        # locator was requested (a lean ``--no-object-detector`` deploy grounds via the locator
        # alone). The continuous-detector node itself stays gated on ``enable_object_detector``
        # below; camera resolution and the locator loop run for both.
        #
        # The object-detection perception leg: the ROS-Image detector runs RT-DETR over the
        # agentview RGB tee and publishes ObjectsMetadata to /openral/perception/objects. The
        # world-state node's object-lift (object_lift_enabled defaults True) subscribes that
        # topic, resolves the detection camera from the robot description via ``sensor_id``,
        # and raises 2-D boxes into the /openral/world_voxels grid in the ``map`` frame. Purely
        # additive: the detector emits no Action chunks and the safety kernel never sees its
        # output. The COCO-80 label map is read from the rtdetr-coco-r18 rSkill manifest at
        # launch-build time so the node's class indices map to the same names the model was
        # exported with. ``yaml`` is imported locally on purpose: a default (detector-off)
        # launch must never import yaml or read rskill.yaml, so the base graph stays
        # byte-for-byte unchanged. Do NOT hoist this import to the module top.
        import yaml

        # Cross-frame lift: detect on (and stamp the detection with) the robot's first
        # *liftable* RGB camera — one whose frame_id is a dedicated ``*_optical_frame`` (the
        # SimSensorBridge broadcasts its live extrinsics, so the world-state lifter can project
        # the world voxel map into it). The detection's ``sensor_id`` MUST be that camera, not a
        # depth sensor, so the lifter resolves the right intrinsics/extrinsics. Generic over
        # robots; prefers an optical-frame RGB camera but falls back to the robot's first RGB
        # camera so the detector still gets frames. (franka_panda publishes ``top``/``wrist``,
        # neither optical-framed; the old hardcoded ``agentview_left`` fallback was a dead
        # topic — no cached frame, every ``locate_in_view`` returned found=False, looping the
        # reasoner.)
        from openral_core.exceptions import ROSConfigError

        det_camera = _primary_rgb_camera(publishing)
        if not det_camera:
            raise ROSConfigError(
                f"object detector enabled but robot {description.name!r} has no RGB camera "
                f"that publishes on this deploy (hal_mode={hal_mode})"
            )
        det_image_topic = camera_topic(det_camera)
        # The lift projects through the camera's live frame + K (the detector stamps them
        # on each batch), not the manifest SensorSpec — on a real OpenArm `top` is a sim
        # stand-in (frame "world", fx 640) for a 1920x1080 ZED image at fx 1498.
        det_info_topic = _live_camera_info_topic(
            next(s for s in publishing if s.name == det_camera), hal_mode
        )
        det_camera_infos = [f"{det_camera}={det_info_topic}"] if det_info_topic else [""]

        # Shared QoS / clock note: clock domain follows the graph-wide flag
        # (see _resolve_clock_origin). The node stamps its output from the input
        # Image's header.stamp; its ONLY use of self.get_clock() is the
        # max_rate_hz publish throttle. With no live /clock this stays
        # wall-clock — use_sim_time=True would pin get_clock().now() at 0 and
        # every frame is dropped at the rate gate → the detector never publishes.
        if object_detector_manifest:
            # 2026-06-09 — manifest-driven backend (RT-DETR ONNX or the
            # open-vocab LocateAnything VLM sidecar). The node loads labels /
            # model_id / contract from the manifest; we only forward the manifest
            # path, the (VLM-ignored) onnx override, and the query.
            with pathlib.Path(object_detector_manifest).open(encoding="utf-8") as handle:
                man = yaml.safe_load(handle) or {}
            # Throttle by the detector engine so the single-threaded callback never
            # backs up: the VLM sidecar (LocateAnything) is slow (~1-2 s / frame),
            # the in-process OmDet-Turbo zero-shot backend is ~hundreds of ms, and
            # the RT-DETR ONNX path is fast. See the manifest's DetectorEngine.
            engine = (man.get("detector") or {}).get("engine")
            max_rate_hz = {"vlm_sidecar": 0.5, "zeroshot_hf": 2.0}.get(engine, 5.0)
            det_params = {
                "image_topic": det_image_topic,
                "sensor_id": det_camera,
                "manifest_path": object_detector_manifest,
                "onnx_path": object_detector_onnx,
                "query": object_detector_query,
                "max_rate_hz": max_rate_hz,
            }
        else:
            rskill_yaml = pathlib.Path(_RSKILLS_DIR) / "rtdetr-coco-r18" / "rskill.yaml"
            with rskill_yaml.open("r", encoding="utf-8") as handle:
                rskill_manifest = yaml.safe_load(handle)
            # Read via .get so a missing/renamed detector.labels key fails with the
            # same legible message as the empty case (a bare KeyError would be
            # cryptic at launch time). Still fail-fast — never an empty label map.
            coco80_labels = (rskill_manifest or {}).get("detector", {}).get("labels")
            if not coco80_labels:
                raise ValueError(
                    f"{rskill_yaml}: detector.labels is missing or empty; the "
                    "detector cannot label any detection without a "
                    "class-index → name map."
                )
            det_params = {
                "image_topic": det_image_topic,
                "sensor_id": det_camera,
                # --object-detector-onnx only relocates THIS model's weights:
                # model_id + the COCO-80 labels are fixed to rtdetr-coco-r18.
                "onnx_path": object_detector_onnx,
                "model_id": "rtdetr-coco-r18",
                # Keep weak/uncertain detections out of the world model: only
                # objects with sigmoid score ≥ 0.5 are published.
                "score_threshold": 0.5,
                "input_size": 640,
                "max_rate_hz": 5.0,
                "labels": coco80_labels,
            }

        det_params["use_sim_time"] = use_sim_time
        # Register the cached frame under the REAL camera name, not the node's
        # "default" fallback. The reasoner reads live camera names from
        # PERCEPTION (e.g. "top") and passes them to locate_in_view; without
        # this the frame caches under "default" and every locate misses with
        # "no frame for camera 'top'" (found=False) regardless of the query.
        det_params["primary_camera"] = det_camera
        det_params["camera_infos"] = det_camera_infos
        # Managed lifecycle node: autostarted to ACTIVE (detector
        # loaded) like the rest of the graph, but the reasoner can DEACTIVATE it
        # via LifecycleTransitionTool to free the detector's VRAM before a
        # co-resident grab policy loads on an 8 GB GPU.
        # The continuous detector node is the VRAM-heavy always-on leg — only
        # create it when explicitly enabled. The on-demand locators below come up
        # regardless (they load per-query and evict), so a lean deploy still grounds.
        if enable_object_detector:
            object_detector = LifecycleNode(
                package="openral_perception_ros",
                executable="ros_image_detector_node.py",
                name="openral_ros_image_detector",
                namespace="",
                parameters=[det_params],
                additional_env=otel_env,
                output="screen",
            )
            extra_nodes.append(object_detector)
            autostart += _autostart_lifecycle(object_detector, "openral_ros_image_detector")

        # On-demand locator nodes: one per --object-detector-locator,
        # each serving its own namespaced /openral/perception/<alias>/locate_in_view
        # (the reasoner picks one via LocateInViewTool.detector). They share the
        # continuous detector's camera/topic; the node's mode wiring (detector
        # invocation mode)
        # makes them serve-only (no continuous publish leg). Throttle by engine.
        for spec in locator_specs:
            locator_rate_hz = {"vlm_sidecar": 0.5, "zeroshot_hf": 2.0}.get(spec["engine"], 5.0)
            locator_params = {
                "image_topic": det_image_topic,
                "sensor_id": det_camera,
                # Cache under the real camera name so locate_in_view(camera="top")
                # hits — see the continuous detector's primary_camera note above.
                "primary_camera": det_camera,
                "camera_infos": det_camera_infos,
                "manifest_path": spec["manifest"],
                "onnx_path": object_detector_onnx,
                "query": object_detector_query,
                "max_rate_hz": locator_rate_hz,
                "locate_in_view_service": f"/openral/perception/{spec['segment']}/locate_in_view",
                "query_topic": f"/openral/perception/{spec['segment']}/detector_query",
                "detector_id": spec["alias"],
                "use_sim_time": use_sim_time,
            }
            locator_node = LifecycleNode(
                package="openral_perception_ros",
                executable="ros_image_detector_node.py",
                name=spec["node"],
                namespace="",
                parameters=[locator_params],
                additional_env=otel_env,
                output="screen",
            )
            extra_nodes.append(locator_node)
            autostart += _autostart_lifecycle(locator_node, spec["node"])

    if vision_attachment_enabled:
        # The segmenter the HAL's vision attachment bridge calls at grasp events
        # (/openral/perception/segment_in_view). It projects through the driver's live
        # CameraInfo (camera_infos), never the manifest's nominal intrinsics.
        va_camera = LaunchConfiguration("vision_attachment_camera").perform(context)
        segmenter = LifecycleNode(
            package="openral_perception_ros",
            executable="segmenter_node.py",
            name="openral_segmenter",
            namespace="",
            parameters=[
                {
                    "robot_yaml": robot_yaml,
                    "manifest_path": LaunchConfiguration(
                        "vision_attachment_segmenter_manifest"
                    ).perform(context),
                    "cameras": [
                        f"{va_camera}="
                        + LaunchConfiguration("vision_attachment_rgb_topic").perform(context)
                    ],
                    "camera_infos": [
                        f"{va_camera}="
                        + LaunchConfiguration("vision_attachment_rgb_camera_info_topic").perform(
                            context
                        )
                    ],
                    "primary_camera": va_camera,
                    "device": LaunchConfiguration("vision_attachment_segmenter_device").perform(
                        context
                    ),
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.append(segmenter)
        autostart += _autostart_lifecycle(segmenter, "openral_segmenter")

    if enable_reward_monitor:
        # Reward monitor runs PARALLEL to the VLA (not a lifecycle/VRAM
        # peer the reasoner frees before a policy; it stays co-active). Plain Node:
        # subscribes the agentview RGB stream, buffers a rolling window, loads
        # the reward backend from the manifest, and serves
        # /openral/perception/query_task_progress for the reasoner to poll.
        reward_manifest = reward_monitor_manifest or str(
            pathlib.Path(_RSKILLS_DIR) / "robometer-4b" / "rskill.yaml"
        )
        # Resolve the camera the monitor scores: the first RGB camera that publishes on
        # this deploy (manifest order; on a real cell only bound cameras), so Robometer
        # follows the same default view order as the deploy graph; do not special-case wrist.
        from openral_core.exceptions import ROSConfigError

        # From the cameras that publish on this deploy (a bound camera on a real cell).
        reward_camera = next((s.name for s in publishing if s.modality == "rgb"), "")
        if not reward_camera:
            raise ROSConfigError(
                f"reward monitor enabled but robot {description.name!r} has no RGB camera "
                f"that publishes on this deploy (hal_mode={hal_mode})"
            )
        reward_image_topic = camera_topic(reward_camera)
        reward_monitor = Node(
            package="openral_perception_ros",
            executable="reward_monitor_node.py",
            name="openral_reward_monitor",
            namespace="",
            parameters=[
                {
                    "manifest_path": reward_manifest,
                    "image_topic": reward_image_topic,
                    "task": reward_monitor_task,
                    # When the critic producer is also up, feed it real
                    # Robometer progress as a CriticScore stream (else stay query-only).
                    "enable_critic_score": enable_critic,
                    # 2026-06-29 — only score while a VLA is executing (the reasoner
                    # publishes /openral/reward/active around each execute_rskill), so
                    # the reward VLM doesn't grind on an idle scene and the Tier-C
                    # watchdog isn't fed idle noise.
                    "gate_scoring_on_execution": True,
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.append(reward_monitor)

    if enable_scene_vlm:
        # Scene VLM runs PARALLEL to the VLA, like the reward monitor: it is a
        # read-only reasoning aid that publishes nothing continuously and answers
        # only on demand, so it is NOT a lifecycle/VRAM peer the reasoner frees
        # before dispatching a policy.
        #
        # Every publishing RGB camera is offered, not just the first: the reasoner
        # picks a viewpoint by camera id per query ("is the bowl on the shelf?"
        # wants a different view than "did the gripper close?"), and the node
        # caches each stream's latest frame precisely so it can answer about any
        # of them. The detector leg above resolves cameras the same way.
        scene_vlm_cameras = [
            f"{s.name}={camera_topic(s.name)}" for s in publishing if s.modality == "rgb"
        ]
        if not scene_vlm_cameras:
            from openral_core.exceptions import ROSConfigError

            # The node has no default camera (ADR-0108); refuse here, at launch, rather
            # than let it die at start-up behind a graph that otherwise came up.
            raise ROSConfigError(
                f"scene VLM enabled but robot {description.name!r} has no RGB camera that "
                f"publishes on this deploy (hal_mode={hal_mode})"
            )
        scene_vlm_params: dict[str, object] = {
            "manifest_path": scene_vlm_manifest
            or str(pathlib.Path(_RSKILLS_DIR) / "qwen35-4b-nf4" / "rskill.yaml"),
            "use_sim_time": use_sim_time,
        }
        scene_vlm_params["cameras"] = scene_vlm_cameras
        scene_vlm_params["primary_camera"] = scene_vlm_cameras[0].split("=", 1)[0]
        extra_nodes.append(
            Node(
                package="openral_perception_ros",
                executable="scene_vlm_node.py",
                name="openral_scene_vlm",
                namespace="",
                parameters=[scene_vlm_params],
                additional_env=otel_env,
                output="screen",
            )
        )

    if enable_critic:
        # Tier-C critic producer. Plain Node co-active with the graph:
        # subscribes /openral/critic/score (any reward model — Robometer, a future
        # SARM — publishes there), routes each sample through a CriticWatchdogGroup,
        # and emits a Tier-C FailureTrigger on /openral/failure/critic on a stall.
        critic_producer = Node(
            package="openral_reasoner_ros",
            executable="critic_producer_node.py",
            name="openral_critic_producer",
            namespace="",
            parameters=[
                {
                    "stall_patience": int(critic_stall_patience),
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.append(critic_producer)

    # Deploy memory bundle: seed the saved 2D occupancy grid.
    # When ``map_path`` points at a nav2 ``map.yaml`` AND live SLAM isn't already
    # owning ``/map``, bring up a standalone nav2 ``map_server`` that latches ``/map``
    # (TRANSIENT_LOCAL) from the first tick, so the nav costmap + the reasoner's
    # occupancy-grid-refined approach-pose grid have the saved prior immediately. With
    # SLAM on, slam_toolbox / cuVSLAM owns ``/map`` and we skip the seed to avoid two
    # publishers. The grid stays advisory: the C++ kernel keeps its own ephemeral
    # collision grid; this map never feeds it.
    if map_path and not enable_slam:
        map_server = LifecycleNode(
            package="nav2_map_server",
            executable="map_server",
            name="openral_map_server",
            namespace="",
            parameters=[
                {
                    "yaml_filename": map_path,
                    "topic_name": "map",
                    "frame_id": "map",
                    "use_sim_time": use_sim_time,
                }
            ],
            output="screen",
        )
        extra_nodes.append(map_server)
        autostart += _autostart_lifecycle(map_server, "openral_map_server")

    nodes: list = [
        safety_kernel,
        # Independent E-stop producers; never conditional on an opt-in flag,
        # on hal_mode, or on the dashboard being up (CLAUDE.md section 3).
        deadman_watchdog,
        hardware_estop,
        human_estop_forwarder,
        runtime,
        hal,
        *extra_nodes,
    ]
    if enable_reasoner:
        nodes[2:2] = [reasoner, prompt_router]
    # Dashboard is opt-out (default on). ``openral deploy sim --no-dashboard``
    # threads ``enable_dashboard:=false`` to skip the spawn entirely —
    # useful for headless CI and avoids the
    # ``[Errno 98] address already in use`` collision that would occur
    # if a previous run's dashboard still holds the port.
    if enable_dashboard:
        nodes.insert(0, dashboard)

    # Read-only Foxglove live-scene bridge. Off by default;
    # ``openral deploy sim --foxglove`` opts in.
    #
    # STALE-BRIDGE ORDERING (VERIFICATION.md "Stale-bridge
    # gotcha"): foxglove-sdk-cpp v0.18.0 advertises channels when a topic is
    # first seen, but if the publisher disappears and reappears (e.g. because the
    # bridge starts before the topic producer) the channel is re-advertised but
    # no data flows. Wrapping the bridge in a TimerAction(period=5.0) ensures it
    # starts AFTER the topic producers (HAL, SLAM, octomap, robot_state_publisher)
    # have had time to advertise their topics on the ROS graph.
    if enable_foxglove:
        foxglove_bridge_node = Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="openral_foxglove_bridge",
            output="screen",
            parameters=[
                {
                    "address": "127.0.0.1",
                    "port": int(foxglove_port),
                    "tls": False,
                    "capabilities": READ_ONLY_CAPABILITIES,
                    # What a viewer may fetch to draw the URDF. Upstream's
                    # default refuses a dot in a directory segment, which
                    # rejects every versioned asset path — see topics.py.
                    "asset_uri_allowlist": ASSET_URI_ALLOWLIST,
                    "topic_whitelist": BUCKET1_TOPIC_WHITELIST,
                    # Keep the upstream 10 MB send buffer for camera frames.
                    "send_buffer_limit": 10_000_000,
                    "max_qos_depth": 10,
                    "include_hidden": False,
                    # Graph-wide clock domain (see _resolve_clock_origin).
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=_foxglove_asset_env(robot_description_xml),
        )
        nodes.append(TimerAction(period=5.0, actions=[foxglove_bridge_node]))

        layout_path = _write_foxglove_layout(
            bound_rgb_camera_names, description.name, description.base_frame
        )
        if layout_path is not None:
            print(
                f"[deploy_e2e] foxglove: ws://127.0.0.1:{foxglove_port} — import the "
                f"scene-matched layout from {layout_path} "
                f"(cameras: {', '.join(bound_rgb_camera_names)})",
                flush=True,
            )

        # Bucket-2 converter. The layout's voxel panels read
        # `/openral/world_voxels_cloud`, a `sensor_msgs` re-publication of the
        # custom `openral_msgs/OccupancyVoxels` — Foxglove renders the standard
        # type natively and the custom one not at all. Nothing else in the
        # graph produces it, so without this the panels sit empty on every
        # deploy while the underlying world state is perfectly healthy.
        # It also draws `/openral/attachment_state` (held payload primitives,
        # grasp/place regions) on `/openral/viz/attachments`, posed on the attach
        # link's TF frame — so it gets the octomap bridge's `attach_link_tf_frames`
        # renames (omitted when empty: an empty list has no ROS param type).
        # Read-only viz: it subscribes and publishes viz topics, and actuates
        # nothing.
        nodes.append(
            Node(
                package="openral_foxglove_bringup",
                executable="bucket2_markers",
                name="openral_bucket2_markers",
                output="log",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        **(
                            {"attach_link_tf_frames": attach_link_tf_frames}
                            if attach_link_tf_frames
                            else {}
                        ),
                    }
                ],
                additional_env=otel_env,
            )
        )

    return [*nodes, *autostart]


def generate_launch_description() -> LaunchDescription:
    """Robot-agnostic deploy-sim launch graph; resolves args via OpaqueFunction."""
    args = [
        DeclareLaunchArgument(
            "robot_yaml",
            description="Absolute path to robots/<robot_id>/robot.yaml.",
        ),
        DeclareLaunchArgument(
            "hal_package",
            description="ament package providing the HAL lifecycle node.",
        ),
        DeclareLaunchArgument(
            "hal_executable",
            description="Executable name inside ``hal_package``.",
        ),
        DeclareLaunchArgument(
            "hal_node_name",
            description=(
                "Fully-qualified node name the HAL registers under; drives lifecycle transitions."
            ),
        ),
        DeclareLaunchArgument(
            "hal_params_file",
            description=(
                "YAML parameter file for the HAL (``/**`` wildcard); the CLI "
                "always writes one, even when empty."
            ),
        ),
        DeclareLaunchArgument(
            "workcell_json",
            default_value="",
            description="DeployScene JSON carrying deploy-time safety/ACM overrides.",
        ),
        DeclareLaunchArgument(
            "reset_to_pose_service",
            default_value="",
            description=(
                "Service the skill_runner calls before the first inference "
                "tick to snap the HAL's qpos to the rSkill starting pose."
            ),
        ),
        DeclareLaunchArgument(
            "place_declaration_json",
            default_value="",
            description=(
                "Serialized openral_core.PlaceDeclaration (ADR-0097) the "
                "skill_runner scopes to each goal it dispatches, for a direct "
                "dispatch with no reasoner in the loop. Empty = no "
                "declaration; no place-phase support-contact witness can arm."
            ),
        ),
        DeclareLaunchArgument(
            "grasp_declaration_json",
            default_value="",
            description=(
                "Serialized openral_core.GraspDeclaration (real pick-and-place "
                "design §2.1) the skill_runner scopes to each goal it "
                "dispatches. Empty = no declaration, no grasp exemption."
            ),
        ),
        DeclareLaunchArgument(
            "approach_skill_id",
            default_value="",
            description=(
                "MoveIt approach rSkill URI (e.g. "
                "rskills/rskill-moveit-joints) the skill_runner dispatches to "
                "plan a collision-free motion to each skill's starting_pose. "
                "Empty = kernel-checked joint ramp."
            ),
        ),
        DeclareLaunchArgument(
            "preload_rskill_id",
            default_value="",
            description=(
                "rSkill the skill_runner resolves and loads right after "
                "activation (worker thread), so the first goal finds it "
                "GPU-resident instead of paying a multi-minute cold load "
                "inside the deadman watchdog's first-chunk window. Goals "
                "are rejected until rskill_runner.preload_done is logged. "
                "Empty = no preload."
            ),
        ),
        DeclareLaunchArgument(
            "preload_rskill_revision",
            default_value="",
            description=(
                "Hub revision pinned for preload_rskill_id; part of the "
                "resident key, so it must equal the revision later goals "
                "send. Empty = the unpinned default revision."
            ),
        ),
        DeclareLaunchArgument(
            "preload_prompt",
            default_value="",
            description=(
                "Prompt the preload warms the policy with. The skill stays "
                "resident per (rskill_id, revision); each goal's own prompt "
                "is passed to it per step, so a different goal prompt does "
                "not reload it."
            ),
        ),
        DeclareLaunchArgument(
            "dataset_out",
            default_value="",
            description=(
                "When set, record the deploy session (proprio + "
                "action + camera frames + episode markers) to this rosbag2 "
                "mcap path. Convert offline with `openral dataset from-bag`. "
                "Empty disables recording."
            ),
        ),
        DeclareLaunchArgument(
            "deploy_config",
            default_value="",
            description=(
                "Path to the DeployScene YAML (`openral deploy sim` and "
                "`openral deploy run` both forward it). Sets the HAL "
                "autostart budget from `backend_options.boot_timeout_s` "
                "and merges scene `sensors:` into the camera set. On "
                "hal_mode:=real only, the runtime node also opens one "
                "SensorReader per deploy-bound SensorSpec (robot "
                "manifest + scene `sensors:`) and publishes each "
                "camera onto its openral_core.camera_topic(<name>) (the "
                "real-hardware counterpart of the sim HAL's "
                "SimSensorBridge); in sim the HAL bridge stays the only "
                "camera source."
            ),
        ),
        DeclareLaunchArgument(
            "dataset_repo_id",
            default_value="",
            description="repo_id for the recorded dataset.",
        ),
        DeclareLaunchArgument(
            "dataset_license",
            default_value="CC-BY-4.0",
            description="SPDX license carried into `openral dataset from-bag`.",
        ),
        DeclareLaunchArgument(
            "dashboard_port",
            default_value="4318",
            description="OTLP/HTTP port for the dashboard child.",
        ),
        DeclareLaunchArgument(
            "reasoner_model",
            # Default to the curated gpt-5.5 registry entry: in live deploy testing it
            # was the most reliable at decomposing a collective operator goal into
            # grounded subtasks (glm-5.2 over-located and never decomposed). Needs
            # OPENRAL_REASONER_API_KEY in the environment; an explicit model-first env
            # still wins.
            default_value=os.environ.get("OPENRAL_REASONER_MODEL") or "gpt-5.5",
            description="OPENRAL_REASONER_MODEL registry key for the reasoner node.",
        ),
        DeclareLaunchArgument(
            "reasoner_endpoint",
            default_value=os.environ.get("OPENRAL_REASONER_ENDPOINT") or "",
            description=(
                "Optional OPENRAL_REASONER_ENDPOINT override — a named endpoint "
                "(openrouter / ollama / vllm / gemini / xai / deepseek / huggingface "
                "/ anthropic) or a URL. Empty uses the curated model's registry "
                "default."
            ),
        ),
        DeclareLaunchArgument(
            "spatial_memory_path",
            default_value="",
            description=(
                "Absolute path to a persisted, hierarchical scene graph "
                "(SceneGraph JSON). When set, the reasoner loads it into a "
                "SpatialMemory and offers the read-only recall_object / "
                "resolve_place query tools against the preloaded map. Empty = "
                "disabled."
            ),
        ),
        DeclareLaunchArgument(
            "spatial_memory_ingest",
            default_value="false",
            description=(
                "When true, the reasoner accumulates a durable "
                "SpatialMemory live from the object-detection producer's "
                "WorldState.detected_objects (auto-creating an empty backend "
                "if no spatial_memory_path was preloaded), so recall_object "
                "recalls what the robot has actually seen. Default false."
            ),
        ),
        DeclareLaunchArgument(
            "memory_md_path",
            default_value="",
            description=(
                "Absolute path to the self-maintained MEMORY.md "
                "(the deploy memory bundle's narrative/semantic modality). When "
                "set, the reasoner loads it as the ## MEMORY context block and "
                "offers the memory_write / memory_search tools. Empty = disabled."
            ),
        ),
        DeclareLaunchArgument(
            "map_path",
            default_value="",
            description=(
                "Absolute path to a saved nav2 map.yaml "
                "(the bundle's 2D occupancy-grid modality). When set and SLAM is "
                "off, a standalone nav2 map_server latches /map from the saved "
                "map so the costmap + the occupancy-grid-refined approach grid have the prior at "
                "boot. With SLAM on it is ignored (SLAM owns /map). Empty = "
                "disabled."
            ),
        ),
        DeclareLaunchArgument(
            "hal_mode",
            default_value="sim",
            description=(
                "Deploy path the reasoner's action-mode palette "
                "gate matches against: ``sim`` (digital-twin; the scene's "
                "robosuite OSC controller synthesises cartesian/OSC modes) "
                "admits cartesian skills, ``real`` admits only the robot's "
                "declared ``supported_control_modes``. ``openral deploy sim`` "
                "passes ``sim``; ``openral deploy run`` passes ``real``."
            ),
        ),
        DeclareLaunchArgument(
            "hardware_estop_device",
            default_value="",
            description=(
                "Device path of the hardware E-stop this host owns -- a GPIO "
                "chip (``/dev/gpiochip0``) or a USB-HID pendant. NOTE: this tree "
                "ships no vendor pendant driver and the launch always spawns the "
                "base hardware_estop_node, so today ANY non-empty value fails "
                "configure; the argument exists for a vendor overlay. "
                "(``/dev/input/by-id/...``). Empty (the default) declares "
                "that there is NO hardware E-stop here: the node is still "
                "spawned, refuses to configure, and logs that fact, so the "
                "absence is visible instead of assumed. Declaring a path that "
                "is not present on the host fails the same way. Note that a "
                "present path is still not enough on its own -- the base "
                "``HardwareEstopNode`` is not a driver, so a vendor subclass "
                "must supply the device read."
            ),
        ),
        DeclareLaunchArgument(
            "gripper_convention",
            default_value="",
            description=(
                "Gripper encoding the simulated scene's environment consumes, "
                "resolved by the CLI from the scene registry. The safety kernel "
                "bounds GRIPPER_* chunks by this convention's range and the runner "
                "refuses a skill that emits another one. Empty: each end "
                "effector's own command_convention (real robots, bare twins)."
            ),
        ),
        DeclareLaunchArgument(
            "clock_origin",
            default_value="host_wall",
            description=(
                "OpenRAL ClockAuthority origin resolved by the CLI: "
                "``simulation`` means the HAL publishes sim elapsed time on "
                "ROS ``/clock`` and the launch maps the graph to "
                "``use_sim_time=true``; ``host_wall`` means ROS system time "
                "and no OpenRAL ``/clock`` publisher. Operators should not "
                "toggle ROS ``use_sim_time`` directly."
            ),
        ),
        DeclareLaunchArgument(
            "enable_slam",
            default_value="false",
            description=(
                "Bring up SLAM as a background service. The "
                "backend is chosen by ``slam_backend``. Auto-transitions to "
                "INACTIVE (lidar backend); the Reasoner promotes to ACTIVE "
                "via LifecycleTransitionTool. Requires the openral_slam_bringup "
                "package built in the workspace (+ ros-${ROS_DISTRO}-slam-toolbox "
                "for the lidar backend / the operator's Isaac ROS install for "
                "the visual backend)."
            ),
        ),
        DeclareLaunchArgument(
            "slam_backend",
            default_value="lidar",
            description=(
                "SLAM backend composed when ``enable_slam`` is "
                "true: ``lidar`` (slam_toolbox, needs /scan), ``visual`` "
                "(cuVSLAM, camera-based, for lidar-less robots), or ``none``. "
                "Normally resolved upstream by deploy_sim.py from "
                "``RobotCapabilities`` (``has_lidar`` / ``has_vision_slam``); "
                "defaults to ``lidar`` to preserve the legacy lidar-only behaviour."
            ),
        ),
        DeclareLaunchArgument(
            "slam_visual_impl",
            default_value="isaac_ros",
            description=(
                "Which cuVSLAM engine the ``visual`` backend composes: "
                "``isaac_ros`` (composable isaac_ros_visual_slam C++ node, needs "
                "the Isaac ROS apt stack) or ``pycuvslam`` (in-process PyCuVSLAM "
                "wheel, rectified stereo only). Ignored unless ``slam_backend`` is "
                "``visual``. From DeployRuntime.slam_visual_impl; defaults to "
                "``isaac_ros``."
            ),
        ),
        DeclareLaunchArgument(
            "slam_stereo_cameras",
            default_value="",
            description=(
                "Optional ``<left>,<right>`` camera names for the visual SLAM "
                "stereo rig; each maps to openral_core.camera_topic(<name>) "
                "(+ camera_info). Required by the stereo visual impls, whose "
                "camera topics carry no default."
            ),
        ),
        DeclareLaunchArgument(
            "slam_mono_camera",
            default_value="",
            description=(
                "Optional single RGB camera name for the visual SLAM **mono "
                "RGBD** path (pycuvslam only): auto-composes the DA3 depth "
                "provider + nvblox so one camera yields pose + occupancy grid + "
                "voxels. From DeployRuntime.slam_mono_camera; mutually exclusive "
                "with slam_stereo_cameras. Empty → stereo/multi-camera."
            ),
        ),
        DeclareLaunchArgument(
            "slam_depth_sidecar_autostart",
            default_value="true",
            description=(
                "Spawn tools/da3_depth_sidecar.py (ZMQ :5771) alongside the mono "
                "RGBD graph. From DeployRuntime.slam_depth_sidecar_autostart; "
                "false = operator-run/shared sidecar. Ignored without "
                "slam_mono_camera."
            ),
        ),
        DeclareLaunchArgument(
            "enable_nav2",
            default_value="false",
            description=(
                "Bring up the Nav2 navigation stack so the "
                "``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose-none`` wrapped-action "
                "rSkill has a ``/navigate_to_pose`` server to dispatch "
                "to. Nav2 auto-activates (lifecycle_manager_navigation "
                "drives its sub-nodes to ACTIVE); the Reasoner triggers "
                "it by dispatching the rSkill, not by lifecycle "
                "transition. Requires ros-${ROS_DISTRO}-nav2-bringup "
                "+ the openral_nav2_bringup package."
            ),
        ),
        DeclareLaunchArgument(
            "enable_octomap",
            default_value="false",
            description=(
                "Bring up the world-collision perception leg: "
                "octomap_server (3-D OcTree from the HAL's depth "
                "PointCloud2) + the openral_octomap_bridge "
                "(octree → /openral/world_voxels), and enable the C++ "
                "safety kernel's capsule-vs-voxel world-collision check. "
                "Requires ros-${ROS_DISTRO}-octomap-server + the "
                "openral_octomap_bridge package built, and a robot whose "
                "manifest declares a depth SensorSpec."
            ),
        ),
        DeclareLaunchArgument(
            "enable_octomap_kernel_check",
            default_value="true",
            description=(
                "When False, the octomap perception leg still "
                "publishes /openral/world_voxels (so the world-state object-lift "
                "works), but the C++ safety kernel's capsule-vs-voxel check stays "
                "OFF (its --no-enable-octomap posture: envelope + self-collision "
                "only). Lets perception use the world map without the dense-scene "
                "false-positive E-stop. Default True preserves the bundled behaviour. "
                "Never weakens the kernel below the --no-enable-octomap baseline."
            ),
        ),
        DeclareLaunchArgument(
            "octomap_cloud_topic",
            default_value="",
            description=(
                "Depth PointCloud2 topic octomap_server consumes "
                "(``cloud_in`` remap). Empty (default) derives, in sim, "
                "the ``points`` camera topic of the manifest's one depth "
                "sensor with intrinsics — the sim sensor bridge's cloud. "
                "Required on real hardware (the depth driver's topic): "
                "octomap refuses to start without it."
            ),
        ),
        DeclareLaunchArgument(
            "world_voxel_deadline_s",
            default_value="",
            description=(
                "How long the safety kernel trusts the last /openral/world_voxels grid "
                "(DeployRuntime.world_voxel_deadline_s). Empty = the schema default, 1.0 s; "
                "hard cap 2.0 s."
            ),
        ),
        DeclareLaunchArgument(
            "max_octree_age_s",
            default_value="",
            description=(
                "How long the octomap bridge republishes the last octree "
                "(DeployRuntime.max_octree_age_s). Empty = the deadline; a value "
                "above the deadline is refused."
            ),
        ),
        DeclareLaunchArgument(
            "world_voxel_data_age_budget_s",
            default_value="",
            description=(
                "How old the sensor data behind a voxel grid may be when the kernel "
                "checks a chunk (DeployRuntime.world_voxel_data_age_budget_s). "
                "Empty = the schema default, 1.5 s; hard cap 3.0 s."
            ),
        ),
        DeclareLaunchArgument(
            "robot_self_filter_padding_m",
            default_value="",
            description=(
                "Real camera path: how far past the collision model a depth return is "
                "removed as the robot (DeployRuntime.robot_self_filter_padding_m). "
                "Empty = the schema default, 0.02 m (measured at rest); hard cap 0.10 m."
            ),
        ),
        DeclareLaunchArgument(
            "enable_object_detector",
            default_value="false",
            description=(
                "Bring up the ROS-Image object detector "
                "(openral_perception_ros/ros_image_detector_node): runs "
                "RT-DETR over the agentview RGB tee and publishes "
                "ObjectsMetadata to /openral/perception/objects, which the "
                "world-state node's object-lift raises into the "
                "/openral/world_voxels grid. Default off; ``openral deploy sim`` "
                "auto-enables it when the --object-detector-onnx weights "
                "exist. Requires the openral_perception_ros package built "
                "and the rtdetr-coco-r18 rSkill ONNX present."
            ),
        ),
        DeclareLaunchArgument(
            "enable_vision_attachment",
            default_value="false",
            description=(
                "Bring up the vision attachment leg (DeployRuntime.vision_attachment): the "
                "SAM 2.1 segmenter lifecycle node the HAL's attachment-evidence bridge calls. "
                "Always turns the safety kernel's attached-payload check on with it "
                "(1000 ms deadline on real). Default off."
            ),
        ),
        DeclareLaunchArgument(
            "grasp_allowance_enabled",
            default_value="false",
            description=(
                "Safety kernel grasp-target exemption (DeployRuntime.grasp_allowance_enabled): "
                "world-voxel cells inside a live GraspDeclaration's producer-measured region "
                "do not trip the manifest's gripper contact links. Default off."
            ),
        ),
        DeclareLaunchArgument(
            "vision_attachment_camera",
            default_value="",
            description="Manifest sensor name the segmenter serves (e.g. head_zed).",
        ),
        DeclareLaunchArgument(
            "vision_attachment_rgb_topic",
            default_value="",
            description="The camera driver's RGB Image topic the segmenter caches.",
        ),
        DeclareLaunchArgument(
            "vision_attachment_rgb_camera_info_topic",
            default_value="",
            description="The driver's CameraInfo for that RGB stream (live K).",
        ),
        DeclareLaunchArgument(
            "vision_attachment_segmenter_manifest",
            default_value="",
            description="kind: segmenter rSkill manifest (absolute path).",
        ),
        DeclareLaunchArgument(
            "vision_attachment_segmenter_device",
            default_value="auto",
            description="Segmenter torch device: auto, cuda or cpu.",
        ),
        DeclareLaunchArgument(
            "vision_attachment_tf_frames",
            default_value="",
            description=(
                "Comma-joined 'link=frame' renames (the scene's vision_attachment.tf_frames, "
                "as the HAL gets them): the octomap bridge's attach_link_tf_frames."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_onnx",
            default_value=str(pathlib.Path(_RSKILLS_DIR) / "rtdetr-coco-r18" / "model.onnx"),
            description=(
                "Absolute path to the RT-DETR ONNX weights the "
                "object detector loads. Defaults to the in-tree "
                "rskills/rtdetr-coco-r18/model.onnx. Ignored unless "
                "enable_object_detector is true."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_manifest",
            default_value="",
            description=(
                "Path to a kind:detector rSkill manifest. "
                "When set, the detector node builds its backend from the manifest "
                "(runtime:onnx -> RT-DETR ONNX; runtime:pytorch -> the open-vocab "
                "LocateAnything VLM sidecar) instead of the hardcoded RT-DETR path. "
                "Ignored unless enable_object_detector is true."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_query",
            default_value="",
            description=(
                "Initial open-vocabulary query for a VLM "
                "detector (e.g. 'red mug'). Empty = the manifest's detector.labels "
                "default. Retarget live by publishing a std_msgs/String to "
                "/openral/perception/detector_query. Ignored by ONNX detectors."
            ),
        ),
        DeclareLaunchArgument(
            "enable_reward_monitor",
            default_value="false",
            description=(
                "Bring up the Robometer reward monitor "
                "(openral_perception_ros/reward_monitor_node) PARALLEL to the VLA. "
                "It buffers the agentview RGB stream and serves "
                "/openral/perception/query_task_progress; the reasoner is told "
                "task_progress_available=True so its LLM may poll per-frame "
                "progress/success whenever it sees fit. Advisory-only — never "
                "actuates. Default off. Requires the openral_perception_ros package "
                "built and Robometer/TOPReward deps in the current env; "
                "co-resident with a VLA needs ~3.3 GB "
                "free VRAM (use a small NF4 VLA on an 8 GB GPU)."
            ),
        ),
        DeclareLaunchArgument(
            "enable_scene_vlm",
            default_value="false",
            description=(
                "Bring up the scene-VLM query service "
                "(openral_perception_ros/scene_vlm_node) alongside the detectors. "
                "It caches every manifest RGB camera's latest frame and serves "
                "/openral/perception/query_scene; the reasoner is told "
                "scene_query_available=True so its LLM may ask open-ended questions "
                "about the current view ('has the robot grasped the mug?'). "
                "Read-only — never actuates. Default off. Needs the "
                "openral_perception_ros package built and the Qwen VLM sidecar "
                "provisionable; ~3 GB VRAM co-resident."
            ),
        ),
        DeclareLaunchArgument(
            "scene_vlm_manifest",
            default_value="",
            description=(
                "Path to a kind:vlm rSkill manifest backing query_scene. "
                "Empty defaults to the in-tree rskills/qwen35-4b-nf4/rskill.yaml. "
                "Ignored unless enable_scene_vlm."
            ),
        ),
        DeclareLaunchArgument(
            "enable_critic",
            default_value="false",
            description=(
                "Bring up the Tier-C critic producer "
                "(openral_reasoner_ros/critic_producer_node). It watches the generic "
                "/openral/critic/score topic that reward models publish (Robometer, "
                "a future SARM, success classifiers), and emits a Tier-C "
                "FailureTrigger on /openral/failure/critic when a critic stalls — the "
                "reasoner already maps that to a forced Tier-C tick. Advisory-only — "
                "never actuates. Default off."
            ),
        ),
        DeclareLaunchArgument(
            "critic_stall_patience",
            default_value="5",
            description=(
                "Consecutive below-threshold, non-improving critic-score "
                "samples (per critic_id) before the producer fires. Ignored unless "
                "enable_critic."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_manifest",
            default_value="",
            description=(
                "Path to a kind:reward rSkill manifest. Empty defaults to "
                "the in-tree rskills/robometer-4b/rskill.yaml. weights_uri may be "
                "hf://org/repo or local:///abs/path (a pre-quantized NF4 checkpoint "
                "loaded directly as 4-bit). Ignored unless enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_task",
            default_value="",
            description=(
                "Default task instruction the reward monitor scores when "
                "a query leaves task empty (e.g. the operator's task goal). The "
                "reasoner normally passes the active task per query. Ignored unless "
                "enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_locators",
            default_value="",
            description=(
                "Comma-separated kind:detector manifest paths for the "
                "on-demand open-vocab locators to bring up alongside the continuous "
                "detector. Each becomes a namespaced lifecycle node serving "
                "/openral/perception/<alias>/locate_in_view, selectable by the "
                "reasoner via LocateInViewTool.detector. Empty = no on-demand "
                "locator. Ignored unless enable_object_detector is true."
            ),
        ),
        DeclareLaunchArgument(
            "enable_reasoner",
            default_value="true",
            description=(
                "Spawn the reasoner and prompt-router lifecycle nodes. Disable "
                "for direct-rSkill deployments that submit an explicit "
                "/openral/execute_rskill goal and do not require LLM planning."
            ),
        ),
        DeclareLaunchArgument(
            "enable_dashboard",
            default_value="true",
            description=(
                "Spawn the live observability dashboard child as part of "
                "the launch graph. Pass false for headless CI runs or "
                "when the operator brings up `openral dashboard` "
                "manually in a separate terminal."
            ),
        ),
        DeclareLaunchArgument(
            "enable_foxglove",
            default_value="false",
            description=(
                "Spawn the read-only foxglove_bridge as part of "
                "the deploy-sim runtime graph. Default off. The bridge binds "
                "to 127.0.0.1:<foxglove_port> and exposes only the Bucket-1 "
                "topic allowlist (no safety/e-stop/action topics). View-only: "
                "clientPublish, services, and parameters capabilities are "
                "omitted. Pass true to enable."
            ),
        ),
        DeclareLaunchArgument(
            "foxglove_port",
            default_value="8765",
            description=(
                "Foxglove WebSocket port "
                "(ws://127.0.0.1:<foxglove_port>). Default 8765. "
                "Ignored unless enable_foxglove is true."
            ),
        ),
        DeclareLaunchArgument(
            "initial_task_prompt",
            default_value="",
            description=(
                "Single operator goal published to /openral/prompt at startup "
                "(cli-level priority 100). Set by ``--initial-task`` on the CLI; "
                "the prompt_router_node forwards it to the reasoner at on_activate "
                "time so the first tick sees the operator's goal without a manual "
                "``openral prompt`` call. Empty (default) = no startup prompt; "
                "the reasoner idles until a manual ``openral prompt`` or dashboard "
                "prompt arrives."
            ),
        ),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=compose_runtime_graph)])
