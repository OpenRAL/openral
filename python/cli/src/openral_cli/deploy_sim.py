"""``openral deploy sim`` — boot the full ROS graph against a digital-twin HAL.

Sibling of ``openral deploy run``: where ``deploy run`` drives a tight Python
tick loop against a HAL + ``SafetyClient``, ``deploy sim`` shells
``ros2 launch openral_rskill_ros sim_e2e.launch.py`` so the operator gets
dashboard + C++ safety kernel + reasoner + prompt router + runtime
(world_state + skill_runner) + HAL in one command, running against the
HAL's digital-twin (MuJoCo viewer) mode.

The launch graph is robot-agnostic — one ``sim_e2e.launch.py`` for every
robot. The CLI's job is to resolve everything robot-specific:

* The robot manifest at ``robots/<robot_id>/robot.yaml``.
* The HAL package + executable + node name + per-robot default
  parameter dict, looked up by ``robot_id`` in ``_ROBOT_HAL_REGISTRY``.
  The lookup asserts the HAL's declared ``supported_robot_names`` matches
  the manifest's ``name`` field — a mismatch (someone wires
  ``openarm`` to the so100 HAL by accident) fails loud.

No envelope YAML file is involved on either side:

* The robot manifest is the single source of truth for the safety
  kernel envelope. ``sim_e2e.launch.py``'s ``compose_runtime_graph``
  callback loads ``robot.yaml`` via Pydantic at launch time, calls
  ``openral_safety.envelope_loader.compute_intersection(robot, skill=None)``
  + ``kernel_params_from_envelope(...)``, and forwards each field of
  the resulting :class:`EnvelopeIntersection` as a ROS parameter on
  the kernel node (see ``cpp/openral_safety_kernel/src/envelope.cpp``
  — `n_dof`, `joint_position_min/max`, `joint_velocity_max`,
  `joint_torque_max`, scalar caps, deadman flag). The legacy
  ``envelope_file:=PATH`` path was removed in ADR-0020 PR-K.

The reasoner is NOT preselected: it walks the in-tree ``rskills/`` and
filters by the robot's capabilities at on_configure. ``openral deploy sim``
intentionally does not accept ``--rskill`` because the reasoner picks
the active rSkill dynamically and switching skills is its job, not the
operator's bring-up command.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from rich.console import Console

if TYPE_CHECKING:
    from openral_core import RobotDescription, RSkillManifest

__all__ = [
    "LaunchInvocation",
    "assert_ros2_packages_discoverable",
    "deploy_sim_command",
    "resolve_launch_invocation",
]

_console = Console(soft_wrap=True)


@dataclass(frozen=True)
class _HalSpec:
    """Per-robot HAL spawn descriptor.

    ``package`` / ``executable`` / ``node_name`` parameterise the HAL
    ``LifecycleNode`` in ``sim_e2e.launch.py``. ``supported_robot_names``
    is the set of ``RobotDescription.name`` values this HAL is willing
    to drive — the CLI asserts the loaded manifest's name is in this
    set so a mis-paired registry entry (openarm HAL routed at an so100
    manifest) fails loud at resolution time, not at the HAL's first
    actuation tick. ``default_params`` is the parameter dict the HAL
    accepts via its ROS parameter interface; operator-supplied
    ``--hal`` overrides win.
    """

    package: str
    executable: str
    node_name: str
    supported_robot_names: frozenset[str]
    default_params: dict[str, object] = field(default_factory=dict)
    # ADR-0025 Stage 3 — HAL nodes that declare a `sim_env_yaml` ROS
    # parameter (today: openral_hal_panda_mobile) opt in via this
    # flag. When True, `openral deploy sim --config <yaml>` injects the
    # resolved config path into hal_params so the HAL builds a live
    # `openral_sim.SimRollout` env in-process. Other HALs leave it
    # False so rclpy doesn't reject the unknown parameter at startup
    # (`automatically_declare_parameters_from_overrides=False` is the
    # default).
    supports_sim_env_yaml: bool = False
    # ADR-0032 — HAL nodes built via `make_lifecycle_main_from_manifest`
    # (franka / ur5e / ur10e / aloha / g1 / h1 / rizon4 / so100 / so101) declare
    # `robot_yaml` + `hal_mode` params and construct their HAL through
    # `build_hal(mode=...)`. When True, `openral deploy sim` injects the resolved
    # manifest path + `hal_mode="sim"`; `openral deploy run` injects
    # `hal_mode="real"`.
    manifest_driven: bool = False
    # issue #191 Phase 2 — a manifest-driven arm that builds a *bare* MuJoCo
    # twin (`MujocoArmHAL.from_description`) in sim rather than scene-attaching:
    # A manifest-driven arm that builds its OWN sim MJCF rather than
    # scene-attaching: so100 / so101 derive a bare `MujocoArmHAL` twin from the
    # manifest's `sim:` block; openarm composes a tabletop MJCF from
    # `scene_defaults.composition` (issue #191 Phase 3b). When True, the
    # manifest-driven injection below skips `sim_env_yaml` so the node builds the
    # explicit `hal.sim` HAL (with the composed mjcf threaded in) instead of a
    # scene-attached `SimAttachedHAL`. Other manifest arms leave it False and
    # scene-attach (ADR-0034).
    bare_twin_sim: bool = False


_ROBOT_HAL_REGISTRY: dict[str, _HalSpec] = {
    # Keys match the directory name under ``robots/<robot_id>/``; the
    # CLI looks up ``robots/<robot_id>/robot.yaml`` against this key. To
    # add a new robot to ``openral deploy sim``: ship a ``openral_hal_<X>``
    # ROS package with a ``lifecycle_node.py`` executable, register it
    # in the ros2-build Justfile target, and add an entry here. The
    # Python HAL adapters under ``python/hal/src/openral_hal/`` are a
    # different layer (used by ``openral deploy run`` / ``openral sim run``);
    # they do not provide ROS lifecycle nodes by themselves.
    "openarm": _HalSpec(
        package="openral_hal_openarm",
        executable="lifecycle_node.py",
        node_name="openral_hal_openarm",
        supported_robot_names=frozenset({"openarm_v2", "openarm"}),
        # issue #191 Phase 3b — migrated onto the manifest-driven node. Scene
        # composition (robot_lift_z / robot_forward_x / white_background) moved to
        # the manifest's `scene_defaults.composition`; HAL kwargs (settle_steps /
        # gravity_enabled / staleness_limit_s) to `hal.parameters`; cameras via
        # SimSensorBridge + OpenArmMujocoHAL.read_images. Only the viewer toggle
        # remains a node ROS param.
        default_params={
            "viewer_enabled": True,
        },
        manifest_driven=True,
        # openarm builds its own MJCF via scene COMPOSITION (the manifest's
        # `scene_defaults.composition`), not scene-attach — so suppress the
        # `sim_env_yaml` injection (see `bare_twin_sim`).
        bare_twin_sim=True,
    ),
    "so100_follower": _HalSpec(
        # issue #191 Phase 2 — migrated onto the manifest-driven node
        # (`make_lifecycle_main_from_manifest`). `hal_mode="sim"` derives a bare
        # `MujocoArmHAL` twin; `hal_mode="real"` builds `SO100FollowerHAL` with
        # `port` / `calibrate_on_connect` from the manifest's `hal.parameters`
        # (so no `port` ROS param — the node would reject the unknown override).
        package="openral_hal_so100",
        executable="lifecycle_node.py",
        node_name="openral_hal_so100",
        supported_robot_names=frozenset({"so100_follower"}),
        manifest_driven=True,
        bare_twin_sim=True,
    ),
    "so101_follower": _HalSpec(
        # The SO-101 is a hardware revision of the SO-100: identical 6-DoF
        # kinematic chain driven by the same lerobot Feetech STS3215 serial
        # backend, so it reuses the ``openral_hal_so100`` ROS lifecycle node
        # (now manifest-driven) verbatim. ``openral deploy sim`` injects this
        # robot's manifest + ``hal_mode="sim"`` and the node builds a bare
        # ``MujocoArmHAL.from_description`` from ``robots/so101_follower/
        # robot.yaml`` (``assets.mjcf`` → ``so101_new_calib``). The SAME node
        # serves so100 (``so_arm100``) and so101 (``so101_new_calib``) from
        # their own MJCF, so no dedicated ``openral_hal_so101`` package exists
        # or is needed (CLAUDE.md §1.13). The robot-name guard below keeps this
        # entry bound to the so101 manifest.
        package="openral_hal_so100",
        executable="lifecycle_node.py",
        node_name="openral_hal_so100",
        supported_robot_names=frozenset({"so101_follower"}),
        manifest_driven=True,
        bare_twin_sim=True,
    ),
    "panda_mobile": _HalSpec(
        # ADR-0024 / ADR-0025 — panda_mobile publishes /joint_states + /odom +
        # /scan and broadcasts the odom -> base_link TF that slam_toolbox + Nav2
        # both need. issue #191 Phase 3 migrated it onto the manifest-driven node:
        # MobileBaseBridge owns /odom + TF + /cmd_vel (gated on the manifest's
        # `base_joints`), SimSensorBridge owns /scan + cameras + depth + viewer.
        package="openral_hal_panda_mobile",
        executable="lifecycle_node.py",
        node_name="openral_hal_panda_mobile",
        supported_robot_names=frozenset({"panda_mobile"}),
        default_params={
            "odom_publish_rate_hz": 20.0,
            # The /scan envelope (rate / beam count / range) is NOT hardcoded
            # here: it's derived from the robot.yaml lidar_2d sensor at resolve
            # time (see _resolve_deploy_target) so robot.yaml stays the single
            # source of truth. viewer_enabled opens a non-blocking
            # mujoco.viewer.launch_passive window (no-op headless).
            "viewer_enabled": True,
        },
        # Scene-attach (SimAttachedHAL via sim_env_yaml) when a scene config is
        # given; bare PandaMobileHAL digital twin otherwise.
        manifest_driven=True,
        supports_sim_env_yaml=True,
    ),
    "franka_panda": _HalSpec(
        package="openral_hal_franka",
        executable="lifecycle_node.py",
        node_name="openral_hal_franka",
        supported_robot_names=frozenset({"franka_panda"}),
        default_params={},
        manifest_driven=True,
    ),
    "ur5e": _HalSpec(
        package="openral_hal_ur5e",
        executable="lifecycle_node.py",
        node_name="openral_hal_ur5e",
        supported_robot_names=frozenset({"ur5e"}),
        default_params={},
        manifest_driven=True,
    ),
    "ur10e": _HalSpec(
        package="openral_hal_ur10e",
        executable="lifecycle_node.py",
        node_name="openral_hal_ur10e",
        supported_robot_names=frozenset({"ur10e"}),
        default_params={},
        manifest_driven=True,
    ),
    "aloha_bimanual": _HalSpec(
        package="openral_hal_aloha",
        executable="lifecycle_node.py",
        node_name="openral_hal_aloha",
        supported_robot_names=frozenset({"aloha_bimanual"}),
        default_params={},
        manifest_driven=True,
    ),
    "aloha_agilex": _HalSpec(
        # RoboTwin owns the SAPIEN robot; deploy-sim only needs a ROS lifecycle
        # host for SimAttachedHAL. This package is intentionally sim-only and
        # generic, so it never claims a real AgileX hardware transport.
        package="openral_hal_scene_attached",
        executable="lifecycle_node.py",
        node_name="openral_hal_scene_attached",
        supported_robot_names=frozenset({"aloha_agilex"}),
        default_params={},
        manifest_driven=True,
        supports_sim_env_yaml=True,
    ),
    "widowx": _HalSpec(
        # SimplerEnv owns the SAPIEN WidowX twin; OpenRAL has no real WidowX HAL.
        # The generic scene-attached node is valid for deploy-sim only because
        # build_hal(mode="sim", sim_env_yaml=...) bypasses hal.sim entirely.
        package="openral_hal_scene_attached",
        executable="lifecycle_node.py",
        node_name="openral_hal_scene_attached",
        supported_robot_names=frozenset({"widowx"}),
        default_params={},
        manifest_driven=True,
        supports_sim_env_yaml=True,
    ),
    "g1": _HalSpec(
        package="openral_hal_g1",
        executable="lifecycle_node.py",
        node_name="openral_hal_g1",
        supported_robot_names=frozenset({"g1"}),
        default_params={},
        manifest_driven=True,
    ),
    "h1": _HalSpec(
        package="openral_hal_h1",
        executable="lifecycle_node.py",
        node_name="openral_hal_h1",
        supported_robot_names=frozenset({"h1"}),
        default_params={},
        manifest_driven=True,
    ),
    "rizon4": _HalSpec(
        package="openral_hal_rizon4",
        executable="lifecycle_node.py",
        node_name="openral_hal_rizon4",
        supported_robot_names=frozenset({"rizon4"}),
        default_params={},
        manifest_driven=True,
    ),
}


@dataclass(frozen=True)
class LaunchInvocation:
    """Resolved ``ros2 launch`` argv + the metadata that built it.

    Returned by :func:`resolve_launch_invocation` so the dispatcher can
    pretty-print under ``--dry-run`` and the unit tests can assert on
    the resolved fields without touching ``subprocess``.
    """

    robot_id: str
    robot_yaml: Path
    robot_manifest_name: str
    hal: _HalSpec
    hal_params: dict[str, object]
    hal_mode: str
    """ADR-0036 — ``"sim"`` (``openral deploy sim``) or ``"real"``
    (``openral deploy run``). Forwarded into the launch as ``hal_mode:=…`` so
    the reasoner's action-mode palette gate matches the HAL this graph
    brings up (sim admits cartesian/OSC skills the scene's robosuite OSC
    controller can execute; real admits only the robot's declared
    ``supported_control_modes``)."""
    reset_to_pose_service: str
    approach_skill_id: str
    """ADR-0053 — MoveIt approach rSkill URI (e.g. ``rskills/rskill-moveit-joints``)
    forwarded into the launch as ``approach_skill_id:=…`` so the skill_runner
    plans a collision-free MoveGroup motion to the next skill's ``starting_pose``
    instead of the teleport snap. Empty (the default) keeps the legacy
    best-effort ``ResetToPose`` snap — opt in with ``--approach-skill-id`` once a
    ``move_group`` is in the graph (ADR-0053 phase 4)."""
    enable_slam: bool
    """ADR-0025 opt-in. Set by ``openral deploy sim --enable-slam``;
    forwarded into the launch as ``enable_slam:=true``."""
    slam_backend: str
    """ADR-0064 — which SLAM backend the launch composes when
    ``enable_slam`` is true: ``"lidar"`` (slam_toolbox, needs ``/scan``),
    ``"visual"`` (cuVSLAM + nvblox, camera-based, for lidar-less robots),
    or ``"none"`` (no SLAM). Resolved from capabilities — ``has_lidar``
    selects ``lidar`` (it wins when both flags are set, needing no AI depth
    model); else ``has_vision_slam`` selects ``visual``. Forwarded as
    ``slam_backend:=…``."""
    enable_nav2: bool
    """ADR-0025 opt-in for the Nav2 navigation stack. Set by
    ``openral deploy sim --enable-nav2``; forwarded into the launch as
    ``enable_nav2:=true``. Defaults to ``has_lidar`` — every robot
    that runs slam_toolbox needs a planner to consume the resulting
    map, so the two are auto-co-enabled."""
    enable_octomap: bool
    """ADR-0030 opt-in for the world-collision perception leg
    (octomap_server + openral_octomap_bridge + the kernel's
    capsule-vs-voxel check). Set by ``openral deploy sim --enable-octomap``;
    forwarded as ``enable_octomap:=true``. Defaults to "auto" = the robot
    manifest declares a depth SensorSpec (nothing to map otherwise)."""
    clock_origin: str
    """ClockAuthority origin forwarded as ``clock_origin:=…``. Derived from
    the deployment: simulator-owned elapsed time for sim backends that expose
    ``sim_time_ns``; host wall time for real deployments or clock-less scenes.
    Operators do not choose ROS ``use_sim_time`` directly."""
    enable_object_detector: bool
    """ADR-0035 object-detection perception leg
    (ros_image_detector_node → /openral/perception/objects → world-state
    object-lift → /openral/world_voxels). **On by default**; disabled with
    ``openral deploy sim --no-object-detector``. Forwarded as
    ``enable_object_detector:=true|false``. Auto-downgrades to ``false`` when no
    backend is available (omdet deps absent *and* the RT-DETR ONNX missing)."""
    object_detector_onnx: Path
    """ADR-0035 — absolute path to the RT-DETR ONNX weights used by the legacy /
    fallback detector path. Forwarded as ``object_detector_onnx:=<path>``.
    Defaults to the in-tree ``rskills/rtdetr-coco-r18/model.onnx``; passing it
    explicitly selects the fixed-label RT-DETR path over the omdet default."""
    object_detector_manifest: str
    """ADR-0037 2026-06-09 — path to a kind:detector rSkill manifest. When set,
    the detector node builds its backend from the manifest (runtime:pytorch →
    the open-vocab LocateAnything VLM sidecar; runtime:onnx → RT-DETR ONNX).
    Forwarded as ``object_detector_manifest:=<path>``. Empty = the RT-DETR ONNX
    fallback. By default (no explicit override) this resolves to the
    ``omdet-turbo-indoor`` manifest when the omdet deps are importable."""
    object_detector_query: str
    """ADR-0037 2026-06-09 — initial open-vocabulary query for a VLM detector
    (e.g. 'red mug'). Forwarded as ``object_detector_query:=<text>``. Empty =
    the manifest's ``detector.labels`` default. Ignored by ONNX detectors."""
    object_detector_locators: tuple[str, ...]
    """ADR-0056 — resolved manifest paths of the ``mode: on_demand`` open-vocab
    locators to bring up alongside the continuous detector. The launch builds one
    namespaced lifecycle node per entry (``/openral/perception/<alias>/locate_in_view``)
    so the reasoner can choose a model via ``LocateInViewTool.detector``. Forwarded
    as ``object_detector_locators:=<comma-joined paths>`` only when non-empty.
    Defaults to the omdet-turbo-locator manifest when the detector is on and the
    omdet deps are importable (LocateAnything is opt-in via an explicit path)."""
    spatial_memory_ingest: bool
    """ADR-0038 opt-in. Set by ``openral deploy sim --spatial-memory-ingest``;
    forwarded as ``spatial_memory_ingest:=true``. The reasoner then accumulates
    a durable ADR-0038 SpatialMemory from the object-lift producer's
    ``WorldState.detected_objects`` so ``recall_object`` recalls what the robot
    has seen. Defaults to "auto" = enabled when the object detector is."""
    enable_foxglove: bool
    """ADR-0059 opt-in. Off by default. Set by ``openral deploy sim --foxglove``;
    forwarded as ``enable_foxglove:=true``. Spawns the read-only
    ``foxglove_bridge`` as part of the deploy-sim runtime graph so operators
    can view the live scene (cameras, /tf, joint states, nav map) in
    Foxglove Studio without an extra bring-up step. Cannot actuate the robot
    (view-only; ``clientPublish``/``services`` capabilities omitted)."""
    foxglove_port: int
    """ADR-0059 — Foxglove WebSocket port. Forwarded as ``foxglove_port:=…``.
    Default 8765 (the ``foxglove_bridge`` upstream default)."""
    initial_task_prompt: str
    """Operator goal delivered to the reasoner at startup (cli priority).

    Sourced only from ``--initial-task`` (or a later live ``/openral/prompt``);
    deploy never derives it from scene tasks (ADR-0073). Forwarded as
    ``initial_task_prompt:=<text>`` to the launch file. Empty = the reasoner
    idles until an operator prompt arrives."""
    enable_reward_monitor: bool
    """ADR-0057/0077 — whether the Robometer reward monitor is brought up
    co-active with the VLA. When true the deploy preflight checks the VLA↔reward
    VRAM pairing (:func:`_preflight_reward_vram_fit`) before bringing up ROS."""
    reward_monitor_manifest: str
    """ADR-0077 — the RESOLVED reward-monitor manifest path. Defaults from the
    capability-matched VLA palette's ``reward_rskill_name`` (the pairing the
    reasoner will honour) when ``--reward-monitor-manifest`` is not given; empty
    when no reward monitor is active. Forwarded as ``reward_monitor_manifest:=…``."""
    argv_template: list[str]
    """``argv_template`` carries ``HAL_PARAMS_FILE_PLACEHOLDER`` where
    the temp HAL-params YAML path goes. The dispatcher substitutes it
    once the file exists."""


def _repo_root_from(start: Path) -> Path:
    """Walk up from ``start`` until a directory with ``robots/`` and ``rskills/``."""
    here = start.resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "rskills").is_dir():
            return ancestor
    raise ROSConfigError(
        f"Could not locate OpenRAL repo root above {start}; "
        "expected a parent containing both robots/ and rskills/."
    )


def _load_scene_robot_id(config: Path) -> str | None:
    """Return the ``robot_id`` declared in a DeployScene YAML, or None.

    Strict DeployScene loading (ADR-0041): ``openral deploy sim --config``
    accepts a DeployScene YAML only (scene + optional robot, no task).
    SimScene / BenchmarkScene YAMLs are rejected with a redirect message.

    Two paths to discover the robot:

    1. ``robot_id:`` declared explicitly in the YAML → use it.
    2. The scene id is registered with a ``fixed_robot=`` in
       ``openral_sim.SCENES`` (every robocasa kitchen / LIBERO / ALOHA
       / MetaWorld / PushT scene) → look it up. Lets robocasa-shaped
       YAMLs (which forbid ``robot_id:`` per the schema's free-axis
       guard) still resolve into ``openral deploy sim`` without the
       operator passing ``--robot`` redundantly.
    """
    from openral_core import DeployScene, load_scene_strict  # reason: defer schema import

    try:
        env = load_scene_strict(str(config), DeployScene)
    except (ROSConfigError, FileNotFoundError) as exc:
        raise ROSConfigError(f"failed to load --config {config}: {exc}") from exc
    if env.robot_id is not None:
        return env.robot_id
    # Fixed-robot scene fallback. The sim registry is the source of
    # truth for which scene ids hard-fix a robot.
    try:
        from openral_sim import SCENES  # reason: defer optional dep
    except ImportError:
        return None
    if env.scene.id in SCENES:
        return SCENES.fixed_robot(env.scene.id)
    return None


def _scan_params_from_description(description: RobotDescription) -> dict[str, object]:
    """Map a robot's ``lidar_2d`` sensor to HAL ``scan_*`` ROS params.

    ADR-0025 single source of truth — ``openral deploy sim`` forwards these
    to the HAL instead of hardcoding a per-robot scan envelope. Returns
    an empty dict when the robot declares no LiDAR (non-mobile robots,
    no scan synthesis), so the call site is a no-op for them.
    """
    lidar = description.lidar_sensor
    if lidar is None:
        return {}
    params: dict[str, object] = {}
    if lidar.rate_hz:
        params["scan_publish_rate_hz"] = lidar.rate_hz
    if lidar.n_channels is not None:
        params["scan_n_beams"] = lidar.n_channels
    if lidar.range_max_m is not None:
        params["scan_max_range_m"] = lidar.range_max_m
    if lidar.range_min_m is not None:
        params["scan_min_range_m"] = lidar.range_min_m
    return params


def _scene_backend_has_sim_clock(config: Path | None) -> bool:
    """Return whether a DeployScene backend can be the OpenRAL simulation clock authority."""
    if config is None:
        return False
    from openral_core import DeployScene, PhysicsBackend, load_scene_strict

    scene = load_scene_strict(str(config), DeployScene)
    return scene.scene.backend in {
        PhysicsBackend.MUJOCO,
        PhysicsBackend.MUJOCO_MJX,
        PhysicsBackend.ISAACSIM,
        PhysicsBackend.SAPIEN,
    }


def _resolve_clock_origin(*, hal_mode: str, config: Path | None) -> str:
    """Resolve the OpenRAL clock authority origin for the launch graph.

    Real deployments use host wall time. Sim deployments use simulator elapsed
    time when the deploy scene backend exposes a sim clock. Scene-attached HALs
    and bare MuJoCo twins both expose ``sim_time_ns``; clock-less scenes stay in
    host-wall time so ROS node clocks never pin at zero.
    """
    if hal_mode != "sim":
        return "host_wall"
    return "simulation" if _scene_backend_has_sim_clock(config) else "host_wall"


def _omdet_runtime_available() -> bool:
    """True when the OmDet-Turbo continuous detector's runtime deps are importable.

    The default object detector is omdet-turbo-indoor (open-vocabulary, grounds
    arbitrary indoor/kitchen objects rather than the fixed COCO-80 of RT-DETR).
    Its in-process zero-shot backend
    (``openral_runner.backends.gstreamer.omdet_turbo_detector.OmDetTurboDetector``)
    needs ``transformers`` + ``timm`` — the ``omdet`` dependency group. When they
    are absent (a checkout that only synced the base group),
    :func:`resolve_launch_invocation` gracefully falls back to the in-tree
    RT-DETR COCO ONNX so ``deploy sim`` still brings up a detector instead of the
    node hard-failing at backend build.

    Patched in tests to exercise both branches without touching the environment.
    """
    # Local import: the probe is cheap and scoped to this one decision.
    import importlib.util

    return all(importlib.util.find_spec(mod) is not None for mod in ("transformers", "timm"))


def _object_detector_onnx_present(path: Path) -> bool:
    """True when the fallback RT-DETR COCO ONNX weights are on disk.

    The weights (``rskills/rtdetr-coco-r18/model.onnx``, ~2 MB) are gitignored, so
    they are present on a weights-fetched dev host but absent in a bare CI
    checkout. :func:`resolve_launch_invocation` downgrades the detector leg off
    when neither omdet deps nor these weights can build a backend. Factored out so
    tests can exercise the fallback-selection logic without the gitignored binary.
    """
    return path.is_file()


def _resolve_slam_backend(*, has_lidar: bool, has_vision_slam: bool, enable_slam: bool) -> str:
    """ADR-0064 — pick the SLAM backend the launch composes.

    Returns one of ``"lidar"`` (slam_toolbox; needs ``/scan``), ``"visual"``
    (cuVSLAM + nvblox; camera-based, for lidar-less robots), or ``"none"``.

    ``has_lidar`` wins when both flags are set: the 2D-lidar leg is the
    cheaper, proven path and needs no AI depth model. A resolved
    ``enable_slam`` of ``False`` (an explicit ``--no-enable-slam``) forces
    ``"none"`` so the forwarded ``slam_backend:=`` arg never contradicts the
    ``enable_slam:=false`` arg.

    Args:
        has_lidar: ``RobotCapabilities.has_lidar``.
        has_vision_slam: ``RobotCapabilities.has_vision_slam``.
        enable_slam: The resolved SLAM-on decision (manifest auto or flag).

    Returns:
        The backend identifier forwarded as the ``slam_backend`` launch arg.

    Example:
        >>> _resolve_slam_backend(has_lidar=False, has_vision_slam=True, enable_slam=True)
        'visual'
        >>> _resolve_slam_backend(has_lidar=True, has_vision_slam=True, enable_slam=True)
        'lidar'
        >>> _resolve_slam_backend(has_lidar=True, has_vision_slam=False, enable_slam=False)
        'none'
    """
    if not enable_slam:
        return "none"
    if has_lidar:
        return "lidar"
    if has_vision_slam:
        return "visual"
    return "none"


def _memory_bundle_launch_args(memory_dir: str) -> list[str]:
    """Derive the sim_e2e.launch.py bundle args from a deploy memory-bundle dir (ADR-0072 §3b).

    The bundle is a directory holding any of ``MEMORY.md`` (semantic memory),
    ``scene_graph.json`` (3D world-state graph), and ``map.yaml`` (2D occupancy grid).
    Each artifact is forwarded to its own consumer's launch arg. The dir must exist
    (the robot writes ``MEMORY.md`` into it); ``scene_graph.json`` / ``map.yaml`` are
    forwarded only when present, so a fresh bundle (empty dir) just starts the reasoner
    with empty memory and no preloaded scene/map.
    """
    from openral_core.exceptions import ROSConfigError

    d = Path(memory_dir).expanduser()
    if not d.is_dir():
        raise ROSConfigError(
            f"--memory-dir {memory_dir!r} is not an existing directory. Create the bundle "
            "dir first (the robot writes MEMORY.md into it; place scene_graph.json / map.yaml "
            "there to preload the scene graph + occupancy grid)."
        )
    # memory_md_path may not exist yet — the reasoner creates it on the first memory_write.
    args = [f"memory_md_path:={d / 'MEMORY.md'}"]
    scene_graph = d / "scene_graph.json"
    if scene_graph.is_file():
        args.append(f"spatial_memory_path:={scene_graph}")
    map_yaml = d / "map.yaml"
    if map_yaml.is_file():
        args.append(f"map_path:={map_yaml}")
    return args


# ADR-0077 — the in-tree directory of the default reward/progress-monitor rSkill
# the deploy pairs with a VLA when nothing names one. Mirrors the reasoner's
# launch default (``rskills/robometer-4b/rskill.yaml``, sim_e2e.launch.py).
_DEFAULT_REWARD_RSKILL_DIR = "robometer-4b"


def _detect_gpu_total_vram_gb() -> float:
    """Total VRAM (GB) of GPU 0 via ``nvidia-smi``, or ``0.0`` when unavailable.

    Torch-free probe (the CLI must not import torch just to size the GPU) — a
    deliberate mirror of ``openral_reasoner_ros.reasoner_node._detect_gpu_total_vram_gb``
    (a private, ROS-package-local helper the CLI cannot import without pulling in
    rclpy). Used by the ADR-0077 deploy preflight. Any failure (no nvidia-smi, no
    GPU, parse error) returns ``0.0`` → the caller skips the pair check rather than
    blocking a launch on a host where the budget cannot be read.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    lines = out.stdout.strip().splitlines()
    if not lines:
        return 0.0
    try:
        return float(lines[0].strip()) / 1024.0  # MiB → GiB
    except ValueError:
        return 0.0


def _capability_matched_manifests(
    repo_root: Path,
    description: RobotDescription,
    *,
    commercial_deployment: bool = False,
) -> list[RSkillManifest]:
    """In-tree rSkill manifests that match this robot's reasoner palette.

    Loads every ``rskills/*/rskill.yaml`` and runs the same
    capability/role/license filter the reasoner seeds at ``on_configure``
    (:func:`openral_reasoner.palette.build_tool_palette`), returning the matched
    manifests. ``openral deploy sim`` does not preselect a VLA — the reasoner picks
    one at runtime from exactly this set — so reward resolution + the ADR-0077 VRAM
    preflight both reason over it (the "VLA known at launch" is the *palette*, not a
    single policy). Unloadable manifests are skipped (the reasoner skips them too).
    """
    from openral_core import RSkillManifest
    from openral_reasoner.palette import build_tool_palette

    manifests: list[RSkillManifest] = []
    for path in sorted((repo_root / "rskills").glob("*/rskill.yaml")):
        try:
            manifests.append(RSkillManifest.from_yaml(str(path)))
        except (OSError, ValueError):
            continue
    if not manifests:
        return []
    palette = build_tool_palette(
        installed_skills=manifests,
        robot_capabilities=description.capabilities,
        commercial_deployment=commercial_deployment,
    )
    matched = set(palette.execute_rskill_ids)
    return [m for m in manifests if m.name in matched]


def _resolve_reward_monitor_manifest(
    *,
    repo_root: Path,
    description: RobotDescription,
    explicit_manifest: str | None,
) -> str:
    """Resolve the reward-monitor manifest, defaulting from the VLA pairing (ADR-0077 §4).

    The pairing used to be implicit: the reward model was chosen by a flag
    (``--reward-monitor-manifest``) wholly decoupled from the VLA the reasoner
    picks. ADR-0077 records the pairing on the VLA manifest
    (``reward_rskill_name``); this honours it at launch. Because ``deploy sim``
    does not preselect a single VLA, we read the pairing across the
    capability-matched VLA palette:

    * An explicit ``--reward-monitor-manifest`` always wins (operator override).
    * Else, if the palette's VLAs agree on a single ``reward_rskill_name``, resolve
      that rSkill ``name`` to its in-tree manifest path.
    * Else (no VLA names a reward model, the named model is not in-tree, or the
      palette VLAs disagree) fall back to the deployment default
      (``robometer-4b``). A disagreement is warned — the reasoner additionally
      warns per-VLA at dispatch when a VLA's ``reward_rskill_name`` differs from
      the loaded reward model.

    Returns the resolved manifest path as a string (empty only when even the
    default is missing from the tree).
    """
    if explicit_manifest:
        return explicit_manifest

    from openral_core import RSkillManifest

    default_path = repo_root / "rskills" / _DEFAULT_REWARD_RSKILL_DIR / "rskill.yaml"
    default = str(default_path.resolve()) if default_path.is_file() else ""

    # name → manifest path for every in-tree kind:reward rSkill.
    reward_index: dict[str, Path] = {}
    for path in sorted((repo_root / "rskills").glob("*/rskill.yaml")):
        try:
            man = RSkillManifest.from_yaml(str(path))
        except (OSError, ValueError):
            continue
        if man.kind == "reward":
            reward_index[man.name] = path

    named = {
        m.reward_rskill_name
        for m in _capability_matched_manifests(repo_root, description)
        if m.kind == "vla" and m.reward_rskill_name
    }
    if not named:
        return default
    if len(named) > 1:
        _console.print(
            "[yellow]warning:[/yellow] capability-matched VLAs name different reward "
            f"models {sorted(named)!r} (ADR-0077); defaulting the reward monitor to "
            f"{_DEFAULT_REWARD_RSKILL_DIR!r}. The reasoner re-checks each VLA's pairing "
            "at dispatch."
        )
        return default
    (target_name,) = tuple(named)
    target_path = reward_index.get(target_name)
    if target_path is None:
        _console.print(
            f"[yellow]warning:[/yellow] VLA(s) pair with reward model {target_name!r} "
            "(ADR-0077) but no in-tree kind:reward rSkill declares that name; "
            f"defaulting the reward monitor to {_DEFAULT_REWARD_RSKILL_DIR!r}."
        )
        return default
    return str(target_path.resolve())


def _preflight_reward_vram_fit(  # noqa: PLR0912  # reason: linear per-VLA classification (fit / oom / undeclared) + the three notify branches read clearest inline
    *,
    repo_root: Path,
    description: RobotDescription,
    reward_manifest_path: str,
    gpu_total_gb: float,
    commercial_deployment: bool = False,
) -> None:
    """Fail fast before launch when no VLA can co-reside with the reward model (ADR-0077 §4).

    A VLA emits no success signal of its own, so it must run with its reward model
    resident alongside it (ADR-0074). The reasoner enforces this per-VLA at
    dispatch (``_refuse_unfittable_vla``) — but only *after* ROS is up. This is the
    pre-LAUNCH gate: build the same capability-matched VLA palette the reasoner
    will, and run :func:`openral_core.schemas.assert_vla_reward_fits` for each VLA
    against the reward model + the GPU budget.

    The contract mirrors :func:`_preflight_palette_deps`: it is advisory per-VLA
    (the reasoner drops a non-fitting VLA from dispatch anyway) and a HARD gate only
    when the palette would be empty of *runnable* policies — i.e. **no** matched VLA
    can dispatch with the reward model resident. In that case the deploy could
    actuate nothing, so we notify and ``typer.Exit(1)`` before bringing up ROS
    instead of booting a graph that dispatches a VLA blind or OOMs mid-run.

    Skipped (returns) when ``gpu_total_gb <= 0.0`` (budget unreadable — defer to the
    reasoner's runtime check), when no reward model is active, or when the robot has
    no capability-matched VLA palette to check.
    """
    if gpu_total_gb <= 0.0 or not reward_manifest_path:
        return
    from openral_core import RSkillManifest
    from openral_core.exceptions import ROSGPUMemoryError
    from openral_core.schemas import assert_vla_reward_fits

    try:
        reward = RSkillManifest.from_yaml(reward_manifest_path)
    except (OSError, ValueError) as exc:
        _console.print(
            f"[red]config error:[/red] reward monitor manifest {reward_manifest_path!r} "
            f"failed to load (ADR-0077 preflight): {exc}"
        )
        raise typer.Exit(code=1) from exc

    vlas = [
        m
        for m in _capability_matched_manifests(
            repo_root, description, commercial_deployment=commercial_deployment
        )
        if m.kind == "vla"
    ]
    if not vlas:
        return

    fits: list[str] = []
    oom: list[str] = []
    undeclared: list[str] = []
    for vla in vlas:
        try:
            combined = assert_vla_reward_fits(vla, reward, gpu_total_gb)
        except ROSGPUMemoryError as exc:
            oom.append(f"{vla.name}: {exc}")
        except ROSConfigError:
            # min_vram_gb undeclared for the active dtype — the pair cannot be
            # verified, so the reasoner will refuse this VLA at dispatch too.
            undeclared.append(vla.name)
        else:
            fits.append(f"{vla.name} ({combined:.2f} GB)")

    if not fits:
        _console.print()
        _console.print(
            "[red]preflight failed:[/red] no capability-matched VLA can co-reside with "
            f"the reward model {reward.name!r} on this GPU "
            f"({gpu_total_gb:.2f} GB total) — every paired policy would be refused at "
            "dispatch, so the deploy could actuate nothing (ADR-0077)."
        )
        for line in oom:
            _console.print(f"  • too large: {line}")
        if undeclared:
            _console.print(
                "  • undeclared min_vram_gb (cannot verify the required co-residency): "
                f"{undeclared!r}"
            )
        _console.print(
            "  Remedies: use a smaller-footprint VLA/reward pair or a larger GPU; "
            "declare min_vram_gb on the VLA manifest(s); or run with "
            "--no-enable-reward-monitor (accepting the VLA runs without a live reward "
            "signal)."
        )
        raise typer.Exit(code=1)

    if oom:
        _console.print(
            f"[yellow]preflight:[/yellow] {len(oom)} VLA(s) cannot fit beside the reward "
            f"model {reward.name!r} on {gpu_total_gb:.2f} GB and will be refused at "
            "dispatch (ADR-0077):"
        )
        for line in oom:
            _console.print(f"  • {line}")
    if undeclared:
        _console.print(
            f"[yellow]preflight:[/yellow] {len(undeclared)} VLA(s) do not declare "
            "min_vram_gb for their active dtype, so the reward pairing cannot be "
            f"verified and the reasoner will refuse them while a reward model is active "
            f"(ADR-0077): {undeclared!r}"
        )
    _console.print(
        f"[green]preflight:[/green] {len(fits)} VLA(s) fit beside reward "
        f"{reward.name!r} on {gpu_total_gb:.2f} GB: {fits!r}"
    )


def resolve_launch_invocation(  # noqa: PLR0912, PLR0915  # reason: a flat resolve sequence (robot_id → manifest → per-feature slam/nav2/octomap + sim/real hal_mode gating); splitting hurts readability
    *,
    config: Path | None = None,
    robot_override: str | None,
    dashboard_port: int,
    reset_to_pose_service: str | None,
    approach_skill_id: str | None = None,
    dataset_out: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_license: str | None = None,
    hal_param_overrides: dict[str, object] | None = None,
    hal_mode: str = "sim",
    enable_slam: bool | None = None,
    enable_nav2: bool | None = None,
    enable_octomap: bool | None = None,
    enable_octomap_kernel_check: bool = True,
    enable_object_detector: bool | None = None,
    object_detector_onnx: Path | None = None,
    object_detector_manifest: str | None = None,
    object_detector_query: str | None = None,
    enable_reward_monitor: bool = False,
    reward_monitor_manifest: str | None = None,
    reward_monitor_task: str | None = None,
    enable_critic: bool = False,
    object_detector_locators: list[str] | None = None,
    spatial_memory_ingest: bool | None = None,
    memory_dir: str | None = None,
    enable_dashboard: bool = True,
    enable_foxglove: bool = False,
    foxglove_port: int = 8765,
    initial_task_prompt: str | None = None,
) -> LaunchInvocation:
    """Resolve every input into the ``ros2 launch`` argv to execute.

    Shared by ``openral deploy sim`` (``hal_mode="sim"``, a ``DeployScene``
    ``config``) and ``openral deploy run`` (``hal_mode="real"``, ``robot_override``
    from a ``RobotEnvironment`` — no sim scene; ADR-0032). In real mode the
    sim-twin / scene-attach injections are skipped so the HAL node builds the
    real hardware HAL via ``build_hal(mode="real")``.

    Returned ``argv_template`` carries a ``HAL_PARAMS_FILE_PLACEHOLDER``
    sentinel the caller substitutes after writing the ephemeral HAL params
    YAML. No envelope file is ever written — the launch reads ``robot_yaml``
    and feeds the kernel via ROS params.
    """
    from openral_core import RobotDescription  # reason: defer schema import

    if hal_mode not in ("sim", "real"):
        raise ROSConfigError(f"hal_mode must be 'sim' or 'real', got {hal_mode!r}.")

    scene_robot_id = _load_scene_robot_id(config) if config is not None else None
    robot_id = robot_override or scene_robot_id
    if not robot_id:
        raise ROSConfigError(
            "robot_id is undefined: pass ``--robot <id>`` or (for ``deploy sim``) "
            "set ``robot_id:`` in the DeployScene YAML / use a fixed-robot scene."
        )

    # The deploy startup prompt comes ONLY from the operator (--initial-task /
    # a live /openral/prompt). Deploy never reads sim-predefined scene tasks —
    # that is `sim run`'s job (ADR-0073 amendment / deploy ≠ benchmark).
    _resolved_initial_prompt: str = initial_task_prompt or ""

    # ADR-0034 — a --robot override that differs from the scene's declared robot
    # composes a different arm than the scene was authored for. The scene's cameras
    # + asset mounts (e.g. tabletop_push's wrist_camera_mount_body="gripper") are
    # tuned for the declared robot, so on the override they may be mis-mounted or
    # unmatched — some /openral/cameras/* topics will be empty. Warn loudly.
    if robot_override and scene_robot_id and robot_override != scene_robot_id:
        _console.print(
            f"[yellow]warning:[/yellow] --robot {robot_override!r} overrides the scene's "
            f"declared robot_id {scene_robot_id!r}; the scene's cameras + asset mounts are "
            f"authored for {scene_robot_id!r} and may not match {robot_override!r} (expect "
            "empty /openral/cameras/* for non-matching sensors). Override only with a "
            "kinematically-compatible arm on a free-axis scene."
        )

    hal = _ROBOT_HAL_REGISTRY.get(robot_id)
    if hal is None:
        supported = ", ".join(sorted(_ROBOT_HAL_REGISTRY))
        raise ROSConfigError(
            f"robot {robot_id!r} has no HAL entry in _ROBOT_HAL_REGISTRY. "
            f"Supported: {supported}. The registry key matches the "
            "``robots/<robot_id>/`` directory name — e.g. ``--robot "
            "franka_panda`` (not ``--robot franka``). To add a new robot, "
            "ship a ROS ``openral_hal_<X>`` package with a "
            "``lifecycle_node.py`` executable, add it to the ros2-build "
            "Justfile target, and register it here."
        )

    repo_root = _repo_root_from(Path(__file__))
    robot_yaml = repo_root / "robots" / robot_id / "robot.yaml"
    if not robot_yaml.is_file():
        raise ROSConfigError(
            f"robot manifest not found at {robot_yaml}; expected robots/{robot_id}/robot.yaml."
        )

    # Validate the manifest carries the e2e fields (joint position /
    # velocity / effort limits) and assert the manifest's declared name
    # is one the registered HAL accepts — protects against a typo
    # routing the wrong HAL at a robot.
    description = RobotDescription.from_yaml(str(robot_yaml))
    description.validate_for_e2e_pipeline()

    # ADR-0032 — fail fast before shelling the launch if real mode is asked of
    # a simulation-only robot (better UX than a graph that dies at HAL
    # configure with the same ROSCapabilityMismatch).
    if hal_mode == "real" and description.hal.real is None:
        raise ROSCapabilityMismatch(
            f"robot {robot_id!r} has no real-hardware HAL (hal.real is null); it is "
            "simulation-only. Use `openral deploy sim` instead of `openral deploy run`."
        )

    # ADR-0025 (#11) — SLAM is ON BY DEFAULT for every robot that *can* run it:
    # i.e. one that declares a lidar (the scan source slam_toolbox needs). This
    # is the firm default — a SLAM-capable robot always brings up the `map` frame
    # the object lift / spatial-memory ingest depend on, unless the operator
    # opts out with `--no-enable-slam`. Fixed-base arms (no mobile base, no lidar)
    # correctly stay off — there is no base to localise and nothing to map.
    # `enable_slam is None` means "auto": honour the manifest; an explicit flag wins.
    # ADR-0064 — SLAM is on for any robot that can localise/map: a lidar
    # (slam_toolbox) OR camera-based visual SLAM (cuVSLAM+nvblox, for
    # lidar-less robots). Fixed-base arms with neither correctly stay off.
    if enable_slam is None:
        enable_slam = bool(
            description.capabilities.has_lidar or description.capabilities.has_vision_slam
        )
    # ADR-0064 — backend selection (pure helper, unit-tested directly).
    slam_backend = _resolve_slam_backend(
        has_lidar=bool(description.capabilities.has_lidar),
        has_vision_slam=bool(description.capabilities.has_vision_slam),
        enable_slam=enable_slam,
    )
    # ADR-0025 — Nav2 auto-enables alongside slam_toolbox: every
    # lidar-equipped mobile robot needs a planner to consume the map.
    # Operators that want the map alone (recording / inspection) pass
    # ``--no-enable-nav2``.
    if enable_nav2 is None:
        enable_nav2 = enable_slam
    # ADR-0030 — the octomap world-collision leg auto-enables when the
    # robot manifest declares a usable depth SensorSpec (a camera the HAL
    # can ray-cast a PointCloud2 from); there is nothing to map otherwise.
    # ``--enable-octomap`` / ``--no-enable-octomap`` overrides.
    if enable_octomap is None:
        enable_octomap = any(
            s.modality in ("depth", "point_cloud") and s.intrinsics is not None
            for s in description.sensors
        )
    clock_origin = _resolve_clock_origin(hal_mode=hal_mode, config=config)

    # ADR-0035/0037 — the object-detection leg is ON by default (deploy sim is a
    # perception-driven stack; ``--no-object-detector`` turns it off). The default
    # backend is the open-vocabulary ``omdet-turbo-indoor`` continuous detector,
    # which grounds arbitrary indoor/kitchen objects instead of the fixed COCO-80
    # of RT-DETR; when its runtime deps (transformers/timm — the ``omdet`` group)
    # are not importable we gracefully fall back to the in-tree RT-DETR COCO ONNX
    # so the graph still comes up. Explicit ``--object-detector-manifest`` /
    # ``--object-detector-onnx`` override the default selection.
    default_rtdetr_onnx = repo_root / "rskills" / "rtdetr-coco-r18" / "model.onnx"
    default_omdet_manifest = repo_root / "rskills" / "omdet-turbo-indoor" / "rskill.yaml"
    if object_detector_manifest:
        resolved_object_detector_manifest = str(Path(object_detector_manifest).resolve())
        resolved_object_detector_onnx = (
            object_detector_onnx.resolve()
            if object_detector_onnx is not None
            else default_rtdetr_onnx
        )
    elif object_detector_onnx is not None:
        # Explicit ONNX → the legacy fixed-label RT-DETR path (no manifest).
        resolved_object_detector_manifest = ""
        resolved_object_detector_onnx = object_detector_onnx.resolve()
    elif _omdet_runtime_available():
        resolved_object_detector_manifest = str(default_omdet_manifest.resolve())
        resolved_object_detector_onnx = default_rtdetr_onnx
    else:
        # Graceful fallback: omdet deps absent → in-tree RT-DETR COCO-80 ONNX.
        resolved_object_detector_manifest = ""
        resolved_object_detector_onnx = default_rtdetr_onnx

    if enable_object_detector is None:
        enable_object_detector = True
    # Downgrade to off (rather than let the node hard-fail at backend build) when
    # the detector is requested but no usable backend is present — a checkout that
    # has neither the omdet deps nor the gitignored RT-DETR ONNX weights.
    if (
        enable_object_detector
        and not resolved_object_detector_manifest
        and not _object_detector_onnx_present(resolved_object_detector_onnx)
    ):
        _console.print(
            "[yellow]object detector requested but no backend is available[/yellow] "
            "(omdet deps not importable and RT-DETR ONNX missing at "
            f"{resolved_object_detector_onnx}); disabling the detector leg. Run "
            "`just sync --group omdet --inexact` for the open-vocab default, fetch "
            "the RT-DETR weights, or pass --no-object-detector to silence."
        )
        enable_object_detector = False

    # When the leg is off (explicit --no-object-detector or the downgrade above),
    # do not forward a continuous-detector manifest: ros2 launch would otherwise
    # carry a manifest for a node that never starts (and, with omdet deps present,
    # a non-empty manifest:= for a disabled leg is simply wrong).
    if not enable_object_detector:
        resolved_object_detector_manifest = ""

    # ADR-0056 — on-demand open-vocab locators co-resident alongside the
    # continuous detector. Each token is a manifest path (``…/rskill.yaml``) or a
    # short alias resolved to ``rskills/<alias>/rskill.yaml``; the launch builds one
    # namespaced locate_in_view node per entry so the reasoner can pick a model.
    # Default = omdet-turbo-locator when the detector is on and the omdet deps are
    # importable (LocateAnything is opt-in — NVIDIA non-commercial, 5 GB VRAM).
    resolved_object_detector_locators: list[str] = []
    if enable_object_detector:
        if object_detector_locators is not None:
            locator_tokens = object_detector_locators
        elif _omdet_runtime_available():
            locator_tokens = ["omdet-turbo-locator"]
        else:
            locator_tokens = []
        for token in locator_tokens:
            candidate = Path(token)
            manifest_path = (
                candidate
                if candidate.suffix == ".yaml"
                else repo_root / "rskills" / token / "rskill.yaml"
            )
            resolved_object_detector_locators.append(str(manifest_path.resolve()))

    # ADR-0038 — auto-enable durable spatial-memory ingest whenever the object
    # detector runs (the producer that feeds it); an explicit flag overrides.
    if spatial_memory_ingest is None:
        spatial_memory_ingest = enable_object_detector

    if description.name not in hal.supported_robot_names:
        raise ROSConfigError(
            f"HAL/robot mismatch: robot_id={robot_id!r} (registry → "
            f"{hal.package!r}) declares supported_robot_names="
            f"{sorted(hal.supported_robot_names)}, but "
            f"{robot_yaml} has name={description.name!r}. Either fix the "
            "manifest's `name:` field or register the right HAL for this "
            "robot in openral_cli.deploy_sim._ROBOT_HAL_REGISTRY."
        )

    hal_params: dict[str, object] = {**hal.default_params}
    if hal_param_overrides:
        hal_params.update(hal_param_overrides)

    # ADR-0025 — derive the /scan envelope from robot.yaml's lidar_2d
    # sensor (single source of truth) instead of hardcoding it in the
    # HAL registry. ``setdefault`` so an explicit ``--hal scan_*=…``
    # operator override still wins.
    for _scan_key, _scan_value in _scan_params_from_description(description).items():
        hal_params.setdefault(_scan_key, _scan_value)

    # ADR-0025 Stage 3 — forward the sim config to HALs that declare
    # `sim_env_yaml` support. Gated on the per-HAL opt-in flag because
    # rclpy rejects unknown parameters at startup
    # (`automatically_declare_parameters_from_overrides=False` is the
    # default), so blanket-forwarding would break every other HAL.
    if hal.supports_sim_env_yaml and hal_mode == "sim" and config is not None:
        hal_params.setdefault("sim_env_yaml", str(config.resolve()))

    # ADR-0032 — manifest-driven nodes build their HAL via build_hal(mode).
    # `deploy sim` → hal_mode="sim"; `deploy run` → hal_mode="real". The node
    # raises ROSCapabilityMismatch for a sim-only-vs-real mismatch.
    if hal.manifest_driven:
        hal_params.setdefault("robot_yaml", str(robot_yaml))
        hal_params.setdefault("hal_mode", hal_mode)
        # ADR-0034 — deploy sim is inherently a scene; forward the resolved
        # config so the manifest-driven node scene-attaches (SimAttachedHAL)
        # instead of building a bare twin. Sim mode only; real never attaches.
        # `bare_twin_sim` arms (so100 / so101) opt out: they build a bare
        # `MujocoArmHAL` twin from their `sim:` block (issue #191 Phase 2),
        # preserving the pre-migration `supports_sim_robot_yaml` behaviour.
        if hal_mode == "sim" and config is not None and not hal.bare_twin_sim:
            hal_params.setdefault("sim_env_yaml", str(config.resolve()))
        # ADR-0066 — forward the DeployScene's own MJCF composition (its arena)
        # to the manifest-driven node so the SCENE owns its environment instead
        # of the robot manifest. Sim-mode bare-twin robots only (scene-attach
        # robots build the scene's SimRollout directly via sim_env_yaml).
        if hal_mode == "sim" and config is not None and hal.bare_twin_sim:
            from openral_core import DeployScene

            scene_composition = DeployScene.from_yaml(str(config)).composition
            if scene_composition is not None:
                hal_params.setdefault("scene_composition_json", scene_composition.model_dump_json())

    service = reset_to_pose_service or f"/openral/{robot_id}/reset_to_pose"
    # Empty by default — the legacy ResetToPose snap stays until a move_group is
    # wired into the graph (ADR-0053 phase 4); opt in with --approach-skill-id.
    approach_skill = approach_skill_id or ""

    argv_template: list[str] = [
        "ros2",
        "launch",
        "openral_rskill_ros",
        "sim_e2e.launch.py",
        f"robot_yaml:={robot_yaml}",
        f"hal_package:={hal.package}",
        f"hal_executable:={hal.executable}",
        f"hal_node_name:={hal.node_name}",
        "hal_params_file:=HAL_PARAMS_FILE_PLACEHOLDER",
        f"reset_to_pose_service:={service}",
        f"dashboard_port:={dashboard_port}",
        # ADR-0036 — forward the deploy path so the reasoner's action-mode
        # palette gate matches the HAL this graph brings up. ``deploy sim``
        # → ``hal_mode="sim"`` (default; the scene's robosuite OSC controller
        # synthesises cartesian/OSC modes); ``deploy run`` → ``"real"``.
        f"hal_mode:={hal_mode}",
        f"enable_slam:={'true' if enable_slam else 'false'}",
        f"slam_backend:={slam_backend}",
        f"enable_nav2:={'true' if enable_nav2 else 'false'}",
        f"enable_octomap:={'true' if enable_octomap else 'false'}",
        f"enable_octomap_kernel_check:={'true' if enable_octomap_kernel_check else 'false'}",
        # ADR-0048 — OpenRAL clock authority. The launch maps this to ROS
        # use_sim_time internally: simulation → use_sim_time=true + HAL /clock;
        # host_wall → system time and no OpenRAL /clock publisher.
        f"clock_origin:={clock_origin}",
        f"enable_object_detector:={'true' if enable_object_detector else 'false'}",
        f"object_detector_onnx:={resolved_object_detector_onnx}",
        # ADR-0057 — reward monitor co-active with the VLA; the reasoner polls
        # /openral/perception/query_task_progress when task_progress_available.
        f"enable_reward_monitor:={'true' if enable_reward_monitor else 'false'}",
        # ADR-0064 — Tier-C critic producer; emits FailureTrigger on
        # /openral/failure/critic when a reward model's score stalls.
        f"enable_critic:={'true' if enable_critic else 'false'}",
        f"spatial_memory_ingest:={'true' if spatial_memory_ingest else 'false'}",
        f"enable_dashboard:={'true' if enable_dashboard else 'false'}",
        # ADR-0059 — read-only Foxglove live-scene bridge. Off by default;
        # ``--foxglove`` opts in. The bridge starts after the topic producers
        # (HAL, SLAM, octomap, robot_state_publisher) via a TimerAction in the
        # launch to avoid the foxglove-sdk-cpp v0.18.0 stale-bridge bug.
        f"enable_foxglove:={'true' if enable_foxglove else 'false'}",
        f"foxglove_port:={foxglove_port}",
    ]
    # ``ros2 launch`` rejects an empty ``name:=`` argument, so only forward the
    # optional detector overrides when set; the launch file defaults both to "".
    if resolved_object_detector_manifest:
        argv_template.append(f"object_detector_manifest:={resolved_object_detector_manifest}")
    if object_detector_query:
        argv_template.append(f"object_detector_query:={object_detector_query}")
    # ADR-0077 §4 — resolve the reward-monitor manifest from the VLA pairing when
    # the operator did not pin one. ``deploy sim`` does not preselect a VLA, so the
    # default is derived from the capability-matched VLA palette's
    # ``reward_rskill_name`` (the pairing the reasoner will honour) instead of an
    # independent flag. Only when the reward monitor is active; otherwise empty.
    resolved_reward_monitor_manifest = (
        _resolve_reward_monitor_manifest(
            repo_root=repo_root,
            description=description,
            explicit_manifest=reward_monitor_manifest,
        )
        if enable_reward_monitor
        else ""
    )
    # Forward the resolved reward manifest (ros2 launch rejects an empty ``name:=``
    # value; the launch file defaults it to "" and the reasoner/monitor fall back
    # to the robometer default when unset).
    if resolved_reward_monitor_manifest:
        argv_template.append(f"reward_monitor_manifest:={resolved_reward_monitor_manifest}")
    # The reward monitor's always-on critic_score path scores against its
    # `task` param; an empty task makes `_publish_critic_score` silently skip
    # every tick (it never scores, never spawns the robometer sidecar). Default
    # it to the operator goal so a deploy with `--initial-task` gets a
    # background progress signal out of the box (an explicit
    # `--reward-monitor-task` still wins; the reasoner's `query_task_progress`
    # polls already carry the live subtask when it dispatches with a deadline).
    effective_reward_task = reward_monitor_task
    if not effective_reward_task and _resolved_initial_prompt:
        effective_reward_task = _resolved_initial_prompt.strip()
    if effective_reward_task:
        argv_template.append(f"reward_monitor_task:={effective_reward_task}")
    # ADR-0056 — only forward the locator list when non-empty (ros2 launch rejects
    # an empty ``name:=`` value; the launch file defaults it to "").
    if resolved_object_detector_locators:
        argv_template.append(
            "object_detector_locators:=" + ",".join(resolved_object_detector_locators)
        )
    # ADR-0053 — only forward the approach skill when opted in (empty default;
    # ros2 launch rejects an empty ``name:=`` value, and the launch file
    # defaults ``approach_skill_id`` to "").
    if approach_skill:
        argv_template.append(f"approach_skill_id:={approach_skill}")

    # ADR-0019 — only forward the dataset args when recording is opted in
    # (empty defaults; ros2 launch rejects an empty ``name:=`` value, and the
    # launch file defaults all three so omitting them disables recording).
    if dataset_out:
        argv_template.append(f"dataset_out:={dataset_out}")
        if dataset_repo_id:
            argv_template.append(f"dataset_repo_id:={dataset_repo_id}")
        if dataset_license:
            argv_template.append(f"dataset_license:={dataset_license}")

    # ADR-0072 Decision 3b — the deploy memory bundle. ``--memory-dir`` (CLI) wins;
    # otherwise the DeployScene's own ``memory_dir`` field. Derive the per-modality
    # launch paths by convention and forward them (each to its consumer's arg).
    effective_memory_dir = memory_dir
    if effective_memory_dir is None and config is not None:
        from openral_core import DeployScene

        effective_memory_dir = DeployScene.from_yaml(str(config)).memory_dir
    if effective_memory_dir:
        argv_template.extend(_memory_bundle_launch_args(effective_memory_dir))

    # Forward the startup prompt only when non-empty (ADR-0073). The launch
    # file defaults ``initial_task_prompt`` to "" (no prompt), so omitting it
    # leaves the reasoner in idle mode until an operator prompt arrives.
    if _resolved_initial_prompt:
        argv_template.append(f"initial_task_prompt:={_resolved_initial_prompt}")

    return LaunchInvocation(
        robot_id=robot_id,
        robot_yaml=robot_yaml,
        robot_manifest_name=description.name,
        hal=hal,
        enable_slam=enable_slam,
        slam_backend=slam_backend,
        enable_nav2=enable_nav2,
        enable_octomap=enable_octomap,
        clock_origin=clock_origin,
        enable_object_detector=enable_object_detector,
        object_detector_onnx=resolved_object_detector_onnx,
        object_detector_manifest=resolved_object_detector_manifest,
        object_detector_query=object_detector_query or "",
        object_detector_locators=tuple(resolved_object_detector_locators),
        spatial_memory_ingest=spatial_memory_ingest,
        hal_params=hal_params,
        hal_mode=hal_mode,
        reset_to_pose_service=service,
        approach_skill_id=approach_skill,
        enable_foxglove=enable_foxglove,
        foxglove_port=foxglove_port,
        initial_task_prompt=_resolved_initial_prompt,
        enable_reward_monitor=enable_reward_monitor,
        reward_monitor_manifest=resolved_reward_monitor_manifest,
        argv_template=argv_template,
    )


def _prepare_launch_env() -> dict[str, str]:
    """Build the environment for the ``ros2 launch`` subprocess (deploy sim + deploy run).

    Shared by both shelling paths so the wiring stays identical:

    * Export ``OPENRAL_VENV_SITE`` + prepend the venv site / bin so the launch
      parser and every spawned node import ``openral_core`` from the workspace
      venv (the editable ``.pth`` files are processed via ``site.py``).
    * ADR-0034 — default the **expandable-segments CUDA allocator**. The
      ``runtime_node`` loads VLA weights (pi05 / molmoact2 …) onto the GPU; on a
      tight 8 GiB card the default allocator fragments and OOMs at the forward
      pass even for an NF4 model that otherwise fits (molmoact2-libero-nf4 peaks
      ~7.6 GiB; without this it dies on a small alloc with ~165 MiB stuck in
      reserved-but-unallocated). ``setdefault`` so an operator override wins;
      both env-var spellings are set because the name was renamed across torch
      releases (``PYTORCH_CUDA_ALLOC_CONF`` → ``PYTORCH_ALLOC_CONF``).
    * Clean stale Fast-DDS SHM (``_apply_rmw_default``).
    """
    env = os.environ.copy()
    venv_site = sysconfig.get_paths()["purelib"]
    env["OPENRAL_VENV_SITE"] = venv_site
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{venv_site}{os.pathsep}{existing}" if existing else venv_site
    venv_bin = os.path.dirname(sys.executable)
    existing_path = env.get("PATH", "")
    env["PATH"] = f"{venv_bin}{os.pathsep}{existing_path}" if existing_path else venv_bin
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _apply_rmw_default(env)
    return env


def run_launch_invocation(invocation: LaunchInvocation, *, run_preflight: bool = True) -> int:
    """Write the HAL params YAML, set the venv env, and shell ``ros2 launch``.

    Returns the launch exit code. The shared shelling path used by both
    ``openral deploy sim`` and ``openral deploy run``
    (ADR-0032). Exports ``OPENRAL_VENV_SITE`` + prepends the venv bin/PATH so
    the launch parser and spawned nodes import ``openral_core`` from the
    workspace venv (the editable ``.pth`` files are processed via ``site.py``).
    ``run_preflight`` probes the rSkill palette extras.
    """
    if run_preflight:
        repo_root = _repo_root_from(Path(__file__))
        _preflight_palette_deps(
            repo_root=repo_root,
            robot_yaml=Path(invocation.robot_yaml),
        )
        # ADR-0077 §4 — VLA↔reward VRAM pair preflight (deploy run path). No-op
        # unless a reward monitor is active and the GPU budget is readable.
        if invocation.enable_reward_monitor and invocation.reward_monitor_manifest:
            from openral_core import RobotDescription

            _preflight_reward_vram_fit(
                repo_root=repo_root,
                description=RobotDescription.from_yaml(str(invocation.robot_yaml)),
                reward_manifest_path=invocation.reward_monitor_manifest,
                gpu_total_gb=_detect_gpu_total_vram_gb(),
            )
    hal_params_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115  # reason: HAL reads after this scope
        mode="w",
        prefix=f"openral-hal-params-{invocation.robot_id}-",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    try:
        yaml.safe_dump(
            {"/**": {"ros__parameters": invocation.hal_params}},
            hal_params_tmp,
            sort_keys=False,
        )
        hal_params_tmp.close()
        argv = [
            arg.replace("HAL_PARAMS_FILE_PLACEHOLDER", hal_params_tmp.name)
            for arg in invocation.argv_template
        ]
        _console.print(f"  hal_params_tmp:{hal_params_tmp.name}")
        _console.print(f"  argv: {shlex.join(argv)}")
        venv_env = _prepare_launch_env()
        return _run_launch(argv, venv_env)
    finally:
        with contextlib.suppress(OSError):
            Path(hal_params_tmp.name).unlink(missing_ok=True)


def _parse_hal_overrides(raw: list[str] | None) -> dict[str, object]:
    """Parse ``--hal key=value`` flags into a typed override dict.

    Accepts bool / int / float / string. JSON-encoded values are tried
    first so ``--hal cameras='["top"]'`` works.
    """
    out: dict[str, object] = {}
    for entry in raw or []:
        if "=" not in entry:
            raise ROSConfigError(f"--hal {entry!r} is malformed; expected ``key=value``.")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _ros2_pkg_prefix(pkg: str) -> str | None:
    """Return the install prefix for ``pkg`` per ``ros2 pkg prefix``, or None.

    Uses ``ros2 pkg prefix`` because it is the same lookup ``ros2 launch``
    performs internally. A package is "discoverable" iff this exits 0
    and prints a path.
    """
    try:
        completed = subprocess.run(
            ["ros2", "pkg", "prefix", pkg],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    out = completed.stdout.strip()
    return out or None


def _reap_orphans_with_log() -> None:
    """Kill orphan openral-graph processes and log the count.

    Reap orphan graph processes from a prior crashed/Ctrl-C'd run.
    Without this, a stale HAL still holds the robocasa MJCF env +
    ``/dev/shm/fastrtps_*`` lockfiles, and the new launch surfaces
    as ``[RTPS_TRANSPORT_SHM Error] Failed init_port
    fastrtps_port7000`` on the safety_kernel, slam_toolbox,
    prompt_router, etc.
    """
    killed = _kill_orphan_openral_graph_processes()
    if killed:
        _console.print(
            f"[yellow]reaped {killed} orphan openral-graph process(es) from a prior run[/yellow]"
        )


# Argv substrings that identify a process as belonging to the openral
# deploy graph. Each entry must be specific enough that a coincidental
# unrelated invocation never matches — the reaper additionally scopes to
# the calling user's PIDs (the ``st_uid`` guard) so a shared host is safe.
_ORPHAN_GRAPH_NEEDLES: tuple[str, ...] = (
    "sim_e2e.launch.py",
    "openral_rskill_ros/runtime_node",
    "install/lib/openral_hal_",
    "openral_reasoner_ros/reasoner_node.py",
    "openral_prompt_router/prompt_router_node.py",
    "openral_safety_kernel/safety_kernel_node",
    "openral dashboard",
    "async_slam_toolbox_node",
    # Nav2 sub-nodes spawned by openral_nav2_bringup's
    # IncludeLaunchDescription. The upstream binaries live under
    # ``/opt/ros/<distro>/lib/nav2_*``; matching the path keeps the
    # predicate specific (won't catch unrelated ``nav2_*`` python
    # imports). Without these, a previously-crashed Nav2 graph leaves
    # zombie controller_server / planner_server / collision_monitor /
    # bt_navigator / opennav_docking processes alive, and the next
    # ``openral deploy sim`` hangs in ``Configuring controller_server``
    # while multiple lifecycle_managers fight over the same nodes.
    "/lib/nav2_controller/",
    "/lib/nav2_smoother/",
    "/lib/nav2_planner/",
    "/lib/nav2_route/",
    "/lib/nav2_behaviors/",
    "/lib/nav2_bt_navigator/",
    "/lib/nav2_waypoint_follower/",
    "/lib/nav2_velocity_smoother/",
    "/lib/nav2_collision_monitor/",
    "/lib/opennav_docking/",
    "/lib/nav2_lifecycle_manager/",
    # ADR-0027 TF chain spawned by ``sim_e2e.launch.py``. These were the
    # silent gap that caused the rldx-rc365 "arm reaches 40 cm high" bug:
    # a ``static_transform_publisher`` orphaned from a run *before* the
    # URDF mount-z was zeroed kept publishing the stale ``base_link →
    # panda_link0 z=0.4`` on the TRANSIENT_LOCAL ``/tf_static`` topic, and
    # the next launch's correct ``z=0.0`` publisher couldn't override it
    # (tf2 picks non-deterministically among same-name static frames).
    # Reaping the renamed static publisher (``static_<base>_to_<root>``)
    # and the URDF ``robot_state_publisher`` closes that hole. Scoped to
    # the ``tf2_ros`` / ``robot_state_publisher`` executables under the
    # calling user so we never touch an unrelated TF graph.
    "/lib/tf2_ros/static_transform_publisher",
    "/lib/robot_state_publisher/robot_state_publisher",
    # rldx out-of-process sidecar (ADR auto-spawn). It runs in its OWN
    # session (``start_new_session=True`` in ``openral_sim.policies.rldx``)
    # so the launch group's SIGINT never reaches it; if the runtime node
    # dies before its adapter ``close()`` runs, the sidecar keeps the
    # GR00T/RLDX weights resident and starves the GPU (~6.5 GiB) of the
    # next run. The cache dir is openral-specific, so this is unambiguous.
    "/.cache/openral/rldx-sidecar/",
    # Robometer reward sidecar (ADR-0057). Same out-of-process pattern as
    # rldx: ``reward_monitor_node`` spawns it in its own session, so killpg on
    # the launch group never reaches it, and it forks one torch-inductor
    # ``compile_worker`` per CPU. The venv path appears in the server's AND
    # every compile_worker's cmdline, so this single needle reaps the whole
    # sidecar tree (~3.3 GiB GPU) if the graceful ``close()`` doesn't run.
    "/.cache/openral/robometer-sidecar/",
    # Perception / critic graph nodes spawned by ``sim_e2e.launch.py``. These
    # were absent from the sweep, so under a heavy graph whose graceful
    # shutdown doesn't finish within ``grace_s`` they orphaned (the reward
    # monitor holds the sidecar; the detector holds its model). Scoped to the
    # in-tree node entry points so we never touch an unrelated process.
    "openral_perception_ros/reward_monitor_node.py",
    "openral_perception_ros/ros_image_detector_node.py",
    "openral_reasoner_ros/critic_producer_node.py",
)


def _cmdline_is_openral_graph_process(cmdline: str) -> bool:
    """Return True when ``cmdline`` matches an openral deploy-graph process.

    Pure predicate over a space-joined ``/proc/<pid>/cmdline`` string so
    the needle set (:data:`_ORPHAN_GRAPH_NEEDLES`) is unit-testable
    without spawning real processes.
    """
    return any(needle in cmdline for needle in _ORPHAN_GRAPH_NEEDLES)


def _kill_orphan_openral_graph_processes() -> int:
    """SIGKILL orphaned openral-graph processes from a prior ``openral deploy sim``.

    A graceful ``Ctrl-C`` doesn't always propagate through ``ros2
    launch`` to every child — under load, the launch dispatcher
    exits before the HAL has finished tearing down its in-process
    robocasa env, and the HAL python process keeps running. On the
    next ``openral deploy sim``:

    * the prior dashboard still holds ``127.0.0.1:4318`` →
      ``[Errno 98] address already in use`` on the new dashboard;
    * the prior HAL still holds ``/dev/shm/fastrtps_*`` lockfiles →
      ``[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7000``
      on the new safety_kernel/slam_toolbox/prompt_router;
    * the prior HAL still holds the MJCF env handle → the new HAL's
      configure step hangs waiting for the robocasa loader.

    Match orphans by argv signature so we never reach into other
    users' processes or unrelated python invocations. Best-effort:
    SIGKILL failures (permission denied, race) are silently
    skipped. Returns the count of processes killed for logging.
    """
    me = os.getuid()
    killed = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            st = entry.stat()
        except (FileNotFoundError, PermissionError):
            continue
        if st.st_uid != me:
            continue  # other users' processes are not ours to kill
        try:
            cmdline = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(
                    errors="replace",
                )
            )
        except (FileNotFoundError, PermissionError):
            continue
        if not _cmdline_is_openral_graph_process(cmdline):
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)
            killed += 1
    return killed


def _terminate_launch_group(proc: subprocess.Popen[bytes], *, grace_s: float = 12.0) -> None:
    """SIGINT the launch's session, then SIGKILL any straggler after a grace.

    ``proc`` was spawned with ``start_new_session=True`` so its PGID
    equals its PID and names the whole launch tree. SIGINT (not SIGTERM)
    is what ``ros2 launch`` translates into a graceful lifecycle
    shutdown; we give it ``grace_s`` to drain, then SIGKILL the group so
    no node — or the launch-spawned static_transform_publisher /
    robot_state_publisher — survives to orphan and poison the next run.
    """
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(proc.pid, signal.SIGINT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)


def _run_launch(argv: list[str], env: dict[str, str], *, grace_s: float = 12.0) -> int:
    """Run ``ros2 launch`` in its own session, reaping the whole tree on exit.

    The legacy ``subprocess.run(argv)`` left the launch a sibling in the
    CLI's process group: a ``kill -INT`` of this CLI (or any non-terminal
    exit) signalled only the CLI, and ``ros2 launch`` plus every node it
    spawned — the HAL (holding the MuJoCo render context, ~1.2 GiB GPU),
    robot_state_publisher, the ``static_<base>_to_<root>`` static TF
    publisher — were reparented to init and kept running. Those orphans
    accumulated across dev iterations and poisoned the TRANSIENT_LOCAL
    ``/tf_static`` topic (see ``_kill_orphan_openral_graph_processes``).

    Teardown runs in three escalating stages so nothing survives,
    whatever the process-group topology:

    1. Forward SIGINT/SIGTERM to the launch's session so ``ros2 launch``
       runs its graceful shutdown (each node's ``on_shutdown`` fires; the
       HAL releases its MJCF env; the skill adapter's ``close()``
       terminates the rldx sidecar).
    2. After ``grace_s`` escalate to SIGKILL on the launch's process
       group (``_terminate_launch_group``).
    3. Sweep by argv signature (``_kill_orphan_openral_graph_processes``).
       This is the bulletproof backstop: ``ros2 launch`` spawns its nodes
       in their OWN process groups (and the rldx sidecar in its own
       session), so neither ``killpg`` reaches them directly — under a
       heavy graph the graceful shutdown often doesn't finish before the
       launch exits, and stage 1+2 alone leak nodes. The signature sweep
       matches every graph process regardless of parentage.

    Returns the launch process's exit code (0 if it exited via signal
    with no recorded returncode).
    """
    proc = subprocess.Popen(argv, env=env, start_new_session=True)

    def _forward(_signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGINT)

    prev_int = signal.signal(signal.SIGINT, _forward)
    prev_term = signal.signal(signal.SIGTERM, _forward)
    try:
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        _terminate_launch_group(proc, grace_s=grace_s)
        # ``ros2 launch`` spawns its nodes in their OWN process groups, so
        # ``killpg`` on the launch's group only reaps them via ros2
        # launch's *graceful* shutdown — which, under a heavy graph
        # (nav2 + slam + sidecar), routinely does not finish before the
        # launch process itself exits, leaving the nodes (and the
        # own-session rldx sidecar) orphaned. Sweep by argv signature as
        # the bulletproof backstop: it matches every graph process
        # regardless of parentage or process group. Safe because two
        # deploy graphs can't coexist (they collide on DDS/ports), so on
        # exit the only matching processes are this run's survivors.
        _kill_orphan_openral_graph_processes()
    return proc.returncode if proc.returncode is not None else 0


def _apply_rmw_default(env: dict[str, str]) -> None:
    """Clean stale Fast-DDS SHM lockfiles before spawning ``ros2 launch``.

    ROS 2 Jazzy defaults to Fast-DDS, whose per-participant SHM
    files live under ``/dev/shm/fastrtps_*``. When a prior launch
    exited uncleanly (Ctrl-C, OOM kill, ``pkill -9``), those
    lockfiles persist, and the next Fast-DDS participant fails to
    bind with ``[RTPS_TRANSPORT_SHM Error] Failed init_port
    fastrtps_port7000: open_and_lock_file failed`` — breaking every
    subsequent ``openral deploy sim`` until the operator manually
    cleans them. CLAUDE.md §2 names Cyclone as the OpenRAL default,
    but the launch_ros lifecycle_event_manager in Jazzy has a
    sharp edge with Cyclone (deserialisation race on the
    auto-CONFIGURE → ACTIVATE chain) that takes down the
    prompt_router; until that's resolved we stay on Fast-DDS and
    aggressively clean its stale state.

    Best-effort: only files owned by the calling user are
    unlinked — other users' SHM segments are silently skipped.
    Operators that explicitly opt into Cyclone or Zenoh via
    ``RMW_IMPLEMENTATION`` keep theirs untouched.
    """
    if "rmw_cyclonedds" in env.get("RMW_IMPLEMENTATION", ""):
        return  # operator opted into Cyclone; nothing to clean
    if "rmw_zenoh" in env.get("RMW_IMPLEMENTATION", ""):
        return
    _clean_stale_fastrtps_shm()


def _clean_stale_fastrtps_shm() -> None:
    """Best-effort: remove stale Fast-DDS SHM lock files in ``/dev/shm``.

    Fast-DDS lock files are owned by the user that created them — we
    silently skip anything we can't unlink (another user's file) so
    this never escalates to ``sudo``-required cleanup. Cyclone-DDS
    deployments never call this path. Only fires when the operator
    explicitly opts back into Fast-DDS via ``RMW_IMPLEMENTATION``.
    """
    shm = Path("/dev/shm")
    if not shm.is_dir():
        return
    for entry in shm.iterdir():
        if not entry.name.startswith("fastrtps_"):
            continue
        with contextlib.suppress(OSError, PermissionError):
            entry.unlink()


def _required_ros2_packages(invocation: LaunchInvocation) -> list[str]:
    """Build the package-list the preflight discovery check must validate.

    Pulled out of :func:`deploy_sim_command` for line-count hygiene
    and so future opt-in bringup wrappers extend a single list.
    """
    pkgs = ["openral_rskill_ros", invocation.hal.package]
    if invocation.enable_slam:
        pkgs.append("openral_slam_bringup")
    if invocation.enable_nav2:
        pkgs.append("openral_nav2_bringup")
    return pkgs


def assert_ros2_packages_discoverable(
    packages: Iterable[str],
    *,
    prefix_lookup: Callable[[str], str | None] = _ros2_pkg_prefix,
) -> None:
    """Raise ``ROSConfigError`` listing every ``pkg`` ``ros2`` cannot find.

    Catches the most common operator failure for ``openral deploy sim``:
    ``ros2`` itself is on PATH (so the system overlay is sourced) but
    the OpenRAL workspace overlay (``install/setup.bash``) is not, so
    ``ros2 launch`` can't find ``openral_rskill_ros`` or the per-robot
    ``openral_hal_<X>`` package. The same error fires for a stale build
    that simply hasn't included the requested HAL package yet.

    ``prefix_lookup`` is injectable so unit tests can drive the path
    with a deterministic fake without shelling out (CLAUDE.md §1.11
    process-boundary fake).
    """
    missing = [pkg for pkg in packages if prefix_lookup(pkg) is None]
    if not missing:
        return
    quoted = ", ".join(repr(p) for p in missing)
    raise ROSConfigError(
        f"ros2 cannot find ROS package(s): {quoted}. The OpenRAL workspace "
        "overlay is not sourced (or the build is stale). From the repo root, "
        "run:\n"
        "  just ros2-build && source install/setup.bash\n"
        "then re-run ``openral deploy sim``. Note: the package is named "
        "``openral_rskill_ros`` (not ``rskill``) — the Python sub-package "
        "under ``python/rskill/`` is a different layer."
    )


def _preflight_palette_deps(  # noqa: PLR0912, PLR0915  # reason: linear flow — split would obscure the prompt → install → re-probe contract
    *,
    repo_root: Path,
    robot_yaml: Path,
    commercial_deployment: bool = False,
) -> None:
    """Prompt to install missing extras before the reasoner palette empties.

    Mirrors :meth:`ReasonerNode._maybe_seed_palette_from_search_paths`:
    loads ``<repo_root>/rskills/*/rskill.yaml``, builds the
    capability-filtered :class:`~openral_reasoner.palette.ToolPalette`
    against the robot's :class:`~openral_core.RobotCapabilities`, then
    probes each capability-matching manifest's ``model_family`` for
    importability. Surfaces missing extras *before* the launch
    instead of letting the reasoner silently drop them at
    ``on_configure`` time (the operator only finds out via
    ``palette_empty`` ticks once everything is up).

    This is ADVISORY, not a gate. The reasoner ALREADY drops
    unimportable rSkills at ``on_configure``
    (:func:`openral_sim.policy_deps.filter_importable_manifests`) and
    runs the importable remainder. The palette is robot-WIDE — a single
    franka config matches six model families (act / molmoact2 / pi05 /
    rldx / smolvla / xvla), so a partially-installed venv is the common
    case and demanding every family's extras to run ONE skill is the
    wrong contract. So we warn-and-drop, and hard-fail only when the
    palette would be left empty (nothing dispatchable).

    Behaviour (when ≥1 matching skill is blocked on missing extras):

    * default / ``OPENRAL_AUTO_INSTALL_DEPS=1`` → install the union of
      missing groups via ``just sync --all-packages --group …``
      (cwd=repo_root), re-probe, and continue. Same env var honoured by
      :mod:`openral_sim._assets` / :mod:`openral_sim._deps`. A non-zero
      ``just sync`` is a real failure → ``typer.Exit``.
    * ``OPENRAL_AUTO_INSTALL_DEPS=0`` on a TTY → ``typer.confirm`` the
      same install; on yes install+re-probe; on no → drop blocked skills.
    * ``OPENRAL_AUTO_INSTALL_DEPS=0`` non-TTY → drop the blocked skills
      and proceed, printing the install command as a hint.
    * In every "proceed" path above: if EVERY matching skill is blocked
      (palette would be empty) → print the install command and
      ``typer.Exit(1)`` instead, since the graph could dispatch nothing.
    * No capability-matching skills are blocked → silent return.
    """
    from openral_core import RobotDescription, RSkillManifest
    from openral_reasoner.palette import build_tool_palette
    from openral_sim.policy_deps import (
        can_import_policy_family,
        model_family_install_groups,
        model_family_install_hint,
    )

    rskills_dir = repo_root / "rskills"
    manifest_paths = sorted(rskills_dir.glob("*/rskill.yaml"))
    if not manifest_paths:
        return

    try:
        description = RobotDescription.from_yaml(str(robot_yaml))
    except (OSError, ValueError):
        # Robot.yaml will fail loudly downstream — don't double-report here.
        return

    manifests: list[RSkillManifest] = []
    for path in manifest_paths:
        try:
            manifests.append(RSkillManifest.from_yaml(str(path)))
        except (OSError, ValueError):
            # Same "skip unloadable" behaviour as the reasoner seed.
            continue

    matching = build_tool_palette(
        installed_skills=manifests,
        robot_capabilities=description.capabilities,
        commercial_deployment=commercial_deployment,
    )
    matching_ids = matching.execute_rskill_ids
    if not matching_ids:
        # Palette would be empty for capability / role / license reasons,
        # not deps. Reasoner already logs this clearly at on_configure.
        return

    blocked: list[tuple[str, str]] = []  # (manifest_name, model_family)
    install_groups: set[str] = set()
    for m in manifests:
        if m.name not in matching_ids:
            continue
        family = getattr(m, "model_family", None) or ""
        ok, _ = can_import_policy_family(family)
        if ok:
            continue
        blocked.append((m.name, family))
        install_groups.update(model_family_install_groups(family))

    if not blocked:
        return

    _console.print()
    _console.print(
        f"[yellow]preflight:[/yellow] {len(blocked)} of "
        f"{len(matching_ids)} capability-matched rSkill(s) are missing "
        "Python extras:"
    )
    for name, family in blocked:
        _console.print(f"  • {name}  (model_family={family!r})")
        _console.print(f"      {model_family_install_hint(family)}")

    install_cmd: list[str] | None = None
    if install_groups:
        groups_argv: list[str] = []
        for g in sorted(install_groups):
            groups_argv.extend(["--group", g])
        # Route through ``just sync`` (not bare ``uv sync``) for two
        # reasons:
        #   1. ``--all-packages`` is REQUIRED so the workspace members
        #      (openral-core, openral-cli, ...) survive the install.
        #      ``uv sync --group <X>`` without ``--all-packages``
        #      uninstalls every workspace member — the next ROS launch
        #      then fails with ``No module named 'openral_core'``
        #      (the exact symptom the preflight is meant to prevent).
        #   2. The libero/robocasa groups pull in ``hf-libero==0.1.3``,
        #      whose sdist installs both modern and legacy uninstall
        #      metadata. The ``just sync`` recipe repairs that
        #      before+after via ``scripts/repair_hf_libero_install.py``
        #      so the next ``uv sync --all-packages`` doesn't bail out
        #      with ``Unable to uninstall hf-libero==0.1.3``.
        #   3. ``--inexact`` makes the install ADDITIVE. Without it, ``uv
        #      sync --group <X>`` is exact-match on the dependency-group set
        #      and uninstalls every package not in group ``X`` — including
        #      run-critical packages from sibling groups: the OmDet-Turbo
        #      detector's ``timm`` (group ``omdet``), ``robosuite`` (group
        #      ``robocasa``), rldx's ``pyzmq``/``msgpack``. The observed
        #      failure: installing the ``rldx`` palette extras wiped ``timm``,
        #      so the detector ImportError'd on every frame and
        #      ``/openral/perception/objects`` stayed empty (issue #12). The
        #      robocasa AUTO_INSTALL plan already uses ``--inexact`` for this
        #      exact reason (openral_sim._deps._robocasa_kitchen_plan).
        install_cmd = ["just", "sync", "--all-packages", "--inexact", *groups_argv]

    # Install by default; set OPENRAL_AUTO_INSTALL_DEPS=0 to prompt on a
    # TTY or skip on non-TTY (honoured by openral_sim._assets / _deps).
    auto_install = os.environ.get("OPENRAL_AUTO_INSTALL_DEPS", "1") == "1"
    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    attempt_install = install_cmd is not None and (
        auto_install
        or (
            is_interactive
            and typer.confirm(
                f"Install missing extras now with `{shlex.join(install_cmd)}`?",
                default=True,
            )
        )
    )

    if attempt_install:
        assert install_cmd is not None  # narrowed by attempt_install guard
        # Propagate the operator's install-consent to the launched graph. The
        # scene backend's on_configure asset/dep install (openral_sim._assets,
        # gated on OPENRAL_AUTO_INSTALL_DEPS) must then proceed WITHOUT a second
        # prompt — otherwise the graph blocks before the MuJoCo viewer opens and
        # the operator is forced to set the env var by hand. run_launch_invocation
        # builds the launch env via os.environ.copy(), so setting it here carries
        # the single "yes, install" answer to every downstream group/asset install,
        # not just this rSkill-extras step.
        os.environ["OPENRAL_AUTO_INSTALL_DEPS"] = "1"
        _console.print(f"[cyan]running:[/cyan] {shlex.join(install_cmd)}")
        # cwd=repo_root so ``just`` resolves the workspace ``Justfile``
        # regardless of where the user invoked ``openral deploy sim`` from.
        completed = subprocess.run(install_cmd, check=False, cwd=str(repo_root))
        if completed.returncode != 0:
            # The operator explicitly asked to install (env var / confirm)
            # and it failed — surface it, don't silently drop and boot a
            # degraded graph.
            _console.print(
                f"[red]uv sync failed (exit {completed.returncode}).[/red] "
                "Fix the install and re-run ``openral deploy sim``."
            )
            raise typer.Exit(code=completed.returncode)
        # Flush importer caches so the freshly-installed packages are
        # discoverable on the re-probe (the .venv is the same one this
        # process runs from; new files on disk need an invalidate to
        # show up via PathFinder), then recompute the blocked set.
        importlib.invalidate_caches()
        blocked = [(n, f) for n, f in blocked if not can_import_policy_family(f)[0]]
        if not blocked:
            _console.print("[green]preflight:[/green] extras installed; continuing launch.")
            return
        # Partial success — fall through to the drop/empty decision with
        # the reduced ``blocked`` set.

    # We did NOT fully resolve the missing extras (declined / non-TTY /
    # no install command / partial install). Preflight is advisory: drop
    # the blocked skills and proceed so the reasoner runs the importable
    # remainder — UNLESS that remainder is empty, in which case the graph
    # could dispatch nothing and we fail fast with the install command.
    kept_count = len(matching_ids) - len(blocked)
    if kept_count <= 0:
        _console.print()
        if install_cmd is not None:
            _console.print(
                "[red]preflight failed:[/red] every capability-matched rSkill is "
                "blocked on missing extras — the reasoner palette would be empty. "
                f"Install them:\n  {shlex.join(install_cmd)}"
            )
        else:
            _console.print(
                "[red]preflight failed:[/red] every capability-matched rSkill is "
                "blocked — see per-skill hints above."
            )
        raise typer.Exit(code=1)

    _console.print()
    _console.print(
        f"[yellow]preflight:[/yellow] proceeding — {len(blocked)} skill(s) will be "
        f"dropped from the reasoner palette; {kept_count} remain dispatchable."
    )
    if install_cmd is not None:
        _console.print(f"  to enable the dropped skill(s): {shlex.join(install_cmd)}")


def deploy_sim_command(
    config: Path = typer.Option(  # reason: typer Option idiom
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        dir_okay=False,
        help=(
            "Path to a DeployScene YAML (scenes/deploy/, scene + optional "
            "robot, no task). Strict: SimScene / BenchmarkScene YAMLs are "
            "rejected with a redirect to `openral sim run` / "
            "`openral benchmark scene`."
        ),
    ),
    robot: str | None = typer.Option(
        None,
        "--robot",
        help=(
            "Override the DeployScene's ``robot_id``. Required when the YAML omits ``robot_id``."
        ),
    ),
    dashboard_port: int = typer.Option(
        4318,
        "--dashboard-port",
        help="OTLP/HTTP port passed through to the launch's dashboard child.",
    ),
    reset_to_pose_service: str | None = typer.Option(
        None,
        "--reset-to-pose-service",
        help=(
            "Override the HAL ``reset_to_pose`` service path. Defaults to "
            "``/openral/<robot_id>/reset_to_pose``."
        ),
    ),
    approach_skill_id: str | None = typer.Option(
        None,
        "--approach-skill-id",
        help=(
            "ADR-0053 — MoveIt approach rSkill URI (e.g. "
            "``rskills/rskill-moveit-joints``). When set, the runner plans a "
            "collision-free MoveGroup motion to each skill's starting_pose "
            "instead of the teleport snap (needs a running move_group). Empty "
            "keeps the legacy ResetToPose snap."
        ),
    ),
    dataset_out: str | None = typer.Option(
        None,
        "--dataset-out",
        help=(
            "ADR-0019 — record the deploy session (proprio + action + camera "
            "frames + episode markers) to this rosbag2 mcap path. Convert to a "
            "LeRobotDataset v3 offline with `openral dataset from-bag`. Empty "
            "disables recording."
        ),
    ),
    dataset_repo_id: str | None = typer.Option(
        None,
        "--dataset-repo-id",
        help="ADR-0019 — repo_id for the recorded dataset (default openral/dataset-<robot>).",
    ),
    dataset_license: str | None = typer.Option(
        None,
        "--dataset-license",
        help="ADR-0019 — SPDX license carried into `openral dataset from-bag` (default CC-BY-4.0).",
    ),
    hal: list[str] = typer.Option(  # reason: typer Option idiom
        None,
        "--hal",
        help=(
            "Per-robot HAL parameter override, ``key=value`` (repeatable). "
            "Value is parsed as JSON when possible (so ``--hal "
            "viewer_enabled=false`` works); otherwise treated as a string. "
            "Overrides the per-robot defaults in ``_ROBOT_HAL_REGISTRY``."
        ),
    ),
    enable_slam: bool | None = typer.Option(
        None,
        "--enable-slam/--no-enable-slam",
        help=(
            "ADR-0025 — bring up slam_toolbox as a Reasoner-managed "
            "background service. **Auto by default**: enabled when the "
            "robot's manifest declares ``capabilities.has_lidar: true``. "
            "Pass ``--enable-slam`` / ``--no-enable-slam`` to override the "
            "manifest. The launcher auto-transitions slam_toolbox to "
            "INACTIVE; the Reasoner promotes to ACTIVE via "
            'LifecycleTransitionTool(node="/openral_slam_toolbox"). '
            "Requires ros-${ROS_DISTRO}-slam-toolbox apt-installed and "
            "the openral_slam_bringup package colcon-built in the "
            "workspace."
        ),
    ),
    enable_nav2: bool | None = typer.Option(
        None,
        "--enable-nav2/--no-enable-nav2",
        help=(
            "ADR-0025 — bring up the Nav2 navigation stack so the "
            "``OpenRAL/rskill-nav2-navigate-to-pose`` wrapped-action "
            "rSkill has a ``/navigate_to_pose`` server to dispatch to. "
            "**Auto by default**: tracks ``--enable-slam`` (lidar-"
            "equipped robots need a planner to consume the map). "
            "Requires ros-${ROS_DISTRO}-nav2-bringup apt-installed "
            "and the openral_nav2_bringup package colcon-built."
        ),
    ),
    object_detector_locator: list[str] | None = typer.Option(
        None,
        "--object-detector-locator",
        help=(
            "ADR-0056 — on-demand open-vocab locator to bring up alongside the "
            "continuous detector (repeatable). A manifest path or a short alias "
            "(e.g. 'omdet-turbo-locator', 'locateanything-3b-nf4'). Each becomes a "
            "namespaced locate_in_view node the reasoner picks via the tool's "
            "'detector' field. Default = omdet-turbo-locator when the detector is "
            "on and the omdet deps are present; LocateAnything is opt-in (NVIDIA "
            "non-commercial, 5 GB VRAM, needs the sidecar venv)."
        ),
    ),
    enable_octomap: bool | None = typer.Option(
        None,
        "--enable-octomap/--no-enable-octomap",
        help=(
            "ADR-0030 — bring up the world-collision perception leg: "
            "octomap_server (3-D OcTree from the HAL's depth PointCloud2) "
            "+ openral_octomap_bridge (octree → /openral/world_voxels) + "
            "the C++ safety kernel's capsule-vs-voxel check. **Auto by "
            "default**: enabled when the robot manifest declares a depth "
            "SensorSpec. Requires ros-${ROS_DISTRO}-octomap-server "
            "apt-installed and the openral_octomap_bridge package "
            "colcon-built."
        ),
    ),
    enable_octomap_kernel_check: bool = typer.Option(
        True,
        "--enable-octomap-kernel-check/--no-enable-octomap-kernel-check",
        help=(
            "ADR-0030/0035 — when --no-enable-octomap-kernel-check, the octomap "
            "perception leg still publishes /openral/world_voxels (so the "
            "world-state object-lift works) but the C++ safety kernel's "
            "capsule-vs-voxel check stays OFF (its --no-enable-octomap posture: "
            "envelope + self-collision only). Use with --enable-octomap to let "
            "perception use the world map without the dense-scene false-positive "
            "E-stop. Default on (bundled ADR-0030 behaviour)."
        ),
    ),
    enable_object_detector: bool = typer.Option(
        True,
        "--object-detector/--no-object-detector",
        help=(
            "ADR-0035 — bring up the ROS-Image object detector "
            "(openral_perception_ros/ros_image_detector_node): publishes "
            "ObjectsMetadata to /openral/perception/objects, which the "
            "world-state node's object-lift raises into /openral/world_voxels. "
            "**On by default.** The default backend is the open-vocabulary "
            "omdet-turbo-indoor continuous detector (falls back to the in-tree "
            "RT-DETR COCO ONNX when the omdet deps are absent). Pass "
            "--no-object-detector to turn the leg off. Requires the "
            "openral_perception_ros package colcon-built."
        ),
    ),
    object_detector_onnx: Path | None = typer.Option(
        None,
        "--object-detector-onnx",
        help=(
            "ADR-0035 — path to the RT-DETR ONNX weights for the legacy / "
            "fallback detector path. Defaults to the in-tree "
            "rskills/rtdetr-coco-r18/model.onnx. Passing a path explicitly "
            "selects the fixed-label RT-DETR backend over the omdet default."
        ),
    ),
    object_detector_manifest: str | None = typer.Option(
        None,
        "--object-detector-manifest",
        help=(
            "ADR-0037 2026-06-09 — path to a kind:detector rSkill manifest "
            "(e.g. rskills/locateanything-3b-nf4/rskill.yaml). When set, the "
            "detector node is manifest-driven: runtime:pytorch brings up the "
            "open-vocabulary LocateAnything VLM sidecar; runtime:onnx uses "
            "RT-DETR. A manifest path auto-enables the detector leg (no ONNX "
            "file needed). The VLM sidecar needs an isolated transformers==4.57.1 "
            "venv (OPENRAL_LOCATEANYTHING_SIDECAR_VENV) + OPENRAL_ALLOW_NONCOMMERCIAL=1."
        ),
    ),
    object_detector_query: str | None = typer.Option(
        None,
        "--object-detector-query",
        help=(
            "ADR-0037 2026-06-09 — initial open-vocabulary query for a VLM "
            "detector (e.g. 'red mug'). Empty = the manifest's detector.labels "
            "default. Retarget live by publishing a std_msgs/String to "
            "/openral/perception/detector_query."
        ),
    ),
    enable_reward_monitor: bool = typer.Option(
        False,
        "--enable-reward-monitor/--no-enable-reward-monitor",
        help=(
            "ADR-0057 — bring up the Robometer reward monitor "
            "(openral_perception_ros/reward_monitor_node) PARALLEL to the VLA: it "
            "buffers the agentview RGB stream and serves "
            "/openral/perception/query_task_progress, and the reasoner is told "
            "task_progress_available=True so its LLM may poll per-frame "
            "progress/success whenever it sees fit. Advisory-only. Default off. "
            "Needs the openral_perception_ros package colcon-built and a "
            "provisioned Robometer sidecar venv (OPENRAL_ROBOMETER_SIDECAR_VENV); "
            "co-resident with a VLA wants a small NF4 VLA on an 8 GB GPU (~3.3 GB)."
        ),
    ),
    enable_critic: bool = typer.Option(
        False,
        "--enable-critic/--no-enable-critic",
        help=(
            "ADR-0064 — bring up the Tier-C critic producer "
            "(openral_reasoner_ros/critic_producer_node). It watches the generic "
            "/openral/critic/score topic that reward models publish (Robometer, a "
            "future SARM, success classifiers) and emits a Tier-C FailureTrigger on "
            "/openral/failure/critic when a critic stalls — the reasoner already maps "
            "that to a forced Tier-C tick (replanning). Advisory-only. Default off."
        ),
    ),
    reward_monitor_manifest: str | None = typer.Option(
        None,
        "--reward-monitor-manifest",
        help=(
            "ADR-0057 — path to a kind:reward rSkill manifest. Empty defaults to "
            "the in-tree rskills/robometer-4b/rskill.yaml. weights_uri may be "
            "hf://org/repo or local:///abs/path (a pre-quantized NF4 checkpoint "
            "loaded directly as 4-bit). Ignored unless --enable-reward-monitor."
        ),
    ),
    reward_monitor_task: str | None = typer.Option(
        None,
        "--reward-monitor-task",
        help=(
            "ADR-0057 — default task instruction the reward monitor scores when a "
            "query leaves task empty. The reasoner normally passes the active task "
            "per query. Ignored unless --enable-reward-monitor."
        ),
    ),
    spatial_memory_ingest: bool | None = typer.Option(
        None,
        "--spatial-memory-ingest/--no-spatial-memory-ingest",
        help=(
            "ADR-0038 — have the reasoner accumulate a durable ADR-0038 "
            "SpatialMemory from the object-lift producer's "
            "WorldState.detected_objects so recall_object recalls what the robot "
            "has seen, and the dashboard shows a scene-objects card + SLAM-map "
            "markers. **Auto by default**: enabled whenever the object "
            "detector is."
        ),
    ),
    memory_dir: str | None = typer.Option(
        None,
        "--memory-dir",
        help=(
            "ADR-0072 — path to a deploy memory bundle directory. The reasoner "
            "loads MEMORY.md (semantic memory + memory_write/search tools) from it; "
            "if the dir also holds scene_graph.json it preloads the 3D world-state "
            "graph (recall_object), and if it holds map.yaml a nav2 map_server seeds "
            "the 2D occupancy grid. The dir must exist (the robot writes MEMORY.md "
            "into it). Overrides the DeployScene's own memory_dir."
        ),
    ),
    dashboard: bool = typer.Option(
        True,
        "--dashboard/--no-dashboard",
        help=(
            "Auto-spawn the live observability dashboard alongside "
            "``ros2 launch`` and point OTLP exporters at it. Default: "
            "on. The dashboard binds to ``--dashboard-port`` (default "
            "4318) and serves the OpenRAL UI + OTLP/HTTP receiver. "
            "``--no-dashboard`` is a true headless mode: the dashboard "
            "child is skipped AND ``OTEL_EXPORTER_OTLP_ENDPOINT`` is "
            "omitted from every node's env, so the OTel SDK short-"
            "circuits to no-op (no BatchSpanProcessor → no shutdown "
            "stall on dead-port retries). Set ``OTEL_EXPORTER_OTLP_"
            "ENDPOINT`` in the parent shell to forward to an external "
            "collector instead."
        ),
    ),
    foxglove: bool = typer.Option(
        False,
        "--foxglove/--no-foxglove",
        help=(
            "ADR-0059 — spawn the read-only Foxglove WebSocket bridge "
            "as part of the deploy-sim runtime graph. Default: off. "
            "When enabled, open Foxglove Studio and connect via "
            "``ws://127.0.0.1:<foxglove-port>`` to see live cameras, "
            "joint states, /tf, and the navigation map. **View-only** "
            "(cannot actuate the robot): ``clientPublish``, "
            "``services``, and ``parameters`` capabilities are omitted "
            "from the bridge. Only Bucket-1 topics are exposed "
            "(safety/e-stop/action topics are never forwarded). "
            "Requires ``foxglove_bridge`` installed in the workspace "
            "(ros-${ROS_DISTRO}-foxglove-bridge)."
        ),
    ),
    foxglove_port: int = typer.Option(
        8765,
        "--foxglove-port",
        help=(
            "ADR-0059 — Foxglove WebSocket port (ws://127.0.0.1:<port>). "
            "Default 8765 (the foxglove_bridge upstream default). "
            "Ignored unless ``--foxglove`` is set."
        ),
    ),
    initial_task: str | None = typer.Option(
        None,
        "--initial-task",
        help=(
            "Single natural-language goal the reasoner decomposes into ordered subtasks "
            "via ``decompose_mission``, e.g. ``--initial-task 'pick the bowl and place "
            "it on the plate, then push the mug back'``. Passed as "
            "``initial_task_prompt`` to the launch. When omitted, no startup prompt is "
            "set and the reasoner idles until a manual ``openral prompt`` or dashboard "
            "prompt arrives."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Print the resolved ``ros2 launch`` argv + HAL params and exit "
            "without writing the HAL params temp file or shelling out."
        ),
    ),
) -> None:
    r"""Boot the full ROS graph against a robot's digital-twin HAL.

    Example::

        openral deploy sim --config scenes/deploy/openarm_tabletop.yaml
        openral deploy sim --config scenes/deploy/openarm_tabletop.yaml \
                       --hal viewer_enabled=false
    """
    try:
        overrides = _parse_hal_overrides(hal)
        invocation = resolve_launch_invocation(
            config=config,
            robot_override=robot,
            dashboard_port=dashboard_port,
            reset_to_pose_service=reset_to_pose_service,
            approach_skill_id=approach_skill_id,
            dataset_out=dataset_out,
            dataset_repo_id=dataset_repo_id,
            dataset_license=dataset_license,
            hal_param_overrides=overrides,
            enable_slam=enable_slam,
            enable_nav2=enable_nav2,
            enable_octomap=enable_octomap,
            enable_octomap_kernel_check=enable_octomap_kernel_check,
            enable_object_detector=enable_object_detector,
            object_detector_onnx=object_detector_onnx,
            object_detector_manifest=object_detector_manifest,
            object_detector_query=object_detector_query,
            enable_reward_monitor=enable_reward_monitor,
            reward_monitor_manifest=reward_monitor_manifest,
            reward_monitor_task=reward_monitor_task,
            enable_critic=enable_critic,
            object_detector_locators=object_detector_locator,
            spatial_memory_ingest=spatial_memory_ingest,
            memory_dir=memory_dir,
            enable_dashboard=dashboard,
            enable_foxglove=foxglove,
            foxglove_port=foxglove_port,
            initial_task_prompt=initial_task,
        )
    except ROSConfigError as exc:
        _console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(
        f"[cyan]deploy sim[/cyan] → robot=[bold]{invocation.robot_id}[/bold] "
        f"(manifest.name=[bold]{invocation.robot_manifest_name}[/bold]) "
        f"hal_package=[bold]{invocation.hal.package}[/bold] "
        f"hal_node_name=[bold]{invocation.hal.node_name}[/bold]"
    )
    _console.print(f"  robot_yaml:    {invocation.robot_yaml}")
    _console.print(f"  reset_service:    {invocation.reset_to_pose_service}")
    _console.print(f"  approach_skill:   {invocation.approach_skill_id or '(snap)'}")
    _console.print(f"  hal_params:    {invocation.hal_params}")
    _console.print(
        "  slam:          "
        + (
            "[green]enabled[/green] (auto from robot.capabilities.has_lidar — "
            "Reasoner drives → ACTIVE via LifecycleTransitionTool)"
            if invocation.enable_slam
            else "[dim]disabled[/dim] (robot.capabilities.has_lidar=false; "
            "pass --enable-slam to force on)"
        )
    )
    _console.print(
        "  nav2:          "
        + (
            "[green]enabled[/green] (Nav2 advertises /navigate_to_pose; "
            "Reasoner dispatches OpenRAL/rskill-nav2-navigate-to-pose)"
            if invocation.enable_nav2
            else "[dim]disabled[/dim] (tracks --enable-slam; pass --enable-nav2 to force on)"
        )
    )
    _console.print(
        "  octomap:       "
        + (
            "[green]enabled[/green] (octomap_server + bridge → "
            "/openral/world_voxels; kernel voxel check on when the robot "
            "has collision capsules)"
            if invocation.enable_octomap
            else "[dim]disabled[/dim] (no depth SensorSpec; pass --enable-octomap to force on)"
        )
    )
    _console.print(
        "  detector:      "
        + (
            f"[green]enabled[/green] (ros_image_detector_node → "
            f"/openral/perception/objects → object-lift; onnx="
            f"{invocation.object_detector_onnx})"
            if invocation.enable_object_detector
            else "[dim]disabled[/dim] (onnx weights not found at "
            f"{invocation.object_detector_onnx}; pass --enable-object-detector "
            "to force on)"
        )
    )
    _console.print(
        "  dashboard:     "
        + (
            f"[green]auto-spawn[/green] at http://127.0.0.1:{dashboard_port}/"
            if dashboard
            else "[dim]disabled[/dim] (pass --dashboard to auto-spawn)"
        )
    )
    _console.print(
        "  foxglove:      "
        + (
            f"[green]enabled[/green] (view-only, cannot actuate) at ws://127.0.0.1:{foxglove_port}"
            if foxglove
            else "[dim]disabled[/dim] (pass --foxglove to enable read-only live scene view)"
        )
    )
    _console.print("  envelope:      synthesised at launch time from robot.yaml (no envelope file)")
    _console.print(
        "  startup_prompt: "
        + (
            f"[green]{invocation.initial_task_prompt!r}[/green] "
            "(from --initial-task; delivered to reasoner at activate)"
            if invocation.initial_task_prompt
            else "[dim](none — reasoner idles until openral prompt or dashboard)[/dim]"
        )
    )

    if dry_run:
        printed = [
            arg.replace("HAL_PARAMS_FILE_PLACEHOLDER", "<hal-params-tmp>")
            for arg in invocation.argv_template
        ]
        _console.print(f"  argv: {shlex.join(printed)}")
        return

    if shutil.which("ros2") is None:
        _console.print(
            "[red]ros2 not found on PATH.[/red] Source your ROS 2 install "
            "(e.g. ``source /opt/ros/jazzy/setup.bash``) and the OpenRAL "
            "workspace overlay (``source install/setup.bash``) before "
            "``openral deploy sim``."
        )
        raise typer.Exit(code=1)

    # Pre-flight: ``ros2`` is on PATH, but is the OpenRAL workspace
    # overlay sourced and current? ``ros2 launch`` would otherwise fail
    # with a terse "Package 'openral_rskill_ros' not found, searching:
    # ['/opt/ros/jazzy']" — the same wording for missing overlay AND
    # for a stale build that omits the HAL. Catch both up front.
    try:
        assert_ros2_packages_discoverable(_required_ros2_packages(invocation))
    except ROSConfigError as exc:
        _console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _reap_orphans_with_log()

    # Probe the in-tree rSkill registry against this robot's
    # capabilities. If every capability-matching skill is blocked by a
    # missing extras group, prompt-and-install on a TTY, or fail with
    # the install command non-interactively. Without this the reasoner
    # silently degrades to ``palette_empty`` ticks once everything is
    # up — the user gets a running but useless deploy.
    _preflight_palette_deps(
        repo_root=_repo_root_from(Path(__file__)),
        robot_yaml=Path(invocation.robot_yaml),
    )

    # ADR-0077 §4 — VLA↔reward VRAM pair preflight. A VLA must run with its reward
    # model resident (ADR-0074); verify the pair fits the GPU BEFORE bringing up
    # ROS. No-op unless the reward monitor is active and the GPU budget is readable;
    # hard-exits (before launch) when no capability-matched VLA can co-reside with
    # the reward model.
    if invocation.enable_reward_monitor and invocation.reward_monitor_manifest:
        from openral_core import RobotDescription

        _preflight_reward_vram_fit(
            repo_root=_repo_root_from(Path(__file__)),
            description=RobotDescription.from_yaml(str(invocation.robot_yaml)),
            reward_manifest_path=invocation.reward_monitor_manifest,
            gpu_total_gb=_detect_gpu_total_vram_gb(),
        )

    # Write the ephemeral HAL params YAML (lifetime = subprocess) and
    # substitute its path into argv. ROS 2 parameter YAML uses the
    # ``/**`` wildcard so the file binds against the HAL's node name
    # regardless of robot. SIM115's "use a context manager" doesn't
    # fit — the file must outlive the Python ``with`` block so the
    # spawned HAL process can open it.
    hal_params_tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115  # reason: HAL reads after this scope
        mode="w",
        prefix=f"openral-hal-params-{invocation.robot_id}-",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    try:
        yaml.safe_dump(
            {"/**": {"ros__parameters": invocation.hal_params}},
            hal_params_tmp,
            sort_keys=False,
        )
        hal_params_tmp.close()

        argv = [
            arg.replace("HAL_PARAMS_FILE_PLACEHOLDER", hal_params_tmp.name)
            for arg in invocation.argv_template
        ]
        _console.print(f"  hal_params_tmp:{hal_params_tmp.name}")
        _console.print(f"  argv: {shlex.join(argv)}")

        # `ros2 launch` runs under the system Python by default; the
        # launch's deferred imports (openral_core, openral_safety) live
        # in the workspace venv (the one `uv run` is using right now).
        # The launch file processes editable-install ``.pth`` files via
        # ``site.addsitedir`` keyed on ``OPENRAL_VENV_SITE``; export
        # that env var alongside PYTHONPATH so both the launch parser
        # and the spawned Python nodes import openral_core correctly.
        # Also prepend the venv's bin dir to PATH so ``#!/usr/bin/env
        # python3`` shebangs on spawned node executables resolve to the
        # venv interpreter — that's the only interpreter that processes
        # the editable ``.pth`` files via site.py at startup. PYTHONPATH
        # alone is not enough: .pth files in PYTHONPATH directories are
        # never processed by Python's site module.
        venv_env = _prepare_launch_env()

        # The dashboard child is spawned by ``sim_e2e.launch.py`` itself
        # (gated on ``enable_dashboard:=true`` forwarded from ``dashboard``
        # above), so this wrapper just runs the launch. The earlier
        # design wrapped this in ``attached_dashboard(enabled=dashboard,
        # ...)``, but that would double-spawn the dashboard and trip
        # ``[Errno 98] address already in use``.
        #
        # ``_run_launch`` (not a bare ``subprocess.run``) puts the launch
        # in its own session and forwards SIGINT/SIGTERM to the group so
        # every node — and the static_transform_publisher /
        # robot_state_publisher / HAL it spawns — is reaped on shutdown
        # instead of orphaning onto ``/tf_static`` + the GPU.
        returncode = _run_launch(argv, venv_env)
        raise typer.Exit(code=returncode)
    finally:
        with contextlib.suppress(OSError):
            Path(hal_params_tmp.name).unlink(missing_ok=True)
