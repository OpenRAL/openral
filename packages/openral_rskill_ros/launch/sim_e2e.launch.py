r"""ADR-0018 — generic end-to-end ROS graph for ``openral deploy sim``.

One launch file for every robot. Every robot-specific bit is a launch
argument resolved inside an ``OpaqueFunction`` so concrete strings
reach ``LifecycleNode(package=, executable=, name=)``:

* ``robot_yaml``          — RobotDescription manifest path. Loaded
                            via Pydantic at launch time; the safety
                            kernel envelope is synthesised from it
                            (``openral_safety.envelope_loader.compute_intersection``)
                            and forwarded as ROS parameters on the
                            kernel node. **No envelope YAML file is
                            written or read.**
* ``hal_package`` / ``hal_executable`` / ``hal_node_name`` — HAL spawn,
                            picked by ``openral deploy sim`` from
                            ``_ROBOT_HAL_REGISTRY[robot_id]``.
* ``hal_params_file``     — Ephemeral ROS parameter YAML the CLI writes
                            with the HAL's per-robot knobs (``/**``
                            wildcard).
* ``reset_to_pose_service``, ``dashboard_port``, ``reasoner_provider``,
  ``reasoner_model`` — shared knobs.

Spawned processes: dashboard + safety_kernel + runtime + reasoner +
prompt_router + HAL. Lifecycle nodes auto-transition UNCONFIGURED →
INACTIVE → ACTIVE.
"""

from __future__ import annotations

import os
import pathlib
import site
import subprocess
import sys
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

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.events.matchers import matches_node_name as _matches_node_name
from lifecycle_msgs.msg import Transition
from openral_foxglove_bringup.topics import BUCKET1_TOPIC_WHITELIST, READ_ONLY_CAPABILITIES

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
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


def _autostart_lifecycle(node: LifecycleNode, node_name: str) -> list:
    """Event handlers that drive ``node`` UNCONFIGURED → INACTIVE → ACTIVE once.

    The activate handler is scoped to the **configure** transition
    (``start_state="configuring"``) so it fires exactly once at boot, after
    ``on_configure`` lands the node in ``inactive``. A bare ``goal_state=
    "inactive"`` matcher would also re-fire on a *runtime* deactivate
    (``active → deactivating → inactive``), which fights ADR-0050 VRAM eviction:
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


def _resolve_sim_clock(value: str) -> bool:
    """Single source of truth for the composed graph's clock domain.

    ``True`` iff a sim ``/clock`` will actually be published this run, in
    which case every node runs on ``use_sim_time``. deploy-sim has **no**
    ``/clock`` publisher today (the sim only advances while a skill steps,
    and the HAL stamps ``/scan`` + TF on wall-clock), so this defaults
    ``False`` and the whole graph stays on wall-clock — coherent with the
    HAL. Flipping a single node to sim-time without a ``/clock`` pins its
    clock at 0, which is exactly the nav-collision root cause (the local
    costmap rejected the "future" wall-clock scans and stayed empty).
    Mirrors the ``enable_*`` truthy idiom used throughout
    :func:`compose_runtime_graph`.
    """
    return value.strip().lower() in ("1", "true", "yes")


def _build_nav2_include(robot_yaml: str, *, use_sim_time: bool) -> object:
    """Construct the IncludeLaunchDescription for upstream Nav2 (ADR-0025).

    Pulled out of :func:`compose_runtime_graph` for line-count
    hygiene. Unlike slam_toolbox (which idles until activate), Nav2
    is always-on: its in-stack ``lifecycle_manager_navigation``
    brings the planner / controller / behavior / smoother /
    velocity_smoother sub-nodes to ACTIVE automatically. The
    Reasoner *triggers* Nav2 by dispatching the
    ``OpenRAL/rskill-nav2-navigate-to-pose`` wrapped-action rSkill,
    not by lifecycle-transitioning the planner.

    ``use_sim_time`` is the graph-wide clock-domain flag (see
    :func:`_resolve_sim_clock`); it is **not** hardcoded here. With no
    ``/clock`` on the bus it must be ``False`` so Nav2's controller loop
    and costmaps run on wall-clock — matching the HAL's wall-clock
    ``/scan`` + odom→base_link TF. (``true`` + no ``/clock`` pins every
    Nav2 node at t=0, "loop rate inf Hz", empty costmap → collision.)
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
        # this robot's footprint_radius / base_kinematics (ADR-0025) —
        # generic across mobile bases, no hand-vendored per-robot file.
        launch_arguments={
            "use_sim_time": "true" if use_sim_time else "false",
            "robot_yaml": robot_yaml,
        }.items(),
    )


def _resolve_urdf_path(ref: str, manifest_dir: pathlib.Path) -> str | None:
    """Resolve a ``RobotDescription.assets.urdf.ref`` to a concrete URDF path.

    Thin wrapper over ``openral_core.assets.resolve_asset`` (ADR-0058). Returns
    ``None`` for the ``ros2://robot_description`` dynamic marker (the URDF is on
    the ``/robot_description`` topic at runtime — no file to read). ``file:`` refs
    resolve against the robot's manifest dir, then the repo root.
    """
    from openral_core.assets import AssetRefError, resolve_asset

    try:
        path = resolve_asset(ref, "urdf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        print(f"[sim_e2e] could not resolve urdf ref {ref!r}: {exc}", flush=True)
        return None
    return None if path is None else str(path)


def compose_runtime_graph(context: LaunchContext, *_args: object, **_kwargs: object) -> list:  # noqa: PLR0915  # reason: launch compose is naturally linear — arg resolution + node construction + autostart wiring in one place is the clearest expression of the boot order
    """Resolve every launch arg, load ``robot.yaml``, build the graph.

    Bound to an :class:`~launch.actions.OpaqueFunction` in
    :func:`generate_launch_description` so the launch args resolve to
    concrete strings before they reach
    :class:`launch_ros.actions.LifecycleNode` (which doesn't accept
    :class:`~launch.substitutions.LaunchConfiguration` in every field).
    The name mirrors :func:`openral_rskill_ros.compose_runtime` — same
    "build the runtime in one place" semantics, scoped to the launch
    layer instead of the in-process composer.
    """
    # Deferred import: launch files are imported by `ros2 launch` even
    # without a sourced workspace, so keep openral_core / openral_safety
    # off the module top.
    from openral_core import RobotDescription
    from openral_safety.envelope_loader import (
        collision_params_from_description,
        compute_intersection,
        ee_link_index_from_collision_params,
        kernel_params_from_envelope,
    )

    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)
    hal_package = LaunchConfiguration("hal_package").perform(context)
    hal_executable = LaunchConfiguration("hal_executable").perform(context)
    hal_node_name = LaunchConfiguration("hal_node_name").perform(context)
    hal_params_file = LaunchConfiguration("hal_params_file").perform(context)
    reset_to_pose_service = LaunchConfiguration("reset_to_pose_service").perform(context)
    approach_skill_id = LaunchConfiguration("approach_skill_id").perform(context)
    dashboard_port = LaunchConfiguration("dashboard_port").perform(context)
    reasoner_provider = LaunchConfiguration("reasoner_provider").perform(context)
    reasoner_model = LaunchConfiguration("reasoner_model").perform(context)
    spatial_memory_path = LaunchConfiguration("spatial_memory_path").perform(context)
    spatial_memory_ingest = LaunchConfiguration("spatial_memory_ingest").perform(
        context
    ).lower() in ("1", "true", "yes")
    # ADR-0036 — deploy-path selector for the reasoner's action-mode
    # palette gate. ``openral deploy sim`` shells this launch with
    # ``hal_mode:=sim`` (digital-twin path: the scene's robosuite OSC
    # controller synthesises cartesian/OSC action modes, so cartesian
    # skills are admissible); ``openral deploy run`` passes ``hal_mode:=real``
    # so the reasoner admits only skills whose action modes ∈ the robot's
    # ``supported_control_modes``. Default ``"sim"`` matches the launch's
    # digital-twin heritage.
    hal_mode = LaunchConfiguration("hal_mode").perform(context)
    enable_slam = LaunchConfiguration("enable_slam").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
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
    # ADR-0030/0035 — decouple the octomap PERCEPTION leg (publishing
    # /openral/world_voxels for the world-state object-lift) from the SAFETY
    # KERNEL's capsule-vs-voxel check. Default True preserves the bundled
    # ADR-0030 behaviour; set False to publish the voxel map for object-lift
    # while keeping the kernel voxel check OFF (its posture under
    # --no-enable-octomap: envelope + self-collision only). Lets perception use
    # the world map without the kitchen false-positive E-stop. Never weakens the
    # kernel below the --no-enable-octomap baseline.
    enable_octomap_kernel_check = LaunchConfiguration("enable_octomap_kernel_check").perform(
        context
    ).lower() in ("1", "true", "yes")
    octomap_cloud_topic = LaunchConfiguration("octomap_cloud_topic").perform(context)
    # ADR-0035 — object-detection perception leg. Off by default; when on,
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
    # ADR-0057 — reward-monitor leg. Off by default; when on, a reward_monitor_node
    # runs PARALLEL to the VLA, buffering the agentview RGB stream, and the reasoner
    # is told task_progress_available=True so its LLM may poll
    # /openral/perception/query_task_progress (the query_task_progress tool) whenever
    # it sees fit. Advisory-only — never actuates.
    enable_reward_monitor = LaunchConfiguration("enable_reward_monitor").perform(
        context
    ).lower() in ("1", "true", "yes")
    reward_monitor_manifest = LaunchConfiguration("reward_monitor_manifest").perform(context)
    reward_monitor_task = LaunchConfiguration("reward_monitor_task").perform(context)
    reward_monitor_image_topic = LaunchConfiguration("reward_monitor_image_topic").perform(context)
    reward_monitor_sidecar_port = LaunchConfiguration("reward_monitor_sidecar_port").perform(
        context
    )
    # ADR-0056 — comma-separated on-demand locator manifest paths. Each becomes a
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
    # ADR-0059 — read-only Foxglove live-scene bridge. Off by default;
    # ``openral deploy sim --foxglove`` opts in.
    enable_foxglove = LaunchConfiguration("enable_foxglove").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    foxglove_port = LaunchConfiguration("foxglove_port").perform(context)
    # Graph-wide clock domain (single source of truth). deploy-sim has no
    # /clock publisher, so this is False by default and EVERY node below —
    # HAL, Nav2, slam_toolbox, robot_state_publisher, octomap, detector —
    # runs on wall-clock coherently. A future sim-/clock publisher (its own
    # ADR) flips ``enable_sim_clock:=true`` to put the whole graph on
    # sim-time at once, instead of the scattered per-node literals that let
    # Nav2 silently disagree and drive the base into an obstacle.
    use_sim_time = _resolve_sim_clock(LaunchConfiguration("enable_sim_clock").perform(context))

    # Synthesise the kernel envelope from the manifest. ``skill=None``
    # because ``openral deploy sim`` does not preselect an rSkill — the
    # reasoner picks dynamically. The robot ceiling is the right boot-
    # time envelope; future per-skill tightening will hot-swap via a
    # kernel reload, not by mounting a different envelope at boot.
    description = RobotDescription.from_yaml(robot_yaml)
    description.validate_for_e2e_pipeline()  # loud failure on missing fields
    envelope = compute_intersection(description, skill=None)
    # ADR-0030 — self-collision model. Prefer lowering from the robot's MJCF
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
            # hand-authored capsules + ACM instead (ADR-0030, safety §3).
            if mjcf_params.get("self_collision_enabled"):
                collision_params = mjcf_params
            else:
                print(
                    "[sim_e2e] MJCF has no primitive collision geometry; "
                    "keeping the manifest self-collision model.",
                    flush=True,
                )
        except Exception as exc:  # never let a geometry hiccup block the boot
            print(
                f"[sim_e2e] MJCF self-collision lowering failed: {exc!r}; using manifest geometry",
                flush=True,
            )
    kernel_params = {**kernel_params_from_envelope(envelope), **collision_params}
    # ADR-0040 — the actuated joint order (length n_dof) so the kernel can map
    # /joint_states (named) into q_meas in the action's dof index space, the seed
    # the geometric check needs to reconstruct non-position chunks. Same order as
    # the per-joint envelope arrays + collision_dof_index. `collision_seed_dt_s`
    # is the velocity-integration look-ahead step; 0.0 keeps the conservative
    # reactive (measured-config) check only. This is deliberate: the only
    # JOINT_VELOCITY emitter in-tree is the robocasa BASE chunk, whose dofs are
    # listed in collision_base_dofs and zeroed before FK — so integrating them is
    # a no-op (ADR-0040 audit). Enabling dt>0 helps only a future fixed-base
    # velocity arm AND requires validating that the chunk's velocity units match
    # this dt; integrating with the wrong dt would mispredict and could
    # under-report, so it stays off (fail-safe) until that validation lands
    # (ADR-0040 Phase 2b).
    kernel_params["collision_joint_names"] = [j.name for j in description.joints]
    kernel_params["collision_seed_dt_s"] = 0.0
    # deploy-sim publishes /joint_states only as fast as the sim steps, which
    # slows to ~3 Hz under heavy VLA inference (the sim advances on the same host
    # the policy runs on). A 200 ms seed deadline would fail-closed on every
    # chunk; 1 s is a safe backstop here because when stepping is slow the arm
    # also moves slowly in sim-time, so a wall-stale seed is still spatially
    # accurate. Real hardware (30 Hz+ /joint_states) never approaches this bound.
    kernel_params["collision_state_deadline_ms"] = 1000.0
    # ADR-0040 — dof indices of the planar mobile-base joints (manifest
    # base_joints). The kernel zeroes these before the base-relative collision FK
    # so a mobile manipulator's arm is checked in the base_link frame the
    # world/voxel grid lives in. Empty for fixed-base arms.
    #
    # ``collision_base_dofs`` is omitted when empty: launch_ros's
    # evaluate_parameter_dict normalises a Python list to a typed array and
    # an EMPTY list collapses to ``()``, which ensure_argument_type rejects
    # ("got '()' of type tuple"). The list is empty for every fixed-base
    # arm (openarm, so101, franka_panda, ur5e, ur10e, …) — i.e. the
    # majority of in-tree robots — so passing it unconditionally crashed
    # the whole launch before any node started. The kernel declares its
    # own ``[]`` default for this parameter, so omitting it is equivalent
    # to "no base dofs to zero" (same semantics as the
    # ``lifecycle_peer_node_ids`` guard 90 lines below).
    _base_joint_set = set(getattr(description, "base_joints", None) or [])
    _collision_base_dofs = [
        i for i, j in enumerate(description.joints) if j.name in _base_joint_set
    ]
    if _collision_base_dofs:
        kernel_params["collision_base_dofs"] = _collision_base_dofs
    # ADR-0040 Phase 3 — predictive Cartesian: the EE control link for the
    # Jacobian look-ahead (deepest collision link = wrist/tip). -1 (no collision
    # model) leaves predictive Cartesian off; the reactive measured-config check
    # is the floor regardless. Base dofs above are blocked from the arm Jacobian.
    kernel_params["collision_ee_link_index"] = ee_link_index_from_collision_params(collision_params)
    # ADR-0030 — when octomap is enabled, turn on the kernel's
    # allocation-free capsule-vs-voxel world-collision check and have it
    # subscribe /openral/world_voxels (published by the octomap bridge
    # below). max_cells covers the bridge's default 2×2×2 m @ 0.05 grid
    # (64 k cells) with headroom; margin inflates obstacles conservatively.
    # Fail-closed staleness/over-capacity semantics are the kernel's.
    #
    # The check tests each robot link CAPSULE against the grid, so it needs a
    # collision model with links: the kernel hard-fails ``on_configure`` if a
    # geometric check is enabled but ``collision_n_links == 0``. Robots that
    # declare a depth sensor but no collision geometry (e.g. panda_mobile)
    # still get the map produced (octomap_server + bridge launch below for
    # observability), but the kernel voxel check stays off so the kernel
    # configures cleanly on its scalar envelope.
    has_collision_capsules = int(collision_params.get("collision_n_links", 0)) > 0
    if enable_octomap and has_collision_capsules and enable_octomap_kernel_check:
        kernel_params = {
            **kernel_params,
            "world_voxel_enabled": True,
            # 2 cm buffer on top of the already-conservative capsule radii.
            # 5 cm was too eager for a cluttered kitchen (vetoed close work);
            # 2 cm lets the arm approach surfaces while still catching imminent
            # contact. The gripper + base are exempt from the model (see
            # robots/panda_mobile/robot.yaml), so the gripper can reach targets.
            "world_voxel_margin_m": 0.02,
            "world_voxel_max_cells": 262144,
            "world_voxel_deadline_ms": 1000.0,
        }

    # ADR-0017 — run identity for the dashboard's Identity card. These
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
    reasoner_env = {
        **otel_env,
        "OPENRAL_REASONER_LLM_PROVIDER": reasoner_provider,
        "OPENRAL_REASONER_LLM_MODEL": reasoner_model,
    }

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
    # ADR-0025 — lifecycle peer node ids the Reasoner should surface to
    # the LLM via `LifecycleTransitionTool`. Today only slam_toolbox is
    # opt-in; future managed services (RTAB-Map, perception trees) will
    # append themselves here under their own `enable_<svc>` launch args.
    lifecycle_peer_node_ids: list[str] = []
    # ADR-0050 — GPU peers the reasoner AUTO-deactivates before a VLA dispatch
    # and reactivates after (distinct from the LLM-facing palette peers above).
    vram_lifecycle_peers: list[str] = []
    if enable_slam:
        lifecycle_peer_node_ids.append("openral_slam_toolbox")
    if enable_object_detector:
        # ADR-0050 — expose the detector as a lifecycle peer so the reasoner can
        # DEACTIVATE it (freeing the detector's VRAM) before dispatching a
        # co-resident grab policy on a memory-constrained GPU.
        lifecycle_peer_node_ids.append("openral_ros_image_detector")
        # …and AUTO-free it: the reasoner deactivates the detector before each
        # execute_rskill and reactivates it on completion, so an 8 GB card does
        # not OOM with the detector (~1.3 GB) co-resident with the VLA (~4.5 GB).
        vram_lifecycle_peers.append("openral_ros_image_detector")
        # ADR-0056 — each on-demand locator is its own lifecycle node, so it is an
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
        # ADR-0036 — tell the reasoner which deploy path it is on so its
        # action-mode palette gate matches the HAL this launch brings up.
        "hal_mode": hal_mode,
    }
    if lifecycle_peer_node_ids:
        reasoner_params["lifecycle_peer_node_ids"] = lifecycle_peer_node_ids
    # ADR-0050 — same empty-list-omission rule as lifecycle_peer_node_ids
    # (launch_ros rejects an empty typed array); the reasoner defaults to [].
    if vram_lifecycle_peers:
        reasoner_params["vram_lifecycle_peers"] = vram_lifecycle_peers
    # ADR-0039 — preload a persisted scene graph as the reasoner's read-only
    # spatial-memory query backend when a path is provided.
    if spatial_memory_path:
        reasoner_params["spatial_memory_path"] = spatial_memory_path
    # ADR-0038 — accumulate the durable scene graph live from the ADR-0035
    # producer's WorldState.detected_objects (auto-creates an empty backend when
    # no path is preloaded).
    reasoner_params["spatial_memory_ingest"] = spatial_memory_ingest
    # ADR-0043 — offer the read-only locate_in_view tool to the LLM when an object
    # detector is in the graph (it exposes /openral/perception/locate_in_view).
    reasoner_params["detector_available"] = enable_object_detector
    # ADR-0057 — offer the read-only query_task_progress tool only when a reward
    # monitor is co-active (otherwise the tool would dispatch to a dead service).
    reasoner_params["task_progress_available"] = enable_reward_monitor
    # ADR-0056 — the default on-demand locator the reasoner routes to when a
    # locate_in_view call leaves ``detector`` empty (the first locator brought up).
    if locator_specs:
        reasoner_params["default_on_demand_detector"] = locator_specs[0]["alias"]
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
        additional_env=otel_env,
        output="screen",
    )
    hal = LifecycleNode(
        package=hal_package,
        executable=hal_executable,
        name=hal_node_name,
        namespace="",
        # Clock domain via the graph-wide flag. The HAL is the clock
        # authority — it stamps /scan, odom→base_link TF and joint_states.
        # Default (wall-clock) is unchanged; flipping enable_sim_clock makes
        # those stamps sim-time, coherent with a future /clock publisher.
        parameters=[hal_params_file, {"use_sim_time": use_sim_time}],
        additional_env=otel_env,
        output="screen",
    )
    # Derive ``camera_names`` from the robot manifest's RGB sensors so
    # the WorldState aggregator subscribes to the topics the HAL
    # actually publishes. Hard-coding ``[top, left_wrist, right_wrist]``
    # broke the panda_mobile / robocasa-kitchen path: that robot's
    # ``robots/panda_mobile/robot.yaml`` declares
    # ``camera1 / camera2 / camera3`` (robocasa renders
    # ``robot0_agentview_left_image`` etc. and the adapter remaps them
    # to ``cameraN``), so WorldState was subscribing to /image topics
    # nothing publishes — and the rldx adapter's
    # ``observation.images['camera1']`` lookup later raised
    # ``ROSConfigError: rldx adapter expects observation.images
    # ['camera1']; got []``. Fall back to the legacy triple if the
    # manifest declares no RGB sensors (e.g. pure-base robots).
    rgb_camera_names = [s.name for s in description.sensors if s.modality == "rgb"]
    if not rgb_camera_names:
        rgb_camera_names = ["top", "left_wrist", "right_wrist"]
    runtime = Node(
        package="openral_rskill_ros",
        executable="runtime_node",
        parameters=[
            {
                "robot_yaml": robot_yaml,
                "camera_names": rgb_camera_names,
                "rskill_search_paths": [_RSKILLS_DIR],
                "reset_to_pose_service": reset_to_pose_service,
                "approach_skill_id": approach_skill_id,
                # Forward enable_slam so the runtime's compose_runtime
                # attaches the SlamMapBridge → dashboard slam.occupancy_grid.
                "enable_slam_bridge": enable_slam,
                # ADR-0030 — when octomap is on the centers topic exists, so
                # attach the WorldCloudBridge → dashboard world.pointcloud.
                "enable_world_cloud_bridge": enable_octomap,
                # ADR-0048 Phase 2 — the runtime node (WorldState aggregator +
                # the GStreamer/runner sensor readers + skill_runner) must share
                # the graph-wide clock domain. Under enable_sim_clock the HAL
                # stamps camera/state data on sim time; a wall-clock runtime
                # would see it as ~1.78e9 s stale and drop every frame at the
                # WorldState staleness gate. Default false keeps it wall-clock.
                "use_sim_time": use_sim_time,
            }
        ],
        additional_env=otel_env,
        output="screen",
    )

    autostart: list = []
    autostart += _autostart_lifecycle(safety_kernel, "openral_safety_kernel")
    autostart += _autostart_lifecycle(reasoner, "openral_reasoner")
    autostart += _autostart_lifecycle(prompt_router, "openral_prompt_router")
    # HAL autostart goes through ``tools/lifecycle_autostart.py`` rather
    # than ``_autostart_lifecycle`` because launch_ros's
    # ``lifecycle_event_manager`` race on Jazzy (same one documented for
    # slam_toolbox below) silently swallows the ACTIVATE transition on
    # robocasa-kitchen first-boots: the HAL's ``on_configure`` takes
    # ~6 s (MuJoCo + robosuite import + env.reset), and by the time the
    # FSM publishes ``transition_event(inactive)``, the
    # ``OnStateTransition(goal_state="inactive")`` event handler's
    # ``EmitEvent(ChangeState=ACTIVATE)`` is dropped. End-state: HAL
    # stuck in INACTIVE, no ``on_activate``, no /joint_states, no
    # /odom, no /openral/cameras/*/image publishers. Nav2 + dashboard
    # cameras can't come up. Mirror the slam_toolbox workaround.
    hal_autostart_path = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
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
                # The HAL's ``on_configure`` runs synchronously on its
                # executor and can block for over a minute on a robocasa-
                # kitchen first-boot (MuJoCo + robosuite import, env.reset,
                # and — on a cold/rebuilt env — a uv resolve+build of
                # robocasa that alone logs ~27 s). The autostart's per-
                # transition spin must outlast that or it times out mid-
                # configure and false-fails with "did not advance the FSM".
                "--transition-timeout-s",
                "300.0",
            ],
            output="log",
        )
    )

    # ADR-0027/0057 — robot_state_publisher: when the robot.yaml carries an
    # ``assets.urdf`` ref, launch ``robot_state_publisher`` so the per-link
    # arm + sensor TF chain lands on ``/tf`` (consumed by the
    # ``openral_state_adapter`` registry at step time; also by
    # Nav2 / MoveIt / RViz when present). The ref can be either:
    #
    # * a ``file:<relpath>`` (vendored URDF, resolved against the manifest dir
    #   then the repo root) or a ``rd:<module>`` ref (pulled from the
    #   ``robot_descriptions`` package, no large file checked in-tree);
    # * the ``ros2://robot_description`` dynamic marker — declared by the
    #   detection assembler when the robot publishes its own URDF on the
    #   ``/robot_description`` topic. ``resolve_asset`` returns ``None`` for it,
    #   so RSP is skipped (the URDF is already on the bus).
    extra_nodes: list = []
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
                    parameters=[
                        {
                            "robot_description": robot_description_xml,
                            # Graph-wide clock domain (see _resolve_sim_clock).
                            # Must match the HAL: with no /clock, sim-time would
                            # pin RSP's TF stamps at 0 while the HAL publishes
                            # odom→base_link on wall-clock — the split that
                            # broke Nav2's TF lookups into the costmap frame.
                            "use_sim_time": use_sim_time,
                            # ADR-0027 — publish_frequency at 30 Hz matches
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
            # ``base_to_root_xyz_rpy`` + ``root_frame`` (ADR-0058), spawn a
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

    # ADR-0025 — opt-in slam_toolbox as a Reasoner-managed background
    # service. Auto-transitions UNCONFIGURED → INACTIVE only; activation
    # is the Reasoner's job (LifecycleTransitionTool).
    if enable_slam:
        # Deferred share-dir lookup so deployments without the
        # openral_slam_bringup package built still launch successfully
        # when enable_slam is left at its default (false).
        from ament_index_python.packages import get_package_share_directory

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
            # Graph-wide clock domain (see _resolve_sim_clock); overrides the
            # yaml's use_sim_time so slam_toolbox shares the HAL's wall-clock
            # /scan + odom TF. Sim-time without a /clock pins its pose-graph at
            # 0 → empty map → Nav2 plans through obstacles.
            parameters=[slam_params_path, {"use_sim_time": use_sim_time}],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.append(slam_node)
        # Auto-CONFIGURE + ACTIVATE slam_toolbox externally via a
        # tiny in-process Python helper that uses rclpy.lifecycle to
        # drive the transitions with retries. Using
        # ``ros2 lifecycle set`` directly was racey on robocasa-kitchen
        # boots: the kitchen install subprocess prints ~60 lines to
        # stdout before slam_toolbox's service is fully advertised,
        # so a fixed-delay TimerAction fired ``ros2 lifecycle set``
        # while the node was still ``Node not found``, exiting 1 and
        # surfacing as ``[ERROR] [ros2-9]: process has died, exit
        # code 1``. The rclpy helper waits for the service, retries,
        # and never logs at ERROR level on transient absence.
        #
        # Using rclpy avoids launch_ros's ``lifecycle_event_manager``
        # which on Jazzy logs a spurious ``[ERROR] Failed to make
        # transition 'TRANSITION_CONFIGURE'`` even when slam_toolbox's
        # ``on_configure`` returns SUCCESS (the change_state response
        # arrives with ``success=false`` on the first call due to a
        # service-responder race upstream).
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
                    entities=[_build_nav2_include(robot_yaml, use_sim_time=use_sim_time)],
                ),
            ),
        )
        # ADR-0026 follow-up — the reasoner_node seeds its rSkill
        # palette at on_configure (~5 s after launch), long before
        # Nav2 finishes its 15-30 s lifecycle bringup. The graph-
        # availability filter drops the
        # ``OpenRAL/rskill-nav2-navigate-to-pose`` rSkill because
        # ``/navigate_to_pose`` isn't yet advertised — so the LLM
        # never sees the Nav2 tool and replies "I do not have a
        # tool available to perform base movement". Spawn a small
        # helper that polls the ROS graph for ``/navigate_to_pose``
        # and fires Empty on ``/openral/skill_registry_changed``
        # once it appears; the reasoner re-seeds the palette and
        # the Nav2 rSkill becomes dispatchable.
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

    if enable_octomap:
        # ADR-0030 — the world-collision perception leg. octomap_server
        # builds a 3-D OcTree from the HAL's depth PointCloud2
        # (``synthesize_depth_pointcloud`` → ``octomap_cloud_topic``), and
        # the openral_octomap_bridge lowers that octree into the dense
        # ``/openral/world_voxels`` grid the kernel rasterizes capsules
        # against. Keeps the octomap dependency OUT of the real-time
        # kernel. ``frame_id`` is the fixed tree frame (odom, already on
        # /tf via the HAL's odom→base_link broadcast); ``cloud_in`` is
        # remapped to the robot's depth topic. Requires
        # ros-${ROS_DISTRO}-octomap-server + the openral_octomap_bridge
        # package built — opt-in, default off, like slam/nav2.
        octomap_server = Node(
            package="octomap_server",
            executable="octomap_server_node",
            name="openral_octomap_server",
            namespace="",
            parameters=[
                {
                    "resolution": 0.05,
                    "frame_id": "odom",
                    "base_frame_id": "base_link",
                    "sensor_model.max_range": 4.0,
                    # Keep the map fresh for manipulation: octomap ray-clears
                    # free space, so a grasped/moved object's old cells decay
                    # back to free once re-observed. A slightly higher
                    # occupancy threshold + speckle filter clears transient /
                    # isolated noise voxels faster so they don't linger as
                    # phantom obstacles in front of the arm.
                    "occupancy_thres": 0.6,
                    "sensor_model.miss": 0.4,
                    "filter_speckles": True,
                    # Graph-wide clock domain (see _resolve_sim_clock). With no
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
            remappings=[("cloud_in", octomap_cloud_topic)],
            additional_env=otel_env,
            output="screen",
        )
        octomap_bridge = Node(
            package="openral_octomap_bridge",
            executable="octomap_voxel_bridge",
            name="openral_octomap_voxel_bridge",
            namespace="",
            parameters=[
                {
                    "base_frame": "base_link",
                    "octomap_topic": "/octomap_binary",
                    "output_topic": "/openral/world_voxels",
                    "resolution": 0.05,
                    # Graph-wide clock domain — matches octomap_server above
                    # (sim-time without a /clock pins its TF lookups at 0).
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.extend([octomap_server, octomap_bridge])

    if enable_object_detector:
        # ADR-0035 — the object-detection perception leg. The ROS-Image
        # detector runs RT-DETR over the agentview RGB tee and publishes
        # ObjectsMetadata to /openral/perception/objects. The world-state
        # node's object-lift (object_lift_enabled defaults True) subscribes
        # that topic, resolves the detection camera from the robot
        # description via ``sensor_id``, and raises 2-D boxes into the
        # /openral/world_voxels grid in the ``map`` frame. Purely additive:
        # the detector emits no Action chunks and the safety kernel never
        # sees its output. The COCO-80 label map is read from the
        # rtdetr-coco-r18 rSkill manifest at launch-build time so the node's
        # class indices map to the same names the model was exported with.
        # yaml is imported locally on purpose: a default (detector-off) launch
        # must never import yaml or read rskill.yaml, so the base graph stays
        # byte-for-byte unchanged. Do NOT hoist this import to the module top.
        import yaml

        # ADR-0035 cross-frame lift — detect on (and stamp the detection with)
        # the robot's first *liftable* RGB camera: one whose frame_id is a
        # dedicated ``*_optical_frame`` (the SimSensorBridge broadcasts its live
        # extrinsics, so the world-state lifter can project the world voxel map
        # into it). The detection's ``sensor_id`` MUST be that camera — not a
        # depth sensor — so the lifter resolves the right intrinsics/extrinsics.
        # Generic over robots; falls back to the historical ``agentview_left``.
        det_camera = "agentview_left"
        try:
            with pathlib.Path(robot_yaml).open(encoding="utf-8") as _rh:
                _robot_doc = yaml.safe_load(_rh) or {}
            for _s in _robot_doc.get("sensors", []):
                if _s.get("modality") == "rgb" and str(_s.get("frame_id", "")).endswith(
                    "_optical_frame"
                ):
                    det_camera = str(_s["name"])
                    break
        except (OSError, yaml.YAMLError):
            pass
        det_image_topic = f"/openral/cameras/{det_camera}/image"

        # Shared QoS / clock note: clock domain follows the graph-wide flag
        # (see _resolve_sim_clock). The node stamps its output from the input
        # Image's header.stamp; its ONLY use of self.get_clock() is the
        # max_rate_hz publish throttle. With no live /clock this stays
        # wall-clock — use_sim_time=True would pin get_clock().now() at 0 and
        # every frame is dropped at the rate gate → the detector never publishes.
        if object_detector_manifest:
            # ADR-0037 2026-06-09 — manifest-driven backend (RT-DETR ONNX or the
            # open-vocab LocateAnything VLM sidecar). The node loads labels /
            # model_id / contract from the manifest; we only forward the manifest
            # path, the (VLM-ignored) onnx override, and the query.
            with pathlib.Path(object_detector_manifest).open(encoding="utf-8") as handle:
                man = yaml.safe_load(handle) or {}
            # Throttle by the detector engine so the single-threaded callback never
            # backs up: the VLM sidecar (LocateAnything) is slow (~1-2 s / frame),
            # the in-process OmDet-Turbo zero-shot backend is ~hundreds of ms, and
            # the RT-DETR ONNX path is fast. ADR-0037 DetectorEngine.
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
        # ADR-0050 — managed lifecycle node: autostarted to ACTIVE (detector
        # loaded) like the rest of the graph, but the reasoner can DEACTIVATE it
        # via LifecycleTransitionTool to free the detector's VRAM before a
        # co-resident grab policy loads on an 8 GB GPU.
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

        # ADR-0056 — on-demand locator nodes: one per --object-detector-locator,
        # each serving its own namespaced /openral/perception/<alias>/locate_in_view
        # (the reasoner picks one via LocateInViewTool.detector). They share the
        # continuous detector's camera/topic; the node's mode wiring (ADR-0051)
        # makes them serve-only (no continuous publish leg). Throttle by engine.
        for spec in locator_specs:
            locator_rate_hz = {"vlm_sidecar": 0.5, "zeroshot_hf": 2.0}.get(spec["engine"], 5.0)
            locator_params = {
                "image_topic": det_image_topic,
                "sensor_id": det_camera,
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

    if enable_reward_monitor:
        # ADR-0057 — reward monitor runs PARALLEL to the VLA (not a lifecycle/VRAM
        # peer the reasoner frees before a policy; it stays co-active). Plain Node:
        # subscribes the agentview RGB stream, buffers a rolling window, auto-spawns
        # the Robometer NF4 sidecar from the manifest, and serves
        # /openral/perception/query_task_progress for the reasoner to poll.
        reward_manifest = reward_monitor_manifest or str(
            pathlib.Path(_RSKILLS_DIR) / "robometer-4b" / "rskill.yaml"
        )
        # Resolve the camera the monitor scores. An explicit override wins; else
        # default to the robot's first RGB camera from robot.yaml (the same camera
        # the VLA consumes), so the monitor "just works" across robots — falling
        # back to the historical agentview_left only if robot.yaml has none.
        reward_image_topic = reward_monitor_image_topic
        if reward_image_topic == "/openral/cameras/agentview_left/image":
            import yaml  # local: the base graph (no detector/reward) never imports it

            reward_camera = "agentview_left"
            try:
                with pathlib.Path(robot_yaml).open(encoding="utf-8") as _rh:
                    _rdoc = yaml.safe_load(_rh) or {}
                for _s in _rdoc.get("sensors", []):
                    if _s.get("modality") == "rgb" and _s.get("name"):
                        reward_camera = str(_s["name"])
                        break
            except (OSError, yaml.YAMLError):
                pass
            reward_image_topic = f"/openral/cameras/{reward_camera}/image"
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
                    "sidecar_port": int(reward_monitor_sidecar_port),
                    "use_sim_time": use_sim_time,
                }
            ],
            additional_env=otel_env,
            output="screen",
        )
        extra_nodes.append(reward_monitor)

    nodes: list = [
        safety_kernel,
        runtime,
        reasoner,
        prompt_router,
        hal,
        *extra_nodes,
    ]
    # Dashboard is opt-out (default on). ``openral deploy sim --no-dashboard``
    # threads ``enable_dashboard:=false`` to skip the spawn entirely —
    # useful for headless CI and avoids the
    # ``[Errno 98] address already in use`` collision that would occur
    # if a previous run's dashboard still holds the port.
    if enable_dashboard:
        nodes.insert(0, dashboard)

    # ADR-0059 — read-only Foxglove live-scene bridge. Off by default;
    # ``openral deploy sim --foxglove`` opts in.
    #
    # STALE-BRIDGE ORDERING (ADR-0059 decision 3, VERIFICATION.md "Stale-bridge
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
                    "topic_whitelist": BUCKET1_TOPIC_WHITELIST,
                    # Keep the upstream 10 MB send buffer for camera frames.
                    "send_buffer_limit": 10_000_000,
                    "max_qos_depth": 10,
                    "include_hidden": False,
                    # Graph-wide clock domain (see _resolve_sim_clock).
                    "use_sim_time": use_sim_time,
                }
            ],
        )
        nodes.append(TimerAction(period=5.0, actions=[foxglove_bridge_node]))

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
            "reset_to_pose_service",
            default_value="",
            description=(
                "Service the skill_runner calls before the first inference "
                "tick to snap the HAL's qpos to the rSkill starting pose."
            ),
        ),
        DeclareLaunchArgument(
            "approach_skill_id",
            default_value="",
            description=(
                "ADR-0053 — MoveIt approach rSkill URI (e.g. "
                "rskills/rskill-moveit-joints) the skill_runner dispatches to "
                "plan a collision-free motion to each skill's starting_pose. "
                "Empty = legacy ResetToPose snap."
            ),
        ),
        DeclareLaunchArgument(
            "dashboard_port",
            default_value="4318",
            description="OTLP/HTTP port for the dashboard child.",
        ),
        DeclareLaunchArgument(
            "reasoner_provider",
            default_value="ollama",
            description="OPENRAL_REASONER_LLM_PROVIDER for the reasoner node.",
        ),
        DeclareLaunchArgument(
            "reasoner_model",
            default_value="gemma4:31b-cloud",
            description="OPENRAL_REASONER_LLM_MODEL for the reasoner node.",
        ),
        DeclareLaunchArgument(
            "spatial_memory_path",
            default_value="",
            description=(
                "ADR-0039 — absolute path to a persisted ADR-0038 scene graph "
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
                "ADR-0038 — when true, the reasoner accumulates a durable "
                "SpatialMemory live from the ADR-0035 producer's "
                "WorldState.detected_objects (auto-creating an empty backend "
                "if no spatial_memory_path was preloaded), so recall_object "
                "recalls what the robot has actually seen. Default false."
            ),
        ),
        DeclareLaunchArgument(
            "hal_mode",
            default_value="sim",
            description=(
                "ADR-0036 — deploy path the reasoner's action-mode palette "
                "gate matches against: ``sim`` (digital-twin; the scene's "
                "robosuite OSC controller synthesises cartesian/OSC modes) "
                "admits cartesian skills, ``real`` admits only the robot's "
                "declared ``supported_control_modes``. ``openral deploy sim`` "
                "passes ``sim``; ``openral deploy run`` passes ``real``."
            ),
        ),
        DeclareLaunchArgument(
            "enable_sim_clock",
            default_value="false",
            description=(
                "Single source of truth for the whole graph's clock domain. "
                "When true, every node (HAL, Nav2, slam_toolbox, "
                "robot_state_publisher, octomap, detector) runs on "
                "``use_sim_time`` — only valid once a sim ``/clock`` is "
                "published. deploy-sim has no ``/clock`` publisher today, so "
                "the default is false (wall-clock graph-wide, matching the "
                "HAL's wall-clock /scan + TF). Setting this true without a "
                "/clock pins every node at t=0 → empty costmap → the base "
                "drives into obstacles (the nav-collision bug)."
            ),
        ),
        DeclareLaunchArgument(
            "enable_slam",
            default_value="false",
            description=(
                "ADR-0025 — bring up slam_toolbox as a Reasoner-managed "
                "background service. Auto-transitions to INACTIVE; the "
                "Reasoner promotes to ACTIVE via LifecycleTransitionTool. "
                "Requires ros-${ROS_DISTRO}-slam-toolbox and the "
                "openral_slam_bringup package built in the workspace."
            ),
        ),
        DeclareLaunchArgument(
            "enable_nav2",
            default_value="false",
            description=(
                "ADR-0025 — bring up the Nav2 navigation stack so the "
                "``OpenRAL/rskill-nav2-navigate-to-pose`` wrapped-action "
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
                "ADR-0030 — bring up the world-collision perception leg: "
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
                "ADR-0030/0035 — when False, the octomap perception leg still "
                "publishes /openral/world_voxels (so the world-state object-lift "
                "works), but the C++ safety kernel's capsule-vs-voxel check stays "
                "OFF (its --no-enable-octomap posture: envelope + self-collision "
                "only). Lets perception use the world map without the dense-scene "
                "false-positive E-stop. Default True preserves bundled ADR-0030. "
                "Never weakens the kernel below the --no-enable-octomap baseline."
            ),
        ),
        DeclareLaunchArgument(
            "octomap_cloud_topic",
            default_value="/openral/cameras/front_depth/points",
            description=(
                "Depth PointCloud2 topic octomap_server consumes "
                "(``cloud_in`` remap). Matches the HAL's depth publisher "
                "for the robot's depth SensorSpec."
            ),
        ),
        DeclareLaunchArgument(
            "enable_object_detector",
            default_value="false",
            description=(
                "ADR-0035 — bring up the ROS-Image object detector "
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
            "object_detector_onnx",
            default_value=str(pathlib.Path(_RSKILLS_DIR) / "rtdetr-coco-r18" / "model.onnx"),
            description=(
                "ADR-0035 — absolute path to the RT-DETR ONNX weights the "
                "object detector loads. Defaults to the in-tree "
                "rskills/rtdetr-coco-r18/model.onnx. Ignored unless "
                "enable_object_detector is true."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_manifest",
            default_value="",
            description=(
                "ADR-0037 2026-06-09 — path to a kind:detector rSkill manifest. "
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
                "ADR-0037 2026-06-09 — initial open-vocabulary query for a VLM "
                "detector (e.g. 'red mug'). Empty = the manifest's detector.labels "
                "default. Retarget live by publishing a std_msgs/String to "
                "/openral/perception/detector_query. Ignored by ONNX detectors."
            ),
        ),
        DeclareLaunchArgument(
            "enable_reward_monitor",
            default_value="false",
            description=(
                "ADR-0057 — bring up the Robometer reward monitor "
                "(openral_perception_ros/reward_monitor_node) PARALLEL to the VLA. "
                "It buffers the agentview RGB stream and serves "
                "/openral/perception/query_task_progress; the reasoner is told "
                "task_progress_available=True so its LLM may poll per-frame "
                "progress/success whenever it sees fit. Advisory-only — never "
                "actuates. Default off. Requires the openral_perception_ros package "
                "built and a provisioned Robometer sidecar venv "
                "(OPENRAL_ROBOMETER_SIDECAR_VENV); co-resident with a VLA needs ~3.3 GB "
                "free VRAM (use a small NF4 VLA on an 8 GB GPU)."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_manifest",
            default_value="",
            description=(
                "ADR-0057 — path to a kind:reward rSkill manifest. Empty defaults to "
                "the in-tree rskills/robometer-4b/rskill.yaml. weights_uri may be "
                "hf://org/repo or local:///abs/path (a pre-quantized NF4 checkpoint "
                "loaded directly as 4-bit). Ignored unless enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_task",
            default_value="",
            description=(
                "ADR-0057 — default task instruction the reward monitor scores when "
                "a query leaves task empty (e.g. the operator's task goal). The "
                "reasoner normally passes the active task per query. Ignored unless "
                "enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_image_topic",
            default_value="/openral/cameras/agentview_left/image",
            description=(
                "ADR-0057 — camera RGB topic the reward monitor buffers; must match "
                "the camera the co-active VLA consumes. Ignored unless "
                "enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "reward_monitor_sidecar_port",
            default_value="5769",
            description=(
                "ADR-0057 — ZMQ port for the Robometer reward sidecar the monitor "
                "auto-spawns. Ignored unless enable_reward_monitor."
            ),
        ),
        DeclareLaunchArgument(
            "object_detector_locators",
            default_value="",
            description=(
                "ADR-0056 — comma-separated kind:detector manifest paths for the "
                "on-demand open-vocab locators to bring up alongside the continuous "
                "detector. Each becomes a namespaced lifecycle node serving "
                "/openral/perception/<alias>/locate_in_view, selectable by the "
                "reasoner via LocateInViewTool.detector. Empty = no on-demand "
                "locator. Ignored unless enable_object_detector is true."
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
                "ADR-0059 — spawn the read-only foxglove_bridge as part of "
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
                "ADR-0059 — Foxglove WebSocket port "
                "(ws://127.0.0.1:<foxglove_port>). Default 8765. "
                "Ignored unless enable_foxglove is true."
            ),
        ),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=compose_runtime_graph)])
