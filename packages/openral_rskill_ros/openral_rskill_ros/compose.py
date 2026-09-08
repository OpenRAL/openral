"""Single-process composer for rskill_runner + world_state.

Locks the contract that ``WorldStateAggregator`` is the *only* subscriber of
``/joint_states`` and bridges it in-process via ``.snapshot()`` to the rskill — which
requires the world_state lifecycle node and the rskill_runner_node to share **one**
aggregator instance in the same OS process so the snapshot call never crosses a ROS topic
boundary.

:func:`compose_runtime` is the single function both production launches and integration
tests call. It loads the robot's ``RobotDescription`` from its on-disk ``robot.yaml``
(CLAUDE.md §1.11 — real manifests under ``robots/``, never a placeholder), constructs one
:class:`WorldStateAggregator`, hands the same instance by reference to
:class:`_WorldStateLifecycleNode` and :class:`RskillRunnerNode`, and returns both nodes for
the caller to attach to an ``rclpy.executors.MultiThreadedExecutor``.

Does **not** drive lifecycle transitions itself; the caller (launch file's ``runtime_node``
entry point or a test ``trigger_configure`` sequence) configures + activates after composing.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openral_core import RobotDescription
from openral_world_state import WorldStateAggregator

if TYPE_CHECKING:
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode

    from openral_rskill_ros.rskill_runner_node import RskillRunnerNode, SkillResolver

__all__ = ["ComposedRuntime", "compose_runtime", "compose_so100_runtime"]


@dataclass
class ComposedRuntime:
    """Bundle returned by :func:`compose_runtime`.

    Attributes:
        description: The :class:`RobotDescription` shared by both
            nodes.
        aggregator: The single :class:`WorldStateAggregator` shared
            in-process.
        world_state_node: The colocated
            :class:`_WorldStateLifecycleNode` (publishes the typed
            ``/openral/world_state_*`` topics).
        skill_runner_node: The colocated :class:`RskillRunnerNode` that
            owns the ``ExecuteRskill`` action server.
    """

    description: RobotDescription
    aggregator: WorldStateAggregator
    world_state_node: _WorldStateLifecycleNode
    skill_runner_node: RskillRunnerNode
    slam_bridge: object
    """rclpy → OTLP bridge subscribing to ``/map``.

    Always constructed with the runtime so the dashboard renders any compatible
    ``nav_msgs/OccupancyGrid`` publisher, regardless of whether the mapper was
    launched by OpenRAL or separately by the operator.
    """
    world_cloud_bridge: object | None = None
    """Optional rclpy → OTLP bridge subscribing to
    ``/octomap_point_cloud_centers``. Constructed when
    :func:`compose_runtime` is called with
    ``enable_world_cloud_bridge=True``. ``None`` otherwise. Production
    launches enable it through the same ``--enable-octomap`` CLI flag
    that brings up octomap_server itself."""
    dataset_recorder_bridge: object | None = None
    """Optional bus-attached recorder writing a rosbag2 mcap of
    the deploy session (proprio + action + camera frames + episode
    markers). Constructed when :func:`compose_runtime` is called with a
    ``dataset_out`` path. ``None`` otherwise. Production launches enable it
    through the ``openral deploy sim/run --dataset-out`` CLI flag. The
    caller must invoke ``.destroy()`` on teardown so the bag is finalized."""


def compose_runtime(
    robot_yaml: str | pathlib.Path,
    *,
    skill_resolver: SkillResolver | None = None,
    skill_resolver_factory: Callable[[Any], SkillResolver] | None = None,
    enable_world_cloud_bridge: bool = False,
    world_cloud_topic: str = "",
    slam_source_node: str = "",
    dataset_out: str | pathlib.Path | None = None,
    dataset_repo_id: str | None = None,
    dataset_license: str = "CC-BY-4.0",
    dataset_fps: float | None = None,
    image_staleness_limit_s: float | None = None,
) -> ComposedRuntime:
    """Build the composed world_state + skill_runner runtime for any robot.

    Args:
        robot_yaml: Path to a ``robots/<id>/robot.yaml``, loaded via
            :meth:`RobotDescription.from_yaml` (full Pydantic validation). Relative or
            absolute; ``runtime_node`` passes an absolute path from the ROS parameter.
        skill_resolver: Optional override of the default production skill resolver. Tests
            pass a local-only resolver to avoid HF Hub network access; ``None`` runs the
            default ``rSkill.from_pretrained``-shaped resolver.
        skill_resolver_factory: Optional factory ``(host_node) -> SkillResolver`` for a
            resolver that closes over the just-constructed ``RskillRunnerNode`` — required
            for wrapped-ROS skills whose adapter needs the host rclpy node to create
            per-skill ActionClients on the same spin. Mutually exclusive with
            ``skill_resolver``.
        enable_world_cloud_bridge: When ``True``, attach a
            :class:`~openral_runner.world_cloud_bridge.WorldCloudBridge` so the octomap
            occupied voxel cloud (``/octomap_point_cloud_centers``) renders into the
            dashboard via the ``world.pointcloud`` OTel span family. ``False`` (default)
            skips the subscription cost when octomap is off.
        world_cloud_topic: PointCloud2 topic the world-cloud bridge renders. Empty keeps
            octomap's latched centers default; the mono visual-SLAM launch points it at
            nvblox's (VOLATILE) ESDF voxel cloud.
        slam_source_node: Node name stamped on the dashboard SLAM card's
            ``openral.slam.source_node`` attribute. Empty keeps the ``openral_slam_toolbox``
            default; the visual-SLAM launch sets ``openral_nvblox`` when nvblox builds
            ``/map``.
        dataset_out: When set, attach a
            :class:`~openral_runner.dataset_recorder_bridge.DatasetRecorderBridge` recording
            the deploy session (proprio + action + camera frames + episode markers) to this
            rosbag2 ``.mcap`` file (single file, not a bag directory; parent must exist, file
            must not). Segmented by the ``/openral/episode`` markers an *executing* rSkill
            emits, so a session with no skill run writes nothing (reported at ``destroy()``).
            Caller must call ``runtime.dataset_recorder_bridge.destroy()`` on teardown to
            finalize the bag. ``None`` disables recording.
        dataset_repo_id: repo_id stamped into the recorded frames / eventual LeRobotDataset.
            Defaults to ``openral/dataset-<robot>``.
        dataset_license: SPDX license carried into the offline ``openral dataset from-bag``
            conversion. Defaults to ``CC-BY-4.0``.
        dataset_fps: Recording cadence. Defaults to the robot's
            ``action_spec.control_freq_hz`` or 30.0.
        image_staleness_limit_s: Camera-specific freshness window for the shared world-state
            aggregator. ``None`` keeps its general default.

    Returns:
        A :class:`ComposedRuntime` bundle. The caller attaches both nodes to a single
        ``rclpy.executors.MultiThreadedExecutor``, then drives the managed-lifecycle
        transitions.
    """
    # Deferred import — keeps the module import-safe on hosts without
    # rclpy (matches CLAUDE.md §1.11 / §5.4 "real component or skip").
    import rclpy
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode

    from openral_rskill_ros.rskill_runner_node import RskillRunnerNode

    description = RobotDescription.from_yaml(str(robot_yaml))
    aggregator = WorldStateAggregator(
        description,
        image_staleness_limit_s=image_staleness_limit_s,
    )
    world_state_node = _WorldStateLifecycleNode(aggregator=aggregator)
    # Override the world_state node's default ``robot_name`` parameter
    # so its /diagnostics ``hardware_id`` matches the composed runtime.
    world_state_node.set_parameters(
        [rclpy.parameter.Parameter("robot_name", value=description.name)],
    )
    # Resolver wiring: explicit ``skill_resolver`` wins (tests pass a
    # local-only resolver to dodge HF Hub); otherwise build the
    # production resolver via ``skill_resolver_factory`` after the
    # node exists so the closure captures the node handle. This is
    # the only way to construct a resolver that branches on
    # ``manifest.kind`` for wrapped-ROS skills (``ros_action`` /
    # ``ros_service``) — those need ``ros_node`` at construction time
    # to create the ActionClient against the same rclpy spin.
    if skill_resolver is None and skill_resolver_factory is not None:
        # Build a placeholder node-less factory the RskillRunnerNode
        # init will swap once it has constructed itself. The runner's
        # __init__ falls back to its internal ``make_default_skill_resolver(self)``
        # when ``skill_resolver`` is ``None`` — we override that
        # right after construction so the factory we passed wins.
        skill_runner_node = RskillRunnerNode(
            robot_description=description,
            aggregator=aggregator,
            skill_resolver=None,
        )
        # Deliberate compose-time private wire: RskillRunnerNode exposes
        # ``_skill_resolver`` for swap so the factory can close over the
        # just-constructed host node (wrapped-ROS skills need it).
        skill_runner_node._skill_resolver = skill_resolver_factory(skill_runner_node)
    else:
        skill_runner_node = RskillRunnerNode(
            robot_description=description,
            aggregator=aggregator,
            skill_resolver=skill_resolver,
        )
    # Share the RskillRunnerNode's executor so the /map
    # subscription's callbacks fire alongside the existing
    # /joint_states + /openral/estop subscriptions without a second rclpy spin.
    from openral_runner.slam_bridge import SlamMapBridge

    slam_bridge: object = SlamMapBridge(
        skill_runner_node,
        base_frame=description.base_frame,
        footprint_radius_m=description.footprint_radius,
        footprint_polygon=description.footprint_polygon,
        # Label the dashboard card with the node that actually built /map
        # (nvblox on the visual backend); empty keeps the slam_toolbox default.
        **({"source_node_name": slam_source_node} if slam_source_node else {}),
    )
    world_cloud_bridge: object | None = None
    if enable_world_cloud_bridge:
        # Share the RskillRunnerNode's executor so the
        # /octomap_point_cloud_centers subscription + TF listener spin
        # alongside the existing runner subscriptions, no second rclpy spin.
        from openral_runner.world_cloud_bridge import WorldCloudBridge

        # Default topic = octomap's occupied-voxel centers; a visual-SLAM deploy
        # points this at nvblox's ESDF voxel cloud so the same dashboard card
        # shows the vision-built voxels.
        # base_frame from the MANIFEST, like SlamMapBridge above. The default
        # is "base_link", which only exists on a mobile base; a fixed-base arm
        # names its own root (openarm_base, panda_link0, pelvis) and the bridge
        # then dropped every cloud on a TF lookup while the graph reported
        # healthy and the dashboard card stayed empty.
        if world_cloud_topic:
            world_cloud_bridge = WorldCloudBridge(
                skill_runner_node,
                topic=world_cloud_topic,
                base_frame=description.base_frame,
                source_node_name="openral_nvblox",
                latched=False,  # nvblox publishes its ESDF cloud VOLATILE
            )
        else:
            world_cloud_bridge = WorldCloudBridge(
                skill_runner_node, base_frame=description.base_frame
            )
    dataset_recorder_bridge: object | None = None
    if dataset_out is not None:
        # Attach a bus recorder sharing the runner's executor +
        # aggregator. Robot-agnostic: every shape is derived from the bus
        # data + this ``description`` (no observation_spec dependency,
        # since Rosbag2Sink writes raw arrays — `openral dataset from-bag`
        # materialises the LeRobotDataset offline).
        from openral_dataset import RolloutRecorder, Rosbag2Sink
        from openral_runner.dataset_recorder_bridge import DatasetRecorderBridge

        action_spec = getattr(description, "action_spec", None)
        fps = (
            float(dataset_fps)
            if dataset_fps
            else (
                float(action_spec.control_freq_hz)
                if action_spec is not None and action_spec.control_freq_hz
                else 30.0
            )
        )
        recorder = RolloutRecorder(
            robot=description,
            task_string="",
            fps=fps,
            sinks=[Rosbag2Sink(bag_path=dataset_out)],
            repo_id=dataset_repo_id or f"openral/dataset-{description.name}",
        )
        dataset_recorder_bridge = DatasetRecorderBridge(
            skill_runner_node,
            robot=description,
            aggregator=aggregator,
            recorder=recorder,
            output_path=str(dataset_out),
        )
        del dataset_license  # carried into the bag's LeRobot conversion via `from-bag`
    return ComposedRuntime(
        description=description,
        aggregator=aggregator,
        world_state_node=world_state_node,
        skill_runner_node=skill_runner_node,
        slam_bridge=slam_bridge,
        world_cloud_bridge=world_cloud_bridge,
        dataset_recorder_bridge=dataset_recorder_bridge,
    )


def compose_so100_runtime(
    *,
    skill_resolver: SkillResolver | None = None,
) -> ComposedRuntime:
    """SO-100 convenience wrapper around :func:`compose_runtime`.

    Resolves the in-tree ``robots/so100_follower/robot.yaml`` relative
    to this module so callers (tests, scripts) do not need to thread a
    repo-root path through. Production launches use
    :func:`compose_runtime` directly via the ``runtime_node`` entry
    point with a ROS-parameter-supplied ``robot_yaml``.

    Args:
        skill_resolver: Optional skill resolver override (see
            :func:`compose_runtime`).
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    return compose_runtime(
        repo_root / "robots" / "so100_follower" / "robot.yaml",
        skill_resolver=skill_resolver,
    )
