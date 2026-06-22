"""ADR-0019 — bus-attached LeRobot/rosbag recorder for the deploy graph.

:class:`DatasetRecorderBridge` is the deploy-side counterpart to the
``SimRunner`` recording path. It mirrors
:class:`openral_runner.world_cloud_bridge.WorldCloudBridge`: constructed
against an existing ``rclpy.node.Node`` so its subscriptions share the
runtime's executor (no second spin), and it owns no actuation logic — it
only *observes* the bus the ``rskill_runner_node`` already publishes.

Per tick it joins three already-on-the-graph signals into one
:class:`openral_dataset.DatasetFrame`:

* **proprioception + camera frames** — read from the shared
  :class:`~openral_world_state.WorldStateAggregator` snapshot
  (``joint_state`` + ``image_frames``), the same in-process snapshot the
  runner feeds the policy. No separate camera-topic publisher is required.
* **action** — the already-flattened ``openral_msgs/ActionChunk.flat``
  the HAL publishes on ``/openral/candidate_action`` (first ``n_dof`` row =
  the next-applied action). This dodges per-control-mode ``Action`` field
  extraction entirely.
* **episode boundaries** — ``openral_msgs/Episode`` PHASE_START / PHASE_END
  markers the ``rskill_runner_node`` publishes around each ``ExecuteRskill``
  goal.

The bridge is **embodiment-agnostic**: every shape is derived from the
recorded data + the injected ``RobotDescription`` — nothing here is
robot-specific. It writes through a :class:`openral_dataset.Rosbag2Sink`
(no ``features_from_robot`` / ``observation_spec`` requirement at record
time), so it works for robots whose proprio layout lives only in the
active rSkill's ``state_contract`` rather than ``RobotDescription``;
``openral dataset from-bag`` materialises the LeRobotDataset offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from openral_core import RobotDescription
    from openral_dataset import RolloutRecorder
    from openral_world_state.aggregator import WorldStateAggregator

__all__ = ["DatasetRecorderBridge"]

_log = structlog.get_logger(__name__)

# Episode.phase enum — mirrors packages/msgs/msg/Episode.msg.
_PHASE_START = 0
_PHASE_END = 1

ACTION_TOPIC_DEFAULT = "/openral/candidate_action"
EPISODE_TOPIC_DEFAULT = "/openral/episode"


def _sensor_name_to_slot(description: RobotDescription | None) -> dict[str, str]:
    """Map each RGB sensor NAME to its VLA slot (``camera1`` / ``camera2`` / ...).

    The aggregator keys ``image_frames`` by sensor name; the dataset sink
    keys images by the slot (``vla_feature_key`` suffix). Mirrors
    ``rskill_runner_node._sensor_name_to_vla_slot`` but kept local so the
    Layer-5 runner package does not import the Layer-3 ROS skill package.
    """
    if description is None:
        return {}
    out: dict[str, str] = {}
    for sensor in description.sensors:
        if getattr(sensor, "modality", None) != "rgb":
            continue
        vfk = getattr(sensor, "vla_feature_key", None)
        out[sensor.name] = str(vfk).rsplit(".", 1)[-1] if vfk else sensor.name
    return out


class DatasetRecorderBridge:
    """rclpy → :class:`RolloutRecorder` bridge for the deploy graph.

    Args:
        node: Host ``rclpy.node.Node``; subscriptions are created on it so
            they share the runtime executor. :meth:`destroy` releases them.
        robot: The runtime's :class:`RobotDescription` (sensor → slot map).
        aggregator: The shared :class:`WorldStateAggregator` the runner
            feeds; read (never written) for proprio + image frames.
        recorder: A configured :class:`openral_dataset.RolloutRecorder`
            (typically fronting a :class:`openral_dataset.Rosbag2Sink`).
        action_topic: ``ActionChunk`` topic. Defaults to
            :data:`ACTION_TOPIC_DEFAULT`.
        episode_topic: ``Episode`` marker topic. Defaults to
            :data:`EPISODE_TOPIC_DEFAULT`.
    """

    def __init__(
        self,
        node: Any,  # noqa: ANN401  # reason: rclpy.node.Node not importable without a sourced ROS 2 workspace
        *,
        robot: RobotDescription,
        aggregator: WorldStateAggregator,
        recorder: RolloutRecorder,
        action_topic: str = ACTION_TOPIC_DEFAULT,
        episode_topic: str = EPISODE_TOPIC_DEFAULT,
    ) -> None:
        """Subscribe to the action + episode topics on the shared node."""
        from openral_msgs.msg import ActionChunk, Episode  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        self._node = node
        self._robot = robot
        self._aggregator = aggregator
        self._recorder = recorder
        self._sensor_to_slot = _sensor_name_to_slot(robot)
        self._episode_open = False
        self._n_frames = 0

        # Control data class — RELIABLE / KEEP_LAST=1, matching the
        # ROSPublishingHAL candidate_action publisher (CLAUDE.md §2 QoS).
        action_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Episode markers are sparse + must not be dropped — KEEP_LAST=10.
        episode_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._episode_sub: Any = node.create_subscription(
            Episode, episode_topic, self._on_episode, episode_qos
        )
        self._action_sub: Any = node.create_subscription(
            ActionChunk, action_topic, self._on_action, action_qos
        )

    def destroy(self) -> None:
        """Close any open episode, finalize the recorder, release subs. Idempotent."""
        if self._episode_open:
            # Open episode at teardown → mark failure (deactivate-with-open
            # contract, same as the offline converter's EOF handling).
            try:
                self._recorder.episode_end(success=False)
            except Exception:  # reason: teardown must not raise
                _log.exception("dataset_recorder_bridge.episode_end_failed")
            self._episode_open = False
        try:
            self._recorder.finalize()
        except Exception:  # reason: teardown must not raise
            _log.exception("dataset_recorder_bridge.finalize_failed")
        for sub in (self._action_sub, self._episode_sub):
            if sub is not None:
                self._node.destroy_subscription(sub)
        self._action_sub = None
        self._episode_sub = None

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_episode(self, msg: Any) -> None:  # noqa: ANN401  # reason: openral_msgs/Episode IDL
        """Open/close a recorder episode on PHASE_START / PHASE_END."""
        phase = int(msg.phase)
        if phase == _PHASE_START:
            if self._episode_open:
                # Stray re-open (missed PHASE_END) → close the previous as a
                # failure before starting the new one.
                self._recorder.episode_end(success=False)
            self._recorder.episode_start(task_string=str(msg.task_string))
            self._episode_open = True
            self._n_frames = 0
        elif phase == _PHASE_END:
            if not self._episode_open:
                return
            self._recorder.episode_end(success=bool(msg.success))
            self._episode_open = False
            _log.debug(
                "dataset_recorder_bridge.episode_closed",
                success=bool(msg.success),
                n_frames=self._n_frames,
            )

    def _on_action(self, msg: Any) -> None:  # noqa: ANN401  # reason: openral_msgs/ActionChunk IDL
        """Join the latest snapshot with this action chunk → one frame."""
        if not self._episode_open:
            return
        snapshot = self._aggregator.snapshot()
        joint_state = getattr(snapshot, "joint_state", None)
        if joint_state is None or not joint_state.position:
            return  # no proprio yet — skip until the aggregator is warm
        state = np.asarray(joint_state.position, dtype=np.float32)

        # First row of the chunk = the next-applied action (row-major
        # [horizon][n_dof]). n_dof guards against an empty/short flat.
        n_dof = int(getattr(msg, "n_dof", 0)) or len(msg.flat)
        action = np.asarray(list(msg.flat[:n_dof]), dtype=np.float32)

        images = self._decode_images(getattr(snapshot, "image_frames", None))
        self._recorder.record_frame(
            observation_state=state,
            images=images,
            action=action,
        )
        self._n_frames += 1

    def _decode_images(self, image_frames: Any) -> dict[str, np.ndarray[Any, Any]]:  # noqa: ANN401
        """Decode aggregator ``image_frames`` (sensor-name keyed) → slot-keyed HWC arrays."""
        out: dict[str, np.ndarray[Any, Any]] = {}
        if not image_frames:
            return out
        for name, frame in image_frames.items():
            data = getattr(frame, "data", None)
            if data is None:
                continue  # topic/handle delivery — no inline pixels to record
            arr = np.frombuffer(data, dtype=np.uint8).reshape(
                int(frame.height), int(frame.width), int(frame.channels)
            )
            out[self._sensor_to_slot.get(name, name)] = arr
        return out
