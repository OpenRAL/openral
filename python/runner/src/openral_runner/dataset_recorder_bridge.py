"""Bus-attached LeRobot/rosbag recorder for the deploy graph.

``DatasetRecorderBridge`` is the deploy-side counterpart to the
``SimRunner`` recording path. It mirrors
``openral_runner.world_cloud_bridge.WorldCloudBridge``: constructed
against an existing ``rclpy.node.Node`` so its subscriptions share the
runtime's executor (no second spin), and it owns no actuation logic — it
only *observes* the bus the ``rskill_runner_node`` already publishes.

Per tick it joins three already-on-the-graph signals into one
``openral_dataset.DatasetFrame``:

* **proprioception + camera frames** — read from the shared
  ``WorldStateAggregator`` snapshot
  (``joint_state`` + ``image_frames``), the same in-process snapshot the
  runner feeds the policy. No separate camera-topic publisher is required.
* **action** — the per-tick action, reassembled from the
  ``openral_msgs/ActionChunk`` stream on ``/openral/candidate_action``. For
  slot-dispatched skills the node emits one chunk per slot per
  tick in a fixed order (e.g. LIBERO = 6-D cartesian_delta + 1-D gripper;
  RoboCasa composite = cartesian + gripper + body_twist); the bridge
  accumulates a tick's slots and concatenates their next-applied rows
  (``flat[:n_dof]``) into one full action vector, detecting the tick boundary
  by the slot cycle (a repeated ``(control_mode, ee_name)`` key starts the
  next tick). Single-``ActionChunk`` skills (joint-position robots) flush one
  frame per chunk. Robot- and control-mode-agnostic — no per-mode ``Action``
  field extraction (the HAL already flattened each slot into ``flat``).
* **episode boundaries** — ``openral_msgs/Episode`` PHASE_START / PHASE_END
  markers the ``rskill_runner_node`` publishes around each ``ExecuteRskill``
  goal.

The bridge is **embodiment-agnostic**: every shape is derived from the
recorded data + the injected ``RobotDescription`` — nothing here is
robot-specific. It writes through a ``openral_dataset.Rosbag2Sink``
(no ``features_from_robot`` / ``observation_spec`` requirement at record
time), so it works for robots whose proprio layout lives only in the
active rSkill's ``state_contract`` rather than ``RobotDescription``;
``openral dataset from-bag`` materialises the LeRobotDataset offline.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from openral_core import sensor_name_to_slot
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import FrameEncoding, SensorFrame

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


# Raw inline layouts the aggregator hands out, by element dtype. ``JPEG`` /
# ``PNG`` are decoded with Pillow; device-handle (CUDA_*) encodings carry no
# host pixel run and are skipped (logged) by ``decode_inline_frame``.
_INLINE_FRAME_DTYPES: dict[FrameEncoding, Any] = {
    FrameEncoding.BGR8: np.uint8,
    FrameEncoding.RGB8: np.uint8,
    FrameEncoding.MONO8: np.uint8,
    FrameEncoding.RAW: np.uint8,
    FrameEncoding.DEPTH16: np.uint16,
}
_COMPRESSED_ENCODINGS = frozenset({FrameEncoding.JPEG, FrameEncoding.PNG})


# What `DatasetRecorder.record_frame` accepts per camera key: `(H, W, 3) uint8`.
_HWC_NDIM = 3
_RGB_CHANNELS = 3


def _decode_compressed(frame: SensorFrame, data: bytes) -> np.ndarray[Any, Any] | str:
    """Decode a JPEG/PNG payload to ``HxWxC`` uint8 RGB (or mono), else a skip reason."""
    try:
        from PIL import Image  # noqa: PLC0415  # reason: lazy — only compressed frames pay it
    except ImportError:
        return "pillow_not_installed"
    try:
        with Image.open(io.BytesIO(data)) as img:
            decoded = np.asarray(img.convert("L" if frame.channels == 1 else "RGB"))
    except (OSError, ValueError) as exc:  # reason: PIL raises these on corrupt/unknown data
        return f"undecodable: {exc}"
    if decoded.shape[:2] != (frame.height, frame.width):
        got = f"{decoded.shape[1]}x{decoded.shape[0]}"
        return f"decoded {got} != declared {frame.width}x{frame.height}"
    return decoded.reshape(frame.height, frame.width, -1)


def decode_inline_frame(frame: SensorFrame) -> np.ndarray[Any, Any] | None:
    """Decode one ``SensorFrame`` with inline ``data`` into an ``HxWxC`` array.

    The element dtype comes from ``frame.encoding``: ``DEPTH16`` is uint16
    millimetres, the 8-bit colour/mono layouts are uint8. Reading everything
    as uint8 was what aborted the first real-hardware OpenArm dispatch
    (qorin1, 2026-09-22): the ZED depth sensor's ``16UC1`` frame is
    1280x720x1 at two bytes per pixel, and ``reshape(720, 1280, 1)`` on a
    uint8 view of it raised ``ValueError: cannot reshape array of size
    1843200`` while the policy only ever wanted the RGB slots next to it.
    ``JPEG`` / ``PNG`` payloads (an MJPEG USB camera) are decoded with Pillow
    to RGB (``channels == 3``) or mono (``channels == 1``) uint8.

    Returns ``None`` silently for a frame with no inline pixels (topic /
    handle delivery — not a skip). Returns ``None`` and logs
    ``runner.frame_skipped`` (sensor, encoding, reason) for a payload it cannot
    decode: a device-handle encoding carrying inline bytes, a corrupt or
    wrongly-sized compressed image, or a raw payload whose byte count is not
    exactly ``height * width * channels * itemsize`` (a truncated or row-padded
    frame — ``SensorFrame`` validates the dimensions but not the payload
    length). Callers decide whether a skip is fatal (the skill runner raises
    when the sensor feeds a required policy slot).
    """
    data = frame.data
    if data is None:
        return None
    if frame.encoding in _COMPRESSED_ENCODINGS:
        decoded = _decode_compressed(frame, data)
        if isinstance(decoded, str):
            _skip(frame, decoded)
            return None
        return decoded
    dtype = _INLINE_FRAME_DTYPES.get(frame.encoding)
    if dtype is None:
        _skip(frame, "no host pixel layout for this encoding")
        return None
    shape = (int(frame.height), int(frame.width), int(frame.channels))
    expected = shape[0] * shape[1] * shape[2] * np.dtype(dtype).itemsize
    if len(data) != expected:
        _skip(frame, f"payload {len(data)} B != expected {expected} B")
        return None
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _skip(frame: SensorFrame, reason: str) -> None:
    # ponytail: logs every skipped frame; rate-limit if a non-required sensor
    # ever skips steadily (a required one raises in the runner instead).
    _log.warning(
        "runner.frame_skipped",
        sensor=frame.sensor_id,
        encoding=frame.encoding.value,
        reason=reason,
    )


class DatasetRecorderBridge:
    """rclpy → ``RolloutRecorder`` bridge for the deploy graph.

    Args:
        node: Host ``rclpy.node.Node``; subscriptions are created on it so
            they share the runtime executor. ``destroy`` releases them.
        robot: The runtime's ``RobotDescription`` (sensor → slot map).
        aggregator: The shared ``WorldStateAggregator`` the runner
            feeds; read (never written) for proprio + image frames.
        recorder: A configured ``openral_dataset.RolloutRecorder``
            (typically fronting a ``openral_dataset.Rosbag2Sink``).
        output_path: Where the recorder's sink writes, echoed in the
            armed / summary log lines so the operator can find (or fail to
            find) the artefact by path. Purely informational.
        action_topic: ``ActionChunk`` topic. Defaults to
            ``ACTION_TOPIC_DEFAULT``.
        episode_topic: ``Episode`` marker topic. Defaults to
            ``EPISODE_TOPIC_DEFAULT``.
    """

    def __init__(
        self,
        node: Any,  # noqa: ANN401  # reason: rclpy.node.Node not importable without a sourced ROS 2 workspace
        *,
        robot: RobotDescription,
        aggregator: WorldStateAggregator,
        recorder: RolloutRecorder,
        output_path: str | None = None,
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
        self._output_path = output_path
        self._sensor_to_slot = sensor_name_to_slot(robot)
        self._episode_open = False
        self._n_frames = 0
        # Run totals, reported at destroy() — including the zero case, which
        # is the whole point: a session where no rSkill ever executed opens no
        # episode and writes no file, and used to say nothing at all.
        self._n_episodes = 0
        self._n_frames_total = 0
        # Per-tick action reassembly (slot dispatch): the node emits
        # one ActionChunk per slot per tick (e.g. LIBERO = cartesian_delta +
        # gripper; RoboCasa composite = cartesian + gripper + body_twist), in a
        # fixed slot order. We accumulate the slots of the current tick and
        # concatenate them into one full action vector; a repeated slot key
        # signals the next tick's cycle has started → flush the assembled frame.
        self._accum: list[tuple[tuple[int, str], list[float]]] = []
        self._pending_snapshot: Any = None
        self._current_tick: int = 0  # tick_index of the accumulating group (0 = none)

        # Control data class — RELIABLE. Deep queue (not KEEP_LAST=1) so a
        # tick's rapid multi-slot ActionChunk burst is never coalesced/dropped
        # before the reassembler sees every slot.
        action_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
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
        # Say so BEFORE the run: recording is armed, and it is contingent on a
        # skill actually reaching execution (the only publisher of the episode
        # markers we segment on). Without this the dependency is invisible
        # until the operator goes looking for a file that was never written.
        _log.info(
            "dataset_recorder.armed",
            output_path=self._output_path,
            episode_topic=episode_topic,
            hint=(
                "recording starts at the first executing rSkill "
                f"({episode_topic} PHASE_START); a run where every dispatch is "
                "rejected records nothing"
            ),
        )

    def destroy(self) -> None:
        """Close any open episode, finalize the recorder, release subs.

        Idempotent — a second call returns before re-finalizing or
        re-emitting the summary.
        """
        if self._action_sub is None and self._episode_sub is None:
            return
        if self._episode_open:
            # Open episode at teardown → mark failure (deactivate-with-open
            # contract, same as the offline converter's EOF handling).
            try:
                self._flush_accumulated()  # commit any pending tick first
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
        # Always report — the zero case is the one an operator most needs to
        # see, and it is exactly the case that used to exit silently.
        if self._n_episodes:
            _log.info(
                "dataset_recorder.summary",
                output_path=self._output_path,
                n_episodes=self._n_episodes,
                n_frames=self._n_frames_total,
            )
        else:
            _log.warning(
                "dataset_recorder.nothing_recorded",
                output_path=self._output_path,
                n_episodes=0,
                n_frames=0,
                hint=(
                    "no rSkill reached execution, so no episode marker ever "
                    "fired and no file was written — check the "
                    "rskill_runner.goal_rejected / goal_failed logs above"
                ),
            )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_episode(self, msg: Any) -> None:  # noqa: ANN401  # reason: openral_msgs/Episode IDL
        """Open/close a recorder episode on PHASE_START / PHASE_END."""
        phase = int(msg.phase)
        if phase == _PHASE_START:
            if self._episode_open:
                # Stray re-open (missed PHASE_END) → flush + close the previous
                # as a failure before starting the new one.
                self._flush_accumulated()
                self._recorder.episode_end(success=False)
            # Drop any half-accumulated tick from a prior episode.
            self._accum = []
            self._pending_snapshot = None
            self._current_tick = 0
            self._recorder.episode_start(task_string=str(msg.task_string))
            self._episode_open = True
            self._n_frames = 0
            self._n_episodes += 1
        elif phase == _PHASE_END:
            if not self._episode_open:
                return
            self._flush_accumulated()  # commit the episode's final tick
            self._recorder.episode_end(success=bool(msg.success))
            self._episode_open = False
            _log.debug(
                "dataset_recorder_bridge.episode_closed",
                success=bool(msg.success),
                n_frames=self._n_frames,
            )

    def _on_action(self, msg: Any) -> None:  # noqa: ANN401  # reason: openral_msgs/ActionChunk IDL
        """Accumulate this tick's slot chunks; flush a full frame per tick.

        Tick boundary detection, in order of preference:

        1. **``ActionChunk.tick_index``** (set by the node) — the
           authoritative, unambiguous key: every slot chunk of one inference
           tick carries the same 1-based index, so a change of index ends the
           tick. Robust even if two slots share a ``(control_mode, ee_name)``.
        2. **Slot cycle** (fallback when ``tick_index == 0``, e.g. an older
           publisher) — a repeated ``(control_mode, ee_name)`` key signals the
           next tick's slot order has restarted.

        The flushed frame concatenates all slots' next-applied rows
        (``flat[:n_dof]``) in emission order into one full action vector.
        """
        if not self._episode_open:
            return
        tick_index = int(getattr(msg, "tick_index", 0) or 0)
        slot_key = (int(getattr(msg, "control_mode", 0)), str(getattr(msg, "ee_name", "") or ""))
        # First row of the chunk = the next-applied action (row-major
        # [horizon][n_dof]). n_dof guards against an empty/short flat.
        n_dof = int(getattr(msg, "n_dof", 0)) or len(msg.flat)
        values = [float(v) for v in msg.flat[:n_dof]]

        if tick_index >= 1:
            boundary = bool(self._accum) and tick_index != self._current_tick
        else:
            boundary = any(key == slot_key for key, _ in self._accum)
        if boundary:
            self._flush_accumulated()
        if not self._accum:
            # First slot of a new tick — capture the proprio + images the
            # policy saw for THIS tick (chunks of a tick arrive within µs, so
            # the snapshot at the first slot is the tick's observation).
            self._current_tick = tick_index
            self._pending_snapshot = self._aggregator.snapshot()
        self._accum.append((slot_key, values))

    def _flush_accumulated(self) -> None:
        """Record one frame from the accumulated tick (concatenated slots)."""
        accum, snapshot = self._accum, self._pending_snapshot
        self._accum = []
        self._pending_snapshot = None
        self._current_tick = 0
        if not accum or snapshot is None:
            return
        joint_state = getattr(snapshot, "joint_state", None)
        if joint_state is None or not joint_state.position:
            return  # no proprio for this tick — drop it
        state = np.asarray(joint_state.position, dtype=np.float32)
        action = np.asarray([v for _key, vals in accum for v in vals], dtype=np.float32)
        images = self._decode_images(getattr(snapshot, "image_frames", None))
        try:
            self._recorder.record_frame(
                observation_state=state,
                images=images,
                action=action,
            )
        except (ValueError, ROSConfigError) as exc:
            # A per-frame shape mismatch (e.g. a robot that gained a defined
            # action_spec.dim AND runs a multi-slot skill whose slots sum to a
            # different dim — none today). Surface it
            # loudly + stop recording this episode rather than crash the
            # shared executor or silently mis-record (CLAUDE.md §1.4).
            _log.error(
                "dataset_recorder_bridge.record_frame_rejected",
                error=str(exc),
                action_dim=int(action.shape[0]),
                state_dim=int(state.shape[0]),
                slot_keys=[k for k, _ in accum],
                hint="reassembled action/state shape does not match the robot's spec",
            )
            return
        self._n_frames += 1
        self._n_frames_total += 1

    def _decode_images(self, image_frames: Any) -> dict[str, np.ndarray[Any, Any]]:  # noqa: ANN401
        """Decode aggregator ``image_frames`` (sensor-name keyed) → slot-keyed HWC arrays."""
        out: dict[str, np.ndarray[Any, Any]] = {}
        if not image_frames:
            return out
        for name, frame in image_frames.items():
            arr = decode_inline_frame(frame)
            if arr is None:
                continue  # topic/handle delivery, or a skip `decode_inline_frame` logged
            # `DatasetRecorder.record_frame` takes `(H, W, 3) uint8` RGB per
            # camera key; a DEPTH16 or mono frame in the same world state (the
            # OpenArm bench's `head_zed`) is not a dataset image feature and
            # would be rejected per frame, stopping the episode.
            if arr.dtype != np.uint8 or arr.ndim != _HWC_NDIM or arr.shape[2] != _RGB_CHANNELS:
                continue
            out[self._sensor_to_slot.get(name, name)] = arr
        return out
