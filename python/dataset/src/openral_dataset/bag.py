"""Rosbag2Sink — mcap-backed :class:`DatasetSink` for online hardware recording.

ADR-0019 PR3. Writes every :class:`RolloutRecorder` event into a
``.mcap`` file that the offline :class:`Rosbag2ToLeRobotConverter`
(PR4) replays into a LeRobotDataset v3. The same file is readable by
``ros2 bag info`` / ``mcap-cli`` / Foxglove / any rosbag2-mcap consumer
because mcap is the file format — we just use it without going through
the ``rosbag2_py`` Python wrapper.

Why mcap directly and not ``rosbag2_py``:

* ``mcap`` is a PyPI library; ``rosbag2_py`` is a ROS 2 system package
  shipped via apt. Using the bare mcap library lets the sink work on
  hosts that haven't sourced ROS 2 (developer laptops, CI runners) and
  makes the unit tests runnable end-to-end without ROS infrastructure.
* The on-disk format is identical — ``rosbag2`` with the mcap storage
  backend writes the same mcap stream that this sink writes. The
  ``ros2 bag info`` tooling reads it back without complaint.
* Schema encoding is ``jsonschema`` (one of mcap's canonical
  encodings), not ``ros2msg`` IDL. This means a future ROS-side
  publisher can ALSO subscribe to these topics and write the SAME
  messages with the ``ros2msg`` encoding alongside ours; consumers
  switch on schema encoding. PR4's converter accepts either.
* Hot-path safety — the writer thread is a daemon with a bounded
  ``queue.Queue``; ``write_frame`` enqueues only. The sink never blocks
  the inference tick on disk I/O.

Topics written:

* ``/openral/tick`` — :class:`openral_msgs.msg.Tick` per-tick metadata
  (episode_idx, step_idx, reward, terminated, truncated, trace_id).
* ``/openral/episode`` — :class:`openral_msgs.msg.Episode` markers
  emitted at episode_start and episode_end (phase=0 / phase=1).

Per CLAUDE.md §1.11 (no mocks) — tests exercise a real
:class:`mcap.writer.Writer` against a tmp_path and re-read with a real
:class:`mcap.reader.make_reader`.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError

from openral_dataset.recorder import DatasetSink

if TYPE_CHECKING:
    from openral_dataset.recorder import DatasetFrame, EpisodeHeader, EpisodeSummary

__all__ = ["Rosbag2Sink"]

_log = structlog.get_logger(__name__)

# Topic names — kept module-private constants so PR4's converter can
# import them by symbol rather than re-typing the strings. A typo would
# silently produce a bag that the converter rejects with
# ROSPagrConfigError("bag has no /openral/episode markers").
TOPIC_TICK: Final[str] = "/openral/tick"
TOPIC_EPISODE: Final[str] = "/openral/episode"
# Per-camera image frames (ADR-0019 PR4-follow-up). One message per
# (episode, step, camera) carrying the inline HWC uint8 pixels so the
# converter can rebuild a video-bearing LeRobotDataset from the bag
# alone — no separate `/joint_states` / camera-topic join required.
TOPIC_IMAGE: Final[str] = "/openral/dataset/image"

# JSON-encoded schemas. The format is mcap's `jsonschema` encoding —
# any mcap reader / Foxglove / mcap-cli decodes them without ROS 2.
# Field names mirror packages/msgs/msg/{Tick,Episode}.msg verbatim so a
# future ROS-side publisher writes the SAME schema with `ros2msg`
# encoding (the converter accepts either).
_TICK_SCHEMA: Final[dict[str, Any]] = {
    "title": "openral_msgs/Tick",
    "type": "object",
    "properties": {
        "stamp_ns": {"type": "integer"},
        "tick_idx": {"type": "integer"},
        "episode_idx": {"type": "integer"},
        "step_idx": {"type": "integer"},
        "reward": {"type": "number"},
        "terminated": {"type": "boolean"},
        "truncated": {"type": "boolean"},
        "action_applied": {"type": "boolean"},
        "trace_id": {"type": "string"},
        "span_id": {"type": "string"},
        # Inline observation/action arrays (ADR-0019 PR4-follow-up). Absent
        # on legacy metadata-only bags — the converter falls back to a
        # zero vector of the robot's declared shape when missing.
        "observation_state": {"type": "array", "items": {"type": "number"}},
        "action": {"type": "array", "items": {"type": "number"}},
    },
    "required": [
        "stamp_ns",
        "episode_idx",
        "step_idx",
        "terminated",
        "truncated",
    ],
}

# Per-camera image frame schema. Pixels are carried as a base64-encoded
# raw HWC uint8 buffer (``encoding="raw_u8"``) so the message stays a
# self-describing JSON object readable by any mcap reader without ROS or
# an image codec; the converter decodes it back to ``(H, W, C) uint8``.
_IMAGE_SCHEMA: Final[dict[str, Any]] = {
    "title": "openral_msgs/DatasetImage",
    "type": "object",
    "properties": {
        "stamp_ns": {"type": "integer"},
        "episode_idx": {"type": "integer"},
        "step_idx": {"type": "integer"},
        "camera": {"type": "string"},
        "height": {"type": "integer"},
        "width": {"type": "integer"},
        "channels": {"type": "integer"},
        "encoding": {"type": "string"},
        "data_b64": {"type": "string"},
    },
    "required": [
        "episode_idx",
        "step_idx",
        "camera",
        "height",
        "width",
        "channels",
        "encoding",
        "data_b64",
    ],
}

_EPISODE_SCHEMA: Final[dict[str, Any]] = {
    "title": "openral_msgs/Episode",
    "type": "object",
    "properties": {
        "stamp_ns": {"type": "integer"},
        "episode_idx": {"type": "integer"},
        "task_string": {"type": "string"},
        "phase": {"type": "integer", "enum": [0, 1]},
        "success": {"type": "boolean"},
    },
    "required": ["stamp_ns", "episode_idx", "task_string", "phase"],
}

# Episode.phase constants — must match packages/msgs/msg/Episode.msg.
PHASE_START: Final[int] = 0
PHASE_END: Final[int] = 1

# Image array rank thresholds (HWC convention) — named so the shape
# probe in `_enqueue_image` stays lint-clean (no magic-value comparison).
_NDIM_H: Final[int] = 1
_NDIM_HW: Final[int] = 2
_NDIM_HWC: Final[int] = 3

# Bounded writer queue. 1024 is generous — at 30 Hz tick rate that's
# >30 s of slack while the writer thread catches up. A full queue logs
# a warning and drops the oldest entry (writer-thread DLQ pattern) so
# the hot path never blocks.
_QUEUE_MAXSIZE: Final[int] = 1024

# Daemon thread join timeout on stop(). Long enough that a slow last
# flush completes; short enough that a hung writer doesn't deadlock the
# runner during teardown.
_WRITER_JOIN_TIMEOUT_S: Final[float] = 5.0

# Install hint shown when mcap is missing. Kept here (not inside
# the import block) so the message is grep-able.
_MCAP_INSTALL_HINT: Final[str] = (
    "mcap>=1.2 is required for Rosbag2Sink. "
    "Install via `uv pip install 'openral-dataset[ros]'` "
    "or `uv pip install 'mcap>=1.2'`."
)


# Sentinel posted to the writer queue to signal "drain and exit".
class _Stop:
    """Sentinel object for terminating the writer thread."""


_STOP = _Stop()


class Rosbag2Sink(DatasetSink):
    """mcap-backed sink fed by :class:`RolloutRecorder` fan-out.

    Writes openral-flavoured rosbag2-compatible mcap files. Compatible
    with `ros2 bag info`, Foxglove, mcap-cli; readable in pure Python
    via :func:`mcap.reader.make_reader` (which is how PR4's converter
    consumes it).

    Args:
        bag_path: Path to the output ``.mcap`` file. Parent directory
            must exist; the file itself must not exist (mcap refuses to
            overwrite).
        compression: Compression for the mcap chunks. Defaults to
            ``"zstd"`` (the rosbag2-mcap default; lz4 also supported by
            the mcap library). Pass ``None`` for uncompressed bags
            during debugging.

    Raises:
        ROSConfigError: If ``mcap`` is not importable, the parent
            directory doesn't exist, or the bag_path is already a file.

    Example:
        >>> from pathlib import Path
        >>> # End-to-end exercised in python/dataset/tests/test_bag.py
        >>> Path("/tmp/example.mcap").exists()  # doctest: +SKIP
        False
    """

    def __init__(
        self,
        *,
        bag_path: Path | str,
        compression: str | None = "zstd",
    ) -> None:
        """Stash configuration; no mcap import or I/O until :meth:`open_episode`."""
        try:
            import mcap  # noqa: F401  # reason: presence probe
        except ImportError as exc:
            raise ROSConfigError(_MCAP_INSTALL_HINT) from exc

        self._bag_path = Path(bag_path)
        if self._bag_path.exists():
            raise ROSConfigError(
                f"Rosbag2Sink: bag_path {self._bag_path} already exists; "
                "remove it or pick a fresh path (mcap refuses to overwrite)"
            )
        if not self._bag_path.parent.exists():
            raise ROSConfigError(
                f"Rosbag2Sink: parent directory {self._bag_path.parent} does not exist"
            )
        self._compression = compression

        # Deferred — opens on the first open_episode() call.
        self._writer: Any | None = None
        self._file_handle: Any | None = None
        self._tick_channel_id: int | None = None
        self._episode_channel_id: int | None = None
        self._image_channel_id: int | None = None
        self._queue: queue.Queue[dict[str, Any] | _Stop] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._writer_thread: threading.Thread | None = None
        self._sequence_per_channel: dict[int, int] = {}
        self._tick_seq_counter: int = 0
        self._finalized: bool = False
        self._current_episode_idx: int = -1
        self._current_task_string: str = ""
        # Statistics — useful in tests and the hardware-side
        # lifecycle node's diagnostics topic.
        self._n_ticks_written: int = 0
        self._n_episode_markers_written: int = 0
        self._n_images_written: int = 0
        self._n_dropped: int = 0

    # ── Properties (test / diagnostics surface) ────────────────────────────

    @property
    def bag_path(self) -> Path:
        """Output bag path (read-only)."""
        return self._bag_path

    @property
    def n_ticks_written(self) -> int:
        """Number of /openral/tick messages successfully written."""
        return self._n_ticks_written

    @property
    def n_episode_markers_written(self) -> int:
        """Number of /openral/episode messages (start + end) written."""
        return self._n_episode_markers_written

    @property
    def n_images_written(self) -> int:
        """Number of /openral/dataset/image messages successfully written."""
        return self._n_images_written

    @property
    def n_dropped(self) -> int:
        """Number of messages dropped because the writer queue was full."""
        return self._n_dropped

    # ── DatasetSink protocol ────────────────────────────────────────────────

    def open_episode(self, header: EpisodeHeader) -> None:
        """Open the bag on the first call; emit an Episode(PHASE_START) marker."""
        if self._writer is None:
            self._open_writer()
        self._current_episode_idx = header.episode_idx
        self._current_task_string = header.task_string
        self._enqueue_episode_marker(
            stamp_ns=header.stamp_ns,
            episode_idx=header.episode_idx,
            task_string=header.task_string,
            phase=PHASE_START,
            success=False,
        )

    def write_frame(self, frame: DatasetFrame) -> None:
        """Enqueue a /openral/tick message for the writer thread.

        Non-blocking: if the queue is full (writer is stuck on disk
        I/O), the oldest entry is dropped and a warning is logged.
        The inference hot path NEVER blocks here.
        """
        self._tick_seq_counter += 1
        self._enqueue(
            {
                "topic": TOPIC_TICK,
                "stamp_ns": frame.stamp_ns,
                "data": {
                    "stamp_ns": frame.stamp_ns,
                    "tick_idx": self._tick_seq_counter,
                    "episode_idx": frame.episode_idx,
                    "step_idx": frame.frame_idx,
                    "reward": float(frame.reward),
                    "terminated": bool(frame.terminated),
                    "truncated": bool(frame.truncated),
                    "action_applied": True,
                    # ISSUE-109: captured in record_frame inside the
                    # rskill.tick span and carried on the frame, because
                    # the actual mcap write runs off-thread where the OTel
                    # context is no longer in scope.
                    "trace_id": frame.trace_id,
                    "span_id": frame.span_id,
                    # Inline arrays so the bag is self-sufficient for
                    # conversion (ADR-0019 PR4-follow-up).
                    "observation_state": _to_float_list(frame.observation_state),
                    "action": _to_float_list(frame.action),
                },
            }
        )
        for camera, image in frame.images.items():
            self._enqueue_image(
                stamp_ns=frame.stamp_ns,
                episode_idx=frame.episode_idx,
                step_idx=frame.frame_idx,
                camera=camera,
                image=image,
            )

    def close_episode(self, summary: EpisodeSummary) -> None:
        """Emit an Episode(PHASE_END) marker carrying the success flag."""
        self._enqueue_episode_marker(
            stamp_ns=summary.stamp_ns,
            episode_idx=summary.episode_idx,
            task_string=self._current_task_string,
            phase=PHASE_END,
            success=summary.success,
        )

    def finalize(self) -> None:
        """Drain the queue, stop the writer thread, close the mcap file. Idempotent."""
        if self._finalized:
            return
        if self._writer_thread is not None:
            # Sentinel triggers a clean drain → close.
            self._queue.put(_STOP)
            self._writer_thread.join(timeout=_WRITER_JOIN_TIMEOUT_S)
            if self._writer_thread.is_alive():
                _log.warning(
                    "rosbag2_sink.writer_thread_join_timeout",
                    bag_path=str(self._bag_path),
                    timeout_s=_WRITER_JOIN_TIMEOUT_S,
                )
            self._writer_thread = None
        self._finalized = True
        _log.info(
            "rosbag2_sink.finalized",
            bag_path=str(self._bag_path),
            n_ticks=self._n_ticks_written,
            n_episode_markers=self._n_episode_markers_written,
            n_images=self._n_images_written,
            n_dropped=self._n_dropped,
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _open_writer(self) -> None:
        """Open the mcap file + register Tick / Episode schemas and channels."""
        from mcap.writer import CompressionType, Writer

        self._file_handle = self._bag_path.open("wb")
        compression_enum = self._compression_enum(CompressionType)
        self._writer = Writer(self._file_handle, compression=compression_enum)
        self._writer.start(profile="openral", library="openral_dataset.bag")
        tick_schema_id = self._writer.register_schema(
            name="openral_msgs/Tick",
            encoding="jsonschema",
            data=json.dumps(_TICK_SCHEMA).encode("utf-8"),
        )
        episode_schema_id = self._writer.register_schema(
            name="openral_msgs/Episode",
            encoding="jsonschema",
            data=json.dumps(_EPISODE_SCHEMA).encode("utf-8"),
        )
        self._tick_channel_id = self._writer.register_channel(
            topic=TOPIC_TICK, message_encoding="json", schema_id=tick_schema_id
        )
        self._episode_channel_id = self._writer.register_channel(
            topic=TOPIC_EPISODE, message_encoding="json", schema_id=episode_schema_id
        )
        image_schema_id = self._writer.register_schema(
            name="openral_msgs/DatasetImage",
            encoding="jsonschema",
            data=json.dumps(_IMAGE_SCHEMA).encode("utf-8"),
        )
        self._image_channel_id = self._writer.register_channel(
            topic=TOPIC_IMAGE, message_encoding="json", schema_id=image_schema_id
        )
        # Spawn the writer daemon now that the channels are registered.
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"rosbag2_sink:{self._bag_path.name}",
            daemon=True,
        )
        self._writer_thread.start()
        _log.debug(
            "rosbag2_sink.opened",
            bag_path=str(self._bag_path),
            compression=self._compression or "none",
        )

    def _compression_enum(self, cls: Any) -> Any:
        """Translate the string compression name to ``CompressionType``."""
        if self._compression is None:
            return cls.NONE
        name = self._compression.upper()
        try:
            return cls[name]
        except KeyError as exc:
            raise ROSConfigError(
                f"unknown mcap compression {self._compression!r}; "
                f"valid values: {[m.name.lower() for m in cls]}"
            ) from exc

    def _enqueue_episode_marker(
        self,
        *,
        stamp_ns: int,
        episode_idx: int,
        task_string: str,
        phase: int,
        success: bool,
    ) -> None:
        self._enqueue(
            {
                "topic": TOPIC_EPISODE,
                "stamp_ns": stamp_ns,
                "data": {
                    "stamp_ns": stamp_ns,
                    "episode_idx": episode_idx,
                    "task_string": task_string,
                    "phase": phase,
                    "success": bool(success),
                },
            }
        )

    def _enqueue_image(
        self,
        *,
        stamp_ns: int,
        episode_idx: int,
        step_idx: int,
        camera: str,
        image: Any,
    ) -> None:
        """Enqueue one camera frame as a base64 raw-u8 image message."""
        import base64

        arr = np.ascontiguousarray(image, dtype=np.uint8)
        height = int(arr.shape[0]) if arr.ndim >= _NDIM_H else 0
        width = int(arr.shape[1]) if arr.ndim >= _NDIM_HW else 0
        channels = int(arr.shape[2]) if arr.ndim >= _NDIM_HWC else 1
        self._enqueue(
            {
                "topic": TOPIC_IMAGE,
                "stamp_ns": stamp_ns,
                "data": {
                    "stamp_ns": stamp_ns,
                    "episode_idx": episode_idx,
                    "step_idx": step_idx,
                    "camera": camera,
                    "height": height,
                    "width": width,
                    "channels": channels,
                    "encoding": "raw_u8",
                    "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                },
            }
        )

    def _enqueue(self, msg: dict[str, Any]) -> None:
        """Non-blocking put with bounded-queue drop semantics.

        If the queue is full the oldest entry is silently dropped and a
        warning is logged. This is deliberate: blocking the hot path on
        disk I/O is worse than losing a frame. The dropped count
        surfaces via :attr:`n_dropped` for HIL-test assertions.
        """
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            self._n_dropped += 1
            _log.warning(
                "rosbag2_sink.queue_full_dropped",
                bag_path=str(self._bag_path),
                n_dropped=self._n_dropped,
            )
            # Try once more — if it fails again, give up on this message
            # too (very rare; would mean the writer thread crashed).
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                self._n_dropped += 1

    def _writer_loop(self) -> None:
        """Drain the queue until the _STOP sentinel arrives."""
        assert self._writer is not None  # invariant: thread spawned post-open
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, _Stop):
                    break
                self._write_one(item)
        except Exception:
            _log.exception("rosbag2_sink.writer_loop_crashed", bag_path=str(self._bag_path))
        finally:
            try:
                if self._writer is not None:
                    self._writer.finish()
            except Exception:
                _log.exception(
                    "rosbag2_sink.writer_finish_failed",
                    bag_path=str(self._bag_path),
                )
            if self._file_handle is not None:
                with contextlib.suppress(Exception):
                    self._file_handle.close()

    def _write_one(self, msg: dict[str, Any]) -> None:
        """Translate a queued envelope to mcap and write it."""
        assert self._writer is not None
        topic = msg["topic"]
        if topic == TOPIC_TICK:
            channel_id = self._tick_channel_id
        elif topic == TOPIC_EPISODE:
            channel_id = self._episode_channel_id
        elif topic == TOPIC_IMAGE:
            channel_id = self._image_channel_id
        else:
            _log.warning("rosbag2_sink.unknown_topic", topic=topic)
            return
        assert channel_id is not None
        log_time_ns = int(msg["stamp_ns"])
        seq = self._sequence_per_channel.get(channel_id, 0)
        self._sequence_per_channel[channel_id] = seq + 1
        # Wall-clock publish time: mcap requires both fields; matching
        # rosbag2 we set them equal so a downstream tool that
        # discriminates "captured" vs "originated" gets a sensible
        # answer either way.
        payload = json.dumps(msg["data"]).encode("utf-8")
        self._writer.add_message(
            channel_id=channel_id,
            log_time=log_time_ns,
            data=payload,
            publish_time=log_time_ns,
            sequence=seq,
        )
        if topic == TOPIC_TICK:
            self._n_ticks_written += 1
        elif topic == TOPIC_EPISODE:
            self._n_episode_markers_written += 1
        else:
            self._n_images_written += 1


def _to_float_list(arr: Any) -> list[float]:
    """Flatten an array-like to a JSON-serialisable list of floats."""
    return [float(x) for x in np.asarray(arr, dtype=np.float64).ravel().tolist()]
