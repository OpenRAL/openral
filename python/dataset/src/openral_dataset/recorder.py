"""RolloutRecorder — in-memory per-rollout accumulator with multi-sink fan-out.

The recorder is decoupled from any specific sink. The same recorder feeds:

* :class:`openral_dataset.LeRobotDatasetSink` (online sim path) — writes a
  LeRobotDataset v3.0 (codebase_version="3.0") directly from the live
  rollout.
* :class:`openral_dataset.Rosbag2Sink` (PR3, online hardware path) — writes
  an mcap rosbag2 of joints, world_state, camera streams, and
  ``/openral/tick`` per-tick metadata.
* :class:`openral_dataset.Rosbag2ToLeRobotConverter` (PR4, offline) — reads a
  bag back and feeds the same ``DatasetSink`` interface to produce a
  v3 dataset.

The recorder owns no I/O. ``record_frame()`` calls fan out to every
attached sink synchronously, which keeps the sink-side concurrency
explicit (the rosbag2 sink uses a daemon thread + bounded queue so the
hot path never blocks on disk).

Per CLAUDE.md §1.11 (no mocks) — the recorder is exercised against real
``RobotDescription`` fixtures from ``robots/`` in
``python/dataset/tests/``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_observability import semconv
from opentelemetry import trace

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = [
    "DatasetFrame",
    "DatasetSink",
    "EpisodeHeader",
    "EpisodeSummary",
    "RolloutRecorder",
]

_log = structlog.get_logger(__name__)

# Sensor modalities that count as "image" streams for dataset binding.
# Mirrors openral_dataset.schema_map._IMAGE_MODALITIES; duplicated here
# to keep that module a pure no-deps schema mapper.
_IMAGE_MODALITIES: frozenset[str] = frozenset({"rgb", "depth", "rgbd", "thermal", "ir"})


@dataclass(frozen=True)
class EpisodeHeader:
    """Per-episode metadata pushed to sinks at ``episode_start``.

    Attributes:
        episode_idx: Zero-based episode index within this recorder lifetime.
        task_string: Natural-language task instruction (becomes the
            ``task`` column in LeRobot v3 rows; mapped to ``task_index``
            via the dataset's ``meta/tasks.parquet`` table).
        fps: Recording cadence in Hz (used by the LeRobot v3 video encoder
            and by the converter's tick-grid generator).
        robot_name: Echo of ``RobotDescription.name`` for sink-side
            book-keeping. Pydantic-validated upstream.
        stamp_ns: Episode-start wall clock in nanoseconds.
    """

    episode_idx: int
    task_string: str
    fps: float
    robot_name: str
    stamp_ns: int


@dataclass(frozen=True)
class DatasetFrame:
    """One per-tick frame pushed to sinks at ``record_frame``.

    The shape conventions match LeRobot v3.0:
      * ``observation_state`` — ``(state_dim,) float32``
      * ``images[cam_key]`` — ``(H, W, 3) uint8`` HWC RGB
      * ``action`` — ``(action_dim,) float32`` (action chunk first row, the
        row that will actually be applied next)

    Attributes:
        episode_idx: Episode this frame belongs to.
        frame_idx: Zero-based frame index within the episode.
        observation_state: Proprioception (joint state) vector.
        images: Mapping from camera ``vla_feature_key`` (without the
            ``observation.images.`` prefix) to a HWC uint8 RGB array.
        action: Action vector (first row of the action chunk).
        reward: Per-step reward (sim-only; ``0.0`` on hardware).
        terminated: Env-signalled natural completion (sim) /
            ``False`` on hardware.
        truncated: Step-budget hit (sim) / ``False`` on hardware.
        stamp_ns: Frame wall clock in nanoseconds.
        trace_id: 32-hex-char OTel trace id of the producing ``rskill.tick``
            span, or ``""`` when no valid span was in scope. Persisted by
            every sink so a written frame can pivot back into its trace.
        span_id: 16-hex-char OTel span id of the producing tick, or ``""``.
    """

    episode_idx: int
    frame_idx: int
    observation_state: NDArray[np.float32]
    images: Mapping[str, NDArray[np.uint8]]
    action: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    stamp_ns: int
    trace_id: str = ""
    span_id: str = ""


@dataclass(frozen=True)
class EpisodeSummary:
    """Per-episode close-out pushed to sinks at ``episode_end``.

    Attributes:
        episode_idx: Episode index.
        success: Whether the episode counted as a success (sim's
            ``task.success_key`` resolution / hardware's
            ``episode_end(success=...)`` argument).
        n_frames: Number of frames written during this episode.
        stamp_ns: Episode-end wall clock in nanoseconds.
    """

    episode_idx: int
    success: bool
    n_frames: int
    stamp_ns: int


class DatasetSink(Protocol):
    """Fan-out target for :class:`RolloutRecorder`.

    A sink receives one ``open_episode`` call per episode, zero or more
    ``write_frame`` calls during the episode, and exactly one
    ``close_episode`` call when the episode ends (regardless of
    success). ``finalize`` is called once at recorder shutdown to flush
    pending I/O.

    All callbacks are synchronous from the recorder's perspective; sinks
    that need async / threaded I/O (e.g. the PR3 ``Rosbag2Sink``)
    implement their own backpressure internally.
    """

    def open_episode(self, header: EpisodeHeader) -> None:
        """Begin a new episode. Idempotent only within one episode."""

    def write_frame(self, frame: DatasetFrame) -> None:
        """Append one frame to the current episode."""

    def close_episode(self, summary: EpisodeSummary) -> None:
        """Finalise the current episode (success flag, frame count)."""

    def finalize(self) -> None:
        """Flush pending I/O and release resources. Idempotent."""


class RolloutRecorder:
    """Per-rollout accumulator with multi-sink fan-out.

    Owns no I/O. ``record_frame`` writes the three OTel semantic
    attributes ``DATASET_REPO_ID`` / ``DATASET_EPISODE_IDX`` /
    ``DATASET_FRAME_IDX`` on the current span (the ``rskill.tick`` span
    that wraps the caller), realising the placeholder semconv constants
    declared at ``openral_observability.semconv:143-145``.

    Args:
        robot: Normative robot description; used by sinks to bind feature
            shapes via :func:`openral_dataset.features_from_robot`.
        task_string: Default natural-language task instruction
            (overridable per-episode in :meth:`episode_start`).
        fps: Recording cadence in Hz. Locks the LeRobot v3 encoder's
            fps and the converter's tick-grid spacing.
        sinks: One or more :class:`DatasetSink` implementations.
        repo_id: Optional HF Hub repo id (e.g. ``openral/dataset-pick-cube``).
            When set, lands on the OTel span attribute
            ``openral.dataset.repo_id`` for trace-to-dataset joins.

    Example:
        >>> from openral_core import RobotDescription
        >>> # doctest: +SKIP
        >>> robot = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")
        >>> rec = RolloutRecorder(robot=robot, task_string="t", fps=30.0, sinks=[])
    """

    def __init__(
        self,
        *,
        robot: RobotDescription,
        task_string: str,
        fps: float,
        sinks: Sequence[DatasetSink],
        repo_id: str | None = None,
    ) -> None:
        """Initialise the recorder with a robot, sinks, and a default task string."""
        if fps <= 0.0:
            raise ValueError(f"fps must be positive; got {fps!r}")
        self._robot = robot
        self._default_task = task_string
        self._fps = float(fps)
        self._sinks: tuple[DatasetSink, ...] = tuple(sinks)
        self._repo_id = repo_id

        self._episode_idx: int = -1  # Pre-first-episode sentinel.
        self._frame_idx: int = 0
        self._frames_in_episode: int = 0
        self._episode_open: bool = False
        self._finalized: bool = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        """Recording cadence in Hz."""
        return self._fps

    @property
    def robot_name(self) -> str:
        """Echo of ``RobotDescription.name``."""
        return self._robot.name

    @property
    def repo_id(self) -> str | None:
        """HF Hub repo id, if configured."""
        return self._repo_id

    @property
    def n_sinks(self) -> int:
        """Number of attached sinks."""
        return len(self._sinks)

    @property
    def expected_state_shape(self) -> tuple[int, ...]:
        """State-vector shape required by attached sinks.

        Pulled from :class:`openral_core.ObservationSpec.state_shape`.
        Callers that can't readily produce the proprioception vector
        (e.g. sim envs that don't surface ``state`` in their obs dict)
        use this to build a same-shape zero placeholder rather than
        crash the rollout.
        """
        obs_spec = self._robot.observation_spec
        if obs_spec is None or not obs_spec.state_shape:
            return (0,)
        return tuple(obs_spec.state_shape)

    def expected_image_keys(self) -> tuple[str, ...]:
        """Camera keys (without ``observation.images.`` prefix) the sinks expect.

        Derived from :class:`openral_core.RobotDescription.sensors` —
        every sensor with both an image modality (``rgb`` / ``depth`` /
        ``rgbd`` / ``thermal`` / ``ir``) and a ``vla_feature_key`` becomes
        a camera key.

        Sim callers use this to build the ``images`` mapping for
        :meth:`record_frame` — typically by writing the same
        ``env.render()`` output to every declared camera key (sim envs
        usually expose one observation viewpoint that mirrors what the
        VLA sees through every robot camera).
        """
        keys: list[str] = []
        for sensor in self._robot.sensors:
            if sensor.modality not in _IMAGE_MODALITIES:
                continue
            if sensor.vla_feature_key is None:
                continue
            keys.append(sensor.vla_feature_key.removeprefix("observation.images."))
        return tuple(keys)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def __enter__(self) -> RolloutRecorder:
        """Enter the recorder context. Returns ``self`` for chaining."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close an open episode (as failure) and finalize sinks.

        If an exception interrupted an open episode, close it as a
        failure so sinks see a clean lifecycle. The exception still
        propagates because we don't return ``True``.
        """
        if self._episode_open:
            self.episode_end(success=False)
        self.finalize()

    def episode_start(self, *, task_string: str | None = None) -> int:
        """Open a new episode.

        Args:
            task_string: Override the recorder's default task string for
                this episode. Useful when one run spans multiple tasks
                (e.g. a BT executor running pick → place → return).

        Returns:
            The new ``episode_idx``.

        Raises:
            RuntimeError: If an episode is already open. Call
                :meth:`episode_end` first.
        """
        if self._finalized:
            raise RuntimeError("RolloutRecorder is finalized; cannot start new episodes")
        if self._episode_open:
            raise RuntimeError(
                f"episode {self._episode_idx} is still open; call episode_end() first"
            )
        self._episode_idx += 1
        self._frame_idx = 0
        self._frames_in_episode = 0
        self._episode_open = True
        task = task_string if task_string is not None else self._default_task
        header = EpisodeHeader(
            episode_idx=self._episode_idx,
            task_string=task,
            fps=self._fps,
            robot_name=self._robot.name,
            stamp_ns=time.time_ns(),
        )
        for sink in self._sinks:
            sink.open_episode(header)
        _log.debug(
            "rollout_episode_start",
            episode_idx=self._episode_idx,
            task=task,
            robot=self._robot.name,
            fps=self._fps,
            n_sinks=len(self._sinks),
        )
        return self._episode_idx

    def record_frame(
        self,
        *,
        observation_state: NDArray[np.float32],
        images: Mapping[str, NDArray[np.uint8]],
        action: NDArray[np.float32],
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        stamp_ns: int | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> int:
        """Append one frame to the current episode.

        Args:
            observation_state: ``(state_dim,) float32``. Shape is
                verified against ``RobotDescription.observation_spec.state_shape``.
            images: Mapping from camera key (without the
                ``observation.images.`` prefix) to ``(H, W, 3) uint8`` RGB.
            action: ``(action_dim,) float32``. Shape is verified against
                ``RobotDescription.action_spec.dim``.
            reward: Per-step reward (see :class:`DatasetFrame.reward`).
            terminated: Env-signalled natural completion (see
                :class:`DatasetFrame.terminated`).
            truncated: Step-budget hit (see :class:`DatasetFrame.truncated`).
            stamp_ns: Frame wall clock in nanoseconds; defaults to
                :func:`time.time_ns` when ``None``.
            trace_id: Override for the per-frame OTel ``trace_id`` (32 hex).
                When ``None`` (the online path) the id is captured from the
                active ``rskill.tick`` span. The offline bag→LeRobot
                converter passes the bag tick's original id here so the
                replayed frame keeps its source trace rather than the
                converter process's own span context.
            span_id: Override for the per-frame OTel ``span_id`` (16 hex);
                same capture/override semantics as ``trace_id``.

        Returns:
            The new frame_idx within the current episode.

        Raises:
            RuntimeError: If no episode is open. Call :meth:`episode_start` first.
            ValueError: If ``observation_state`` or ``action`` shapes do
                not match the robot's specs.
        """
        if not self._episode_open:
            raise RuntimeError("no episode open; call episode_start() first")

        # Light shape validation — catches the most common wiring bug
        # (state and action transposed) at the recorder, before sinks
        # see anything. The observation / action specs are optional on
        # the schema; when absent we skip the check rather than reject.
        obs_spec = self._robot.observation_spec
        if obs_spec is not None:
            expected_state_shape = tuple(obs_spec.state_shape)
            if expected_state_shape and tuple(observation_state.shape) != expected_state_shape:
                raise ValueError(
                    f"observation_state shape {observation_state.shape!r} does not match "
                    f"RobotDescription.observation_spec.state_shape={expected_state_shape!r}"
                )
        action_spec = self._robot.action_spec
        if action_spec is not None:
            action_dim = action_spec.dim
            if action_dim and (action.ndim != 1 or action.shape[0] != action_dim):
                raise ValueError(
                    f"action shape {action.shape!r} does not match "
                    f"RobotDescription.action_spec.dim={action_dim!r}"
                )

        # Forward link — capture the producing tick's (trace_id, span_id)
        # once, here, while we are guaranteed to be inside the caller's
        # rskill.tick span. Sinks that defer the actual write to a worker
        # thread (Rosbag2Sink) cannot read the context later, so it must
        # ride on the frame. Explicit ids (offline converter) win over the
        # live span; absent both, the field degrades to "".
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if trace_id is None:
            trace_id = f"{ctx.trace_id:032x}" if ctx.is_valid else ""
        if span_id is None:
            span_id = f"{ctx.span_id:016x}" if ctx.is_valid else ""

        frame = DatasetFrame(
            episode_idx=self._episode_idx,
            frame_idx=self._frame_idx,
            observation_state=observation_state,
            images=dict(images),
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            stamp_ns=stamp_ns if stamp_ns is not None else time.time_ns(),
            trace_id=trace_id,
            span_id=span_id,
        )
        for sink in self._sinks:
            sink.write_frame(frame)

        # Reverse link — realise the placeholder semconv constants: every
        # recorder write attaches the dataset-coordinate to the active
        # rskill.tick span so the Jaeger trace can be joined to the
        # on-disk frame.
        if span.is_recording():
            if self._repo_id is not None:
                span.set_attribute(semconv.DATASET_REPO_ID, self._repo_id)
            span.set_attribute(semconv.DATASET_EPISODE_IDX, self._episode_idx)
            span.set_attribute(semconv.DATASET_FRAME_IDX, self._frame_idx)

        emitted_idx = self._frame_idx
        self._frame_idx += 1
        self._frames_in_episode += 1
        return emitted_idx

    def episode_end(self, *, success: bool) -> EpisodeSummary:
        """Close the current episode.

        Args:
            success: Episode-level success flag. Sinks tag every frame
                of this episode with ``next.success = success`` (the
                converter does the equivalent reconstruction offline).

        Returns:
            The :class:`EpisodeSummary` (also handed to every sink).

        Raises:
            RuntimeError: If no episode is open.
        """
        if not self._episode_open:
            raise RuntimeError("no episode open; cannot end")
        summary = EpisodeSummary(
            episode_idx=self._episode_idx,
            success=bool(success),
            n_frames=self._frames_in_episode,
            stamp_ns=time.time_ns(),
        )
        for sink in self._sinks:
            sink.close_episode(summary)
        self._episode_open = False
        _log.debug(
            "rollout_episode_end",
            episode_idx=self._episode_idx,
            success=summary.success,
            n_frames=summary.n_frames,
        )
        return summary

    def finalize(self) -> None:
        """Flush all sinks. Idempotent. Safe to call multiple times."""
        if self._finalized:
            return
        if self._episode_open:
            # Episodes must be explicitly closed; finalising an open
            # episode is a wiring bug, not a recoverable state.
            raise RuntimeError(
                f"episode {self._episode_idx} is still open at finalize(); "
                "call episode_end() before finalize()"
            )
        for sink in self._sinks:
            sink.finalize()
        self._finalized = True
