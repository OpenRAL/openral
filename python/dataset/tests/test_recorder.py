"""Tests for ``openral_dataset.RolloutRecorder``.

Per CLAUDE.md §1.11: real ``RobotDescription`` from
``robots/so100_follower/``, real ``DatasetSink`` implementations
(no MagicMock — uses a tiny ``_CaptureSink`` that records protocol
calls into a list).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_dataset.recorder import (
    DatasetFrame,
    DatasetSink,
    EpisodeHeader,
    EpisodeSummary,
    RolloutRecorder,
)


@dataclass
class _CaptureSink(DatasetSink):
    """Real DatasetSink implementation that records every protocol call.

    NOT a mock — implements the protocol surface exactly and keeps an
    ordered list of (event, payload) tuples for assertions. The point
    of CLAUDE.md §1.11 is to avoid MagicMock-style ducks that "pass"
    while doing nothing; a real recorder/dispatcher with a recording
    list satisfies the rule.
    """

    opened: list[EpisodeHeader] = field(default_factory=list)
    frames: list[DatasetFrame] = field(default_factory=list)
    closed: list[EpisodeSummary] = field(default_factory=list)
    finalized: int = 0

    def open_episode(self, header: EpisodeHeader) -> None:
        self.opened.append(header)

    def write_frame(self, frame: DatasetFrame) -> None:
        self.frames.append(frame)

    def close_episode(self, summary: EpisodeSummary) -> None:
        self.closed.append(summary)

    def finalize(self) -> None:
        self.finalized += 1


def _zero_frame(robot: RobotDescription) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Build (state, images, action) of the shapes the SO-100 expects."""
    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images: dict[str, np.ndarray] = {
        "camera1": np.zeros((32, 32, 3), dtype=np.uint8),
        "camera2": np.zeros((32, 32, 3), dtype=np.uint8),
    }
    return state, images, action


def test_recorder_basic_lifecycle(so100_robot: RobotDescription) -> None:
    sink = _CaptureSink()
    rec = RolloutRecorder(
        robot=so100_robot,
        task_string="pick the cube",
        fps=30.0,
        sinks=[sink],
    )
    state, images, action = _zero_frame(so100_robot)

    ep_idx = rec.episode_start()
    assert ep_idx == 0
    fidx = rec.record_frame(observation_state=state, images=images, action=action)
    assert fidx == 0
    rec.record_frame(observation_state=state, images=images, action=action, reward=1.0)
    summary = rec.episode_end(success=True)

    assert summary.episode_idx == 0
    assert summary.success is True
    assert summary.n_frames == 2
    assert len(sink.opened) == 1
    assert sink.opened[0].task_string == "pick the cube"
    assert sink.opened[0].robot_name == so100_robot.name
    assert sink.opened[0].fps == 30.0
    assert len(sink.frames) == 2
    assert sink.frames[1].reward == pytest.approx(1.0)
    assert len(sink.closed) == 1
    assert sink.closed[0].success is True

    rec.finalize()
    assert sink.finalized == 1

    # Idempotent finalize per the DatasetSink contract.
    rec.finalize()
    assert sink.finalized == 1


def test_recorder_multiple_episodes_advance_idx(so100_robot: RobotDescription) -> None:
    sink = _CaptureSink()
    rec = RolloutRecorder(
        robot=so100_robot,
        task_string="task",
        fps=30.0,
        sinks=[sink],
    )
    state, images, action = _zero_frame(so100_robot)

    for ep in range(3):
        got_idx = rec.episode_start()
        assert got_idx == ep
        rec.record_frame(observation_state=state, images=images, action=action)
        rec.episode_end(success=(ep % 2 == 0))

    rec.finalize()
    assert [s.success for s in sink.closed] == [True, False, True]
    assert [c.episode_idx for c in sink.closed] == [0, 1, 2]


def test_episode_start_twice_raises(so100_robot: RobotDescription) -> None:
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[])
    rec.episode_start()
    with pytest.raises(RuntimeError, match="still open"):
        rec.episode_start()


def test_record_frame_without_open_episode_raises(so100_robot: RobotDescription) -> None:
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[])
    state, images, action = _zero_frame(so100_robot)
    with pytest.raises(RuntimeError, match="no episode open"):
        rec.record_frame(observation_state=state, images=images, action=action)


def test_episode_end_without_open_episode_raises(so100_robot: RobotDescription) -> None:
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[])
    with pytest.raises(RuntimeError, match="no episode open"):
        rec.episode_end(success=True)


def test_wrong_state_shape_raises(so100_robot: RobotDescription) -> None:
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[])
    rec.episode_start()
    bad_state = np.zeros(5, dtype=np.float32)  # SO-100 wants 6
    images = {"camera1": np.zeros((32, 32, 3), dtype=np.uint8)}
    action = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match=r"state_shape"):
        rec.record_frame(observation_state=bad_state, images=images, action=action)


def test_wrong_action_shape_raises(so100_robot: RobotDescription) -> None:
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[])
    rec.episode_start()
    state, images, _ = _zero_frame(so100_robot)
    bad_action = np.zeros(5, dtype=np.float32)  # SO-100 wants 6
    with pytest.raises(ValueError, match=r"action_spec\.dim"):
        rec.record_frame(observation_state=state, images=images, action=bad_action)


def test_context_manager_closes_open_episode_on_exit(so100_robot: RobotDescription) -> None:
    sink = _CaptureSink()
    state, images, action = _zero_frame(so100_robot)
    # Exception during a rollout should NOT leave the recorder in a half-open
    # state — the __exit__ contract closes the open episode as a failure.
    with (
        pytest.raises(ZeroDivisionError),
        RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink]) as rec,
    ):
        rec.episode_start()
        rec.record_frame(observation_state=state, images=images, action=action)
        raise ZeroDivisionError("boom")
    assert len(sink.closed) == 1
    assert sink.closed[0].success is False, "interrupted episode must close with success=False"
    assert sink.finalized == 1


def test_invalid_fps_raises(so100_robot: RobotDescription) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        RolloutRecorder(robot=so100_robot, task_string="t", fps=0.0, sinks=[])
    with pytest.raises(ValueError, match="fps must be positive"):
        RolloutRecorder(robot=so100_robot, task_string="t", fps=-1.0, sinks=[])


def test_repo_id_writes_dataset_repo_id_semconv(so100_robot: RobotDescription) -> None:
    """Recorder writes openral.dataset.repo_id on the current span when configured.

    Asserts the placeholder semconv constants at
    openral_observability.semconv:143-145 are realized.
    """
    from openral_observability import semconv
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    sink = _CaptureSink()
    rec = RolloutRecorder(
        robot=so100_robot,
        task_string="t",
        fps=30.0,
        sinks=[sink],
        repo_id="openral/dataset-test",
    )
    state, images, action = _zero_frame(so100_robot)

    with tracer.start_as_current_span("rskill.tick"):
        rec.episode_start()
        rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get(semconv.DATASET_REPO_ID) == "openral/dataset-test"
    assert attrs.get(semconv.DATASET_EPISODE_IDX) == 0
    assert attrs.get(semconv.DATASET_FRAME_IDX) == 0


def test_record_frame_stamps_active_span_trace_and_span_ids(so100_robot: RobotDescription) -> None:
    """ISSUE-109: the frame handed to sinks carries the producing tick's ids.

    ``record_frame`` runs inside the ``rskill.tick`` span; the forward
    link reads ``get_current_span().get_span_context()`` and stamps the
    32-hex ``trace_id`` + 16-hex ``span_id`` onto the ``DatasetFrame``
    so every sink can persist the cross-process correlation key.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")

    sink = _CaptureSink()
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    with tracer.start_as_current_span("rskill.tick"):
        ctx = trace.get_current_span().get_span_context()
        rec.episode_start()
        rec.record_frame(observation_state=state, images=images, action=action)

    assert len(sink.frames) == 1
    frame = sink.frames[0]
    assert frame.trace_id == f"{ctx.trace_id:032x}"
    assert frame.span_id == f"{ctx.span_id:016x}"
    assert len(frame.trace_id) == 32
    assert len(frame.span_id) == 16


def test_record_frame_no_active_span_yields_empty_ids(so100_robot: RobotDescription) -> None:
    """Outside any valid span the frame carries empty-string ids (not garbage).

    The default OTel context has an invalid span context; the forward
    link must degrade to ``""`` so downstream consumers can treat the
    field as "no trace recorded" rather than persisting the all-zero
    INVALID_SPAN id.
    """
    sink = _CaptureSink()
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    rec.episode_start()
    rec.record_frame(observation_state=state, images=images, action=action)

    assert sink.frames[0].trace_id == ""
    assert sink.frames[0].span_id == ""


def test_record_frame_explicit_ids_override_active_span(so100_robot: RobotDescription) -> None:
    """The offline converter injects the bag's original ids; they win over the live span.

    When ``record_frame`` is called with explicit ``trace_id`` /
    ``span_id`` (the offline bag→LeRobot replay path), those values are
    persisted verbatim instead of capturing the converter process's own
    (unrelated) span context.
    """
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    sink = _CaptureSink()
    rec = RolloutRecorder(robot=so100_robot, task_string="t", fps=30.0, sinks=[sink])
    state, images, action = _zero_frame(so100_robot)

    bag_trace = "0123456789abcdef0123456789abcdef"
    bag_span = "fedcba9876543210"
    with tracer.start_as_current_span("converter.replay"):
        rec.episode_start()
        rec.record_frame(
            observation_state=state,
            images=images,
            action=action,
            trace_id=bag_trace,
            span_id=bag_span,
        )

    assert sink.frames[0].trace_id == bag_trace
    assert sink.frames[0].span_id == bag_span
