"""``RSkillEvalResult.trace_id`` points at the trace that produced the numbers.

The field is what lets a reviewer jump from a committed
``rskills/<id>/eval/<benchmark>.json`` to the run's span tree. Both
aggregators must fill it from the ambient ``cli.command`` root span, and
both must leave it ``None`` in no-op observability mode rather than
writing a zero id that resolves to nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from openral_core import (
    BenchmarkMetadata,
    BenchmarkScene,
    PhysicsBackend,
    SceneSpec,
    TaskSpec,
    VLASpec,
)
from openral_sim.benchmark import _aggregate_results, _aggregate_scene_results
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span

_SUITE_ID = "tiny_mock"


def _scene() -> BenchmarkScene:
    return BenchmarkScene(
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 2, "action_dim": 7},
        ),
        task=TaskSpec(
            id="mock/0",
            scene_id="mock",
            instruction="noop",
            max_steps=20,
            success_key="is_success",
        ),
        robot_id="so100_follower",
        n_episodes=2,
        seed=0,
        metadata=BenchmarkMetadata(
            paper="https://example.invalid/mock-paper",
            honest_scope="2 episodes on the mock scene.",
            display_name="mock-mt1",
            simulator="mock",
        ),
    )


def _vla() -> VLASpec:
    return VLASpec(id="zero", weights_uri="placeholder", extra={"action_dim": 7})


@pytest.fixture
def root_span() -> Iterator[Span]:
    """A real recording ``cli.command`` span, as the CLI callback opens one."""
    tracer = TracerProvider().get_tracer("openral")
    with tracer.start_as_current_span("cli.command") as span:
        yield span


def test_suite_aggregator_records_the_root_trace_id(root_span: Span) -> None:
    result = _aggregate_results(
        [_scene()],
        suite_id=_SUITE_ID,
        vla=_vla(),
        per_task={"mock/0": [True, False]},
        episodes=[],
    )
    assert result.trace_id == f"{root_span.get_span_context().trace_id:032x}"


def test_scene_aggregator_records_the_root_trace_id(root_span: Span) -> None:
    result = _aggregate_scene_results(_scene(), _vla(), [True, False], [], None)
    assert result.trace_id == f"{root_span.get_span_context().trace_id:032x}"


def test_trace_id_is_none_without_a_trace() -> None:
    """No OTLP endpoint → no span tree to deep-link into."""
    assert not trace.get_current_span().get_span_context().is_valid
    result = _aggregate_results(
        [_scene()],
        suite_id=_SUITE_ID,
        vla=_vla(),
        per_task={"mock/0": [True]},
        episodes=[],
    )
    assert result.trace_id is None
