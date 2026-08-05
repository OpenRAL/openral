"""Unit tests for the reasoner's ``world.scene_objects`` emit-on-change gate.

The span is triggered from the 0.2 Hz heartbeat (`_start_tick`), which used to
re-emit the FULL object list as JSON every 5 s even for a static (or empty)
scene — 720 zero-information event rows per hour in the dashboard feed. The
gate keys on the SEMANTIC scene content (id / label / pose to 1 cm /
is_container) with a keepalive so a restarted collector still repopulates.

Real components throughout (CLAUDE.md §1.11): a real ``SpatialMemory`` built
from the ``home_scene_graph.json`` fixture, and a real in-memory OTel exporter
counting actual emitted spans — same harness as
``tests/unit/test_scene_objects_span.py``. The node method is bound to a
minimal holder (no ROS context), same trick as
``tests/unit/test_skill_runner_deadline.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
from openral_core import Pose6D, SceneGraph, SpatialNode, SpatialNodeKind
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs.msg")

from openral_reasoner_ros.reasoner_node import ReasonerNode
from openral_world_state import SpatialMemory

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


class _Holder:
    """Minimal carrier for the attributes ``_emit_scene_objects_span`` touches."""

    _SCENE_OBJECTS_KEEPALIVE_S = ReasonerNode._SCENE_OBJECTS_KEEPALIVE_S

    def __init__(self, memory: SpatialMemory) -> None:
        import rclpy.logging

        self._spatial_memory = memory
        self._scene_objects_key: tuple[object, ...] | None = None
        self._scene_objects_emitted_monotonic: float = 0.0
        self._logger = rclpy.logging.get_logger("test_scene_objects_gate")

    def get_name(self) -> str:
        return "test_reasoner"

    def get_logger(self) -> Any:  # reason: rclpy logger is untyped
        return self._logger


def _make_holder() -> Any:
    graph = SceneGraph.model_validate_json(_FIXTURE.read_text())
    holder = _Holder(SpatialMemory.from_scene_graph(graph))
    holder.emit = MethodType(  # type: ignore[attr-defined]
        ReasonerNode._emit_scene_objects_span, holder
    )
    return holder


def _object_node(node_id: str, x: float, label: str = "mug") -> SpatialNode:
    return SpatialNode(
        node_id=node_id,
        kind=SpatialNodeKind.OBJECT,
        label=label,
        pose=Pose6D(frame_id="map", xyz=(x, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0)),
        first_seen_ns=1,
        last_seen_ns=1,
    )


def test_unchanged_scene_emits_once(memory_exporter: InMemorySpanExporter) -> None:
    """Heartbeat re-invocations with a static scene emit exactly one span."""
    holder = _make_holder()
    for _ in range(5):
        holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 1


def test_scene_change_emits_immediately(memory_exporter: InMemorySpanExporter) -> None:
    """A genuine map change (new object) re-emits on the next heartbeat."""
    holder = _make_holder()
    holder.emit()
    holder._spatial_memory.upsert_node(_object_node("obj-new", 1.0))
    holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 2


def test_sub_centimetre_jitter_is_suppressed(memory_exporter: InMemorySpanExporter) -> None:
    """Detector pose jitter under the 1 cm rounding must not defeat the gate."""
    holder = _make_holder()
    holder._spatial_memory.upsert_node(_object_node("obj-jitter", 1.000))
    holder.emit()
    holder._spatial_memory.upsert_node(_object_node("obj-jitter", 1.003))
    holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 1


def test_keepalive_reemits_unchanged_scene(memory_exporter: InMemorySpanExporter) -> None:
    """After the keepalive interval an unchanged scene re-emits (collector restart)."""
    holder = _make_holder()
    holder.emit()
    holder._scene_objects_emitted_monotonic -= ReasonerNode._SCENE_OBJECTS_KEEPALIVE_S + 1.0
    holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 2
