"""``emit_scene_objects_span`` publishes the durable objects for the dashboard.

Real OTel SDK + in-memory exporter over the real ``home_scene_graph.json``
fixture (CLAUDE.md §1.11, no mocks). The ``world.scene_objects`` span gates the
on-the-wire vocabulary the dashboard's scene-objects card + SLAM-map overlay
read: ``openral.world_state.scene_objects.{count,frame_id,source_node,list}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from openral_core import SceneGraph, SpatialNodeKind
from openral_observability import semconv
from openral_world_state import emit_scene_objects_span, scene_objects_payload
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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


def _graph() -> SceneGraph:
    return SceneGraph.model_validate_json(_FIXTURE.read_text())


def test_payload_keeps_only_object_nodes() -> None:
    graph = _graph()
    payload = scene_objects_payload(graph)
    object_ids = {n.node_id for n in graph.nodes if n.kind is SpatialNodeKind.OBJECT}
    assert object_ids, "the home fixture has object nodes (wine bottle / glass)"
    assert {o["id"] for o in payload} == object_ids
    # Every payload row carries the map-frame pose + recency the dashboard renders.
    required = {"id", "label", "x", "y", "z", "frame_id", "confidence", "last_seen_ns"}
    for row in payload:
        assert required <= row.keys()


def test_emit_publishes_one_span_with_object_list(
    memory_exporter: InMemorySpanExporter,
) -> None:
    graph = _graph()
    count = emit_scene_objects_span(graph, source_node="openral_reasoner")

    spans = memory_exporter.get_finished_spans()
    assert [s.name for s in spans] == [semconv.SPAN_WORLD_SCENE_OBJECTS]
    attrs = spans[0].attributes
    assert attrs is not None
    n_objects = sum(1 for n in graph.nodes if n.kind is SpatialNodeKind.OBJECT)
    assert count == n_objects
    assert attrs[semconv.WORLD_SCENE_OBJECTS_COUNT] == n_objects
    assert attrs[semconv.WORLD_SCENE_OBJECTS_SOURCE_NODE] == "openral_reasoner"

    decoded = json.loads(str(attrs[semconv.WORLD_SCENE_OBJECTS_LIST]))
    assert len(decoded) == n_objects
    labels = {o["label"] for o in decoded}
    assert any("wine" in label.lower() for label in labels)


def test_empty_graph_emits_zero_count_and_default_frame(
    memory_exporter: InMemorySpanExporter,
) -> None:
    count = emit_scene_objects_span(SceneGraph(nodes=[], edges=[]), source_node="demo")
    assert count == 0
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs[semconv.WORLD_SCENE_OBJECTS_COUNT] == 0
    assert attrs[semconv.WORLD_SCENE_OBJECTS_FRAME] == "map"
    assert json.loads(str(attrs[semconv.WORLD_SCENE_OBJECTS_LIST])) == []
