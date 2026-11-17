"""Unit tests for the reasoner's ``world.scene_objects`` emit-on-change gate.

The span fires from the 0.2 Hz heartbeat (`_start_tick`), which used to
re-emit the FULL object list every 5 s even for a static scene — 720
zero-information event rows per hour. The gate keys on SEMANTIC content
(id / label / pose to 1 cm / is_container) with a keepalive so a restarted
collector still repopulates.

Real ``SpatialMemory`` from the ``home_scene_graph.json`` fixture + a real
in-memory OTel exporter (CLAUDE.md §1.11), same harness as
``test_scene_objects_span.py``. The key-gate tests bind the node method to a
minimal holder (no ROS context), same trick as ``test_skill_runner_deadline.py``;
the "seen now" tests drive a real ``ReasonerNode`` through its ingest path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
from openral_core import (
    DetectedObject,
    JointState,
    Pose6D,
    SceneGraph,
    SpatialNode,
    SpatialNodeKind,
    WorldState,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs.msg")

from openral_reasoner_ros.reasoner_node import ReasonerNode
from openral_world_state import SpatialMemory

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


class _Holder:
    """Minimal carrier for the attributes ``_emit_scene_objects_span`` touches."""

    _SCENE_OBJECTS_KEEPALIVE_S = ReasonerNode._SCENE_OBJECTS_KEEPALIVE_S

    def __init__(self, memory: SpatialMemory) -> None:
        import rclpy.logging

        self._spatial_memory = memory
        self._scene_objects_key: tuple[object, ...] | None = None
        self._scene_objects_emitted_monotonic: float = 0.0
        self._latest_batch_ns: int | None = None
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


def test_scene_order_does_not_count_as_change(memory_exporter: InMemorySpanExporter) -> None:
    holder = _make_holder()
    holder.emit()
    graph = holder._spatial_memory.to_scene_graph()
    reordered = graph.model_copy(update={"nodes": list(reversed(graph.nodes))})
    holder._spatial_memory = SpatialMemory.from_scene_graph(reordered)
    holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 1


def test_keepalive_reemits_unchanged_scene(memory_exporter: InMemorySpanExporter) -> None:
    """After the keepalive interval an unchanged scene re-emits (collector restart)."""
    holder = _make_holder()
    holder.emit()
    holder._scene_objects_emitted_monotonic -= ReasonerNode._SCENE_OBJECTS_KEEPALIVE_S + 1.0
    holder.emit()
    assert len(memory_exporter.get_finished_spans()) == 2


@pytest.fixture
def ingesting_node() -> Iterator[ReasonerNode]:
    """A real ``ReasonerNode`` owning a live-ingest spatial memory (not configured).

    ``_maybe_load_spatial_memory`` is the ``on_configure`` step that builds it; the
    rest of ``on_configure`` (LLM client, subscriptions) plays no part here.
    """
    import rclpy
    from rclpy.parameter import Parameter

    rclpy.init()
    node = ReasonerNode(node_name="test_scene_objects_gate")
    try:
        node.set_parameters([Parameter("spatial_memory_ingest", Parameter.Type.BOOL, True)])
        node._maybe_load_spatial_memory()
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _snapshot(stamp_ns: int, *labels_x: tuple[str, float]) -> WorldState:
    return WorldState(
        stamp_ns=stamp_ns,
        joint_state=JointState(name=[], position=[], stamp_ns=stamp_ns),
        detected_objects=[
            DetectedObject(
                label=label,
                confidence=0.9,
                pose=Pose6D(frame_id="map", xyz=(x, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0)),
                bbox_3d=(0.0, 0.0, 0.0, 0.1, 0.1, 0.1),
                track_id=i,
            )
            for i, (label, x) in enumerate(labels_x)
        ],
    )


def _tick(node: ReasonerNode, world_state: WorldState) -> None:
    """The two spatial-memory steps of ``_on_tick``, in its order."""
    node._ingest_detected_objects(world_state)
    node._emit_scene_objects_span()


def _seen_now(span: Any) -> dict[str, bool]:
    import json

    from openral_observability import semconv

    rows = json.loads(str(span.attributes[semconv.WORLD_SCENE_OBJECTS_LIST]))
    return {r["label"]: r["in_latest_batch"] for r in rows}


def test_seen_now_change_emits_but_new_batch_stamp_alone_does_not(
    memory_exporter: InMemorySpanExporter, ingesting_node: ReasonerNode
) -> None:
    """A batch that changes WHICH objects are seen re-emits; a same-set batch does not."""
    _tick(ingesting_node, _snapshot(5, ("cup", 1.0)))
    _tick(ingesting_node, _snapshot(6, ("cup", 1.0)))  # newer batch, same seen set
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2  # the empty map at load, then the cup
    assert _seen_now(spans[1]) == {"cup": True}


def test_an_empty_batch_clears_seen_now(
    memory_exporter: InMemorySpanExporter, ingesting_node: ReasonerNode
) -> None:
    """The cup stays remembered but is no longer "seen now", and that re-emits at once."""
    _tick(ingesting_node, _snapshot(100, ("cup", 1.0)))
    _tick(ingesting_node, _snapshot(200))
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 3
    assert _seen_now(spans[-1]) == {"cup": False}


def test_a_rebuilt_memory_does_not_inherit_the_batch_stamp(
    memory_exporter: InMemorySpanExporter, ingesting_node: ReasonerNode, tmp_path: Path
) -> None:
    """Cleanup + configure reloads the saved map; nothing in it is "seen now"."""
    from rclpy.parameter import Parameter

    _tick(ingesting_node, _snapshot(100, ("cup", 1.0)))
    saved = tmp_path / "scene.json"
    assert ingesting_node._spatial_memory_writer is not None
    ingesting_node._spatial_memory_writer.save(saved)
    ingesting_node.on_cleanup(None)
    ingesting_node.set_parameters(
        [Parameter("spatial_memory_path", Parameter.Type.STRING, str(saved))]
    )
    ingesting_node._maybe_load_spatial_memory()
    assert _seen_now(memory_exporter.get_finished_spans()[-1]) == {"cup": False}
