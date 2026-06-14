"""Emit the durable spatial-memory objects as an OTel span for the dashboard.

ADR-0038 spatial memory is observability-visible: this module renders the
``object``-kind nodes of a :class:`~openral_core.SceneGraph` onto a single
``world.scene_objects`` span. The dashboard (ADR-0017) ingests it over OTLP and
shows the objects both as a table card and as labelled markers on the SLAM 2D
map (same ``map`` frame as the robot pose).

The span is advisory telemetry — never a safety input (CLAUDE.md §1.1).

Today the emitter is driven by the Reasoner from its preloaded scene graph
(``spatial_memory_path``); once the perception object-lift producer lands
(ADR-0035 / PR #229) the World-State node becomes the canonical caller, right
next to ``WorldState.detected_objects`` ingest — same helper, no change here.
"""

from __future__ import annotations

import json

from openral_core import SceneGraph, SpatialNodeKind
from openral_observability import semconv
from opentelemetry import trace

__all__ = ["emit_scene_objects_span", "scene_objects_payload"]

_DEFAULT_FRAME = "map"


def scene_objects_payload(graph: SceneGraph) -> list[dict[str, object]]:
    """Project the ``object``-kind nodes to JSON-friendly dicts.

    Each dict carries the map-frame position, the semantic label, and the
    recency/confidence the dashboard renders. Non-object nodes (places, rooms,
    agents) are skipped — the dashboard surface is "what objects do we
    remember, and where".

    Args:
        graph: A scene-graph snapshot (e.g. ``SpatialMemory.to_scene_graph()``).

    Returns:
        One dict per object node, in graph order.
    """
    objects: list[dict[str, object]] = []
    for node in graph.nodes:
        if node.kind is not SpatialNodeKind.OBJECT:
            continue
        x, y, z = node.pose.xyz
        objects.append(
            {
                "id": node.node_id,
                "label": node.label,
                "x": x,
                "y": y,
                "z": z,
                "frame_id": node.pose.frame_id,
                "confidence": node.confidence,
                "last_seen_ns": node.last_seen_ns,
                "observation_count": node.observation_count,
                "is_container": node.is_container,
            }
        )
    return objects


def emit_scene_objects_span(graph: SceneGraph, *, source_node: str) -> int:
    """Emit one ``world.scene_objects`` span carrying ``graph``'s object nodes.

    Args:
        graph: A scene-graph snapshot to publish.
        source_node: Name of the node emitting the span (shown on the card so
            an operator knows which process produced the map).

    Returns:
        The number of object nodes published.

    Example:
        >>> import time
        >>> from openral_core import DetectedObject, Pose6D
        >>> from openral_world_state import SpatialMemory
        >>> mem = SpatialMemory()
        >>> mug = DetectedObject(
        ...     label="mug",
        ...     confidence=0.9,
        ...     pose=Pose6D(xyz=(1.0, 2.0, 0.8), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        ... )
        >>> _ = mem.ingest_detected_objects([mug], now_ns=time.time_ns())
        >>> emit_scene_objects_span(mem.to_scene_graph(), source_node="demo")
        1
    """
    objects = scene_objects_payload(graph)
    frame = str(objects[0]["frame_id"]) if objects else _DEFAULT_FRAME
    tracer = trace.get_tracer("openral")
    with tracer.start_as_current_span(semconv.SPAN_WORLD_SCENE_OBJECTS) as span:
        span.set_attribute(semconv.WORLD_SCENE_OBJECTS_COUNT, len(objects))
        span.set_attribute(semconv.WORLD_SCENE_OBJECTS_SOURCE_NODE, source_node)
        span.set_attribute(semconv.WORLD_SCENE_OBJECTS_FRAME, frame)
        span.set_attribute(semconv.WORLD_SCENE_OBJECTS_LIST, json.dumps(objects))
    return len(objects)
