"""Persistent object-centric scene-graph spatial memory (ADR-0038 Phase 2).

:class:`SpatialMemory` accumulates the momentary
``WorldState.detected_objects`` into a durable, queryable scene graph and
answers the ADR-0038 read-only query contracts:

- :meth:`SpatialMemory.recall_object` — recall a remembered object by label/text,
  with optional proximity/recency filters, returning the object's ``map``-frame
  pose plus a **camera-facing approach viewpoint** (the standoff pose a mobile
  base drives to so its gripper-mounted camera faces the object) and, when the
  object sits inside an occluding container, the ``inside_container_id`` the
  planner must open first.
- :meth:`SpatialMemory.resolve_place` — resolve a place/room/agent reference
  ("the kitchen", "where I was standing") to a navigation goal plus a
  ``traversable_to`` path.

This is **advisory** Layer-2 world-model state consumed by the S2 Reasoner; it
is never a safety input (CLAUDE.md §1.1) — the safety kernel gates only on the
live, bounded ADR-0030 geometric world. Object poses are anchored in the
durable ``map`` frame (TF resolution happens upstream); the memory never stores
a raw transform.

Persistence is the :class:`~openral_core.SceneGraph` JSON contract
(:meth:`SpatialMemory.save` / :meth:`SpatialMemory.load`). The graph is small
(hundreds-to-thousands of nodes for one robot), so traversal is a plain typed
BFS — no graph-engine dependency. Open-vocabulary embedding retrieval and a
``sqlite-vec`` store (ADR-0038 §5, Phase 4) layer on top of this without
changing the contract.

Example:
    >>> import time
    >>> from openral_core import DetectedObject, RecallObjectQuery, Pose6D
    >>> mem = SpatialMemory()
    >>> mug = DetectedObject(
    ...     label="mug",
    ...     confidence=0.9,
    ...     pose=Pose6D(xyz=(1.0, 0.0, 0.8), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
    ...     track_id=7,
    ... )
    >>> _ = mem.ingest_detected_objects([mug], now_ns=time.time_ns())
    >>> result = mem.recall_object(RecallObjectQuery(label="mug"), now_ns=time.time_ns())
    >>> result.matches[0].label
    'mug'
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    ApproachViewpoint,
    DetectedObject,
    Pose6D,
    RecallObjectMatch,
    RecallObjectQuery,
    RecallObjectResult,
    ResolvePlaceQuery,
    ResolvePlaceResult,
    SceneGraph,
    SpatialEdge,
    SpatialNode,
    SpatialNodeKind,
    SpatialRelationKind,
)
from openral_core.exceptions import ROSObjectNotInMemory

from openral_world_state.embedder import TextEmbedder

DEFAULT_ASSOC_DISTANCE_M = 0.3
"""Default radius (m) for label-based instance association when ``track_id`` is absent."""

DEFAULT_STANDOFF_M = 0.6
"""Default standoff distance (m) from an object for the camera-facing approach viewpoint."""

DEFAULT_CAMERA_FRAME = "gripper_camera"
"""Default tf2 frame the approach viewpoint orients toward."""

DEFAULT_MAP_FRAME = "map"
"""Default durable frame object poses are anchored in."""

DEFAULT_MIN_TEXT_SIMILARITY = 0.85
"""Min CLIP cosine for an embedding-only object match (calibrated for ViT-B/32 openai).

OpenCLIP text-text cosines cluster high (~0.76-0.96), so this floor gates
embedding-only hits; an exact/substring label match always qualifies regardless.
"""

_MIN_DIR_NORM = 1e-6
"""Below this horizontal distance the approach direction is treated as degenerate."""


def _xy_yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a pure yaw rotation about +Z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def compute_approach_viewpoint(
    target: Pose6D,
    *,
    standoff_m: float = DEFAULT_STANDOFF_M,
    camera_frame_id: str = DEFAULT_CAMERA_FRAME,
    approach_from: Pose6D | None = None,
) -> ApproachViewpoint:
    """Compute a standoff pose whose camera faces ``target`` (ADR-0038 §6).

    The viewpoint is placed ``standoff_m`` away from the target in the
    horizontal (x, y) plane and yawed to look at it. When ``approach_from`` is
    given (e.g. the place a robot would stand at), the viewpoint sits on that
    side of the target; otherwise it defaults to approaching from -X.

    Args:
        target: Object pose to view, in the map frame.
        standoff_m: Standoff distance in metres (``> 0``).
        camera_frame_id: tf2 frame of the camera the viewpoint orients toward.
        approach_from: Optional pose giving the side to approach from.

    Returns:
        An :class:`~openral_core.ApproachViewpoint` in the target's frame.
    """
    tx, ty, tz = target.xyz
    if approach_from is not None:
        ax, ay, _az = approach_from.xyz
        dx, dy = ax - tx, ay - ty
        norm = math.hypot(dx, dy)
        ux, uy = (dx / norm, dy / norm) if norm > _MIN_DIR_NORM else (-1.0, 0.0)
    else:
        ux, uy = -1.0, 0.0
    vp_x, vp_y = tx + ux * standoff_m, ty + uy * standoff_m
    # Yaw so the camera (looking down its +X) points from the viewpoint at the target.
    yaw = math.atan2(ty - vp_y, tx - vp_x)
    return ApproachViewpoint(
        pose=Pose6D(xyz=(vp_x, vp_y, tz), quat_xyzw=_xy_yaw_quat(yaw), frame_id=target.frame_id),
        standoff_m=standoff_m,
        camera_frame_id=camera_frame_id,
    )


class SpatialMemory:
    """Accumulating, queryable scene-graph spatial memory (ADR-0038 Phase 2).

    Args:
        assoc_distance_m: Radius for label-based instance association when a
            detection carries no ``track_id``.
        default_standoff_m: Standoff used for approach viewpoints.
        camera_frame_id: tf2 frame approach viewpoints orient toward.
        map_frame: Durable frame object poses are anchored in.
    """

    def __init__(
        self,
        *,
        assoc_distance_m: float = DEFAULT_ASSOC_DISTANCE_M,
        default_standoff_m: float = DEFAULT_STANDOFF_M,
        camera_frame_id: str = DEFAULT_CAMERA_FRAME,
        map_frame: str = DEFAULT_MAP_FRAME,
        embedder: TextEmbedder | None = None,
        min_text_similarity: float = DEFAULT_MIN_TEXT_SIMILARITY,
    ) -> None:
        """Initialize an empty memory.

        ``embedder`` (ADR-0038 §5, optional) enables open-vocabulary matching:
        object labels are embedded on creation and free-text queries match by
        CLIP cosine similarity (>= ``min_text_similarity``) in addition to
        label substring. Without it, matching is label + pose + recency only.
        """
        self._assoc_distance_m = assoc_distance_m
        self._default_standoff_m = default_standoff_m
        self._camera_frame_id = camera_frame_id
        self._map_frame = map_frame
        self._embedder = embedder
        self._min_text_similarity = min_text_similarity
        self._nodes: dict[str, SpatialNode] = {}
        self._edges: dict[tuple[str, str, SpatialRelationKind], SpatialEdge] = {}
        self._embeddings: dict[str, NDArray[np.float32]] = {}
        self._auto_counter = 0

    def _embed_node(self, node: SpatialNode) -> None:
        """Compute + cache the label embedding for an object node (no-op without an embedder)."""
        if self._embedder is None or node.kind is not SpatialNodeKind.OBJECT or not node.label:
            return
        self._embeddings[node.node_id] = self._embedder.embed_text([node.label])[0]

    # ── Mutation ──────────────────────────────────────────────────────────────

    def upsert_node(self, node: SpatialNode) -> None:
        """Insert or replace a node by ``node_id``."""
        self._nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str, kind: SpatialRelationKind) -> None:
        """Add a directed relation; src/dst must already exist (idempotent)."""
        if src not in self._nodes:
            raise KeyError(f"unknown src node: {src!r}")
        if dst not in self._nodes:
            raise KeyError(f"unknown dst node: {dst!r}")
        self._edges[(src, dst, kind)] = SpatialEdge(src=src, dst=dst, kind=kind)

    def ingest_detected_objects(
        self, objects: Sequence[DetectedObject], *, now_ns: int
    ) -> list[str]:
        """Fold a snapshot's detected objects into the graph (ADR-0038 §2).

        Instance association: a detection is matched to an existing ``object``
        node by ``track_id`` when present, else by identical label within
        ``assoc_distance_m``. A match updates the pose, bumps ``last_seen_ns``
        and ``observation_count`` and keeps the higher confidence; otherwise a
        new node is created. Object poses are expected in the map frame.

        Args:
            objects: Detections from a :class:`~openral_core.WorldState`.
            now_ns: Observation timestamp in nanoseconds.

        Returns:
            The node ids touched (created or updated), in input order.
        """
        touched: list[str] = []
        for obj in objects:
            match_id = self._associate(obj)
            if match_id is None:
                node_id = self._new_node_id(obj)
                self._nodes[node_id] = SpatialNode(
                    node_id=node_id,
                    kind=SpatialNodeKind.OBJECT,
                    pose=obj.pose,
                    label=obj.label,
                    confidence=obj.confidence,
                    bbox_3d=obj.bbox_3d,
                    first_seen_ns=now_ns,
                    last_seen_ns=now_ns,
                    observation_count=1,
                )
                self._embed_node(self._nodes[node_id])
                touched.append(node_id)
            else:
                prev = self._nodes[match_id]
                self._nodes[match_id] = prev.model_copy(
                    update={
                        "pose": obj.pose,
                        "bbox_3d": obj.bbox_3d if obj.bbox_3d is not None else prev.bbox_3d,
                        "confidence": max(prev.confidence, obj.confidence),
                        "last_seen_ns": max(prev.last_seen_ns, now_ns),
                        "observation_count": prev.observation_count + 1,
                    }
                )
                touched.append(match_id)
        return touched

    def _associate(self, obj: DetectedObject) -> str | None:
        """Return the id of the existing object node this detection updates, or None.

        The upstream ``track_id`` (ADR-0035 ``ObjectMemory``) is only a
        *within-session* hint: it is a per-session monotonic counter that resets
        whenever the world-state node restarts, so this durable memory must not
        treat it as a stable identity. We use it as a fast path **only when the
        label also matches** (so a recycled id can't merge a cup into a mug's
        node); durable identity is otherwise label + proximity, which also
        re-associates an object that returns under a fresh id (CLAUDE.md §1.4).
        """
        if obj.track_id is not None:
            track_key = f"obj_track_{obj.track_id}"
            node = self._nodes.get(track_key)
            if node is not None and node.label == obj.label:
                return track_key
        best_id: str | None = None
        best_d = self._assoc_distance_m
        for node in self._nodes.values():
            if node.kind is not SpatialNodeKind.OBJECT or node.label != obj.label:
                continue
            d = _xyz_distance(node.pose, obj.pose)
            if d <= best_d:
                best_d, best_id = d, node.node_id
        return best_id

    def _new_node_id(self, obj: DetectedObject) -> str:
        # Use the upstream track_id as the node key only when it's free. A
        # recycled id (world-state restart, ADR-0035 per-session counter) that
        # already names a different object must NOT clobber it — fall back to a
        # fresh auto-id (CLAUDE.md §1.4). See `_associate` for the matching rule.
        if obj.track_id is not None:
            track_key = f"obj_track_{obj.track_id}"
            if track_key not in self._nodes:
                return track_key
        self._auto_counter += 1
        slug = "".join(c if c.isalnum() else "_" for c in obj.label) or "object"
        return f"obj_{slug}_{self._auto_counter}"

    # ── Queries ─────────────────────────────────────────────────────────────────

    def recall_object(self, query: RecallObjectQuery, *, now_ns: int) -> RecallObjectResult:
        """Recall remembered objects matching ``query`` (ADR-0038 §6).

        Matches ``object`` nodes by exact (case-insensitive) or substring label
        against ``query.label`` / ``query.text``, applies the optional recency
        filter, ranks by match quality x confidence (proximity to
        ``query.near`` as a tiebreaker), and attaches a camera-facing approach
        viewpoint and any occluding container. Returns an empty result when
        nothing matches — the caller decides whether to raise
        :class:`~openral_core.exceptions.ROSObjectNotInMemory` or search.
        """
        term = (query.label or query.text).strip().lower()
        query_text = (query.text or query.label).strip()
        query_emb = (
            self._embedder.embed_text([query_text])[0]
            if self._embedder is not None and query_text
            else None
        )
        scored: list[tuple[float, float, SpatialNode]] = []
        for node in self._nodes.values():
            if node.kind is not SpatialNodeKind.OBJECT:
                continue
            if query.max_age_ns is not None and now_ns - node.last_seen_ns > query.max_age_ns:
                continue
            label_q = _label_match_quality(node.label, term)
            emb_sim = 0.0
            if query_emb is not None and node.node_id in self._embeddings:
                emb_sim = float(np.dot(query_emb, self._embeddings[node.node_id]))
            # Include on a label hit, or an embedding hit above the floor.
            if label_q <= 0.0 and emb_sim < self._min_text_similarity:
                continue
            score = max(label_q, emb_sim) * node.confidence
            proximity = _xyz_distance(node.pose, query.near) if query.near is not None else 0.0
            scored.append((score, proximity, node))

        # Rank: score desc, then nearer first, then most-recently seen.
        scored.sort(key=lambda t: (-t[0], t[1], -t[2].last_seen_ns))
        matches = [
            RecallObjectMatch(
                node_id=node.node_id,
                label=node.label,
                pose=node.pose,
                score=min(1.0, max(0.0, score)),
                last_seen_ns=node.last_seen_ns,
                approach=self._approach_for(node),
                inside_container_id=self._occluding_container_of(node.node_id),
            )
            for score, _prox, node in scored[: query.limit]
        ]
        return RecallObjectResult(matches=matches)

    def resolve_place(
        self, query: ResolvePlaceQuery, *, from_node_id: str | None = None
    ) -> ResolvePlaceResult:
        """Resolve a place/room/agent reference to a goal pose + path (ADR-0038 §6).

        Raises:
            ROSObjectNotInMemory: When the reference matches no node (the caller
                degrades to "unknown" / active search — it never fabricates a pose).
        """
        target = self._resolve_reference(query.reference, query.kind)
        if target is None:
            raise ROSObjectNotInMemory(
                f"no scene-graph node resolves reference {query.reference!r}"
            )
        goal_node = target
        if target.kind is not SpatialNodeKind.PLACE:
            place_id = self._at_place_of(target.node_id)
            if place_id is not None:
                goal_node = self._nodes[place_id]
        path = (
            self._traversable_path(from_node_id, goal_node.node_id)
            if from_node_id is not None
            else []
        )
        return ResolvePlaceResult(
            node_id=goal_node.node_id, goal=goal_node.pose, path_node_ids=path
        )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _approach_for(self, node: SpatialNode) -> ApproachViewpoint:
        place_id = self._at_place_of(node.node_id)
        approach_from = self._nodes[place_id].pose if place_id is not None else None
        return compute_approach_viewpoint(
            node.pose,
            standoff_m=self._default_standoff_m,
            camera_frame_id=self._camera_frame_id,
            approach_from=approach_from,
        )

    def _occluding_container_of(self, node_id: str) -> str | None:
        """Id of a container that ``contains`` ``node_id`` and occludes its contents."""
        for src, dst, kind in self._edges:
            if kind is SpatialRelationKind.CONTAINS and dst == node_id:
                container = self._nodes.get(src)
                if container is not None and container.occludes_contents:
                    return src
        return None

    def _at_place_of(self, node_id: str) -> str | None:
        for src, dst, kind in self._edges:
            if kind is SpatialRelationKind.AT_PLACE and src == node_id:
                return dst
        return None

    def _resolve_reference(
        self, reference: str, kind: SpatialNodeKind | None
    ) -> SpatialNode | None:
        # Exact node-id hit first.
        node = self._nodes.get(reference)
        if node is not None and (kind is None or node.kind is kind):
            return node
        ref = reference.strip().lower()
        best: SpatialNode | None = None
        best_quality = 0.0
        for node in self._nodes.values():
            if kind is not None and node.kind is not kind:
                continue
            quality = _label_match_quality(node.label, ref)
            if quality > best_quality:
                best_quality, best = quality, node
        return best

    def _traversable_path(self, start: str, goal: str) -> list[str]:
        """BFS shortest path over ``traversable_to`` edges; [] if none (or start==goal handled)."""
        if start == goal:
            return [start]
        adj: dict[str, list[str]] = {}
        for src, dst, kind in self._edges:
            if kind is SpatialRelationKind.TRAVERSABLE_TO:
                adj.setdefault(src, []).append(dst)
        prev: dict[str, str] = {}
        seen = {start}
        queue: deque[str] = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in adj.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = cur
                if nxt == goal:
                    return _reconstruct_path(prev, start, goal)
                queue.append(nxt)
        return []

    # ── Serialization / persistence ─────────────────────────────────────────────

    def to_scene_graph(self) -> SceneGraph:
        """Snapshot the memory as an immutable :class:`~openral_core.SceneGraph`."""
        return SceneGraph(nodes=list(self._nodes.values()), edges=list(self._edges.values()))

    @classmethod
    def from_scene_graph(
        cls, graph: SceneGraph, *, embedder: TextEmbedder | None = None
    ) -> SpatialMemory:
        """Build a memory from a persisted :class:`~openral_core.SceneGraph`.

        Embeddings are not serialized; when an ``embedder`` is supplied, object
        labels are re-embedded here (cheap, deterministic) so open-vocab queries
        work against a loaded graph.
        """
        mem = cls(embedder=embedder)
        for node in graph.nodes:
            mem._nodes[node.node_id] = node
            mem._embed_node(node)
        for edge in graph.edges:
            mem._edges[(edge.src, edge.dst, edge.kind)] = edge
        return mem

    def save(self, path: str | Path) -> None:
        """Persist the scene graph to ``path`` as JSON (the SceneGraph contract)."""
        Path(path).write_text(self.to_scene_graph().model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path, *, embedder: TextEmbedder | None = None) -> SpatialMemory:
        """Load a memory from a JSON scene graph written by :meth:`save`."""
        graph = SceneGraph.model_validate_json(Path(path).read_text())
        return cls.from_scene_graph(graph, embedder=embedder)


def _xyz_distance(a: Pose6D, b: Pose6D) -> float:
    ax, ay, az = a.xyz
    bx, by, bz = b.xyz
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _label_match_quality(label: str, term: str) -> float:
    """1.0 exact (case-insensitive), 0.7 substring either way, 0.0 no match."""
    if not term:
        return 0.0
    low = label.strip().lower()
    if low == term:
        return 1.0
    if term in low or low in term:
        return 0.7
    return 0.0


def _reconstruct_path(prev: dict[str, str], start: str, goal: str) -> list[str]:
    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path
