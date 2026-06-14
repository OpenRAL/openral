"""Open-vocabulary CLIP matching for SpatialMemory (ADR-0038 §5, Phase 4).

Exercises the real OpenCLIP ViT-B/32 embedder (CLAUDE.md §1.11 — no mocks;
§1.12 — uses the local GPU when present). Skips cleanly when the ``clip`` group
is not installed or the weights cannot be fetched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import DetectedObject, Pose6D, RecallObjectQuery
from openral_core.exceptions import ROSConfigError
from openral_world_state import OpenClipEmbedder, SpatialMemory

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


@pytest.fixture(scope="module")
def clip_embedder() -> OpenClipEmbedder:
    pytest.importorskip("open_clip", reason="install the clip group: uv sync --group clip")
    try:
        return OpenClipEmbedder()
    except ROSConfigError as exc:  # pragma: no cover - network/weights unavailable
        pytest.skip(f"OpenCLIP weights unavailable: {exc}")


def _obj(label: str, xyz: tuple[float, float, float], track_id: int) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=0.9,
        pose=Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        track_id=track_id,
    )


def _populated(embedder: OpenClipEmbedder) -> SpatialMemory:
    mem = SpatialMemory(embedder=embedder)
    mem.ingest_detected_objects(
        [
            _obj("bottle of wine", (3.0, 0.0, 1.0), 1),
            _obj("wine glass", (4.0, 2.0, 1.4), 2),
            _obj("coffee mug", (2.0, 1.0, 0.8), 3),
            _obj("fridge", (3.4, 0.4, 0.9), 4),
        ],
        now_ns=1000,
    )
    return mem


def test_embedder_is_normalized_512d(clip_embedder: OpenClipEmbedder) -> None:
    import numpy as np

    assert clip_embedder.dim == 512
    v = clip_embedder.embed_text(["bottle of wine"])
    assert v.shape == (1, 512)
    assert np.isclose(float(np.linalg.norm(v[0])), 1.0, atol=1e-4)


def test_openvocab_matches_synonym_with_no_substring(clip_embedder: OpenClipEmbedder) -> None:
    """'red wine' is not a substring of any label, but CLIP recalls the wine bottle."""
    mem = _populated(clip_embedder)
    res = mem.recall_object(RecallObjectQuery(text="red wine"), now_ns=2000)
    assert res.matches, "open-vocab query 'red wine' should match the wine bottle"
    assert res.matches[0].node_id == "obj_track_1"  # bottle of wine ranks first
    ids = {m.node_id for m in res.matches}
    assert "obj_track_4" not in ids  # fridge is below the similarity floor


def test_openvocab_ranks_glass_query(clip_embedder: OpenClipEmbedder) -> None:
    mem = _populated(clip_embedder)
    res = mem.recall_object(RecallObjectQuery(text="a glass for wine"), now_ns=2000)
    assert res.matches[0].node_id == "obj_track_2"  # wine glass


def test_without_embedder_substring_only_misses_synonym() -> None:
    """Without an embedder the synonym misses, but exact substring still matches."""
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("bottle of wine", (3.0, 0.0, 1.0), 1)], now_ns=1000)
    assert mem.recall_object(RecallObjectQuery(text="red wine"), now_ns=2000).matches == []
    assert mem.recall_object(RecallObjectQuery(label="wine"), now_ns=2000).matches


def test_load_fixture_with_embedder_reembeds_labels(clip_embedder: OpenClipEmbedder) -> None:
    """A persisted scene graph loaded with an embedder answers open-vocab queries."""
    if not _FIXTURE.exists():
        pytest.skip(f"scene-graph fixture missing: {_FIXTURE}")
    mem = SpatialMemory.load(_FIXTURE, embedder=clip_embedder)
    res = mem.recall_object(RecallObjectQuery(text="red wine"), now_ns=2_000_000_000_000)
    assert any(m.node_id == "wine_bottle" for m in res.matches)
