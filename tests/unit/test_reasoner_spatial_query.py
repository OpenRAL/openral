"""ADR-0039 Phase 2 — read-only spatial-memory query tools in the reasoner.

Covers three pure-Python slices (the ROS ``reasoner_node`` dispatch is Phase 2b):
- palette rendering gates ``recall_object`` / ``resolve_place`` on
  ``ToolPalette.spatial_memory_available``;
- the ``ReasonerToolCall`` decoder routes their payloads to the right variant;
- the ``run_spatial_query`` bridge maps a tool call → ADR-0038 query, runs it
  against a **real** ``SpatialMemory`` (loaded from the home fixture — no mock,
  CLAUDE.md §1.11), and renders an LLM-readable result for the prompt cascade.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openral_core import RecallObjectTool, ResolvePlaceTool
from openral_reasoner import ToolPalette, run_spatial_query
from openral_reasoner.tool_use import _decode_tool_payload, _tool_palette_to_anthropic_tools
from openral_world_state import SpatialMemory

if TYPE_CHECKING:
    from openral_world_state.grid import OccupancyGridIndex

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


def _memory() -> SpatialMemory:
    return SpatialMemory.load(_FIXTURE)


# ── Palette rendering gate ────────────────────────────────────────────────────


def test_query_tools_absent_by_default() -> None:
    """Without a wired backend the query tools are NOT offered to the LLM."""
    names = {t["name"] for t in _tool_palette_to_anthropic_tools(ToolPalette())}
    assert "recall_object" not in names
    assert "resolve_place" not in names


def test_query_tools_present_when_available() -> None:
    """With spatial_memory_available the two read-only tools are offered."""
    palette = ToolPalette(spatial_memory_available=True)
    names = {t["name"] for t in _tool_palette_to_anthropic_tools(palette)}
    assert "recall_object" in names
    assert "resolve_place" in names


# ── Decode routing ────────────────────────────────────────────────────────────


def test_decode_routes_recall_object_and_resolve_place() -> None:
    palette = ToolPalette(spatial_memory_available=True)
    found = _decode_tool_payload(
        tool_name="recall_object", arguments={"query": "the red mug", "limit": 3}, palette=palette
    )
    assert isinstance(found, RecallObjectTool)
    assert found.query == "the red mug"
    place = _decode_tool_payload(
        tool_name="resolve_place", arguments={"reference": "the kitchen"}, palette=palette
    )
    assert isinstance(place, ResolvePlaceTool)
    assert place.reference == "the kitchen"


# ── run_spatial_query bridge over the real wine fixture ───────────────────────


def test_recall_object_bridge_reports_pose_and_occluding_container() -> None:
    text = run_spatial_query(
        RecallObjectTool(query="bottle of wine"), _memory(), now_ns=2_000_000_000_000
    )
    assert "wine_bottle" in text
    assert "INSIDE container 'fridge'" in text  # planner must open the fridge first


def test_recall_object_bridge_miss_suggests_search() -> None:
    text = run_spatial_query(RecallObjectTool(query="teapot"), _memory(), now_ns=2_000_000_000_000)
    assert "not in memory" in text
    assert "search" in text.lower()


def test_resolve_place_bridge_returns_path() -> None:
    text = run_spatial_query(
        ResolvePlaceTool(reference="fridge"),
        _memory(),
        now_ns=2_000_000_000_000,
        from_node_id="living_room_sofa",
    )
    assert "front_of_fridge" in text
    assert "Path:" in text
    assert "kitchen_table" in text


def test_resolve_place_bridge_requester_return_goal() -> None:
    text = run_spatial_query(
        ResolvePlaceTool(reference="the person who asked"), _memory(), now_ns=2_000_000_000_000
    )
    assert "living_room_sofa" in text


def test_resolve_place_bridge_unknown_is_text_not_exception() -> None:
    text = run_spatial_query(
        ResolvePlaceTool(reference="garage"), _memory(), now_ns=2_000_000_000_000
    )
    assert "not in memory" in text


# ── ADR-0044 Phase 4 — occupancy-grid approach refinement in the bridge ──────


def _grid_5x3(occupied: list[tuple[slice, slice]]) -> OccupancyGridIndex:
    """A 5 m x 3 m grid at 0.1 m covering the home fixture's wine-bottle area."""
    import numpy as np

    data = np.zeros((30, 50), dtype=np.int8)
    for rows, cols in occupied:
        data[rows, cols] = 100
    from openral_world_state.grid import OccupancyGridIndex

    return OccupancyGridIndex(data, resolution_m=0.1, origin_xy=(0.0, 0.0))


def _refiner_for(grid: OccupancyGridIndex):
    from openral_world_state.grid import refine_approach_pose

    def refiner(viewpoint, target_xyz):
        return refine_approach_pose(grid, viewpoint, target_xyz, inflation_m=0.15)

    return refiner


def test_recall_object_grid_refines_blocked_approach() -> None:
    """A wall on the ideal approach snaps the rendered viewpoint to a free cell.

    The wine bottle sits at (3.42, 0.38); its geometric approach is (2.82, 0.38).
    A wall at x in [2.7, 3.0] blocks that ideal cell, so the refiner must land on
    the bottle's side of the wall — and the LLM must see the SNAPPED pose.
    """
    grid = _grid_5x3([(slice(0, 8), slice(27, 30))])  # wall x in [2.7, 3.0), y in [0, 0.8)
    text = run_spatial_query(
        RecallObjectTool(query="bottle of wine", rationale="grid test"),
        _memory(),
        now_ns=2_000_000_000,
        refine_approach=_refiner_for(grid),
    )
    assert "approach from (2.82, 0.38" not in text, f"ideal pose leaked through: {text}"
    assert "approach from" in text, f"no refined approach rendered: {text}"
    # The snapped viewpoint must be free + sighted on the grid it was refined on.
    import re

    m = re.search(r"approach from \(([-\d.]+), ([-\d.]+), [-\d.]+\)", text)
    assert m is not None
    x, y = float(m.group(1)), float(m.group(2))
    assert grid.is_free(x, y, inflation_m=0.15)
    assert grid.line_of_sight((x, y), (3.42, 0.38))


def test_recall_object_grid_blocked_renders_blocked_not_fabricated() -> None:
    """No reachable viewpoint → the match is rendered BLOCKED, with no pose."""
    # Entomb the bottle: occupied x in [2.2, 4.7), y in [0, 1.5) — every free
    # cell within the standoff ceiling has the block between it and the target.
    grid = _grid_5x3([(slice(0, 15), slice(22, 47))])
    text = run_spatial_query(
        RecallObjectTool(query="bottle of wine", rationale="grid test"),
        _memory(),
        now_ns=2_000_000_000,
        refine_approach=_refiner_for(grid),
    )
    assert "approach BLOCKED on the occupancy grid" in text, text
    assert "approach from" not in text, f"fabricated viewpoint rendered: {text}"


def test_recall_object_no_refiner_keeps_geometric_approach() -> None:
    """Without a grid (refine_approach=None) the geometric viewpoint passes through."""
    text = run_spatial_query(
        RecallObjectTool(query="bottle of wine", rationale="grid test"),
        _memory(),
        now_ns=2_000_000_000,
    )
    assert "approach from (2.82, 0.38" in text, text
