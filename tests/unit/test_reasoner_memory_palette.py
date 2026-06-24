"""ADR-0072 §3 / Phase 4c — memory tools in the reasoner palette + decode routing.

Pure-Python slices (the ROS ``reasoner_node`` dispatch is covered live in
``tests/integration/test_reasoner_node_end_to_end.py``):
- palette rendering gates ``memory_write`` / ``memory_search`` on
  ``ToolPalette.memory_available`` (off by default — no dispatcher, no tool);
- the ``ReasonerToolCall`` decoder routes their payloads to the right variant.

Run with:
    uv run pytest tests/unit/test_reasoner_memory_palette.py -v
"""

from __future__ import annotations

from openral_core import MemorySearchTool, MemoryWriteTool
from openral_reasoner import ToolPalette
from openral_reasoner.tool_use import _decode_tool_payload, _tool_palette_to_anthropic_tools


def test_memory_tools_absent_by_default() -> None:
    """Without a wired MEMORY.md the write/search tools are NOT offered to the LLM."""
    names = {t["name"] for t in _tool_palette_to_anthropic_tools(ToolPalette())}
    assert "memory_write" not in names
    assert "memory_search" not in names


def test_memory_tools_present_when_available() -> None:
    """With memory_available both the write and the read-only recall tool are offered."""
    palette = ToolPalette(memory_available=True)
    names = {t["name"] for t in _tool_palette_to_anthropic_tools(palette)}
    assert "memory_write" in names
    assert "memory_search" in names


def test_decode_routes_memory_write_and_search() -> None:
    palette = ToolPalette(memory_available=True)
    write = _decode_tool_payload(
        tool_name="memory_write",
        arguments={
            "op": "add",
            "section": "preferences",
            "content": "Clothes go in the bedroom drawer.",
            "importance": 0.9,
        },
        palette=palette,
    )
    assert isinstance(write, MemoryWriteTool)
    assert write.op == "add"
    assert write.section == "preferences"
    search = _decode_tool_payload(
        tool_name="memory_search",
        arguments={"query": "where is the mug", "section": "object_locations", "limit": 3},
        palette=palette,
    )
    assert isinstance(search, MemorySearchTool)
    assert search.query == "where is the mug"
    assert search.limit == 3
