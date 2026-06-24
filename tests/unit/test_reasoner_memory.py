"""Tests for the self-maintained MEMORY.md file model (ADR-0071 §3 / Phase 4b).

Covers :class:`~openral_reasoner.memory.MemoryStore` (apply ops, supersession,
round-trip render/parse, archival search) and the ``## MEMORY`` context section.

Run with:
    uv run pytest tests/unit/test_reasoner_memory.py -v
"""

from __future__ import annotations

from openral_reasoner.context import ContextRenderer
from openral_reasoner.memory import MemoryEntry, MemoryStore

_NOW = "2026-06-24T12:00:00"


def test_add_then_render_round_trips() -> None:
    s = MemoryStore()
    assert (
        s.apply(
            op="add",
            section="preferences",
            content="Clothes in the drawer",
            importance=0.9,
            target=None,
            now=_NOW,
        )
        is None
    )
    s.apply(
        op="add",
        section="lessons",
        content="Grasp mugs by the handle",
        importance=0.8,
        target=None,
        now=_NOW,
    )
    md = s.to_markdown()
    assert "## User Preferences" in md and "Clothes in the drawer" in md
    # Round-trip stable: parse the rendered file back to identical entries.
    assert MemoryStore.from_markdown(md).entries == s.entries


def test_update_replaces_and_archives_old() -> None:
    s = MemoryStore([MemoryEntry("preferences", "old rule", 0.5, _NOW, "current")])
    archived = s.apply(
        op="update",
        section="preferences",
        content="new rule",
        importance=0.7,
        target="old rule",
        now=_NOW,
    )
    assert archived is not None and archived.content == "old rule"
    assert [e.content for e in s.entries] == ["new rule"]


def test_supersede_marks_old_stale_and_keeps_it() -> None:
    s = MemoryStore([MemoryEntry("object_locations", "bottle on counter", 0.9, _NOW, "current")])
    s.apply(
        op="supersede",
        section="object_locations",
        content="bottle in fridge",
        importance=0.9,
        target="bottle on counter",
        now=_NOW,
    )
    by_status = {e.content: e.status for e in s.entries}
    assert by_status == {"bottle on counter": "stale", "bottle in fridge": "current"}


def test_delete_removes_and_archives() -> None:
    s = MemoryStore([MemoryEntry("open_tasks", "water the plants", 0.5, _NOW, "current")])
    archived = s.apply(
        op="delete",
        section="open_tasks",
        content="",
        importance=0.5,
        target="water the plants",
        now=_NOW,
    )
    assert archived is not None and not s.entries


def test_archival_search_ranks_by_importance() -> None:
    archive = [
        MemoryEntry("object_locations", "mug in cupboard", 0.4, "2026-06-20", "stale"),
        MemoryEntry("object_locations", "mug on the table", 0.9, "2026-06-22", "stale"),
        MemoryEntry("preferences", "quiet after 22:00", 0.7, "2026-06-21", "current"),
    ]
    hits = MemoryStore.search(archive, query="mug", section="object_locations", limit=5)
    assert [h.content for h in hits] == ["mug on the table", "mug in cupboard"]


def test_context_renderer_includes_memory_section_when_set() -> None:
    s = MemoryStore()
    s.apply(
        op="add",
        section="home_map",
        content="kitchen — north",
        importance=0.5,
        target=None,
        now=_NOW,
    )
    r = ContextRenderer()
    r.set_memory_block(s.to_context_block())
    out = r.render(world_state=None)
    assert "## MEMORY" in out
    assert "kitchen — north" in out
    # Ordering: MEMORY sits before WORLD_STATE.
    assert out.index("## MEMORY") < out.index("## WORLD_STATE")
    # Omitted when not set.
    assert "## MEMORY" not in ContextRenderer().render(world_state=None)
