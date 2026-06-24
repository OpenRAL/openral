"""ADR-0071 §3 / Phase 4b — the self-maintained ``MEMORY.md`` file model.

A persistent, human-readable **semantic** memory for the S2 reasoner —
complementary to the ADR-0038 *geometric* scene graph. Holds preferences,
corrections/lessons, durable home facts, an object-location log, and open
tasks. The reasoner reads it as the ``## MEMORY`` context block and edits it
through the :class:`~openral_core.MemoryWriteTool` op (never a free-form
rewrite). Advisory only — a wrong memory yields a bad plan the C++ safety
kernel still vetoes (CLAUDE.md §1.1).

The file is the source of truth (human-editable). Each entry is one line in a
fixed, round-trip-stable format::

    ## User Preferences
    - [imp:0.90 ts:2026-06-24 st:current] Clothes go in the bedroom drawer.

so a human edit that keeps the format is parsed, and re-rendering is stable.
"""

from __future__ import annotations

import dataclasses
import re

from openral_core import MemorySection

__all__ = ["MemoryEntry", "MemoryStore"]

# Section discriminator → human-readable MEMORY.md heading (fixed render order).
_SECTION_TITLES: dict[MemorySection, str] = {
    "home_map": "Home Map / Places",
    "preferences": "User Preferences",
    "lessons": "Learned Lessons / Corrections",
    "object_locations": "Object-Location Log",
    "open_tasks": "Open Tasks / Commitments",
}
_TITLE_TO_SECTION: dict[str, MemorySection] = {v: k for k, v in _SECTION_TITLES.items()}

# `- [imp:<f> ts:<token> st:<status>] <content>`
_ENTRY_RE = re.compile(r"^- \[imp:([0-9.]+) ts:(\S+) st:(\w+)\] (.*)$")

_SCHEMA_VERSION = "0.1"


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One remembered fact in a :class:`MemoryStore`."""

    section: MemorySection
    content: str
    importance: float = 0.5
    timestamp: str = ""  # iso8601 (when added/updated); "" renders as "-"
    status: str = "current"  # "current" | "stale" (superseded prior, kept as a search hint)

    def render_line(self) -> str:
        """Render the entry as one round-trip-stable MEMORY.md line."""
        ts = self.timestamp or "-"
        return f"- [imp:{self.importance:.2f} ts:{ts} st:{self.status}] {self.content}"


class MemoryStore:
    """Ordered set of :class:`MemoryEntry` rendered to / parsed from ``MEMORY.md``."""

    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        """Hold an ordered list of memory entries (empty by default)."""
        self._entries: list[MemoryEntry] = list(entries or [])

    # ── parse / render ──────────────────────────────────────────────────────

    @classmethod
    def from_markdown(cls, text: str) -> MemoryStore:
        """Parse a ``MEMORY.md`` body. Unrecognized lines are ignored (lenient)."""
        entries: list[MemoryEntry] = []
        current: MemorySection | None = None
        for raw in text.splitlines():
            line = raw.rstrip()
            if line.startswith("## "):
                current = _TITLE_TO_SECTION.get(line[3:].strip())
                continue
            if current is None:
                continue
            m = _ENTRY_RE.match(line)
            if m is None:
                continue
            imp, ts, status, content = m.groups()
            entries.append(
                MemoryEntry(
                    section=current,
                    content=content,
                    importance=float(imp),
                    timestamp="" if ts == "-" else ts,
                    status=status,
                )
            )
        return cls(entries)

    def _render_sections(self) -> str:
        blocks: list[str] = []
        for section, title in _SECTION_TITLES.items():
            blocks.append(f"## {title}")
            rows = [e.render_line() for e in self._entries if e.section == section]
            blocks.append("\n".join(rows) if rows else "(none)")
            blocks.append("")
        return "\n".join(blocks).rstrip() + "\n"

    def to_markdown(self) -> str:
        """The full on-disk ``MEMORY.md`` (title + schema marker + sections)."""
        header = f"# MEMORY.md\n<!-- schema_version: {_SCHEMA_VERSION} -->\n\n"
        return f"{header}{self._render_sections()}"

    def to_context_block(self) -> str:
        """The ``## MEMORY`` block the reasoner injects into its context."""
        return f"## MEMORY\n{self._render_sections().rstrip()}"

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        """A snapshot of the current entries, in order."""
        return tuple(self._entries)

    # ── edits (MemoryWriteTool ops) ─────────────────────────────────────────

    def apply(
        self,
        *,
        op: str,
        section: MemorySection,
        content: str,
        importance: float,
        target: str | None,
        now: str,
    ) -> MemoryEntry | None:
        """Apply one explicit edit. Returns an entry to archive (or ``None``).

        * ``add`` — append a new current entry.
        * ``update`` — replace the ``target`` entry's content in place (archives
          the old version); appends if ``target`` is absent.
        * ``supersede`` — mark current ``target`` entries ``stale`` (kept in-file
          as a search prior) and append the new current entry. Archives nothing.
        * ``delete`` — remove the ``target`` entry (archives it).
        """
        new = MemoryEntry(section, content, importance, now, "current")
        if op == "add":
            self._entries.append(new)
            return None
        if op == "update":
            for i, e in enumerate(self._entries):
                if e.section == section and e.content == target:
                    self._entries[i] = new
                    return e
            self._entries.append(new)
            return None
        if op == "supersede":
            for i, e in enumerate(self._entries):
                if e.section == section and e.content == target and e.status == "current":
                    self._entries[i] = dataclasses.replace(e, status="stale")
            self._entries.append(new)
            return None
        if op == "delete":
            for i, e in enumerate(self._entries):
                if e.section == section and e.content == target:
                    return self._entries.pop(i)
            return None
        raise ValueError(f"MemoryStore.apply: unknown op {op!r}")

    # ── archival search ─────────────────────────────────────────────────────

    @staticmethod
    def search(
        archive: list[MemoryEntry],
        *,
        query: str,
        section: MemorySection | None,
        limit: int,
    ) -> list[MemoryEntry]:
        """Rank archived entries by a keyword match (MemGPT recall).

        Any query token appearing in an entry's content matches; results are
        sorted by importance then recency and truncated to ``limit``.
        """
        terms = [t for t in query.lower().split() if t]
        hits = [
            e
            for e in archive
            if (section is None or e.section == section)
            and any(t in e.content.lower() for t in terms)
        ]
        hits.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        return hits[:limit]
