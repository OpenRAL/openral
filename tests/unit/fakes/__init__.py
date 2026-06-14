"""Test-only fakes — CLAUDE.md §1.11 boundary doubles for the unit tier.

Production code never imports from here. Each module documents which
real Protocol it satisfies and which slice of that Protocol the fake
implements.
"""

from __future__ import annotations
