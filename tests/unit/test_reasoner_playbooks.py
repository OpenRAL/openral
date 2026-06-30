"""Tests for playbook system-prompt injection (ADR-0072 Decision 1 / Phase 3).

Covers :func:`~openral_reasoner.context.render_playbooks_block` against the real
in-tree ``rskills/find-object`` playbook (no synthetic placeholders, §1.11).

Run with:
    uv run pytest tests/unit/test_reasoner_playbooks.py -v
"""

from __future__ import annotations

import pathlib

from openral_core import RSkillManifest
from openral_reasoner.context import render_playbooks_block

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FIND_OBJECT = _REPO_ROOT / "rskills" / "find-object"


def test_empty_block_is_noop() -> None:
    assert render_playbooks_block([]) == ""


def test_block_injects_real_find_object_playbook() -> None:
    m = RSkillManifest.from_yaml(str(_FIND_OBJECT / "rskill.yaml"))
    assert m.kind == "playbook" and m.playbook is not None
    body = (_FIND_OBJECT / m.playbook.body_uri).read_text()
    label = m.name.split("/")[-1].removeprefix("rskill-")
    block = render_playbooks_block([(f"{label} — {m.playbook.trigger}", body)])

    assert block.startswith("## PLAYBOOKS")
    assert "execute_rskill and the safety kernel" in block  # the guard line
    assert "A playbook is NOT a skill" in block
    assert f"--- playbook: {label} — {m.playbook.trigger} ---" in block
    assert m.name not in block
    # The SOP body (its own headings) is carried verbatim.
    assert "## Steps" in block
    assert "recall_object" in block


def test_multiple_playbooks_each_delimited() -> None:
    block = render_playbooks_block([("a — t1", "## Steps\n1. x"), ("b — t2", "## Steps\n1. y")])
    assert block.count("--- playbook:") == 2
    assert "--- playbook: a — t1 ---" in block
    assert "--- playbook: b — t2 ---" in block
