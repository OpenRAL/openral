"""``role`` is the slot ``kind`` runs in (CLAUDE.md §3): derived when omitted, refused on mismatch.

The rule lives on ``RSkillManifest`` so every parse enforces it — the loader, the reasoner
palette and the perception nodes all read manifests. The loader turns a refusal into a
logged ``ROSConfigError`` (CLAUDE.md §5). Dispatch is a separate, kind-based rule
(``EXECUTE_RSKILL_KINDS``). Real in-tree manifests only; the refusal cases copy one and
rewrite its ``role:`` line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
from openral_core import EXECUTE_RSKILL_KINDS, RSKILL_ROLE_BY_KIND, RSkillKind
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_core.schemas import RSkillManifest
from openral_rskill.loader import load_rskill_manifest, rSkill
from pydantic import ValidationError
from structlog.testing import capture_logs

_REPO = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted((_REPO / "rskills").glob("*/rskill.yaml"))
_VLA = _REPO / "rskills" / "smolvla-libero" / "rskill.yaml"
_VLM = _REPO / "rskills" / "qwen35-4b-nf4" / "rskill.yaml"


def _with_role(src: Path, role: str | None, tmp_path: Path) -> Path:
    """Copy ``src`` with its ``role:`` line set to ``role`` (``None`` drops it)."""
    line = "" if role is None else f'role: "{role}"'
    text, n = re.subn(r"(?m)^role:.*$", line, src.read_text(encoding="utf-8"))
    assert n == 1
    dst = tmp_path / "rskill.yaml"
    dst.write_text(text, encoding="utf-8")
    return dst


def test_every_kind_has_a_role_and_only_s1_kinds_dispatch() -> None:
    assert set(RSKILL_ROLE_BY_KIND) == set(get_args(RSkillKind))
    assert "s0" not in RSKILL_ROLE_BY_KIND.values()
    s1_kinds = {k for k, r in RSKILL_ROLE_BY_KIND.items() if r == "s1"}
    assert EXECUTE_RSKILL_KINDS.issubset(s1_kinds)


@pytest.mark.parametrize("path", _MANIFESTS, ids=lambda p: p.parent.name)
def test_every_intree_manifest_declares_its_kinds_role(path: Path) -> None:
    m = RSkillManifest.from_yaml(str(path))
    assert m.role == RSKILL_ROLE_BY_KIND[m.kind]


@pytest.mark.parametrize("src", [_VLA, _VLM], ids=["vla", "vlm"])
def test_an_omitted_role_is_derived_from_kind(src: Path, tmp_path: Path) -> None:
    m = RSkillManifest.from_yaml(str(_with_role(src, None, tmp_path)))
    assert m.role == RSKILL_ROLE_BY_KIND[m.kind]


@pytest.mark.parametrize(
    ("src", "role", "match"),
    [
        (_VLA, "s0", "S0 cerebellar control ships as ros2_control"),
        (_VLA, "s2", "kind='vla' requires role='s1'"),
        (_VLM, "s1", "kind='vlm' requires role='s2'"),
    ],
)
def test_a_mismatched_role_is_refused(src: Path, role: str, match: str, tmp_path: Path) -> None:
    path = _with_role(src, role, tmp_path)
    # The schema layer raises pydantic's error ...
    with pytest.raises(ValidationError, match=match):
        RSkillManifest.from_yaml(str(path))
    # ... the loader turns it into a typed, logged ROSConfigError.
    for load in (lambda: rSkill.from_yaml(path), lambda: load_rskill_manifest(str(path))):
        with capture_logs() as logs, pytest.raises(ROSConfigError, match=match):
            load()
        refused = next(e for e in logs if e["event"] == "rskill.manifest_refused")
        assert any(match in err for err in refused["errors"])


def test_only_execute_rskill_kinds_are_dispatchable() -> None:
    vla = RSkillManifest.from_yaml(str(_VLA))
    vlm = RSkillManifest.from_yaml(str(_VLM))
    rSkill.check_dispatchable(vla)
    with capture_logs() as logs, pytest.raises(ROSCapabilityMismatch, match="kind='vlm'"):
        rSkill.check_dispatchable(vlm)
    assert logs[0]["event"] == "rskill.dispatch_refused"
