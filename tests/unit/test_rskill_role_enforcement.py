"""``role`` is enforced (CLAUDE.md §3): it must be the slot ``kind`` runs in, never ``s0``.

The rule lives on ``RSkillManifest`` so every parse enforces it — the loader, the
reasoner palette and the perception nodes all read manifests directly. Real in-tree
manifests only; the refusal cases copy one and rewrite its ``role:`` line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
from openral_core import RSKILL_ROLE_BY_KIND, RSkillKind
from openral_core.exceptions import ROSCapabilityMismatch
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


def test_every_kind_has_a_role() -> None:
    assert set(RSKILL_ROLE_BY_KIND) == set(get_args(RSkillKind))
    assert "s0" not in RSKILL_ROLE_BY_KIND.values()


@pytest.mark.parametrize("path", _MANIFESTS, ids=lambda p: p.parent.name)
def test_every_intree_manifest_declares_its_kinds_role(path: Path) -> None:
    m = RSkillManifest.from_yaml(str(path))
    assert m.role == RSKILL_ROLE_BY_KIND[m.kind]


@pytest.mark.parametrize(
    ("src", "role", "match"),
    [
        (_VLA, "s0", "S0 cerebellar control ships as ros2_control"),
        (_VLA, "s2", "kind='vla' requires role='s1'"),
        (_VLM, "s1", "kind='vlm' requires role='s2'"),
        # The schema default is s1, so a VLM that omits `role` is refused too.
        (_VLM, None, "kind='vlm' requires role='s2'"),
    ],
)
def test_mismatched_role_is_refused_by_every_entry_point(
    src: Path, role: str | None, match: str, tmp_path: Path
) -> None:
    path = _with_role(src, role, tmp_path)
    with pytest.raises(ValidationError, match=match):
        RSkillManifest.from_yaml(str(path))
    with pytest.raises(ValidationError, match=match):
        rSkill.from_yaml(path)
    with pytest.raises(ValidationError, match=match):
        load_rskill_manifest(str(path))


def test_s1_dispatch_refuses_an_s2_skill() -> None:
    vlm = RSkillManifest.from_yaml(str(_VLM))
    vla = RSkillManifest.from_yaml(str(_VLA))
    rSkill.check_s1_role(vla.name, vla.role)
    with capture_logs() as logs, pytest.raises(ROSCapabilityMismatch, match="role='s2'"):
        rSkill.check_s1_role(vlm.name, vlm.role)
    assert logs[0]["event"] == "rskill.slot_refused"
