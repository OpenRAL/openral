"""``role`` is the slot ``kind`` runs in (CLAUDE.md §3), and dispatch is decided by kind.

``role`` is derived when omitted. A declared mismatch is refused under schema 0.2 and
migrated with a warning under 0.1; ``s0`` is refused in both. The rule lives on
``RSkillManifest`` so every parse enforces it; the loader turns a refusal into a
``ROSConfigError`` naming the failing field (CLAUDE.md §5). Real in-tree manifests
only; the refusal cases copy one and rewrite lines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
import yaml
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


def _copy(
    src: Path, tmp_path: Path, *, role: str | None = "", version: str = "0.2", extra: str = ""
) -> Path:
    """Copy ``src`` with ``role`` (``None`` drops it, ``""`` keeps it) and ``schema_version``."""
    text = src.read_text(encoding="utf-8")
    if role != "":
        line = "" if role is None else f'role: "{role}"'
        text, n = re.subn(r"(?m)^role:.*$", line, text)
        assert n == 1
    text, n = re.subn(r"(?m)^schema_version:.*$", f'schema_version: "{version}"', text)
    assert n == 1
    dst = tmp_path / "rskill.yaml"
    dst.write_text(text + extra, encoding="utf-8")
    return dst


def test_every_kind_has_a_role_and_only_s1_kinds_dispatch() -> None:
    assert set(RSKILL_ROLE_BY_KIND) == set(get_args(RSkillKind))
    s1_kinds = {k for k, r in RSKILL_ROLE_BY_KIND.items() if r == "s1"}
    assert EXECUTE_RSKILL_KINDS.issubset(s1_kinds)


@pytest.mark.parametrize("path", _MANIFESTS, ids=lambda p: p.parent.name)
def test_every_intree_manifest_declares_its_kinds_role(path: Path) -> None:
    """Checked on the raw YAML: loading a 0.1 file would migrate a mismatch, not fail."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = RSKILL_ROLE_BY_KIND[raw["kind"]]
    assert raw.get("role", expected) == expected


@pytest.mark.parametrize("src", [_VLA, _VLM], ids=["vla", "vlm"])
def test_an_omitted_role_is_derived_from_kind(src: Path, tmp_path: Path) -> None:
    m = RSkillManifest.from_yaml(str(_copy(src, tmp_path, role=None)))
    assert m.role == RSKILL_ROLE_BY_KIND[m.kind]


@pytest.mark.parametrize(
    ("src", "role", "version", "match"),
    [
        (_VLA, "s0", "0.2", "S0 cerebellar control ships as ros2_control"),
        (_VLA, "s0", "0.1", "S0 cerebellar control ships as ros2_control"),
        (_VLA, "s2", "0.2", "kind='vla' requires role='s1'"),
        (_VLM, "s1", "0.2", "kind='vlm' requires role='s2'"),
    ],
)
def test_a_refused_role_is_typed_and_names_the_rule(
    src: Path, role: str, version: str, match: str, tmp_path: Path
) -> None:
    path = _copy(src, tmp_path, role=role, version=version)
    # The schema layer raises pydantic's error ...
    with pytest.raises(ValidationError, match=match):
        RSkillManifest.from_yaml(str(path))
    # ... the loader turns it into a ROSConfigError, logged at DEBUG only (a CLI
    # prints JSON on stdout, where structlog's stock renderer also writes).
    for load in (lambda: rSkill.from_yaml(path), lambda: load_rskill_manifest(str(path))):
        with capture_logs() as logs, pytest.raises(ROSConfigError, match=match):
            load()
        assert not [e for e in logs if e["log_level"] not in ("debug", "info")]


@pytest.mark.parametrize(("src", "role"), [(_VLA, "s2"), (_VLM, "s1")])
def test_a_schema_0_1_role_mismatch_is_migrated_with_a_warning(
    src: Path, role: str, tmp_path: Path
) -> None:
    path = _copy(src, tmp_path, role=role, version="0.1")
    with pytest.warns(UserWarning, match="migrated"):
        m = RSkillManifest.from_yaml(str(path))
    assert m.role == RSKILL_ROLE_BY_KIND[m.kind]
    assert m.schema_version == "0.1"


def test_a_malformed_kind_is_a_validation_error_not_a_type_error() -> None:
    raw = yaml.safe_load(_VLA.read_text(encoding="utf-8"))
    raw.pop("role")
    for bad in (["vla"], {"a": 1}):
        with pytest.raises(ValidationError):
            RSkillManifest.model_validate({**raw, "kind": bad})


def test_the_loader_error_names_the_failing_fields(tmp_path: Path) -> None:
    text = _VLA.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^version:.*$", "", text)
    text = re.sub(r"(?m)^chunk_size:.*$", "chunk_size: sixteen", text)
    path = tmp_path / "rskill.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ROSConfigError) as excinfo:
        rSkill.from_yaml(path)
    assert "version: Field required" in str(excinfo.value)
    assert "chunk_size: " in str(excinfo.value)


def test_invalid_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = _copy(_VLA, tmp_path, extra="\nbroken: [\n")
    with pytest.raises(ROSConfigError, match="invalid YAML"):
        rSkill.from_yaml(path)


def test_only_execute_rskill_kinds_naming_their_embodiments_are_dispatchable() -> None:
    vla = RSkillManifest.from_yaml(str(_VLA))
    rSkill.check_dispatchable(vla)
    vlm = RSkillManifest.from_yaml(str(_VLM))
    with capture_logs() as logs, pytest.raises(ROSCapabilityMismatch, match="kind='vlm'"):
        rSkill.check_dispatchable(vlm)
    assert logs[0]["event"] == "rskill.dispatch_refused"
    anywhere = vla.model_copy(update={"embodiment_tags": ["any"]})
    with pytest.raises(ROSCapabilityMismatch, match="must name the embodiments"):
        rSkill.check_dispatchable(anywhere)
