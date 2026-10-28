"""The shared manifest/scene YAML parser bounds merge-key (``<<``) expansion (CWE-776)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from openral_core import RSkillManifest
from openral_core._safe_yaml import safe_load
from openral_core.exceptions import ROSConfigError
from openral_core.loaders import load_benchmark_suite

REPO = Path(__file__).resolve().parents[2]


def _merge_bomb(levels: int) -> str:
    """~30 bytes per level; stock PyYAML builds 2**levels merged pairs."""
    lines = ["a0: &a0 {k: 1}"]
    lines += [f"a{i}: &a{i} {{<<: [*a{i - 1}, *a{i - 1}]}}" for i in range(1, levels + 1)]
    return "\n".join(lines)


def test_merge_bomb_parses_in_linear_time() -> None:
    start = time.monotonic()
    data = safe_load(_merge_bomb(60))
    assert time.monotonic() - start < 2.0
    assert isinstance(data, dict)
    assert data["a60"] == {"k": 1}


def test_merge_bomb_through_manifest_loader_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "rskill.yaml"
    path.write_text(_merge_bomb(60), encoding="utf-8")
    start = time.monotonic()
    with pytest.raises(ValueError):  # pydantic ValidationError: not a manifest
        RSkillManifest.from_yaml(str(path))
    assert time.monotonic() - start < 2.0


def test_merge_bomb_through_suite_loader_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "bomb.yaml"
    path.write_text(_merge_bomb(60), encoding="utf-8")
    start = time.monotonic()
    with pytest.raises(ROSConfigError):
        load_benchmark_suite(str(path))
    assert time.monotonic() - start < 2.0


def test_merge_semantics_match_pyyaml() -> None:
    doc = (
        "base: &b {x: 1, y: 2}\n"
        "over: &o {y: 3, z: 4}\n"
        "m: {<<: [*o, *b], x: 9}\n"  # first listed mapping wins, own keys win over both
        "n: {<<: *b, 1: one, '1': str-one}\n"
    )
    assert safe_load(doc) == yaml.safe_load(doc)


@pytest.mark.parametrize("path", sorted((REPO / "benchmarks").glob("*.yaml")), ids=lambda p: p.name)
def test_repo_benchmarks_parse_identically(path: Path) -> None:
    """The benchmark suites use merge keys; the bounded parser must not change them."""
    text = path.read_text(encoding="utf-8")
    assert safe_load(text) == yaml.safe_load(text)
