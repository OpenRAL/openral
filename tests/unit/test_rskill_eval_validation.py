"""Unit tests for the ``rskills/<id>/eval/<benchmark>.json`` validator.

Covers the wiring between :class:`openral_core.RSkillEvalResult` and
:meth:`rSkill.from_yaml` (the loader scans ``<skill_dir>/eval/*.json`` and
validates each — see ``python/rskill/src/openral_rskill/loader.py``).

Coverage
--------
- Every in-tree ``rskills/*/eval/*.json`` validates against
  :class:`RSkillEvalResult` (regression guard against shape drift).
- A malformed JSON in ``eval/`` causes :meth:`rSkill.from_yaml` to raise
  :class:`ROSConfigError`.
- A skill with **no** ``eval/`` directory loads cleanly
  (e.g. ``rskills/pi05-robocasa365-human300-nf4``).
"""

from __future__ import annotations

import glob
import textwrap
from pathlib import Path

import pytest
from openral_core import RSkillEvalResult
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import rSkill


def test_every_in_tree_skill_eval_json_validates() -> None:
    """Pin the schema against every ``rskills/*/eval/*.json`` file in tree."""
    paths = sorted(glob.glob("rskills/*/eval/*.json"))
    assert paths, "expected at least one in-tree skill eval JSON"
    for path in paths:
        result = RSkillEvalResult.from_json(path)
        assert result.benchmark.name, f"{path}: empty benchmark.name"
        assert result.source.model_variant, f"{path}: empty source.model_variant"


_RSKILL_YAML = textwrap.dedent("""\
    name: test/rskill-alpha
    version: "0.1.0"
    license: apache-2.0
    role: s1
    kind: vla
    model_family: smolvla
    embodiment_tags: [so100_follower]
    runtime: pytorch
    weights_uri: "hf://test/rskill-alpha"
    chunk_size: 16
    latency_budget:
      per_chunk_ms: 200.0
    actuators_required:
      - kind: joint_position
        control_mode_semantics: {mode: absolute}
    processors:
      preprocessor_uri: "hf://test/rskill-alpha/policy_preprocessor.json"
      postprocessor_uri: "hf://test/rskill-alpha/policy_postprocessor.json"
    description: "Eval-validation test rSkill fixture."
    actions:
      - generalist
""")


def test_skill_with_no_eval_dir_loads(tmp_path: Path) -> None:
    """Skills without an ``eval/`` directory must still load."""
    (tmp_path / "rskill.yaml").write_text(_RSKILL_YAML)
    pkg = rSkill.from_yaml(tmp_path / "rskill.yaml")
    assert pkg.manifest.name == "test/rskill-alpha"


def test_malformed_eval_json_raises(tmp_path: Path) -> None:
    """A malformed JSON file inside ``eval/`` must surface as ROSConfigError."""
    (tmp_path / "rskill.yaml").write_text(_RSKILL_YAML)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "bad.json").write_text("{not even json")
    with pytest.raises(ROSConfigError, match="malformed JSON"):
        rSkill.from_yaml(tmp_path / "rskill.yaml")


def test_eval_json_missing_required_field_raises(tmp_path: Path) -> None:
    """JSON missing required RSkillEvalResult fields must raise ROSConfigError."""
    (tmp_path / "rskill.yaml").write_text(_RSKILL_YAML)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Missing required `source`, `benchmark`, `results`.
    (eval_dir / "incomplete.json").write_text('{"schema_version": "0.1"}')
    with pytest.raises(ROSConfigError, match="invalid skill eval JSON"):
        rSkill.from_yaml(tmp_path / "rskill.yaml")


def test_well_formed_eval_json_loads_cleanly(tmp_path: Path) -> None:
    """Round-trip a minimal-valid RSkillEvalResult through the loader."""
    (tmp_path / "rskill.yaml").write_text(_RSKILL_YAML)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    eval_json = textwrap.dedent("""\
        {
          "schema_version": "0.1",
          "source": {
            "paper": "Acme et al., 1999",
            "model_variant": "alpha-1",
            "evaluated_by": "ci",
            "reproduced_locally": true
          },
          "benchmark": {
            "name": "Acme",
            "protocol": "10-trial mean",
            "robot": "so100_follower",
            "simulator": "n/a"
          },
          "results": {"mean": 0.42}
        }
    """)
    (eval_dir / "acme.json").write_text(eval_json)
    pkg = rSkill.from_yaml(tmp_path / "rskill.yaml")
    assert pkg.manifest.name == "test/rskill-alpha"
