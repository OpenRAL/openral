"""Unit tests for ``openral_sim.benchmark.update_rskill_benchmarks``.

Closes the ``openral benchmark run`` → ``rskill.yaml`` loop without touching any
sim. Uses the real, in-tree ``rskills/pi05-robocasa365-human300-nf4/rskill.yaml``
manifest (copied into ``tmp_path`` so the source-of-truth file is never
mutated by the test) so the test exercises the same schema and YAML shape
the production runner sees — no placeholder strings (CLAUDE.md §1.11).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from openral_core import RSkillManifest
from openral_sim.benchmark import (
    update_rskill_benchmarks,
    update_rskill_benchmarks_from_uri,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_SKILL_DIR = _REPO_ROOT / "rskills" / "pi05-robocasa365-human300-nf4"
_REAL_MANIFEST = _REAL_SKILL_DIR / "rskill.yaml"


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Copy the real rSkill into a tmp directory so the source file stays clean."""
    if not _REAL_MANIFEST.exists():
        pytest.skip(f"real fixture rSkill not present at {_REAL_MANIFEST}")
    dst = tmp_path / "pi05-robocasa365-human300-nf4"
    dst.mkdir()
    shutil.copy2(_REAL_MANIFEST, dst / "rskill.yaml")
    return dst


def test_update_writes_into_empty_benchmarks_block(skill_dir: Path) -> None:
    """Manifest with no ``benchmarks:`` block gets one appended; subsequent
    runs land in the just-created block.

    The pi05-robocasa365-human300-nf4 manifest omits the empty
    ``benchmarks: {}`` line (schema default fills it back in); the
    writeback's "missing block → append" branch is what's exercised here.
    """
    before = (skill_dir / "rskill.yaml").read_text()
    assert "benchmarks:" not in before  # sanity: the fixture omits the block

    written = update_rskill_benchmarks(skill_dir, "robocasa_pnp", 0.42)

    assert written == skill_dir / "rskill.yaml"
    after = written.read_text()

    # Re-validates as a full RSkillManifest with the new score.
    raw = yaml.safe_load(after)
    manifest = RSkillManifest.model_validate(raw)
    assert manifest.benchmarks == {"robocasa_pnp": 0.42}


def test_update_preserves_comments_outside_benchmarks_block(skill_dir: Path) -> None:
    """The surgical edit MUST NOT touch comments anywhere else in the file."""
    before = (skill_dir / "rskill.yaml").read_text()
    # Pull a couple of distinctive comments out of the real manifest.
    assert "# rSkill manifest" in before
    assert "Pre-quantized nf4" in before
    assert "No published benchmark numbers yet" in before

    update_rskill_benchmarks(skill_dir, "robocasa_pnp", 0.55)
    after = (skill_dir / "rskill.yaml").read_text()

    # Every comment except (possibly) the leading-edge of the benchmarks block
    # itself is preserved verbatim. The comment "No published benchmark
    # numbers yet" lives ABOVE `benchmarks: {}` and must survive.
    assert "# rSkill manifest" in after
    assert "Pre-quantized nf4" in after
    assert "No published benchmark numbers yet" in after


def test_update_overwrites_existing_benchmark_key(skill_dir: Path) -> None:
    """Re-running on the same benchmark id replaces, not duplicates, the score."""
    update_rskill_benchmarks(skill_dir, "robocasa_pnp", 0.10)
    update_rskill_benchmarks(skill_dir, "robocasa_pnp", 0.77)

    raw = yaml.safe_load((skill_dir / "rskill.yaml").read_text())
    manifest = RSkillManifest.model_validate(raw)
    assert manifest.benchmarks == {"robocasa_pnp": 0.77}


def test_update_merges_multiple_benchmarks(skill_dir: Path) -> None:
    """Successive calls with different benchmark ids accumulate into the dict."""
    update_rskill_benchmarks(skill_dir, "robocasa_pnp", 0.50)
    # libero_spatial is also a valid BenchmarkName literal — we just exercise
    # the merge logic; no claim the skill actually achieves this rate.
    update_rskill_benchmarks(skill_dir, "libero_spatial", 0.20)

    raw = yaml.safe_load((skill_dir / "rskill.yaml").read_text())
    manifest = RSkillManifest.model_validate(raw)
    assert manifest.benchmarks == {"libero_spatial": 0.20, "robocasa_pnp": 0.50}


def test_update_rejects_unknown_benchmark_id(skill_dir: Path) -> None:
    """Unknown BenchmarkName literal raises before the file is touched."""
    from openral_core.exceptions import ROSConfigError

    before = (skill_dir / "rskill.yaml").read_text()
    with pytest.raises(ROSConfigError):
        update_rskill_benchmarks(skill_dir, "not_a_real_benchmark", 0.5)
    # File is untouched.
    assert (skill_dir / "rskill.yaml").read_text() == before


def test_update_rejects_out_of_range_score(skill_dir: Path) -> None:
    """Scores outside [0.0, 1.0] are caught by the manifest validator."""
    from openral_core.exceptions import ROSConfigError

    before = (skill_dir / "rskill.yaml").read_text()
    with pytest.raises(ROSConfigError):
        update_rskill_benchmarks(skill_dir, "robocasa_pnp", 1.5)
    assert (skill_dir / "rskill.yaml").read_text() == before


def test_update_from_uri_bare_path(skill_dir: Path) -> None:
    """The uri wrapper accepts a bare path and delegates."""
    written = update_rskill_benchmarks_from_uri(str(skill_dir), "robocasa_pnp", 0.33)
    assert written == skill_dir / "rskill.yaml"
    raw = yaml.safe_load(written.read_text())
    assert raw["benchmarks"]["robocasa_pnp"] == 0.33


def test_update_from_uri_rejects_hf_uri() -> None:
    with pytest.raises(ValueError, match="hf://"):
        update_rskill_benchmarks_from_uri("hf://some/repo", "robocasa_pnp", 0.5)


def test_update_raises_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        update_rskill_benchmarks(tmp_path, "robocasa_pnp", 0.5)
