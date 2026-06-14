"""Sim test: real ``openral benchmark run`` end-to-end against RoboCasa + pi05.

Mirrors ``test_franka_panda_smolvla_cli_benchmark.py`` for the
panda_mobile / pi05 / RoboCasa stack. Closes the seam between the
Typer-glued CLI, ``openral_sim.run_benchmark``, and the new
``update_rskill_benchmarks_from_uri`` finaliser that writes the headline
``avg_success_rate`` back into ``rskill.yaml``.

What is asserted
----------------
* ``openral benchmark run`` resolves ``--suite <yaml>``, parses
  ``--rskill <copy of the in-tree skill>``, writes a
  schema-validated ``RSkillEvalResult`` JSON to the requested ``--out``
  path, and updates the manifest's ``benchmarks.robocasa_pnp`` field.
* The emitted JSON re-validates via ``RSkillEvalResult.from_json`` and
  ``source.reproduced_locally`` is ``True``.
* The manifest writeback is surgical — comments outside the
  ``benchmarks:`` block survive verbatim.
* ``eval_config`` echoes the requested protocol overrides — the CLI
  honours overrides instead of silently running the full protocol.

Skips automatically when CUDA, torch, transformers, robocasa, or
robosuite cannot be imported.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip` / `pytest.skip(allow_module_level=True)`: with
# `tests/sim/__init__.py` making this directory a Package, a module-level
# Skipped raised during collection of *this* file marks the **whole
# `tests/sim` Package** as `outcome='skipped'`, which drops every sibling
# test file from collection ("found no collectors for ..."). Deferring the
# decision to `pytestmark` keeps this module importable so its siblings
# remain reachable when the optional `robocasa` package isn't installed.
_REQUIRED_MODULES = ("torch", "transformers", "robocasa", "robosuite")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)


def _robosuite_incompatible() -> str:
    """``robocasa`` needs the openral-vendored robosuite (>=1.5.2, ``get_elements``);
    PyPI 1.5.1 lacks it. ``find_spec`` can't see the missing symbol, so a
    ``uv run --group robocasa`` that re-resolved robosuite to 1.5.1 would ERROR
    instead of skip. The compatible build is installed by the ``openral sim run``
    auto-install path; checking dist metadata is cheap (no heavy import).
    """
    if "robosuite" in _MISSING_MODULES:
        return ""
    try:
        ver = importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError:
        return ""
    parts = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
    if parts < (1, 5, 2):
        return (
            f"robocasa needs robosuite>=1.5.2 (get_elements); found {ver} "
            "— run via `openral sim run`"
        )
    return ""


_INCOMPATIBLE = _robosuite_incompatible()
_CUDA_AVAILABLE = False
if not _MISSING_MODULES and not _INCOMPATIBLE:
    import torch  # gated above

    _CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=("pi05 RoboCasa CLI test requires " + ", ".join(_MISSING_MODULES)),
    ),
    pytest.mark.skipif(
        bool(_INCOMPATIBLE),
        reason=_INCOMPATIBLE or "robosuite incompatible",
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="pi05 RoboCasa CLI test requires CUDA",
    ),
]


_REPO_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_YAML = _REPO_ROOT / "benchmarks" / "robocasa_pnp.yaml"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "pi05-robocasa365-human300-nf4" / "rskill.yaml"


def _make_tiny_robocasa_suite(tmp_path: Path) -> Path:
    """Trim benchmarks/robocasa_pnp.yaml down to 1 seed × short horizon.

    The committed catalogue YAML is 1 task × 10 seeds × 500 steps —
    fine for a paper-equivalent reproduction but a multi-hour run on CI.
    We rewrite it into a tmp_path fixture so the test exercises the same
    code path (``_resolve_benchmark_suite`` reading from ``--suite
    <path>``, ``load_benchmark_suite``, ``run_benchmark``) in a few
    seconds.

    ADR-0042 (June 2026): a benchmark suite is a bare ``list[BenchmarkScene]``
    YAML; ``suite_id`` is derived from the filename stem. We write to
    ``robocasa_pnp.yaml`` (not ``_tiny``) so the stem matches the same
    :data:`openral_core.BenchmarkName` key the manifest writeback targets
    — the test asserts on ``manifest.benchmarks["robocasa_pnp"]`` below.

    The trim mutates the single :class:`BenchmarkScene` in place —
    ``n_episodes=1``, ``seed=0``, and ``task.max_steps=10`` — without
    disturbing the per-scene robot_id / scene / metadata so the suite
    invariants in :func:`openral_core.raise_on_invalid_suite` still
    hold.
    """
    from openral_core import load_benchmark_suite

    scenes = load_benchmark_suite(str(_BENCHMARK_YAML))
    first = scenes[0]
    tiny_scene = first.model_copy(
        update={
            "task": first.task.model_copy(update={"max_steps": 10}),
            "n_episodes": 1,
            "seed": 0,
        }
    )
    out = tmp_path / "robocasa_pnp.yaml"
    out.write_text(yaml.safe_dump([tiny_scene.model_dump(mode="json")]))
    return out


def _copy_skill_to_tmp(tmp_path: Path) -> Path:
    """Copy the in-tree rSkill manifest into a tmp directory.

    The CLI's ``--update-manifest`` finaliser mutates ``rskill.yaml`` in
    place — copying the skill keeps the source-of-truth file untouched
    while still exercising the real production code path.
    """
    if not _LOCAL_MANIFEST.exists():
        pytest.skip(f"rSkill manifest not present at {_LOCAL_MANIFEST}")
    dst = tmp_path / "pi05-robocasa365-human300-nf4"
    dst.mkdir()
    shutil.copy2(_LOCAL_MANIFEST, dst / "rskill.yaml")
    return dst


def test_bh_benchmark_run_robocasa_pi05_end_to_end(tmp_path: Path) -> None:
    """`openral benchmark run` writes a validated RSkillEvalResult AND updates the manifest."""
    from openral_cli.main import app
    from openral_core import RSkillEvalResult, RSkillManifest
    from typer.testing import CliRunner

    if not _BENCHMARK_YAML.exists():
        pytest.skip("benchmarks/robocasa_pnp.yaml not present in this checkout")

    spec_path = _make_tiny_robocasa_suite(tmp_path)
    skill_dir = _copy_skill_to_tmp(tmp_path)
    out_path = tmp_path / "out.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            str(spec_path),
            "--rskill",
            str(skill_dir),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, f"openral benchmark run failed:\n{result.output}"
    assert out_path.exists(), result.output

    payload = json.loads(out_path.read_text())
    eval_result = RSkillEvalResult.model_validate(payload)

    assert eval_result.source.reproduced_locally is True
    assert eval_result.source.reproduction_cli is not None
    assert "openral benchmark run" in eval_result.source.reproduction_cli
    assert eval_result.benchmark.robot == "panda_mobile"

    assert eval_result.eval_config["n_episodes"] == 1
    assert eval_result.eval_config["seeds"] == [0]
    assert eval_result.eval_config["success_key"] == "is_success"

    assert "avg_success_rate" in eval_result.results
    assert eval_result.results["n_tasks"] == 1
    assert eval_result.results["n_episodes_total"] == 1
    assert eval_result.results.get("mean_step_latency_ms_avg", 0.0) > 0.0

    # The CLI finaliser must have written the headline back into rskill.yaml.
    manifest_text = (skill_dir / "rskill.yaml").read_text()
    assert "Pre-quantized nf4" in manifest_text  # comments preserved
    raw = yaml.safe_load(manifest_text)
    manifest = RSkillManifest.model_validate(raw)
    assert "robocasa_pnp" in manifest.benchmarks
    assert 0.0 <= manifest.benchmarks["robocasa_pnp"] <= 1.0


def test_bh_benchmark_run_no_update_manifest_leaves_rskill_untouched(tmp_path: Path) -> None:
    """`--no-update-manifest` skips the rskill.yaml writeback even on success."""
    from openral_cli.main import app
    from typer.testing import CliRunner

    if not _BENCHMARK_YAML.exists():
        pytest.skip("benchmarks/robocasa_pnp.yaml not present in this checkout")

    spec_path = _make_tiny_robocasa_suite(tmp_path)
    skill_dir = _copy_skill_to_tmp(tmp_path)
    before = (skill_dir / "rskill.yaml").read_text()
    out_path = tmp_path / "out.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            str(spec_path),
            "--rskill",
            str(skill_dir),
            "--out",
            str(out_path),
            "--no-update-manifest",
        ],
    )
    assert result.exit_code == 0, f"openral benchmark run failed:\n{result.output}"
    assert out_path.exists(), result.output
    after = (skill_dir / "rskill.yaml").read_text()
    assert after == before, "manifest was mutated despite --no-update-manifest"
