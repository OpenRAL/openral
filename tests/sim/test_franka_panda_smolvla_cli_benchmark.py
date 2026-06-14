"""Sim test: real ``openral benchmark run`` end-to-end against LIBERO + SmolVLA.

Closes the seam between the Typer-glued CLI and ``openral_sim.run_benchmark``.
The underlying ``run_benchmark`` function is already covered by
``tests/unit/test_benchmark_runner.py`` with the mock scene; here we go
the other way and pay the full physics + weights cost so the CLI wiring
itself is exercised on a real (LIBERO MuJoCo, SmolVLA) pair — the most
advanced sim + policy combination currently wired in openral_sim.

What is asserted
----------------
* ``openral benchmark run`` resolves ``--suite <yaml>``, parses
  ``--rskill smolvla-libero``, and writes a
  schema-validated ``RSkillEvalResult`` JSON to the requested ``--out`` path.
* The emitted JSON re-validates via ``RSkillEvalResult.from_json`` and the
  ``source.reproduced_locally`` flag is ``True``.
* ``eval_config`` echoes the requested protocol overrides (single task,
  single seed) — the CLI honours overrides instead of silently running the
  full 100-rollout libero_spatial protocol.
* The aggregated rollup contains a per-task success rate plus the
  ``avg_success_rate`` headline. The mean step latency is positive,
  proving real physics ran (mock scenes would return None / 0).

Skips automatically when CUDA, torch, transformers, num2words, or the
LIBERO suite cannot be imported. Mirrors the policy used by
``test_franka_panda_smolvla_libero.py`` so the same hosts run both.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip` / `pytest.skip(allow_module_level=True)`: with
# `tests/sim/__init__.py` making this directory a Package, a Skipped raised
# at module-import time poisons the whole `tests/sim` Package collection
# ("found no collectors for ..." on every sibling). Deferring the decision
# to `pytestmark` keeps this module importable when optional deps are
# missing, so sibling files remain reachable.
_REQUIRED_MODULES = ("torch", "transformers", "num2words")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)
_CUDA_AVAILABLE = False
if not _MISSING_MODULES:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()


def _libero_robosuite_conflict() -> bool:
    """True when an installed robosuite blocks the LIBERO runtime (it pins 1.4.x).

    A >=1.5 robosuite (e.g. provisioned by a robocasa install) makes LIBERO
    unprovisionable on this host — ``openral benchmark run`` auto-installs the
    ``libero`` group but cannot downgrade robosuite, so its post-install probe
    fails. Skip cleanly rather than go red. On a clean runner robosuite is
    absent, so the ``--group libero`` install supplies 1.4.x and the test runs.
    """
    import importlib.metadata as _md

    if importlib.util.find_spec("robosuite") is None:
        return False
    try:
        return not _md.version("robosuite").startswith("1.4")
    except _md.PackageNotFoundError:
        return False


pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="SmolVLA LIBERO CLI test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="SmolVLA LIBERO CLI test requires CUDA",
    ),
    pytest.mark.skipif(
        _libero_robosuite_conflict(),
        reason="LIBERO needs robosuite 1.4.x; a newer robosuite (robocasa 1.5.x) is installed",
    ),
]


_REPO_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_YAML = _REPO_ROOT / "benchmarks" / "libero_spatial.yaml"
_LOCAL_MANIFEST = _REPO_ROOT / "rskills" / "smolvla-libero" / "rskill.yaml"


def _make_tiny_libero_suite(tmp_path: Path) -> Path:
    """Trim benchmarks/libero_spatial.yaml down to 1 task × 1 seed × 20 steps.

    The committed catalogue YAML is 10 tasks × 10 seeds × 280 steps — fine
    for a paper-equivalent reproduction but a multi-hour run on CI. We
    rewrite it into a tmp_path fixture so the test exercises the same code
    path (``_resolve_benchmark_suite`` reading from ``--suite <path>``,
    ``load_benchmark_suite``, ``run_benchmark``) in a few seconds.

    ADR-0042 (June 2026): a benchmark suite is a bare ``list[BenchmarkScene]``
    YAML; ``suite_id`` is derived from the filename stem. We write to
    ``libero_spatial.yaml`` (not ``_tiny``) so the stem matches a valid
    :data:`openral_core.BenchmarkName` literal — the test uses
    ``--no-update-manifest`` to keep the in-tree ``rskills/smolvla-libero/rskill.yaml``
    untouched, but the resolver still validates the stem on the way in.

    The trim collapses the first :class:`BenchmarkScene` to a
    single-episode, 20-step rollout and discards the remaining nine.
    Per-scene fields (robot_id / scene / metadata) carry through
    untouched so the suite invariants in
    :func:`openral_core.raise_on_invalid_suite` still hold.
    """
    import yaml
    from openral_core import load_benchmark_suite

    scenes = load_benchmark_suite(str(_BENCHMARK_YAML))
    first = scenes[0]
    tiny_scene = first.model_copy(
        update={
            "task": first.task.model_copy(update={"max_steps": 20}),
            "n_episodes": 1,
            "seed": 0,
        }
    )
    out = tmp_path / "libero_spatial.yaml"
    # Persist as a bare YAML list — load_benchmark_suite reads it back the
    # same way, so the round-trip is part of the assertion surface.
    out.write_text(yaml.safe_dump([tiny_scene.model_dump(mode="json")]))
    return out


def test_bh_benchmark_run_libero_smolvla_end_to_end(tmp_path: Path) -> None:
    """`openral benchmark run` writes a validated RSkillEvalResult for LIBERO + SmolVLA."""
    from openral_cli.main import app
    from openral_core import RSkillEvalResult
    from typer.testing import CliRunner

    if not _BENCHMARK_YAML.exists():
        pytest.skip("benchmarks/libero_spatial.yaml not present in this checkout")
    if not _LOCAL_MANIFEST.exists():
        pytest.skip(f"rSkill manifest not present at {_LOCAL_MANIFEST}")

    spec_path = _make_tiny_libero_suite(tmp_path)
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
            str(_LOCAL_MANIFEST.parent),
            "--out",
            str(out_path),
            # ADR-0042: ``suite_id`` is the YAML filename stem
            # (``libero_spatial`` here), which IS a valid BenchmarkName.
            # We still suppress manifest writeback so the in-tree
            # ``rskills/smolvla-libero/rskill.yaml`` is never mutated by a
            # test run — the eval JSON is the entire assertion surface.
            "--no-update-manifest",
        ],
    )
    assert result.exit_code == 0, f"openral benchmark run failed:\n{result.output}"
    assert out_path.exists(), result.output

    payload = json.loads(out_path.read_text())
    eval_result = RSkillEvalResult.model_validate(payload)

    assert eval_result.source.reproduced_locally is True
    assert eval_result.source.reproduction_cli is not None
    assert "openral benchmark run" in eval_result.source.reproduction_cli
    assert eval_result.benchmark.robot == "franka_panda"

    assert eval_result.eval_config["n_episodes"] == 1
    assert eval_result.eval_config["seeds"] == [0]
    assert eval_result.eval_config["success_key"] == "is_success"

    assert "avg_success_rate" in eval_result.results
    assert eval_result.results["n_tasks"] == 1
    assert eval_result.results["n_episodes_total"] == 1
    # Real physics produces a mean-step latency; mock scenes leave it None.
    assert eval_result.results.get("mean_step_latency_ms_avg", 0.0) > 0.0
