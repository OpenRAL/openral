"""Doctest enforcement — implements the CLAUDE.md §5.4 doctest mandate.

CLAUDE.md §5.4 requires every public docstring example to run. Enforced for
the curated, explicitly opt-in ``DOCTEST_TARGETS`` list; adding a package
means appending it here AND to ``just test-doctest`` in the Justfile.

``runtime_onnx.py`` joined once ``import onnxruntime`` was lazified into a
constructor-time import (``_import_ort``); structlog stdout pollution that
blocked ``world_state/aggregator.py``, ``hal/so100_follower.py``, and
``hal/ros_control.py`` is fixed by the repo-root ``conftest.py`` filtering
records below ``WARNING``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repo root resolved relative to this file: tests/unit/test_doctest_runner.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Curated targets — append-only when new doctests are added cleanly ────────

DOCTEST_TARGETS: list[str] = [
    "python/core/src/openral_core",
    "python/cli/src/openral_cli",
    "python/sensors/src/openral_sensors",
    "python/world_state/src/openral_world_state",
    "python/hal/src/openral_hal/protocol.py",
    "python/hal/src/openral_hal/sim_transport.py",
    "python/hal/src/openral_hal/_real_description.py",
    "python/hal/src/openral_hal/franka_panda.py",
    "python/hal/src/openral_hal/franka_panda_real.py",
    "python/hal/src/openral_hal/sawyer_real.py",
    "python/hal/src/openral_hal/aloha.py",
    "python/hal/src/openral_hal/ur.py",
    "python/hal/src/openral_hal/ur_real.py",
    "python/hal/src/openral_hal/so100_sim.py",
    "python/hal/src/openral_hal/lifecycle.py",
    "python/hal/src/openral_hal/so100_follower.py",
    "python/hal/src/openral_hal/ros_control.py",
    "python/rskill/src/openral_rskill/backend_registry.py",
    "python/rskill/src/openral_rskill/base.py",
    "python/rskill/src/openral_rskill/engine_cache.py",
    "python/rskill/src/openral_rskill/loader.py",
    "python/rskill/src/openral_rskill/quantization.py",
    "python/rskill/src/openral_rskill/runtime.py",
    "python/rskill/src/openral_rskill/runtime_onnx.py",
    "python/rskill/src/openral_rskill/runtime_pytorch.py",
    "python/rskill/src/openral_rskill/smolvla.py",
    "python/rskill/src/openral_rskill/executor.py",
    "python/runner/src/openral_runner/clock.py",
    "python/runner/src/openral_runner/safety.py",
]


def test_curated_doctest_targets_all_pass() -> None:
    """Every path in ``DOCTEST_TARGETS`` must have its doctests pass.

    Runs ``pytest --doctest-modules`` as a subprocess (not in-process
    ``pytest.main``) to stay independent of the outer pytest run and avoid
    re-entering its plugin loop; stdout/stderr surface directly on failure.
    """
    targets = [str(_REPO_ROOT / p) for p in DOCTEST_TARGETS]
    for path in targets:
        assert Path(path).exists(), f"DOCTEST_TARGETS entry does not exist: {path}"

    cmd = [sys.executable, "-m", "pytest", "--doctest-modules", "-q", *targets]
    result = subprocess.run(  # reason: trusted args, no shell
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"Doctest run failed (rc={result.returncode}).\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )


def test_doctest_targets_collect_at_least_30_examples() -> None:
    """Smoke check: the curated set exercises a meaningful number of examples.

    Guards against silently stripped example blocks, which would otherwise pass
    ``test_curated_doctest_targets_all_pass`` vacuously.
    """
    targets = [str(_REPO_ROOT / p) for p in DOCTEST_TARGETS]
    cmd = [sys.executable, "-m", "pytest", "--doctest-modules", "--collect-only", "-q", *targets]
    result = subprocess.run(  # reason: trusted args, no shell
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"Doctest collection failed (rc={result.returncode}).\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
    # pytest --collect-only -q ends with "<N> tests collected"
    last = next(
        (line for line in reversed(result.stdout.splitlines()) if "tests collected" in line),
        "",
    )
    n_collected = int(last.split()[0]) if last else 0
    assert n_collected >= 55, (
        f"Expected ≥55 doctest examples across the curated targets, got {n_collected}.\n"
        f"Last line: {last!r}"
    )
