"""Doctest enforcement — implements the CLAUDE.md §5.4 doctest mandate.

CLAUDE.md §5.4 requires *"every public-facing example block in a docstring
is run"*.  This file enforces that rule for the curated set of packages whose
doctests pass today.  When a new package is added, append it to
``DOCTEST_TARGETS`` here and to the corresponding ``just test-doctest``
recipe in the ``Justfile``.

The list is **explicitly opt-in** — adding a path is a deliberate decision
to keep the docstring examples passing forever, not a global flag that
sweeps in every new module silently.

Every public module that ships docstring examples is currently in
``DOCTEST_TARGETS``.  ``runtime_onnx.py`` joined the list once its
``import onnxruntime`` was lazified into a constructor-time import
(see ``_import_ort``); the previous structlog stdout pollution that
blocked ``world_state/aggregator.py``, ``hal/so100_follower.py``, and
``hal/ros_control.py`` is fixed by the repo-root ``conftest.py`` that
filters every record below ``WARNING``.
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
    "python/rskill/src/openral_rskill/base.py",
    "python/rskill/src/openral_rskill/engine_cache.py",
    "python/rskill/src/openral_rskill/loader.py",
    "python/rskill/src/openral_rskill/quantization.py",
    "python/rskill/src/openral_rskill/runtime.py",
    "python/rskill/src/openral_rskill/runtime_onnx.py",
    "python/rskill/src/openral_rskill/runtime_pytorch.py",
    "python/rskill/src/openral_rskill/runtime_tensorrt.py",
    "python/rskill/src/openral_rskill/smolvla.py",
    "python/rskill/src/openral_rskill/executor.py",
    "python/runner/src/openral_runner/clock.py",
    "python/runner/src/openral_runner/safety.py",
]


def test_curated_doctest_targets_all_pass() -> None:
    """Every path in ``DOCTEST_TARGETS`` must have its doctests pass.

    Runs ``pytest --doctest-modules`` as a subprocess so the inner pytest
    invocation is independent of the outer one.  Subprocess (rather than
    in-process ``pytest.main``) avoids re-entering the pytest plugin loop
    and makes the failure mode crisp: stdout / stderr show up directly in
    the failure message.
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
    """Smoke check: the curated set actually exercises a meaningful number of examples.

    A regression that silently strips example blocks from docstrings would
    pass ``test_curated_doctest_targets_all_pass`` (vacuously). This test
    asserts that we are running at least the count we have today, so a
    drop-off triggers a CI failure.
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
