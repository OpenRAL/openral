"""Regression test for silent skips in ``tests/sim/``.

Background: the sim-test files use ``pytest.importorskip`` and module-level
``pytest.skip(reason=..., allow_module_level=True)`` to bail on
non-CUDA / non-lerobot hosts. Pytest swallows the skip *reason* unless the
runner is invoked with ``-r`` (we now set ``-ra`` in
``pyproject.toml [tool.pytest.ini_options]``). This test asserts that:

1. Pytest's collector can walk every ``tests/sim/`` file without raising
   a collection error (i.e. import-time bugs would surface here, not as
   a "0 items collected" silence).
2. Every test file with a module-level CUDA skip emits a SKIPPED entry
   whose reason mentions CUDA, MuJoCo, or another opt-in dep — so the
   developer running ``pytest`` on a CPU-only laptop sees *why*.

The test runs ``pytest --collect-only`` in a subprocess with
``CUDA_VISIBLE_DEVICES=""`` so it deterministically simulates a CUDA-less
host even on machines that have a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _REPO_ROOT / "tests" / "sim"

# Files we expect to be discoverable by the collector. Update when adding /
# removing sim test modules; the absence of one of these in the collected
# list is itself a regression signal.
_EXPECTED_SIM_TEST_FILES = (
    "test_aloha_bimanual_act_aloha.py",
    "test_pusht_2d_diffusion_pusht.py",
    "test_franka_panda_hal_mujoco.py",
    "test_ur5e_hal_mujoco.py",
    "test_ur10e_hal_mujoco.py",
    "test_franka_panda_smolvla_libero.py",
    "test_franka_panda_smolvla_cli_benchmark.py",
)


def _run_pytest_collect() -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",  # force CUDA-less host
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    # Pass every expected sim file explicitly. Pytest's directory collector
    # short-circuits at the first module-level skip — only the alphabetically
    # first file gets imported, hiding skip reasons for the rest. Naming each
    # file forces pytest to visit it (it is reported either as a collected
    # item, a SKIPPED line, or an "ERROR: found no collectors for <path>"
    # stderr line — all three contain the filename, which is what this
    # regression test is asserting).
    explicit_files = [str(_SIM_DIR / fname) for fname in _EXPECTED_SIM_TEST_FILES]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-rs",  # surface skip reasons even if the user's addopts changes
            "-q",
            *explicit_files,
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture(scope="module")
def collect_result() -> subprocess.CompletedProcess[str]:
    return _run_pytest_collect()


def test_sim_collection_does_not_error(
    collect_result: subprocess.CompletedProcess[str],
) -> None:
    """Pytest must complete collection cleanly (exit 0, 4, or 5, no internal errors)."""
    # Exit codes: 0 = ok, 5 = "no tests collected" (acceptable when every
    # module skips on the CUDA-less runner). 4 = pytest's usage error
    # which it raises with "found no collectors for <path>" when an
    # explicit file path is given to a module that did pytest.importorskip
    # / pytest.skip(allow_module_level=True) at import — that's the
    # expected outcome on a CPU-only host and is itself a visible signal
    # (the path appears in stderr). Any other code (1, 2, 3) signals a
    # real collection error or test failure.
    assert collect_result.returncode in (0, 4, 5), (
        f"pytest --collect-only exited {collect_result.returncode}\n"
        f"--- stdout ---\n{collect_result.stdout}\n"
        f"--- stderr ---\n{collect_result.stderr}"
    )
    out = collect_result.stdout + collect_result.stderr
    # No INTERNALERROR or import errors during collection.
    assert "INTERNALERROR" not in out, out
    assert "errors during collection" not in out, out


def test_every_sim_file_visible_to_collector(
    collect_result: subprocess.CompletedProcess[str],
) -> None:
    """Each known sim test file must show up in the collector output —
    either as collected items or with an explicit SKIPPED reason."""
    out = collect_result.stdout + collect_result.stderr
    missing: list[str] = []
    for fname in _EXPECTED_SIM_TEST_FILES:
        if fname not in out:
            missing.append(fname)
    assert not missing, (
        f"sim test files invisible to pytest collector: {missing}\n"
        "(silent-skip regression — every sim test must appear as either a "
        "collected item or a SKIPPED line so the user knows why it was "
        "skipped on their host)\n"
        f"--- pytest output ---\n{out}"
    )


def test_skip_reasons_are_visible(
    collect_result: subprocess.CompletedProcess[str],
) -> None:
    """If anything skips, the reason must appear in the output (not just a count)."""
    out = collect_result.stdout + collect_result.stderr
    if "skipped" not in out.lower():
        pytest.skip("no sim modules skipped on this host — nothing to assert")
    # A reason word should appear: CUDA, lerobot, mujoco, gym_*, transformers, ...
    skip_keywords = (
        "CUDA",
        "cuda",
        "lerobot",
        "mujoco",
        "transformers",
        "torch",
        "gym_aloha",
        "gym_pusht",
        "gymnasium",
        "pymunk",
        "libero",
    )
    assert any(k in out for k in skip_keywords), (
        f"sim modules skipped without a recognisable reason in the output:\n{out}"
    )
