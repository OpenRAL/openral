"""Tests for the opt-in lane accounting tool (``tools/lane_report.py``).

CLAUDE.md §1.11 — these run against the *real* ``tools/test_selection.toml``
capability declarations, and the junit-parsing test drives a *real* pytest
subprocess so the parser is proven against genuine pytest output rather than a
hand-written approximation of it.

The behaviour under test is the fix for issue #163: a lane that ran nothing
must never be indistinguishable from a lane that ran everything and passed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
CONFIG_PATH = TOOLS / "test_selection.toml"


# `tools/` is not an installed package; import it the same way the tool itself
# does at runtime (repo root on sys.path, `tools.` package form) so the test
# exercises the real import path rather than a private copy of the modules.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import lane_report, select_tests  # noqa: E402

CONFIG = select_tests.load_config(CONFIG_PATH)


# --------------------------------------------------------------------------
# Skip classification — the line between "cannot run here" and "is broken".
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        # Every one of these was observed verbatim on a real hosted run
        # (PR #153, run 32757769291).
        ("ACT-ALOHA-full sim test requires CUDA", "cuda"),
        ("GR00T NF4 requires a CUDA GPU", "cuda"),
        ("MolmoAct2 NF4 requires CUDA", "cuda"),
        ("LingBot-VLA 2.0 NF4 requires a CUDA GPU", "cuda"),
        (
            "SAPIEN rendering needs a working Vulkan driver: "
            "vk::createInstanceUnique: ErrorIncompatibleDriver",
            "vulkan",
        ),
        (
            "Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
            "sidecar",
        ),
        ("RLBench/CoppeliaSim sidecar not provisioned", "sidecar"),
        ("could not import 'rclpy': No module named 'rclpy'", "ros"),
        ("panda.srdf not installed", "ros"),
        ("moveit_resources panda.srdf not installed", "ros"),
    ],
)
def test_declared_capability_gaps_are_attributed(reason: str, expected: str) -> None:
    assert lane_report._classify_skip(reason, CONFIG.capability_gaps) == expected


@pytest.mark.parametrize(
    "reason",
    [
        # A missing *installable* package is a provisioning bug, not a hardware
        # limit. It must stay undeclared so the lane stays red — this is the
        # whole point of an allowlist rather than "tolerate every skip".
        "mujoco not installed",
        "robot_descriptions not installed",
        "smolvla-libero rskill not present",
        "LIBERO config not present on this branch",
    ],
)
def test_undeclared_skips_are_not_excused(reason: str) -> None:
    assert lane_report._classify_skip(reason, CONFIG.capability_gaps) is None


# --------------------------------------------------------------------------
# Lane policy.
# --------------------------------------------------------------------------


def _record(lane: str, *, passed: int, skips: list[str]) -> object:
    return lane_report.build_record(
        lane, passed=passed, failed=0, skip_reasons=skips, config=CONFIG
    )


def test_partly_gated_lane_still_passes_and_reports_the_gap() -> None:
    # The exact shape that blocked PR #153: the `sim` lane genuinely ran 162
    # tests and was failed anyway by 13 CUDA-gated skips.
    record = _record("sim", passed=162, skips=["x requires CUDA"] * 13)
    assert record.status == lane_report.STATUS_RAN
    assert record.declared_skips == {"cuda": 13}
    assert record.declared_total == 13


def test_undeclared_skip_fails_the_lane() -> None:
    record = _record("sim", passed=162, skips=["mujoco not installed"])
    assert record.status == lane_report.STATUS_FAILED
    assert "undeclared skip" in (record.note or "")


def test_fully_gated_lane_is_declared_not_run_never_silently_green() -> None:
    record = _record("isaacsim", passed=0, skips=["Isaac Sim sidecar venv not provisioned"] * 21)
    assert record.status == lane_report.STATUS_DECLARED_NOT_RUN
    # It is reported, not absent: the gap is named and counted.
    assert record.declared_skips == {"sidecar": 21}


def test_narrow_selection_of_a_gated_file_is_declared_not_run_not_failed() -> None:
    """A well-covered lane can still have every *selected* test gated.

    Found by proof run 32815008771: a diff touching only `rskills/act-aloha/**`
    selects one `sim` file whose six tests are all CUDA-gated, so the lane had
    0 passed and 6 declared skips. `sim` yields 162 passing tests when all its
    files are selected, but the verdict must be about what the diff SELECTED —
    failing here would punish a PR merely for touching a GPU-only file, the
    exact breakage this policy removes.
    """
    record = _record("sim", passed=0, skips=["x requires CUDA"] * 6)
    assert record.status == lane_report.STATUS_DECLARED_NOT_RUN
    assert record.declared_skips == {"cuda": 6}


def test_lane_that_collected_nothing_at_all_fails() -> None:
    # No passes and no skips means the lane collected no tests — broken, not
    # gated. It must not masquerade as "no satisfiable coverage".
    record = _record("sim", passed=0, skips=[])
    assert record.status == lane_report.STATUS_FAILED
    assert "collected no tests" in (record.note or "")


def test_failing_test_fails_the_lane() -> None:
    record = lane_report.build_record("sim", passed=5, failed=1, skip_reasons=[], config=CONFIG)
    assert record.status == lane_report.STATUS_FAILED


def test_exit_code_only_lane_is_judged_on_its_exit_code() -> None:
    # simpler-env / robocasa-gr1 run their pytest inside verify_test_envs.py.
    ok = lane_report.build_record(
        "simpler-env",
        passed=0,
        failed=0,
        skip_reasons=[],
        config=CONFIG,
        exit_code=0,
        from_exit_code=True,
    )
    assert ok.status == lane_report.STATUS_RAN
    bad = lane_report.build_record(
        "simpler-env",
        passed=0,
        failed=0,
        skip_reasons=[],
        config=CONFIG,
        exit_code=2,
        from_exit_code=True,
    )
    assert bad.status == lane_report.STATUS_FAILED


# --------------------------------------------------------------------------
# junit parsing, against real pytest output.
# --------------------------------------------------------------------------


def test_reads_real_pytest_junit_including_module_level_skip(tmp_path: Path) -> None:
    """A collection skip hides its reason in the element TEXT, not ``message``.

    pytest writes ``message="collection skipped"`` for a module-level
    ``importorskip`` and puts the real reason in the body. Reading only
    ``message`` would classify every such skip as undeclared and redden the
    lane — so parse a real report and assert the reason survives.
    """
    (tmp_path / "test_modskip.py").write_text(
        "import pytest\npytest.importorskip('rclpy')\ndef test_never(): assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "test_mixed.py").write_text(
        "import pytest\n"
        "def test_ok(): assert True\n"
        "def test_gated(): pytest.skip('SmolVLA LIBERO sim test requires CUDA')\n",
        encoding="utf-8",
    )
    xml = tmp_path / "report.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junit-xml={xml}",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    passed, failed, reasons = lane_report.read_junit([xml])
    assert (passed, failed) == (1, 0)
    assert any("requires CUDA" in r for r in reasons)
    # The module-level skip's real reason, recovered from the element text.
    assert any("rclpy" in r for r in reasons)
    # And both classify as declared gaps rather than undeclared noise.
    assert {lane_report._classify_skip(r, CONFIG.capability_gaps) for r in reasons} == {
        "cuda",
        "ros",
    }


# --------------------------------------------------------------------------
# The anti-vacuity gate.
# --------------------------------------------------------------------------


def _attest(tmp_path: Path, *, full_run: str, lanes: dict[str, list[str]], records: list) -> int:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"full_run": full_run, "requirement_targets_json": json.dumps(lanes)}),
        encoding="utf-8",
    )
    return lane_report.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--ledger",
            str(ledger),
            "attest",
            "--selection-json",
            str(selection),
        ]
    )


def test_attest_fails_when_a_full_run_selects_no_lane(tmp_path: Path) -> None:
    """The #163 regression guard, stated directly.

    If ``select_tests.py`` ever stops expanding lanes on a blast-radius diff,
    this is what catches it — instead of the job quietly reporting success
    having executed nothing.
    """
    assert _attest(tmp_path, full_run="true", lanes={}, records=[]) == 1


def test_attest_fails_when_a_selected_lane_produced_no_record(tmp_path: Path) -> None:
    # "Selected but never executed" — vacuous green — is now a hard failure.
    assert _attest(tmp_path, full_run="false", lanes={"sim": ["a.py"]}, records=[]) == 1


def test_attest_passes_when_every_selected_lane_is_accounted_for(tmp_path: Path) -> None:
    ran = _record("sim", passed=10, skips=["x requires CUDA"])
    gated = _record("isaacsim", passed=0, skips=["Isaac Sim sidecar venv not provisioned"])
    rc = _attest(
        tmp_path,
        full_run="true",
        lanes={"sim": ["a.py"], "isaacsim": ["b.py"]},
        records=[ran, gated],
    )
    assert rc == 0


def test_attest_fails_on_a_failed_lane_record(tmp_path: Path) -> None:
    bad = _record("sim", passed=1, skips=["mujoco not installed"])
    assert _attest(tmp_path, full_run="false", lanes={"sim": ["a.py"]}, records=[bad]) == 1


# --------------------------------------------------------------------------
# Config hygiene.
# --------------------------------------------------------------------------


def test_no_lane_is_declared_with_an_empty_glob_list() -> None:
    """A lane with no globs can never run, and says nothing when it doesn't.

    `rldx` was declared with an empty list, so `run_lane rldx` returned at its
    `[ -z "$targets" ]` guard on every run since it was added — a lane that was
    configured, wired into the workflow, and structurally incapable of
    executing. Indistinguishable in the log from a lane that simply had nothing
    selected, which is the same silent-degradation shape as #163 itself.
    """
    empty = sorted(name for name, globs in CONFIG.requirement_globs.items() if not globs)
    assert empty == [], f"lanes declared with no globs can never run: {empty}"


def test_every_capability_gap_declares_how_it_could_be_satisfied() -> None:
    # A declared gap must say what WOULD satisfy it, so "declared-not-run" is
    # an actionable statement rather than a shrug.
    for name, gap in CONFIG.capability_gaps.items():
        assert gap.summary.strip(), f"{name} has no summary"
        assert gap.satisfied_by.strip(), f"{name} does not say what would satisfy it"
        assert gap.skip_patterns, f"{name} declares no skip patterns"
