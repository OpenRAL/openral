"""Tests for the selective-test tool (``tools/select_tests.py``).

CLAUDE.md §1.11 — these run against the *real* workspace: the real
``python/*/pyproject.toml`` dependency declarations, the real ``tests/`` tree,
and the real ``tools/test_selection.toml``. No fixtures invented, no graph
mocked; if the workspace layout changes in a way that breaks selection, these
fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load tools/select_tests.py as a module (tools/ is not an installed package).
_spec = importlib.util.spec_from_file_location(
    "select_tests", REPO_ROOT / "tools" / "select_tests.py"
)
assert _spec is not None and _spec.loader is not None
select_tests = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = select_tests  # let Pydantic resolve forward refs
_spec.loader.exec_module(select_tests)

CONFIG = select_tests.load_config(REPO_ROOT / "tools" / "test_selection.toml")


def test_config_has_blast_radius_and_ignores() -> None:
    assert "pyproject.toml" in CONFIG.full_run_globs
    assert "uv.lock" in CONFIG.full_run_globs
    assert any(g.startswith("cpp/") for g in CONFIG.ignore_globs)


def test_dependency_graph_reflects_real_workspace() -> None:
    graph = select_tests.build_dependency_graph(REPO_ROOT)
    # hal really depends on core (python/hal/pyproject.toml declares openral-core).
    assert "openral_core" in graph["openral_hal"]
    # core is foundational and depends on no other openral package.
    assert graph["openral_core"] == set()


def test_transitive_dependents_includes_chain() -> None:
    graph = {
        "openral_core": set(),
        "openral_hal": {"openral_core"},
        "openral_runner": {"openral_hal"},
    }
    affected = select_tests.transitive_dependents(graph, {"openral_core"})
    assert affected == {"openral_core", "openral_hal", "openral_runner"}


def test_leaf_package_selects_only_its_own_tests() -> None:
    result = select_tests.select(REPO_ROOT, ["python/wam/src/openral_wam/core.py"], CONFIG)
    assert not result.full_run
    assert result.affected_packages == ["openral_wam"]
    assert result.targets == ["python/wam/tests"]


def test_core_change_fans_out_widely() -> None:
    result = select_tests.select(REPO_ROOT, ["python/core/src/openral_core/schemas.py"], CONFIG)
    assert not result.full_run
    # core is depended on by ~every package, so the affected set is broad.
    assert "openral_hal" in result.affected_packages
    assert "openral_rskill" in result.affected_packages
    assert len(result.targets) > 50  # fans out across the top-level tests/ tree
    # every selected target carries a reason (CLAUDE.md §1.4).
    for tgt in result.targets:
        assert result.reasons[tgt]


def test_blast_radius_forces_full_run() -> None:
    result = select_tests.select(REPO_ROOT, ["pyproject.toml"], CONFIG)
    assert result.full_run
    assert result.full_run_reason is not None
    assert result.targets == []


def test_doc_only_change_selects_nothing() -> None:
    result = select_tests.select(REPO_ROOT, ["docs/architecture/repo-map.md"], CONFIG)
    assert not result.full_run
    assert result.targets == []


def test_cpp_change_is_ignored_by_python_selector() -> None:
    result = select_tests.select(REPO_ROOT, ["cpp/openral_safety_kernel/src/kernel.cpp"], CONFIG)
    assert not result.full_run
    assert result.targets == []


def test_unattributed_python_source_forces_full_run() -> None:
    result = select_tests.select(REPO_ROOT, ["scripts/mystery_tool.py"], CONFIG)
    assert result.full_run
    assert "unattributed" in (result.full_run_reason or "")


def test_changed_test_file_is_selected_directly() -> None:
    result = select_tests.select(REPO_ROOT, ["tests/unit/test_select_tests.py"], CONFIG)
    assert "tests/unit/test_select_tests.py" in result.targets


def test_ros_package_change_selects_its_test_dir() -> None:
    target = "packages/openral_hal_so100"
    if not (REPO_ROOT / target / "test").is_dir():
        pytest.skip(f"{target}/test not present in this checkout")
    result = select_tests.select(REPO_ROOT, [f"{target}/openral_hal_so100/node.py"], CONFIG)
    assert f"{target}/test" in result.targets


def test_fixture_change_triggers_loader_test() -> None:
    result = select_tests.select(REPO_ROOT, ["scenes/kitchen/pick.yaml"], CONFIG)
    assert "tests/unit/test_examples_sim_configs_load.py" in result.targets


# --- isolation of fork-in-threaded-process crashers (issue #24) ---------------

_FORK_TEST = "tests/unit/test_cli_dataset_from_bag.py"


def test_config_declares_fork_isolation_globs() -> None:
    # The real toml flags the lerobot dataset CLI tests that fork a pool.
    assert _FORK_TEST in CONFIG.isolate_globs
    assert "tests/unit/test_cli_replay_record_profile.py" in CONFIG.isolate_globs


def test_isolated_test_is_peeled_out_of_targets() -> None:
    # A core change selects the broad CLI surface, which imports the fork tests.
    result = select_tests.select(REPO_ROOT, ["python/core/src/openral_core/schemas.py"], CONFIG)
    assert _FORK_TEST in result.isolated_targets
    # Critically, it must NOT also be a normal target — that is what folds it
    # into the broad partition and trips the teardown crash.
    assert _FORK_TEST not in result.targets
    # Every isolated target still carries a reason (CLAUDE.md §1.4).
    assert result.reasons[_FORK_TEST]


def test_directly_changed_fork_test_runs_isolated_only() -> None:
    # Editing the fork test itself selects it, but only in its own process.
    result = select_tests.select(REPO_ROOT, [_FORK_TEST], CONFIG)
    assert not result.full_run
    assert result.isolated_targets == [_FORK_TEST]
    assert _FORK_TEST not in result.targets


def test_full_run_still_reports_isolated_targets() -> None:
    # The full-suite invocation collects the fork tests too, so they must be
    # reported for separate execution even on a blast-radius change.
    result = select_tests.select(REPO_ROOT, ["pyproject.toml"], CONFIG)
    assert result.full_run
    assert _FORK_TEST in result.isolated_targets
    assert result.targets == []


def test_unrelated_change_does_not_isolate_fork_test() -> None:
    # A leaf change that never reaches the fork tests leaves them out entirely —
    # isolating un-selected tests would defeat selective execution.
    result = select_tests.select(REPO_ROOT, ["python/wam/src/openral_wam/core.py"], CONFIG)
    assert result.isolated_targets == []
