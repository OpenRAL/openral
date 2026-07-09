"""Tests for the test-suite auditor (``tools/audit_tests.py``).

The AST helpers are exercised on real parsed source (``ast.parse`` of real
Python — not a mock), and the collector runs against the *real* repo tree
(CLAUDE.md §1.11). The repo-level assertions double as a regression guard: if
someone ever lands a shadowed or placeholder test, ``test_repo_has_no_dead_tests``
goes red.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "audit_tests", REPO_ROOT / "tools" / "audit_tests.py"
)
assert _spec is not None and _spec.loader is not None
audit_tests = importlib.util.module_from_spec(_spec)
# Register before exec so Pydantic can resolve the module's own forward refs
# (e.g. AuditReport -> TestFuncInfo) via sys.modules under `from __future__`.
sys.modules[_spec.name] = audit_tests
_spec.loader.exec_module(audit_tests)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_is_trivial_detects_pass_and_ellipsis_and_docstring_only() -> None:
    assert audit_tests._is_trivial(_fn("def test_x():\n    pass"))
    assert audit_tests._is_trivial(_fn("def test_x():\n    ..."))
    assert audit_tests._is_trivial(_fn('def test_x():\n    """doc only"""'))
    assert not audit_tests._is_trivial(_fn("def test_x():\n    assert 1 == 1"))


def test_scan_function_finds_assert_and_validation() -> None:
    has_assert, has_val = audit_tests._scan_function(_fn("def test_x():\n    assert foo()"))
    assert has_assert and not has_val

    has_assert, has_val = audit_tests._scan_function(
        _fn("def test_x():\n    Model.model_validate(data)")
    )
    assert not has_assert and has_val

    has_assert, has_val = audit_tests._scan_function(
        _fn("def test_x():\n    with pytest.raises(ValueError):\n        boom()")
    )
    assert has_val

    has_assert, has_val = audit_tests._scan_function(
        _fn("def test_x():\n    self.assertEqual(a, b)")
    )
    assert has_assert


def test_body_hash_is_location_insensitive() -> None:
    a = _fn("def test_a():\n    x = 1\n    assert x")
    b = _fn("def test_b():\n\n\n    x = 1\n    assert x")  # same body, different lines/blanks
    assert audit_tests._body_hash(a) == audit_tests._body_hash(b)


def test_markers_extracts_pytest_mark() -> None:
    fn = _fn("@pytest.mark.slow\n@pytest.mark.sim\ndef test_x():\n    assert True")
    assert set(audit_tests._markers(fn)) == {"slow", "sim"}


def test_collect_runs_over_real_tree() -> None:
    records = audit_tests.collect(REPO_ROOT)
    # The suite is large; this guards against the collector silently finding nothing.
    assert len(records) > 2000
    # qualnames distinguish same-named methods in different classes.
    quals = {(r.path, r.qualname) for r in records}
    assert len(quals) == len(records)  # no (path, qualname) collisions survive as duplicates...


def test_repo_has_no_dead_tests() -> None:
    """Regression guard: the real tree carries no trivial or shadowed tests."""
    report = audit_tests.build_report(audit_tests.collect(REPO_ROOT))
    assert report.trivial == []
    assert report.shadowed == []


def test_shadowing_is_detected() -> None:
    """Two same-named methods in ONE class → the first is reported as dead."""
    records = audit_tests.collect(REPO_ROOT)
    # Synthesize a shadow by appending a duplicate of an existing record.
    victim = records[0]
    dup = victim.model_copy(update={"lineno": victim.lineno + 100})
    report = audit_tests.build_report([victim, dup])
    assert len(report.shadowed) == 1
    assert report.shadowed[0].lineno == victim.lineno  # the earlier def is the dead one


def test_render_markdown_smoke() -> None:
    report = audit_tests.build_report(audit_tests.collect(REPO_ROOT))
    md = audit_tests.render_markdown(report)
    assert "# Test-suite audit" in md
    assert "Shadowed tests" in md
    assert "Duplicate-body groups" in md
