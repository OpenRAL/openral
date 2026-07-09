"""Test-suite auditor — surface redundant / low-signal tests for review.

OpenRAL carries ~2.9k test functions. We want *meaningful* coverage, not
volume (CLAUDE.md §1.7 — tests are part of the contract; §2 truth over
plausibility — we report what is actually there, we do not manufacture
deletions). This tool reads every test with the ``ast`` module and reports:

* **trivial** — body is only ``pass`` / ``...`` / a docstring. Genuinely
  dead; safe to delete.
* **duplicate-body** — two+ test functions with byte-identical normalized
  ASTs. Strong dedup / parametrize signal.
* **no-assertion** — no ``assert``, ``pytest.raises``, ``*.assert*`` call, or
  recognised validation call (``from_yaml`` / ``model_validate`` / ``load`` …).
  These are *candidates* for review, not automatic deletes: a constructor that
  raises on bad input is a real check even without an ``assert``.
* **inventory** — counts per tier / marker / directory, slowest-by-marker.

It is read-only. Pruning is a separate, reviewed commit (CLAUDE.md §1.15 /
§4.2). Regenerate the committed report with ``just test-audit``.

Run::

    uv run python tools/audit_tests.py                 # print summary
    uv run python tools/audit_tests.py --write-report  # refresh docs/contributing/test-audit.md
    uv run python tools/audit_tests.py --json          # machine-readable

Example:
    >>> import ast
    >>> fn = ast.parse("def test_x():\\n    pass").body[0]
    >>> isinstance(fn, ast.FunctionDef) and _is_trivial(fn)
    True
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "docs" / "contributing" / "test-audit.md"

# Test roots: top-level tiers, per-package python tests, ROS package tests.
_TEST_GLOBS = ("tests/**/test_*.py", "python/*/tests/**/test_*.py", "packages/*/test/**/test_*.py")

# Call names that constitute a real check even without a bare `assert` — a
# constructor / validator that raises on bad input. Keeps the no-assertion
# bucket honest (CLAUDE.md §2).
_VALIDATION_CALLS = frozenset(
    {
        "from_yaml",
        "from_json",
        "from_dict",
        "model_validate",
        "model_validate_json",
        "validate",
        "parse",
        "parse_obj",
        "load",
        "loads",
        "raises",
        "warns",
        "approx",
    }
)


class TestFuncInfo(BaseModel):
    """One ``test_*`` function discovered in the tree."""

    path: str
    name: str
    qualname: str  # ``ClassName.test_x`` or ``test_x`` — the pytest collection scope
    lineno: int
    tier: str
    markers: list[str] = Field(default_factory=list)
    has_assertion: bool
    has_validation_call: bool
    is_trivial: bool
    body_hash: str


class DuplicateGroup(BaseModel):
    body_hash: str
    members: list[str]


class AuditReport(BaseModel):
    """Full audit result (CLAUDE.md §2 — Pydantic as the contract)."""

    total_files: int
    total_functions: int
    by_tier: dict[str, int]
    by_marker: dict[str, int]
    by_directory: dict[str, int]
    trivial: list[TestFuncInfo] = Field(default_factory=list)
    no_assertion: list[TestFuncInfo] = Field(default_factory=list)
    shadowed: list[TestFuncInfo] = Field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)


def _tier_of(rel_path: str) -> str:
    parts = rel_path.split("/")
    if parts[0] == "tests" and len(parts) > 1:
        return parts[1]  # unit / integration / sim / hil
    if parts[0] == "python":
        return f"pkg:{parts[1]}"
    if parts[0] == "packages":
        return f"ros:{parts[1]}"
    return "other"


def _is_trivial(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the body is only a docstring and/or ``pass`` / ``...``."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop docstring
    if not body:
        return True
    return all(
        isinstance(stmt, ast.Pass)
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        )
        for stmt in body
    )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _scan_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    """Return ``(has_assertion, has_validation_call)`` for a function body."""
    has_assertion = False
    has_validation = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            has_assertion = True
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name is None:
                continue
            if name.startswith("assert"):
                has_assertion = True
            elif name in _VALIDATION_CALLS:
                has_validation = True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # `with pytest.raises(...)` is a check even though the call is the
            # context expression rather than a plain statement.
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr) in {
                    "raises",
                    "warns",
                }:
                    has_validation = True
    return has_assertion, has_validation


def _markers(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    out: list[str] = []
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        # @pytest.mark.<name> or @pytest.mark.<name>(...)
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            out.append(target.attr)
    return out


def _body_hash(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Normalized hash of the body (sans docstring), location-insensitive."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    dumped = "\n".join(ast.dump(stmt, include_attributes=False) for stmt in body)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()


def _record(node: ast.FunctionDef | ast.AsyncFunctionDef, rel: str, scope: str) -> TestFuncInfo:
    has_assertion, has_validation = _scan_function(node)
    qualname = f"{scope}.{node.name}" if scope else node.name
    return TestFuncInfo(
        path=rel,
        name=node.name,
        qualname=qualname,
        lineno=node.lineno,
        tier=_tier_of(rel),
        markers=_markers(node),
        has_assertion=has_assertion,
        has_validation_call=has_validation,
        is_trivial=_is_trivial(node),
        body_hash=_body_hash(node),
    )


def collect(repo_root: Path) -> list[TestFuncInfo]:
    """Walk every test file and return one record per ``test_*`` function.

    Scope-aware: a method's ``qualname`` carries its enclosing class, so the
    same method name in two different classes is *not* conflated (both run),
    while two same-named functions in one scope are caught as shadowing.
    """
    seen: set[Path] = set()
    records: list[TestFuncInfo] = []
    for glob in _TEST_GLOBS:
        for path in sorted(repo_root.glob(glob)):
            if path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(repo_root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("test_"):
                    records.append(_record(node, rel, scope=""))
                elif isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(
                            method, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and method.name.startswith("test_"):
                            records.append(_record(method, rel, scope=node.name))
    return records


def build_report(records: list[TestFuncInfo]) -> AuditReport:
    by_tier: Counter[str] = Counter(r.tier for r in records)
    by_marker: Counter[str] = Counter(m for r in records for m in r.markers)
    by_directory: Counter[str] = Counter(str(Path(r.path).parent) for r in records)

    by_hash: dict[str, list[TestFuncInfo]] = defaultdict(list)
    for r in records:
        by_hash[r.body_hash].append(r)

    duplicate_groups = [
        DuplicateGroup(
            body_hash=h,
            members=sorted(f"{r.path}::{r.name}" for r in group),
        )
        for h, group in by_hash.items()
        # Ignore trivial bodies here — they are reported separately and a
        # shared `pass` body is not a meaningful "duplicate".
        if len(group) > 1 and not group[0].is_trivial
    ]
    duplicate_groups.sort(key=lambda g: len(g.members), reverse=True)

    # Shadowing: same (file, qualname) defined more than once. Python keeps
    # only the last definition, so every earlier one is dead — it can never be
    # collected by pytest. This is the one truly *safe to delete* duplicate.
    by_qual: dict[tuple[str, str], list[TestFuncInfo]] = defaultdict(list)
    for r in records:
        by_qual[(r.path, r.qualname)].append(r)
    shadowed: list[TestFuncInfo] = []
    for group in by_qual.values():
        if len(group) > 1:
            # all but the last (highest lineno) are shadowed/dead
            ordered = sorted(group, key=lambda r: r.lineno)
            shadowed.extend(ordered[:-1])
    shadowed.sort(key=lambda r: (r.path, r.lineno))

    trivial = [r for r in records if r.is_trivial]
    no_assertion = [
        r for r in records if not r.has_assertion and not r.has_validation_call and not r.is_trivial
    ]
    return AuditReport(
        total_files=len({r.path for r in records}),
        total_functions=len(records),
        by_tier=dict(by_tier.most_common()),
        by_marker=dict(by_marker.most_common()),
        by_directory=dict(by_directory.most_common()),
        trivial=trivial,
        no_assertion=no_assertion,
        shadowed=shadowed,
        duplicate_groups=duplicate_groups,
    )


def render_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("# Test-suite audit")
    lines.append("")
    lines.append(
        "> Generated by `tools/audit_tests.py` (`just test-audit`). Read-only "
        "signal for review — see [`selective-testing.md`](selective-testing.md) "
        "for the companion CI selector. CLAUDE.md §1.7/§1.11: tests are part of "
        "the contract; this report **flags** candidates, it does not delete."
    )
    lines.append("")
    lines.append(f"- **Test files:** {report.total_files}")
    lines.append(f"- **Test functions:** {report.total_functions}")
    lines.append(f"- **Trivial (dead) tests:** {len(report.trivial)}")
    lines.append(f"- **Shadowed (dead) tests:** {len(report.shadowed)}")
    lines.append(f"- **No-assertion candidates:** {len(report.no_assertion)}")
    lines.append(f"- **Duplicate-body groups:** {len(report.duplicate_groups)}")
    lines.append("")

    lines.append("## Inventory by tier")
    lines.append("")
    lines.append("| Tier | Functions |")
    lines.append("| --- | ---: |")
    for tier, count in report.by_tier.items():
        lines.append(f"| `{tier}` | {count} |")
    lines.append("")

    if report.by_marker:
        lines.append("## Inventory by marker")
        lines.append("")
        lines.append("| Marker | Functions |")
        lines.append("| --- | ---: |")
        for marker, count in report.by_marker.items():
            lines.append(f"| `{marker}` | {count} |")
        lines.append("")

    lines.append("## Trivial tests (safe to delete)")
    lines.append("")
    if report.trivial:
        for r in report.trivial:
            lines.append(f"- `{r.path}::{r.name}` (line {r.lineno})")
    else:
        lines.append("_None — no placeholder / `pass`-only tests in the tree._")
    lines.append("")

    lines.append("## Shadowed tests (dead — safe to delete)")
    lines.append("")
    lines.append(
        "Same name defined twice in one scope (file + class). Python keeps only "
        "the last; the earlier definition is never collected by pytest."
    )
    lines.append("")
    if report.shadowed:
        for r in report.shadowed:
            lines.append(f"- `{r.path}::{r.qualname}` (line {r.lineno}) — shadowed")
    else:
        lines.append("_None — no test is silently shadowed by a later redefinition._")
    lines.append("")

    lines.append("## Duplicate-body groups (dedup / parametrize candidates)")
    lines.append("")
    if report.duplicate_groups:
        for group in report.duplicate_groups:
            lines.append(f"- {len(group.members)} identical bodies:")
            for member in group.members:
                lines.append(f"    - `{member}`")
    else:
        lines.append("_None — no two test functions share an identical normalized body._")
    lines.append("")

    lines.append("## No-assertion candidates (review, do not auto-delete)")
    lines.append("")
    lines.append(
        "These neither `assert` nor make a recognised validation call. Some are "
        "real (a side-effecting call that raises); others may be dead. Review "
        "individually."
    )
    lines.append("")
    if report.no_assertion:
        for r in report.no_assertion:
            lines.append(f"- `{r.path}::{r.name}` (line {r.lineno})")
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"write the Markdown report to {REPORT_PATH.relative_to(REPO_ROOT)}",
    )
    args = parser.parse_args(argv)

    records = collect(REPO_ROOT)
    report = build_report(records)

    if args.json:
        print(report.model_dump_json(indent=2))
        return 0

    markdown = render_markdown(report)
    if args.write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    else:
        # Summary to stdout; full detail goes to the report file.
        print(f"files={report.total_files} functions={report.total_functions}")
        print(
            f"trivial={len(report.trivial)} shadowed={len(report.shadowed)} "
            f"no_assertion={len(report.no_assertion)} dup_groups={len(report.duplicate_groups)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
