"""Selective test execution — choose the pytest targets a diff actually needs.

OpenRAL ships ~2.9k tests across ~300 files (CLAUDE.md §2 *Tests*). Running
the whole suite on every PR is wasteful — and, while the GitHub Actions
budget is exhausted (every workflow is ``workflow_dispatch`` only), it is the
difference between affordable and not. This tool maps a git diff to the
*minimal* set of tests that can observe the change, by:

1. Deriving the workspace dependency graph from each
   ``python/<pkg>/pyproject.toml`` (never hand-maintained — it cannot drift).
2. Expanding the set of changed packages by its transitive *dependents* (edit
   ``openral_core`` → every package that imports it is in scope).
3. Selecting tests two ways: per-package ``tests/`` dirs for affected
   packages, and any top-level ``tests/`` file whose ``import openral_*`` set
   intersects the affected packages.

CLAUDE.md §1.4 (explicit beats implicit): every selection carries a reason,
and a *blast-radius* change (root config, lockfile, shared conftest, this
tool's own inputs) forces a full run rather than a clever guess — a wrong
negative would silently skip a real regression.

Run::

    uv run python tools/select_tests.py --base origin/master --head HEAD
    uv run python tools/select_tests.py --files python/core/src/openral_core/schemas.py
    uv run python tools/select_tests.py --base origin/master --github-output  # CI

Example:
    >>> from pathlib import Path
    >>> cfg = load_config(Path(__file__).resolve().parent / "test_selection.toml")
    >>> "pyproject.toml" in cfg.full_run_globs
    True
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source extensions that, when changed but unattributable to any package,
# trigger a conservative full run. Non-source paths (docs, images, data) that
# no test imports are simply ignored.
_SOURCE_SUFFIXES = frozenset(
    {".py", ".pyi", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c", ".pyx"}
)

# Matches `import openral_x` / `from openral_x import ...` (the import name,
# e.g. `openral_core`, not the distribution name `openral-core`).
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(openral_[a-z_]+)", re.MULTILINE)

# Matches a distribution dep like `"openral-core"` in a pyproject dependency list.
_DEP_RE = re.compile(r"openral-[a-z-]+")


class SelectionConfig(BaseModel):
    """Typed view of ``tools/test_selection.toml`` (CLAUDE.md §2 — Pydantic for config)."""

    full_run_globs: list[str] = Field(default_factory=list)
    ignore_globs: list[str] = Field(default_factory=list)
    isolate_globs: list[str] = Field(default_factory=list)
    extra_triggers: dict[str, list[str]] = Field(default_factory=dict)


class SelectionResult(BaseModel):
    """Outcome of a selection pass.

    ``full_run`` short-circuits everything: when true the caller must run the
    whole suite and ``targets`` is empty. Otherwise ``targets`` is the sorted,
    de-duplicated set of pytest paths to run, and ``reasons`` explains *why*
    each was picked (CLAUDE.md §1.4).

    ``isolated_targets`` is the subset of in-scope tests that must run in their
    OWN pytest process (``isolate_globs`` — fork-in-threaded-process crashers,
    issue #24). They are removed from ``targets`` so a broad partition never
    collects them; the caller runs each one separately. On a ``full_run`` they
    are still reported (the full-suite invocation must exclude and re-run them).
    """

    full_run: bool = False
    full_run_reason: str | None = None
    affected_packages: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    isolated_targets: list[str] = Field(default_factory=list)
    reasons: dict[str, list[str]] = Field(default_factory=dict)


def load_config(path: Path) -> SelectionConfig:
    """Load and validate the selection config from a TOML file."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return SelectionConfig.model_validate(raw)


def _dist_to_import(dist: str) -> str:
    """``openral-core`` → ``openral_core`` (distribution name → import name)."""
    return dist.replace("-", "_")


def package_dir_import_names(repo_root: Path) -> dict[str, str]:
    """Map each ``python/<dir>`` to its import name (the ``src/openral_*`` dir).

    Example:
        >>> names = package_dir_import_names(REPO_ROOT)
        >>> names["python/core"]
        'openral_core'
    """
    out: dict[str, str] = {}
    for pkg_dir in sorted((repo_root / "python").glob("*")):
        if not pkg_dir.is_dir():
            continue
        src_candidates = sorted((pkg_dir / "src").glob("openral_*"))
        if src_candidates:
            rel = pkg_dir.relative_to(repo_root).as_posix()
            out[rel] = src_candidates[0].name
    return out


def build_dependency_graph(repo_root: Path) -> dict[str, set[str]]:
    """Derive ``import-name -> {direct openral dependency import-names}``.

    Read straight from each ``python/<pkg>/pyproject.toml``; the graph is never
    written down by hand, so it cannot fall out of sync with the workspace.
    """
    graph: dict[str, set[str]] = {}
    for pkg_dir in sorted((repo_root / "python").glob("*")):
        pyproject = pkg_dir / "pyproject.toml"
        if not pyproject.is_dir() and pyproject.exists():
            src_candidates = sorted((pkg_dir / "src").glob("openral_*"))
            if not src_candidates:
                continue
            import_name = src_candidates[0].name
            text = pyproject.read_text(encoding="utf-8")
            deps = {
                _dist_to_import(m.group(0))
                for m in _DEP_RE.finditer(text)
                if _dist_to_import(m.group(0)) != import_name
            }
            graph[import_name] = deps
    return graph


def transitive_dependents(graph: dict[str, set[str]], changed: set[str]) -> set[str]:
    """All packages that (transitively) depend on any package in ``changed``.

    The returned set includes ``changed`` itself. Edit ``openral_core`` and
    every package importing it — directly or through a chain — comes back.

    Example:
        >>> g = {"a": set(), "b": {"a"}, "c": {"b"}}
        >>> sorted(transitive_dependents(g, {"a"}))
        ['a', 'b', 'c']
    """
    # Reverse the edges: dep -> {packages that depend on it}.
    reverse: dict[str, set[str]] = defaultdict(set)
    for pkg, deps in graph.items():
        for dep in deps:
            reverse[dep].add(pkg)
    seen = set(changed)
    stack = list(changed)
    while stack:
        node = stack.pop()
        for dependent in reverse.get(node, set()):
            if dependent not in seen:
                seen.add(dependent)
                stack.append(dependent)
    return seen


def map_test_imports(repo_root: Path) -> dict[str, set[str]]:
    """Map each top-level ``tests/`` file to the ``openral_*`` packages it imports.

    This is what lets a change to ``openral_hal`` pull in the integration test
    that imports it even though that test lives under the shared ``tests/``
    tree rather than ``python/hal/tests``.
    """
    out: dict[str, set[str]] = {}
    tests_root = repo_root / "tests"
    if not tests_root.exists():
        return out
    for test_file in tests_root.rglob("test_*.py"):
        text = test_file.read_text(encoding="utf-8")
        imports = {m.group(1) for m in _IMPORT_RE.finditer(text)}
        if imports:
            out[test_file.relative_to(repo_root).as_posix()] = imports
    return out


def _classify_path(rel: str, dir_imports: dict[str, str]) -> tuple[str, str] | None:
    """Classify a changed path. Returns ``(kind, key)`` or ``None`` (ignored).

    ``kind`` ∈ {``package`` (key = import name), ``ros`` (key = packages/<name>),
    ``test`` (key = the test path itself), ``unattributed-source``}.
    """
    if rel.startswith("python/"):
        for pkg_rel, import_name in dir_imports.items():
            if rel.startswith(pkg_rel + "/"):
                if "/tests/" in rel or rel.endswith("/tests"):
                    return ("test", rel)
                return ("package", import_name)
    if rel.startswith("packages/"):
        parts = rel.split("/")
        if len(parts) >= 2:
            return ("ros", f"packages/{parts[1]}")
    if rel.startswith("tests/") and Path(rel).name.startswith("test_"):
        return ("test", rel)
    if Path(rel).suffix in _SOURCE_SUFFIXES:
        return ("unattributed-source", rel)
    return None


def _isolate_files(repo_root: Path, config: SelectionConfig) -> list[str]:
    """Repo-relative paths of every existing file matching an ``isolate_glob``.

    Globs are resolved against the working tree (literal paths resolve to
    themselves), so a stale entry pointing at a deleted file is silently dropped
    rather than handed to pytest as a phantom target.
    """
    matched: set[str] = set()
    for glob in config.isolate_globs:
        for path in repo_root.glob(glob):
            if path.is_file():
                matched.add(path.relative_to(repo_root).as_posix())
    return sorted(matched)


def _in_scope_isolated(targets: set[str], isolate_files: list[str]) -> list[str]:
    """Isolate files a selected ``targets`` set would actually execute.

    A file is in scope when it is itself a target or lives under a selected
    directory target (e.g. ``tests/unit`` from a fixture trigger). Files no
    selected target reaches are left out — isolating them would run tests the
    diff never selected, defeating the point of selective execution.
    """
    in_scope: list[str] = []
    for rel in isolate_files:
        for tgt in targets:
            if rel == tgt or rel.startswith(tgt.rstrip("/") + "/"):
                in_scope.append(rel)
                break
    return sorted(in_scope)


def select(
    repo_root: Path,
    changed_files: list[str],
    config: SelectionConfig,
) -> SelectionResult:
    """Resolve a list of changed repo-relative paths to pytest targets."""
    isolate_files = _isolate_files(repo_root, config)

    # 1. Blast radius — any match forces a full run. The full-suite invocation
    #    still collects the isolate files, so report them for separate execution.
    for rel in changed_files:
        for glob in config.full_run_globs:
            if fnmatch.fnmatch(rel, glob):
                return SelectionResult(
                    full_run=True,
                    full_run_reason=f"{rel} matches full-run glob {glob!r}",
                    isolated_targets=isolate_files,
                )

    dir_imports = package_dir_import_names(repo_root)
    graph = build_dependency_graph(repo_root)
    test_imports = map_test_imports(repo_root)

    changed_pkgs: set[str] = set()
    ros_pkgs: set[str] = set()
    direct_tests: set[str] = set()
    reasons: dict[str, list[str]] = defaultdict(list)

    for rel in changed_files:
        if any(fnmatch.fnmatch(rel, glob) for glob in config.ignore_globs):
            continue
        classified = _classify_path(rel, dir_imports)
        if classified is None:
            # Try the non-code fixture triggers before ignoring.
            for glob, trigger_targets in config.extra_triggers.items():
                if fnmatch.fnmatch(rel, glob):
                    for tgt in trigger_targets:
                        direct_tests.add(tgt)
                        reasons[tgt].append(f"fixture {rel} matches {glob!r}")
            continue
        kind, key = classified
        if kind == "package":
            changed_pkgs.add(key)
        elif kind == "ros":
            ros_pkgs.add(key)
        elif kind == "test":
            direct_tests.add(key)
            reasons[key].append(f"test file {rel} changed")
        elif kind == "unattributed-source":
            return SelectionResult(
                full_run=True,
                full_run_reason=f"unattributed source change: {rel}",
            )

    affected = transitive_dependents(graph, changed_pkgs)

    targets: set[str] = set(direct_tests)

    # 2. Per-package test dirs for affected packages.
    for pkg_rel, import_name in dir_imports.items():
        if import_name in affected and (repo_root / pkg_rel / "tests").is_dir():
            tgt = f"{pkg_rel}/tests"
            targets.add(tgt)
            why = "changed" if import_name in changed_pkgs else "depends on a changed package"
            reasons[tgt].append(f"package {import_name} {why}")

    # 3. ROS package test dirs.
    for ros in sorted(ros_pkgs):
        if (repo_root / ros / "test").is_dir():
            tgt = f"{ros}/test"
            targets.add(tgt)
            reasons[tgt].append(f"ROS package {ros} changed")

    # 4. Top-level tests whose imports intersect the affected packages.
    for test_rel, imports in test_imports.items():
        if imports & affected:
            targets.add(test_rel)
            hit = sorted(imports & affected)
            reasons[test_rel].append(f"imports affected package(s): {', '.join(hit)}")

    # 5. Peel fork-in-threaded-process crashers into their own process (#24).
    #    Removed from `targets` so no broad partition collects them; a directory
    #    target that *contains* one stays (the caller --ignores it there).
    isolated = _in_scope_isolated(targets, isolate_files)
    targets -= set(isolated)
    for rel in isolated:
        reasons[rel].append("isolated: forks a multiprocessing pool (issue #24)")

    return SelectionResult(
        full_run=False,
        affected_packages=sorted(affected),
        targets=sorted(targets),
        isolated_targets=isolated,
        reasons={k: reasons[k] for k in sorted(reasons)},
    )


def changed_files_from_git(base: str, head: str, repo_root: Path) -> list[str]:
    """Repo-relative paths changed between ``base`` and ``head`` (merge-base diff)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _render_human(result: SelectionResult) -> str:
    if result.full_run:
        return f"FULL RUN — {result.full_run_reason}"
    if not result.targets and not result.isolated_targets:
        return "No tests selected (no code paths affected)."
    lines = [f"Affected packages: {', '.join(result.affected_packages) or '(none)'}", ""]
    lines.append(f"Selected {len(result.targets)} pytest target(s):")
    for tgt in result.targets:
        lines.append(f"  • {tgt}")
        for reason in result.reasons.get(tgt, []):
            lines.append(f"      ← {reason}")
    if result.isolated_targets:
        lines.append("")
        lines.append(
            f"{len(result.isolated_targets)} isolated target(s) (own process — issue #24):"
        )
        for tgt in result.isolated_targets:
            lines.append(f"  • {tgt}")
            for reason in result.reasons.get(tgt, []):
                lines.append(f"      ← {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", default="origin/master", help="diff base ref (default: origin/master)"
    )
    parser.add_argument("--head", default="HEAD", help="diff head ref (default: HEAD)")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="explicit changed paths (bypasses git diff; useful for testing)",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="emit GitHub Actions outputs (full_run, targets) to $GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

    config = load_config(REPO_ROOT / "tools" / "test_selection.toml")
    if args.files is not None:
        changed = list(args.files)
    else:
        changed = changed_files_from_git(args.base, args.head, REPO_ROOT)

    result = select(REPO_ROOT, changed, config)

    if args.github_output:
        import json
        import os

        out_path = os.environ.get("GITHUB_OUTPUT")
        payload = {
            "full_run": "true" if result.full_run else "false",
            "targets": " ".join(result.targets),
            "targets_json": json.dumps(result.targets),
            "isolated_targets": " ".join(result.isolated_targets),
            "isolated_targets_json": json.dumps(result.isolated_targets),
            "any": (
                "true"
                if (result.full_run or result.targets or result.isolated_targets)
                else "false"
            ),
        }
        if out_path:
            with open(out_path, "a", encoding="utf-8") as fh:
                for key, value in payload.items():
                    fh.write(f"{key}={value}\n")
        print(json.dumps({**payload, "full_run_reason": result.full_run_reason}, indent=2))
    else:
        print(_render_human(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
