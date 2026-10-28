"""Gate the release-please test short-circuit on a version-only diff.

`.github/actions/select-tests` skips every test for a `release-please--*` branch.
A path allowlist alone would let that branch change a dependency in a
`pyproject.toml` or `uv.lock` untested, so each release file is parsed at both
revisions and compared with only OpenRAL's own versions and `openral-*==` pins
blanked out. Any other difference selects tests normally.

Usage: ``python tools/check_release_diff.py BASE HEAD`` — exit 0 when the
``BASE...HEAD`` diff is pure release churn, 1 otherwise (each reason printed).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

import tomllib

_PIN = re.compile(r"^(openral-[A-Za-z0-9_.-]+(?:\[[^\]]*\])?)==\S+$")
_PYPROJECT = re.compile(r"^(?:python/[^/]+/)?pyproject\.toml$")


def _blank_pins(deps: list[str]) -> list[str]:
    return [_PIN.sub(r"\1==X", d.strip()) for d in deps]


def normalize(path: str, text: str) -> Any:
    """Parse a release file and blank the fields a release bump may rewrite.

    Args:
        path: Repo-relative path, which picks the format.
        text: File contents.

    Returns:
        A comparable value; equal at both revisions iff only release fields changed.

    Example:
        >>> a = normalize(
        ...     "pyproject.toml", 'project = {version = "1", dependencies = ["openral-core==1"]}'
        ... )
        >>> b = normalize(
        ...     "pyproject.toml", 'project = {version = "2", dependencies = ["openral-core==2"]}'
        ... )
        >>> a == b
        True
    """
    if path == ".release-please-manifest.json":
        return {k: "X" for k in json.loads(text)}
    data = tomllib.loads(text)
    if path == "uv.lock":
        for pkg in data.get("package", []):
            if pkg.get("name", "").startswith("openral"):
                pkg["version"] = "X"
        return data
    project = data.get("project", {})
    if "version" in project:
        project["version"] = "X"
    if "dependencies" in project:
        project["dependencies"] = _blank_pins(project["dependencies"])
    for extra, deps in project.get("optional-dependencies", {}).items():
        project["optional-dependencies"][extra] = _blank_pins(deps)
    return data


def _show(rev: str, path: str) -> str | None:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=False
    )
    return out.stdout if out.returncode == 0 else None


def violations(base: str, head: str) -> list[str]:
    """Every way the ``base...head`` diff goes beyond a release bump."""
    merge_base = subprocess.run(
        ["git", "merge-base", base, head], capture_output=True, text=True, check=True
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    problems = []
    for path in changed:
        if path == "CHANGELOG.md":
            continue
        if not (_PYPROJECT.match(path) or path in ("uv.lock", ".release-please-manifest.json")):
            problems.append(f"{path}: not a release file")
            continue
        old, new = _show(merge_base, path), _show(head, path)
        if old is None or new is None:
            problems.append(f"{path}: added or removed")
        elif normalize(path, old) != normalize(path, new):
            problems.append(f"{path}: changes more than OpenRAL versions and pins")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``BASE HEAD``; print each violation, return 1 on any."""
    base, head = argv if argv is not None else sys.argv[1:]
    problems = violations(base, head)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
