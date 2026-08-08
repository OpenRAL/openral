"""Lockstep versioning contract across the workspace.

Every distributable package in this monorepo shares one SemVer number, and
every inter-package dependency pins that exact number. Both halves matter:

- The shared number is what ``release-pypi.yml`` publishes as a set and what
  a ``v*.*.*`` tag names.
- The ``==`` pins are the only thing that makes "lockstep" true *for an
  installer*. ``[tool.uv.sources]`` resolves siblings to workspace members
  for local development, but it is stripped at build time — an unpinned
  ``openral-core`` in published metadata lets pip pair ``openral-cli 0.2.0``
  with any future ``openral-core``.

Coverage
--------
- Root and every ``python/*`` package declare the same ``[project] version``.
- Every ``openral-*`` dependency is pinned ``==`` to that version.
- Every ``python/*`` package that ships a distribution appears in the
  ``release-pypi.yml`` publish matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIRS = sorted(
    p for p in (REPO_ROOT / "python").glob("*") if (p / "pyproject.toml").is_file()
)

# `openral-core==0.2.0` / `openral-observability[dashboard]==0.2.0` → name, extras, specifier.
_SIBLING_RE = re.compile(r"^(openral-[a-z0-9-]+)(\[[a-z0-9,-]+\])?(.*)$")


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _root_version() -> str:
    version = _pyproject(REPO_ROOT / "pyproject.toml")["project"]["version"]
    assert isinstance(version, str)
    return version


def test_package_dirs_discovered() -> None:
    """Guard the parametrisation itself — an empty sweep would pass vacuously."""
    assert len(PACKAGE_DIRS) >= 14, f"expected the full workspace, found {PACKAGE_DIRS}"


@pytest.mark.parametrize("pkg_dir", PACKAGE_DIRS, ids=lambda p: p.name)
def test_package_version_matches_root(pkg_dir: Path) -> None:
    """Every workspace member carries the root's version."""
    version = _pyproject(pkg_dir / "pyproject.toml")["project"]["version"]
    assert version == _root_version(), (
        f"{pkg_dir.name} is at {version}, root is at {_root_version()}. "
        "Releases are lockstep — bump every pyproject together."
    )


@pytest.mark.parametrize("pkg_dir", PACKAGE_DIRS, ids=lambda p: p.name)
def test_sibling_dependencies_pin_the_lockstep_version(pkg_dir: Path) -> None:
    """Inter-package deps pin ``==<lockstep version>``, extras included."""
    expected = f"=={_root_version()}"
    project = _pyproject(pkg_dir / "pyproject.toml")["project"]
    deps: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        deps.extend(group)

    for dep in deps:
        match = _SIBLING_RE.match(dep.strip())
        if match is None:
            continue  # third-party dependency — versioned on its own cadence.
        name, _extras, specifier = match.groups()
        assert specifier == expected, (
            f"{pkg_dir.name} declares {dep!r}; expected {name}{_extras or ''}{expected}. "
            "Unpinned siblings let pip mix release versions at install time."
        )


def test_every_package_is_in_the_publish_matrix() -> None:
    """A new ``python/<name>`` package must also be published."""
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/release-pypi.yml").read_text())
    published = set(workflow["jobs"]["build-and-publish"]["strategy"]["matrix"]["package"])
    expected = {f"python/{p.name}" for p in PACKAGE_DIRS}
    assert expected == published, (
        f"publish matrix drift — missing: {sorted(expected - published)}, "
        f"stale: {sorted(published - expected)}"
    )
