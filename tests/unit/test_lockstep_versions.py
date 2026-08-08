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

Both numbers are rewritten by release-please, never by hand — so the
``x-release-please-*`` annotations that let it find them are part of the
contract too. A package that loses an annotation, or that never reaches
``extra-files``, silently stops being bumped and drifts out of lockstep at the
next release.

Coverage
--------
- Root and every ``python/*`` package declare the same ``[project] version``.
- Every ``openral-*`` dependency is pinned ``==`` to that version.
- Every ``python/*`` package that ships a distribution appears in the
  ``release-pypi.yml`` publish matrix.
- Every ``python/*`` package is listed in ``release-please-config.json`` and
  carries the annotations release-please needs.
- The manifest agrees with the tree about the current version.
- ``uv.lock`` records that same version for every workspace member.
- No package hardcodes ``__version__`` — release-please only rewrites
  ``pyproject.toml``, so source has to derive it.
"""

from __future__ import annotations

import json
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


def _release_please_config() -> dict:
    return json.loads((REPO_ROOT / "release-please-config.json").read_text(encoding="utf-8"))


def test_every_package_is_an_extra_file() -> None:
    """release-please only rewrites what ``extra-files`` names."""
    extra = set(_release_please_config()["packages"]["."]["extra-files"])
    expected = {f"python/{p.name}/pyproject.toml" for p in PACKAGE_DIRS}
    assert expected == extra, (
        f"release-please extra-files drift — missing: {sorted(expected - extra)}, "
        f"stale: {sorted(extra - expected)}. An unlisted package never gets bumped."
    )


@pytest.mark.parametrize("pkg_dir", PACKAGE_DIRS, ids=lambda p: p.name)
def test_package_carries_release_please_annotations(pkg_dir: Path) -> None:
    """The ``version`` line and any sibling block are annotated for the updater.

    release-please's generic updater rewrites *only* annotated lines, so a
    missing marker fails open — the file keeps the old version with no error.
    """
    lines = (pkg_dir / "pyproject.toml").read_text(encoding="utf-8").split("\n")

    version_lines = [ln for ln in lines if ln.startswith("version = ")]
    assert len(version_lines) == 1, (
        f"{pkg_dir.name}: expected one version line, got {version_lines}"
    )
    assert version_lines[0].endswith("# x-release-please-version"), (
        f"{pkg_dir.name}: {version_lines[0]!r} is missing the "
        "`# x-release-please-version` annotation — it would not be bumped."
    )

    sibling_idx = [i for i, ln in enumerate(lines) if _SIBLING_RE.match(ln.strip().strip('",'))]
    if not sibling_idx:
        return  # no siblings (core, observability) — nothing to fence.

    starts = [i for i, ln in enumerate(lines) if ln.strip() == "# x-release-please-start-version"]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "# x-release-please-end"]
    assert len(starts) == 1 and len(ends) == 1, (
        f"{pkg_dir.name}: expected exactly one start/end annotation pair, "
        f"got {len(starts)} start(s) and {len(ends)} end(s)."
    )
    assert starts[0] < min(sibling_idx) and max(sibling_idx) < ends[0], (
        f"{pkg_dir.name}: sibling pins fall outside the "
        "x-release-please-start-version block, so they would not be bumped."
    )


def test_manifest_matches_the_tree() -> None:
    """``.release-please-manifest.json`` is release-please's view of "now"."""
    manifest = json.loads((REPO_ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    assert manifest["."] == _root_version(), (
        f"manifest says {manifest['.']}, root pyproject says {_root_version()}. "
        "release-please computes the next bump from the manifest — they must agree."
    )


def test_uv_lock_records_the_lockstep_version() -> None:
    """``uv.lock`` pins every workspace member at the tree's version.

    release-please rewrites pyprojects textually and cannot regenerate a lock
    file, so the two drifted for two releases: master shipped 0.3.1 pyprojects
    against a lock still naming 0.2.0, and `uv sync --frozen` happily installed
    the stale metadata. release-please.yml now runs `uv lock` on the release
    branch; this test is what fails if that step is dropped or silently stops
    working.
    """
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    # Each workspace member is a `[[package]]` whose source is the local tree.
    members = re.findall(
        r'^name = "(openral[a-z-]*)"\nversion = "([^"]+)"\nsource = \{ (?:editable|virtual)',
        lock,
        re.M,
    )
    assert len(members) >= len(PACKAGE_DIRS), (
        f"expected every workspace member in uv.lock, found {members}"
    )
    stale = {name: version for name, version in members if version != _root_version()}
    assert not stale, (
        f"uv.lock is a release behind for {stale}; the tree is at {_root_version()}. "
        "Run `uv lock` and commit the result."
    )


@pytest.mark.parametrize("pkg_dir", PACKAGE_DIRS, ids=lambda p: p.name)
def test_dunder_version_is_derived_not_hardcoded(pkg_dir: Path) -> None:
    """``__version__`` reads ``importlib.metadata``, never a literal.

    release-please rewrites `pyproject.toml`, not source. Eleven packages
    carried a hand-written ``__version__ = "0.2.0"`` that nothing bumped, so
    two releases later the runtime still announced 0.2.0 while the wheel said
    0.3.1 — and ``test_smoke.py``'s metadata comparison failed on master. A
    literal here is a second source of truth; there is only one.
    """
    src = pkg_dir / "src"
    for init in src.glob("*/__init__.py"):
        for line in init.read_text(encoding="utf-8").split("\n"):
            if not line.startswith("__version__"):
                continue
            assert not re.match(r'^__version__\s*=\s*["\']', line), (
                f"{init.relative_to(REPO_ROOT)}: {line!r} hardcodes the version. "
                'Use `_pkg_version("<dist-name>")` from importlib.metadata — '
                "release-please only rewrites pyproject.toml."
            )


def test_selective_tests_short_circuit_the_release_pr() -> None:
    """The release PR is version churn only, and must not drag in the sim lanes.

    Rewriting all 15 pyprojects matches the ``pyproject.toml`` full-run glob and
    marks every package changed, which would otherwise select the whole suite
    plus every opt-in lane — including ones whose sidecars a hosted runner
    cannot provision. The guard has to live *inside* the job: ``select-and-test``
    is a required check, and a required check skipped by a job-level condition
    is never reported, so GitHub blocks the merge forever.
    """
    path = REPO_ROOT / ".github/workflows/test-selective.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = workflow["jobs"]["select-and-test"]

    assert "if" not in job, (
        "select-and-test is a required check; a job-level `if:` leaves it "
        "reported as Expected forever and blocks every merge it skips."
    )

    select = next(s for s in job["steps"] if s.get("id") == "select")
    assert select["env"]["HEAD_REF"] == "${{ github.head_ref }}", (
        "the release-PR guard reads the head branch from HEAD_REF"
    )
    # release-please names the branch from RELEASE_PLEASE_BRANCH_PREFIX,
    # "release-please--branches"; the guard globs the shorter prefix so the
    # --components-- variant is covered too.
    assert "release-please--*)" in select["run"], (
        "the select step must short-circuit release-please's branch"
    )
    for output in ("any=false", "full_run=false"):
        assert output in select["run"], (
            f"the guard must emit {output!r} — the full-suite step is gated on "
            "full_run alone, the rest on any."
        )


def test_release_please_reports_a_stalled_release() -> None:
    """A release that stalls between merge and tag must fail the job, not pass.

    release-please emits no outputs when it finds a merged release PR it cannot
    build a release from, then declines to open a new one. That is byte-for-byte
    the same output as "nothing releasable merged", so the summarise step cannot
    tell them apart from the action alone — it has to ask the repository whether
    a merged PR is still labelled ``autorelease: pending``. v0.3.0 sat stalled
    across two runs while the step reported success, and no tag was ever cut.
    """
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/release-please.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["release-please"]["steps"]
    summarise = next(s for s in steps if s.get("name") == "summarise")
    run = summarise["run"]

    assert "autorelease: pending" in run, (
        "the stalled-release check keys off release-please's pending label"
    )
    assert "--state merged" in run, "only a *merged* pending PR indicates a stall"
    assert "::error::" in run and "exit 1" in run, (
        "a stalled release must fail the job — reporting success is what hid it"
    )
    # The action sets releases_created in manifest mode; release_created is the
    # single-package name. Reading only one silently misses a real release.
    assert "releases_created" in summarise["env"]["RELEASED"], (
        "manifest mode reports via releases_created"
    )
    assert "GH_TOKEN" in summarise["env"], "the label query needs an authenticated gh"
