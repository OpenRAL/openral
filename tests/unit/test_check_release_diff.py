"""The release-please test skip holds only for a version-only diff (`check_release_diff.py`).

Each case commits the repo's real ``python/cli/pyproject.toml`` and ``uv.lock``
into a scratch git repo, applies a release bump or a smuggled change on a branch,
and runs the checker over the real ``git diff``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "check_release_diff", REPO_ROOT / "tools" / "check_release_diff.py"
)
assert _spec is not None and _spec.loader is not None
check_release_diff = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_release_diff
_spec.loader.exec_module(check_release_diff)

# The fixture is the live pyproject, so read its version: release-please bumps it.
_CURRENT = tomllib.loads((REPO_ROOT / "python/cli/pyproject.toml").read_text())["project"][
    "version"
]
_MAJOR, _MINOR, _ = _CURRENT.split(".")
_NEXT = f"{_MAJOR}.{int(_MINOR) + 1}.0"
_VERSION = f'"{_CURRENT}"'


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def release_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for rel in ("python/cli/pyproject.toml", "uv.lock"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((REPO_ROOT / rel).read_text())
    assert _VERSION in (tmp_path / "python/cli/pyproject.toml").read_text()
    _git(tmp_path, "init", "-q", "-b", "master")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "checkout", "-qb", "release-please--branches--master")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _bump(repo: Path, rel: str, edit: Callable[[str], str]) -> list[str]:
    path = repo / rel
    path.write_text(edit(path.read_text()))
    _git(repo, "commit", "-qam", "change")
    return list(check_release_diff.violations("master", "HEAD"))


def _release(text: str) -> str:
    return text.replace(_VERSION, f'"{_NEXT}"').replace(f"=={_CURRENT}", f"=={_NEXT}")


def test_a_lockstep_version_bump_passes(release_repo: Path) -> None:
    (release_repo / "CHANGELOG.md").write_text(f"## {_NEXT}\n")
    _git(release_repo, "add", "CHANGELOG.md")
    assert _bump(release_repo, "python/cli/pyproject.toml", _release) == []
    assert _bump(release_repo, "uv.lock", _release) == []


def test_a_dependency_edit_in_a_pyproject_is_refused(release_repo: Path) -> None:
    def smuggle(text: str) -> str:
        return _release(text).replace('"typer>=0.12"', '"typer>=0.12", "evil-pkg"')

    assert _bump(release_repo, "python/cli/pyproject.toml", smuggle) == [
        "python/cli/pyproject.toml: changes more than OpenRAL versions and pins"
    ]


def test_a_third_party_version_in_the_lock_is_refused(release_repo: Path) -> None:
    def smuggle(text: str) -> str:
        head, sep, tail = text.partition('name = "typer"\nversion = "')
        assert sep, "uv.lock no longer pins typer"
        return head + sep + "9" + tail

    assert _bump(release_repo, "uv.lock", smuggle) == [
        "uv.lock: changes more than OpenRAL versions and pins"
    ]


def test_a_non_release_file_is_refused(release_repo: Path) -> None:
    (release_repo / "tools.py").write_text("print('x')\n")
    _git(release_repo, "add", "tools.py")
    assert _bump(release_repo, "python/cli/pyproject.toml", _release) == [
        "tools.py: not a release file"
    ]


def test_an_openral_pin_off_the_release_version_is_refused(release_repo: Path) -> None:
    """Only a pin that moves in lockstep with ``project.version`` is release churn."""

    def smuggle(text: str) -> str:
        return text.replace(f'"openral-core=={_CURRENT}"', '"openral-core==999.0"')

    assert _bump(release_repo, "python/cli/pyproject.toml", smuggle) == [
        "python/cli/pyproject.toml: changes more than OpenRAL versions and pins"
    ]
