"""Regression tests for the DCO sign-off hook (``.githooks/prepare-commit-msg``).

The hook auto-appends a ``Signed-off-by:`` trailer so the ``Verify Signed-off-by``
CI passes without anyone remembering ``git commit -s``. Two defects corrupted a
real commit (``990d70b``) during a stack rebase:

1. Idempotence keyed on the *formatted* trailer built from ``git config
   user.name``. A configured ``AdrianLlopart`` never matched the ``Adrian
   Llopart`` already recorded in the message, so every rebase step that re-runs
   the hook (``reword``/``edit``, ``rebase --continue``) appended another
   sign-off.
2. ``git interpret-trailers`` reads a lone ``---`` as the start of a patch and
   inserted the trailer *above* it — splitting a body that legitimately contains
   ``---`` and burying the trailer in the middle of the prose.

These tests drive the real hook through real ``git commit`` and ``git rebase``
invocations in a throwaway repository, so they neither need nor are affected by
the hooks installed in the OpenRAL checkout itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "prepare-commit-msg"

NAME = "AdrianLlopart"
EMAIL = "adrianllopart@gmail.com"
TRAILER = f"Signed-off-by: Adrian Llopart <{EMAIL}>"

#: A body whose middle paragraph is separated by ``---``, mirroring the
#: safety-WG rationale that commit 990d70b carried when it was mangled.
DIVIDER_BODY = (
    "fix(safety): keep the workspace-violation rationale intact\n"
    "\n"
    "The safety WG asked for the derivation to travel with the commit.\n"
    "\n"
    "---\n"
    "\n"
    "Below the divider: the conservative bound this change preserves.\n"
)


def _env(repo: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A git environment with the ambient user/system config neutralised."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(repo.parent),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": NAME,
        "GIT_AUTHOR_EMAIL": EMAIL,
        "GIT_COMMITTER_NAME": NAME,
        "GIT_COMMITTER_EMAIL": EMAIL,
    }
    env.update(extra or {})
    return env


def _git(
    repo: Path, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo``, failing loudly on a non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_env(repo, extra_env),
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo whose only active hook is the real ``prepare-commit-msg``."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook_copy = hooks / "prepare-commit-msg"
    hook_copy.write_bytes(HOOK.read_bytes())
    hook_copy.chmod(0o755)

    work = tmp_path / "repo"
    work.mkdir()
    _git(work, "init", "-q", "-b", "master", ".")
    _git(work, "config", "core.hooksPath", str(hooks))
    _git(work, "config", "user.name", NAME)
    _git(work, "config", "user.email", EMAIL)
    (work / "seed.txt").write_text("seed\n")
    _git(work, "add", "seed.txt")
    _git(work, "commit", "-q", "-m", "chore: seed")
    return work


def _commit(repo: Path, message: str, filename: str) -> None:
    (repo / filename).write_text(f"{filename}\n")
    _git(repo, "add", filename)
    msg_file = repo.parent / f"{filename}.msg"
    msg_file.write_text(message)
    _git(repo, "commit", "-q", "-F", str(msg_file))


def _message(repo: Path, rev: str = "HEAD") -> str:
    return _git(repo, "log", "-1", "--format=%B", rev).stdout


def test_appends_sign_off_to_a_plain_commit(repo: Path) -> None:
    """The ordinary fresh-commit case is unchanged: one trailer, at the end."""
    _commit(repo, "feat(hal): add a joint limit probe\n", "a.txt")

    message = _message(repo)
    assert message.count("Signed-off-by:") == 1
    assert message.strip().endswith(f"Signed-off-by: {NAME} <{EMAIL}>")


def test_does_not_duplicate_an_existing_sign_off_spelled_differently(repo: Path) -> None:
    """Idempotence keys on the email, so a differently spelled name still matches.

    The old hook also survived this particular shape — not because its check
    worked, but because ``--if-exists doNothing`` saw a ``Signed-off-by`` key in
    the trailing trailer block and declined. That safety net disappears the
    moment a ``---`` divider hides the existing trailer (see the divider and
    rebase tests), which is exactly how 990d70b ended up with two sign-offs.
    """
    _commit(repo, f"fix(sensors): clamp the stale deadline\n\n{TRAILER}\n", "b.txt")

    message = _message(repo)
    assert message.count("Signed-off-by:") == 1
    assert TRAILER in message


def test_divider_in_the_body_is_not_split_by_the_trailer(repo: Path) -> None:
    """A `---` line is body prose, not a patch separator: the body survives intact."""
    _commit(repo, DIVIDER_BODY, "c.txt")

    message = _message(repo)
    assert message.count("Signed-off-by:") == 1
    # Every body line keeps its order, and nothing is injected above the divider.
    assert DIVIDER_BODY.strip() in message
    assert message.strip().endswith(f"Signed-off-by: {NAME} <{EMAIL}>")


def test_rebase_reword_replay_neither_duplicates_nor_corrupts(repo: Path) -> None:
    """The observed failure: a rebase replay re-runs the hook on the ORIGINAL message."""
    _commit(repo, f"{DIVIDER_BODY}\n{TRAILER}\n", "d.txt")
    pristine = _message(repo)
    assert pristine.count("Signed-off-by:") == 1

    # `reword` (like `edit` and `rebase --continue`) re-runs prepare-commit-msg,
    # feeding it the original message; a plain `pick` fast-forwards and does not.
    sequence_editor = repo.parent / "reword.py"
    sequence_editor.write_text(
        "import pathlib, sys\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "p.write_text(p.read_text().replace('pick ', 'reword ', 1))\n"
    )
    rebase_env = {
        "GIT_SEQUENCE_EDITOR": f"{sys.executable} {sequence_editor}",
        "GIT_EDITOR": "true",
    }
    for _ in range(2):  # a stack rebase replays the same commit more than once
        _git(repo, "rebase", "-i", "master~1", extra_env=rebase_env)

    assert _message(repo) == pristine


@pytest.mark.parametrize("source", ["merge", "squash"])
def test_merge_and_squash_messages_are_left_alone(repo: Path, source: str) -> None:
    """Those aren't a single authored change, so the hook must not sign them off."""
    msg_file = repo.parent / "msg.txt"
    original = "Merge branch 'topic'\n"
    msg_file.write_text(original)
    subprocess.run(
        [str(HOOK), str(msg_file), source],
        cwd=repo,
        env=_env(repo),
        check=True,
        capture_output=True,
    )
    assert msg_file.read_text() == original
