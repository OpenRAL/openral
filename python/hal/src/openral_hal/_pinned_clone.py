"""Shared staging-clone helper for the pinned-SHA asset fetchers.

Concurrency-critical: CI runs pytest partitions as parallel processes against
the shared ``$OPENRAL_CACHE_DIR``, and cloning straight into the final cache
path let one worker ``rmtree`` another's in-flight clone (openral PR #51's
select-and-test run). Each fetch therefore stages in a unique directory and
atomically renames into place, so the final path is never partial. ONE copy of
that dance lives here — it used to be duplicated verbatim across the enactic
and Anvil OpenArm fetchers, where the next fix (e.g. ``os.rename`` raising
``EXDEV`` across filesystems) would have landed in one and left the other racy.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import uuid
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["fetch_pinned_clone"]


def fetch_pinned_clone(
    repo_url: str,
    sha: str,
    repo_dir: Path,
    *,
    submodule: str | None = None,
    what: str = "pinned asset repo",
) -> None:
    """Clone ``repo_url`` at ``sha`` into ``repo_dir`` — staged, then atomic.

    Shallow (``--filter=blob:none``) clone into a per-call staging dir under
    the same cache root, checkout of the pinned SHA (never a branch — the sim
    contract must be reproducible), optional single-submodule init, then an
    atomic rename into ``repo_dir``. A concurrent fetch may have renamed its
    tree first; the loser's rename fails silently and its staging tree is
    removed — ``repo_dir`` either does not exist or is a complete tree, never
    partial. Any pre-existing (possibly half-finished) ``repo_dir`` is removed
    first; callers short-circuit on a validated cache before calling this.

    Args:
        repo_url: Git remote to clone.
        sha: Full commit SHA to check out.
        repo_dir: Final cache path (``<cache>/<sha>`` by convention); the
            staging dir is created alongside it in ``repo_dir.parent``.
        submodule: Optional submodule path to ``update --init`` (relative
            gitdir links survive the rename).
        what: Human label for error messages (e.g. ``"OpenArm v2"``).

    Raises:
        ROSConfigError: When ``git`` is missing from the PATH or the
            clone / checkout / submodule init fails.
    """
    cache = repo_dir.parent
    cache.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise ROSConfigError(
            f"{what} needs `git` on the PATH to clone {repo_url} (pin {sha[:10]}).  "
            f"Install git or pre-populate {repo_dir!s}."
        )

    # Clean any half-finished clone from a previous failed attempt so we
    # never inherit a partial tree.
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    staging = cache / f".staging-{sha}-{uuid.uuid4().hex}"
    try:
        try:
            subprocess.run(
                ["git", "clone", "--filter=blob:none", repo_url, str(staging)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(staging), "checkout", sha],
                check=True,
                capture_output=True,
            )
            if submodule:
                subprocess.run(
                    ["git", "-C", str(staging), "submodule", "update", "--init", submodule],
                    check=True,
                    capture_output=True,
                )
        except subprocess.CalledProcessError as exc:
            # `subprocess.CalledProcessError` carries stderr as bytes.
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            raise ROSConfigError(
                f"Failed to fetch {what} (pin {sha[:10]}) from {repo_url}: {stderr or exc}"
            ) from exc
        # A concurrent fetch may have renamed its tree first; keep the
        # winner's, which the caller's validation vouches for.
        with contextlib.suppress(OSError):
            os.rename(staging, repo_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
