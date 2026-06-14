"""Shared HF Hub publishing helpers — token resolution, scope check, ignore patterns.

Lifted from :mod:`tools.rskill_publisher` so that both ``openral dataset push``
(ADR-0019 PR5) and the existing skill publisher share one canonical path
for token discovery, scope verification, and ignore-pattern filtering. Per
CLAUDE.md §1.13, this de-duplication preempts the next "add another HF
uploader" PR from copying the rSkill version verbatim.

The helpers are conservative by design:

* :func:`resolve_token` accepts an explicit token argument, then falls back
  to ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` environment variables. It
  raises :class:`openral_core.exceptions.ROSConfigError` (with an actionable
  hint) when nothing is found, rather than handing the API a ``None`` and
  watching it produce a generic 401.
* :func:`ensure_private` re-fetches repo metadata after creation and aborts
  if the API reports the repo as public. This catches the edge case where
  ``create_repo(private=True)`` silently fails to apply on a pre-existing
  public repo.
* :data:`IGNORE_PATTERNS` excludes the same secrets / build-artifact glob
  as ``rskill_publisher`` so dataset uploads can't accidentally publish
  ``.env`` files or ``__pycache__/`` directories.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from huggingface_hub import HfApi

__all__ = [
    "IGNORE_PATTERNS",
    "ensure_private",
    "resolve_token",
]

# Files / directories excluded from every HF Hub upload. Same set the
# rSkill publisher uses — keeps secrets out of public-by-accident repos.
IGNORE_PATTERNS: Final[list[str]] = [
    "*.pyc",
    "__pycache__",
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    ".DS_Store",
    "Thumbs.db",
]


def resolve_token(token_arg: str | None = None) -> str:
    """Return the HF token from the argument or the environment.

    Args:
        token_arg: Value passed via a ``--token`` CLI flag (may be None).
            When provided, takes precedence over the environment.

    Returns:
        The resolved token string.

    Raises:
        ROSConfigError: When no token is available (with a hint listing
            the env vars and the CLI flag).

    Example:
        >>> import os
        >>> os.environ["HF_TOKEN"] = "hf_test_token"
        >>> resolve_token()
        'hf_test_token'
        >>> resolve_token("hf_override")
        'hf_override'
        >>> del os.environ["HF_TOKEN"]
    """
    token = token_arg or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ROSConfigError(
            "no HF token found; set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) env var, "
            "or pass --token <hf_xxx>. Token must have 'repo.write' scope. "
            "Generate at https://huggingface.co/settings/tokens."
        )
    return token


def ensure_private(api: HfApi, repo_id: str, *, repo_type: str = "model") -> None:
    """Re-fetch repo metadata and raise if it is not private.

    Critical safety gate: even when ``create_repo`` was called with
    ``private=True``, an existing public repo at the same id will NOT
    be auto-flipped to private. This function verifies the live state
    on the Hub.

    Args:
        api: Authenticated :class:`huggingface_hub.HfApi` client.
        repo_id: The repository to verify (e.g. ``"openral/dataset-foo"``).
        repo_type: ``"model"`` (default — used by rSkills), ``"dataset"``
            (used by ADR-0019 dataset uploads), or ``"space"``.

    Raises:
        ROSConfigError: When the repo is not private or metadata cannot
            be fetched. The caller must abort the upload and let the user
            either delete the existing public repo or rename.
    """
    # HF Hub returns different concrete info types per repo_type, but all
    # carry the .private attribute. Use Any to side-step the union plumbing.
    info: Any
    try:
        if repo_type == "dataset":
            info = api.dataset_info(repo_id)
        elif repo_type == "space":
            info = api.space_info(repo_id)
        else:
            info = api.model_info(repo_id)
    except Exception as exc:  # HF SDK raises many specific types
        raise ROSConfigError(f"privacy check failed for {repo_type} '{repo_id}': {exc!s}") from exc

    if not info.private:
        raise ROSConfigError(
            f"ABORT: {repo_type} '{repo_id}' is public on the HF Hub. "
            "OpenRAL never publishes a public repo automatically. "
            "Delete the existing public repo manually before retrying."
        )
