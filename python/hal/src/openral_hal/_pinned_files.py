"""Shared single-file fetcher for pinned, SHA-256-verified upstream assets.

The pinned-git-SHA + per-file SHA-256 download used to live privately in
``_g1_walking``; the Unitree H2 description (``_unitree_h2_assets``) needs the
same fail-closed fetch, so ONE copy lives here. A raw file URL at a pinned
commit is fetched into a per-process temporary file next to its destination,
hashed, and atomically renamed into place only when the digest matches — the
cache path is either absent or a verified file, never partial or unverified.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["fetch_pinned_file", "is_download_failure", "sha256_file"]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path`` (streamed, 1 MiB chunks).

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "empty"
        >>> _ = p.write_bytes(b"")
        >>> sha256_file(p)[:12]
        'e3b0c44298fc'
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned_file(
    url: str,
    expected_sha256: str,
    destination: Path,
    *,
    what: str,
    user_agent: str = "OpenRAL-pinned-assets",
) -> None:
    """Download ``url`` to ``destination`` unless a verified copy is already cached.

    A cached file whose digest matches is reused without network access. A
    fresh download lands in ``.<name>.<pid>.tmp`` beside the destination and is
    ``os.replace``-d into place only after its SHA-256 matches — concurrent
    fetchers race benignly (each rename installs identical verified bytes).

    Args:
        url: Raw-file URL at a pinned commit (never a branch).
        expected_sha256: Hex SHA-256 the downloaded bytes must have.
        destination: Final cache path; parent directories are created.
        what: Human label for error messages (e.g. ``"Unitree H2 asset 'x.stl'"``).
        user_agent: HTTP ``User-Agent`` header.

    Raises:
        ROSConfigError: The download fails, or the bytes fail SHA-256
            verification (fail-closed: nothing is installed).
    """
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ROSConfigError(f"Cannot create the cache directory for {what}: {exc}") from exc
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise ROSConfigError(f"Could not download pinned {what}: {exc}") from exc
    actual_sha256 = sha256_file(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ROSConfigError(
            f"{what} failed SHA-256 verification: expected {expected_sha256}, got {actual_sha256}."
        )
    try:
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ROSConfigError(f"Cannot install verified {what} into the cache: {exc}") from exc


def is_download_failure(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its ``__cause__`` chain) is a network fetch failure.

    Separates "the pinned file could not be downloaded" (an unavailable
    dependency: tests skip, the asset resolver reports ``AssetFetchError``) from
    "the bytes failed SHA-256 verification" or a local cache error, which are
    integrity / configuration failures that must never be skipped.

    Args:
        exc: An exception raised while fetching, possibly wrapped several times.

    Returns:
        ``True`` when a ``urllib.error.URLError`` (HTTP or connection error),
        ``TimeoutError`` or ``ConnectionError`` appears in the cause chain.

    Example:
        >>> import urllib.error
        >>> from openral_core.exceptions import ROSConfigError
        >>> try:
        ...     try:
        ...         raise urllib.error.URLError("offline")
        ...     except urllib.error.URLError as cause:
        ...         raise ROSConfigError("Could not download pinned x") from cause
        ... except ROSConfigError as err:
        ...     is_download_failure(err)
        True
        >>> is_download_failure(ROSConfigError("x failed SHA-256 verification"))
        False
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False
