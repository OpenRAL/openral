"""The robot UI the dashboard serves at ``/``: a pinned release, fetched on first start.

The UI is the React app in ``python/observability/ui/`` (ADR-0129 §2), released as
``openral-robot-ui-<ver>.tar.gz`` by ``.github/workflows/robot-ui.yml``. The dashboard
pins one release by URL, sha256 and size (``ROBOT_UI``), downloads it once into
``$OPENRAL_CACHE_DIR/dashboard_assets/robot-ui/<sha256>/`` and serves it from there.
``$OPENRAL_DASHBOARD_UI_DIR`` points at a local build instead (UI development).

With neither — no pin yet, or an offline first start — ``/`` serves the built-in
``static/fallback.html``: robot status and an E-STOP button. Fetching never raises:
a failure is a loud ``structlog`` warning and the fallback page.

``ensure_bundle`` is shared with ``viz_assets`` (the pinned Lichtblick build).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path
from typing import NamedTuple

import structlog

__all__ = [
    "ROBOT_UI",
    "UI_DIR_ENV",
    "PinnedAsset",
    "ensure_bundle",
    "ensure_ui_assets",
    "sha256_of",
    "ui_dir",
]

_logger = structlog.get_logger(__name__)

UI_DIR_ENV = "OPENRAL_DASHBOARD_UI_DIR"
_DOWNLOAD_TIMEOUT_S = 60


class PinnedAsset(NamedTuple):
    """A pinned download: where to fetch it and how to verify it."""

    url: str
    sha256: str
    size: int  # bytes; checked before the hash


ROBOT_UI: PinnedAsset | None = PinnedAsset(
    url="https://github.com/OpenRAL/openral/releases/download/robot-ui-9d86ae3/openral-robot-ui-robot-ui-9d86ae3.tar.gz",
    sha256="00a1f00db34f4bf455f4e468cbd405b23be0538cd3f79a9a46af3f0309ac9790",
    size=377400,
)
"""The robot UI release this dashboard serves (``robot-ui.yml``'s tarball), or ``None``
to serve the built-in fallback page. Bumped by ``robot-ui.yml`` (``tools/pin_robot_ui.py``)
whenever the UI changes on master: ``robot-ui-9d86ae3`` until the next pin."""


def _cache_root() -> Path:
    """``$OPENRAL_CACHE_DIR`` (default ``~/.cache/openral``) ``/dashboard_assets``."""
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "dashboard_assets"


def sha256_of(path: Path) -> str:
    """Hex sha256 of the file at ``path``, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(asset: PinnedAsset, dest: Path) -> None:
    """Download ``asset`` to ``dest``, or raise ``ValueError`` on a size/sha256 mismatch."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        # A fixed https URL from a pinned constant, not user input.
        with (
            urllib.request.urlopen(asset.url, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        size, digest = tmp.stat().st_size, sha256_of(tmp)
        if (size, digest) != (asset.size, asset.sha256):
            raise ValueError(
                f"{asset.url}: expected {asset.size} bytes sha256 {asset.sha256}, "
                f"got {size} bytes sha256 {digest}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _without_source_maps(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    """The ``data`` extraction filter (no links out, no devices, no ``..``), minus ``*.map``."""
    return None if member.name.endswith(".map") else tarfile.data_filter(member, path)


def ensure_bundle(name: str, asset: PinnedAsset) -> Path | None:
    """The extracted directory of a pinned web-build tarball, downloading it once.

    The tarball is verified (size, sha256) before it is opened, extracted with the
    ``data`` filter minus source maps, and moved into place only when complete, so a
    half-extracted bundle is never served. The bundle root is the directory holding
    ``index.html`` (the archive root, or its single top-level directory).

    Returns:
        The directory, or ``None`` when it could not be fetched or verified (logged).
    """
    dest = _cache_root() / name / asset.sha256
    if (dest / "index.html").is_file():
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tarball = dest.parent / f"{asset.sha256}.tar.gz"
        _download(asset, tarball)
        with tempfile.TemporaryDirectory(dir=dest.parent, ignore_cleanup_errors=True) as tmp:
            with tarfile.open(tarball) as archive:
                archive.extractall(tmp, filter=_without_source_maps)
            tarball.unlink()
            root = Path(tmp)
            subdirs = [p for p in root.iterdir() if p.is_dir()]
            if not (root / "index.html").is_file() and len(subdirs) == 1:
                root = subdirs[0]
            if not (root / "index.html").is_file():
                raise ValueError(f"{asset.url}: no index.html in the bundle")
            root.replace(dest)
    # http.client.HTTPException: a truncated download (IncompleteRead) is not an OSError.
    except (OSError, ValueError, urllib.error.URLError, tarfile.TarError, HTTPException) as exc:
        _logger.warning("dashboard.bundle_unavailable", bundle=name, url=asset.url, error=str(exc))
        return None
    return dest


def ui_dir() -> Path | None:
    """Where the robot UI is: ``$OPENRAL_DASHBOARD_UI_DIR``, else the fetched pin, else ``None``."""
    local = os.environ.get(UI_DIR_ENV)
    if local:
        return Path(local) if (Path(local) / "index.html").is_file() else None
    if ROBOT_UI is None:
        return None
    pinned = _cache_root() / "robot-ui" / ROBOT_UI.sha256
    return pinned if (pinned / "index.html").is_file() else None


def ensure_ui_assets() -> Path | None:
    """Fetch the pinned robot UI if it is not cached yet; the directory to serve, or ``None``."""
    if os.environ.get(UI_DIR_ENV) or ROBOT_UI is None:
        return ui_dir()
    return ensure_bundle("robot-ui", ROBOT_UI)
