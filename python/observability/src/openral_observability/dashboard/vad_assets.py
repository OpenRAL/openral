"""Fetch the vendored voice-prompt (VAD) binary assets on first dashboard start.

The dashboard's mic button runs speech detection fully client-side via
``@ricky0123/vad-web`` (Silero VAD ONNX models over ``onnxruntime-web``/WASM —
see ``static/vendor/vad/NOTICE.md``). Two Silero ONNX models
(``silero_vad_v5.onnx``, ``silero_vad_legacy.onnx``) and the onnxruntime-web
WASM binary (``ort-wasm-simd-threaded.wasm``) total ~15 MB and were previously
committed to git. They are now fetched on demand into a local cache
(``$OPENRAL_CACHE_DIR/dashboard_assets/vad/``, matching the convention used by
``openral_hal._openarm_v2_assets`` and ``openral_rskill.engine_cache``) and
hard-linked (falling back to a copy) into ``static/vendor/vad/`` so
``StaticFiles`` keeps serving them from the same URL the page already expects.
The small JS glue files (``bundle.min.js``, ``ort.wasm.min.js``, the worklet)
stay committed to git — they are tiny and reviewable as text.

Every URL below is pinned to the exact npm-published file jsDelivr serves for
the version recorded in ``NOTICE.md`` (onnxruntime-web 1.22.0,
``@ricky0123/vad-web`` 0.0.29 — same source vad-web itself vendors its Silero
ONNX models from) and verified against a sha256 of the file that was
previously committed to git before this module existed, so the bytes served
are provably identical to what shipped before. No Hugging Face Hub fallback
is configured: a real, stable, versioned upstream URL exists for all three
files, so there is nothing for a fallback mirror to cover. If a future
version bump ever lacks a stable pinned URL, add an entry under
``https://huggingface.co/datasets/OpenRAL/dashboard-assets/resolve/main/<file>``
here — **that HF dataset repo does not exist yet and must be populated before
it is relied on.**

Failure mode: a network-less host or an upstream outage must never crash the
dashboard (CLAUDE.md §1.11 tolerates "unavailable dependency", never a fake).
:func:`ensure_vad_assets` is best-effort — each failure is a loud
``structlog`` warning, never a silent ``except: pass``, and the mic button
degrades client-side (``/api/config``'s ``voice_prompt_enabled`` flag; see
``dashboard.js``) rather than throwing at click time.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import structlog

__all__ = ["PINNED_VAD_ASSETS", "ensure_vad_assets", "vad_assets_available"]

_logger = structlog.get_logger(__name__)

_STATIC_VAD_DIR = Path(__file__).parent / "static" / "vendor" / "vad"

_DOWNLOAD_TIMEOUT_S = 30


class PinnedAsset(NamedTuple):
    """One vendored binary asset: where to fetch it and how to verify it."""

    url: str
    sha256: str
    size: int  # bytes; a cheap pre-hash sanity/fast-path check


# Pinned to onnxruntime-web 1.22.0 / @ricky0123/vad-web 0.0.29 (the versions
# recorded in static/vendor/vad/NOTICE.md). jsDelivr serves the exact
# npm-published dist file for a pinned version byte-for-byte; the sha256
# values were computed from the files this module replaces, before they were
# removed from git (see the PR that added this module).
PINNED_VAD_ASSETS: dict[str, PinnedAsset] = {
    "ort-wasm-simd-threaded.wasm": PinnedAsset(
        url="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.wasm",
        sha256="71aef04959c5c1b6de461b6538e2058e306610034a85aad2742d0c7fd4533fe4",
        size=11210254,
    ),
    "silero_vad_v5.onnx": PinnedAsset(
        url="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/silero_vad_v5.onnx",
        sha256="2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
        size=2327524,
    ),
    "silero_vad_legacy.onnx": PinnedAsset(
        url="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/silero_vad_legacy.onnx",
        sha256="a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",
        size=1807522,
    ),
}


def _cache_dir() -> Path:
    """Local cache root for the downloaded VAD assets.

    Honours ``$OPENRAL_CACHE_DIR`` (the same override
    ``openral_hal._openarm_v2_assets`` and ``openral_rskill.engine_cache``
    read); falls back to ``~/.cache/openral``.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "dashboard_assets" / "vad"


def sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of the file at *path*, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(name: str, asset: PinnedAsset, dest: Path) -> None:
    """Download *asset* to *dest*, verifying its sha256, or raise.

    Downloads to a ``.part`` sibling first and only renames into place once
    the digest matches, so a crash mid-download never leaves a corrupt file
    that a size/mtime check might mistake for a valid cache hit.
    """
    tmp = dest.with_name(dest.name + ".part")
    try:
        # Fixed https URL from the pinned constant table above, not user input.
        with (
            urllib.request.urlopen(asset.url, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        digest = sha256_of(tmp)
        if digest != asset.sha256:
            msg = (
                f"{name}: sha256 mismatch after downloading {asset.url} "
                f"(expected {asset.sha256}, got {digest})"
            )
            raise ValueError(msg)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _place(cached: Path, target: Path) -> None:
    """Make *cached* available at *target* (hardlink, falling back to copy)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(cached, target)
    except OSError:
        # Cross-device (EXDEV) or a filesystem without hardlink support.
        shutil.copy2(cached, target)


def ensure_vad_assets() -> bool:
    """Fetch and place every pinned VAD asset, best-effort.

    For each entry in :data:`PINNED_VAD_ASSETS`: skip if already served from
    ``static/vendor/vad/`` (a real repo checkout, or a previous run already
    placed it); otherwise re-use a verified cache hit, or download + verify
    into the cache, then hard-link/copy it into the served static directory.

    Never raises. A download or verification failure for one asset is logged
    as a ``structlog`` warning and does not stop the others; failures leave
    that asset (and therefore the voice-prompt feature — see
    :func:`vad_assets_available`) unavailable rather than serving a
    corrupt/unverified file.

    Returns:
        ``True`` iff every pinned asset ended up available at its served
        path; ``False`` when at least one is missing.

    Example:
        >>> from openral_observability.dashboard.vad_assets import ensure_vad_assets
        >>> ensure_vad_assets() in (True, False)
        True
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name, asset in PINNED_VAD_ASSETS.items():
        target = _STATIC_VAD_DIR / name
        # Cheap pre-filter (size) before the sha256 pass, so an unrelated file
        # of a different size skips straight to a re-download without a wasted
        # hash of the whole thing.
        if (
            target.is_file()
            and target.stat().st_size == asset.size
            and sha256_of(target) == asset.sha256
        ):
            continue
        cached = cache / name
        try:
            if not (
                cached.is_file()
                and cached.stat().st_size == asset.size
                and sha256_of(cached) == asset.sha256
            ):
                _download_one(name, asset, cached)
            _place(cached, target)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            all_ok = False
            _logger.warning(
                "dashboard.vad_asset_unavailable",
                asset=name,
                url=asset.url,
                error=str(exc),
            )
    if not all_ok:
        _logger.warning(
            "dashboard.voice_prompt_disabled",
            reason=(
                "one or more voice-prompt (VAD) static assets could not be "
                "downloaded/verified; the dashboard mic button is disabled "
                "client-side (see /api/config voice_prompt_enabled)"
            ),
        )
    return all_ok


def vad_assets_available() -> bool:
    """Whether every pinned VAD asset is currently served from the static dir.

    Cheap (path existence only, no re-hash) — safe to call on every
    ``GET /api/config`` so the page can proactively grey out the mic button
    instead of the operator discovering the failure by clicking it.
    """
    return all((_STATIC_VAD_DIR / name).is_file() for name in PINNED_VAD_ASSETS)
