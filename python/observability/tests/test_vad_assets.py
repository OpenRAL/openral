"""Tests for :mod:`openral_observability.dashboard.vad_assets`.

The offline tests exercise the real download/verification code path via
``file://`` URLs against real temp files (CLAUDE.md §1.11 — no mocks; a
``file://`` URL is the process/network-boundary double the same section
allows). The one network-dependent test hits the actual pinned CDN URLs to
confirm the sha256 pins recorded in the module are still correct; it
``pytest.skip``s when the host is offline rather than failing CI.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest
from openral_observability.dashboard import vad_assets
from openral_observability.dashboard.vad_assets import PinnedAsset


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    """sha256_of reads a real file in chunks and matches a direct hashlib pass."""
    content = b"OpenRAL dashboard VAD asset checksum fixture" * 1000
    f = tmp_path / "fixture.bin"
    f.write_bytes(content)
    assert vad_assets.sha256_of(f) == _sha256_hex(content)


def test_pinned_vad_assets_table_is_well_formed() -> None:
    """The pinned constant table has the three expected assets, each sane."""
    assert set(vad_assets.PINNED_VAD_ASSETS) == {
        "ort-wasm-simd-threaded.wasm",
        "silero_vad_v5.onnx",
        "silero_vad_legacy.onnx",
    }
    for name, asset in vad_assets.PINNED_VAD_ASSETS.items():
        assert asset.url.startswith("https://"), name
        assert len(asset.sha256) == 64, name
        assert asset.sha256 == asset.sha256.lower(), name
        int(asset.sha256, 16)  # raises ValueError if not valid hex
        assert asset.size > 0, name


def test_download_one_verifies_and_writes(tmp_path: Path) -> None:
    """_download_one fetches via file:// (real urlopen path), verifying sha256."""
    content = b"pretend wasm bytes" * 500
    src = tmp_path / "source.bin"
    src.write_bytes(content)
    asset = PinnedAsset(url=src.as_uri(), sha256=_sha256_hex(content), size=len(content))
    dest = tmp_path / "dest.bin"

    vad_assets._download_one("fixture", asset, dest)

    assert dest.read_bytes() == content
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_one_rejects_sha256_mismatch(tmp_path: Path) -> None:
    """A downloaded file whose digest doesn't match the pin is rejected, not placed."""
    content = b"tampered or wrong-version bytes"
    src = tmp_path / "source.bin"
    src.write_bytes(content)
    asset = PinnedAsset(url=src.as_uri(), sha256="0" * 64, size=len(content))
    dest = tmp_path / "dest.bin"

    with pytest.raises(ValueError, match="sha256 mismatch"):
        vad_assets._download_one("fixture", asset, dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_ensure_vad_assets_end_to_end_and_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full ensure_vad_assets() flow against local file:// assets: download once,
    place into the served static dir, then reuse the cache on a second call
    without needing the source file anymore (proves the cache-hit skip path).
    """
    static_dir = tmp_path / "static_vendor_vad"
    cache_root = tmp_path / "cache_root"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    monkeypatch.setattr(vad_assets, "_STATIC_VAD_DIR", static_dir)
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(cache_root))

    content = b"fixture voice model bytes" * 2000
    src = src_dir / "one.bin"
    src.write_bytes(content)
    fixture_assets = {
        "one.bin": PinnedAsset(url=src.as_uri(), sha256=_sha256_hex(content), size=len(content)),
    }
    monkeypatch.setattr(vad_assets, "PINNED_VAD_ASSETS", fixture_assets)

    assert vad_assets.vad_assets_available() is False
    assert vad_assets.ensure_vad_assets() is True
    assert vad_assets.vad_assets_available() is True
    placed = static_dir / "one.bin"
    assert placed.read_bytes() == content

    # Second call must succeed purely from the cache: delete the "upstream"
    # source so a real re-download would fail loudly, then confirm nothing
    # broke — proves the skip-early / cache-hit branch does not touch the URL.
    src.unlink()
    placed.unlink()
    assert vad_assets.ensure_vad_assets() is True
    assert placed.read_bytes() == content


def test_ensure_vad_assets_partial_failure_is_isolated_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad asset returns False and is warned about; a good sibling still lands."""
    static_dir = tmp_path / "static_vendor_vad"
    cache_root = tmp_path / "cache_root"
    monkeypatch.setattr(vad_assets, "_STATIC_VAD_DIR", static_dir)
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(cache_root))

    good_content = b"good asset bytes" * 100
    good_src = tmp_path / "good.bin"
    good_src.write_bytes(good_content)
    fixture_assets = {
        "good.bin": PinnedAsset(
            url=good_src.as_uri(), sha256=_sha256_hex(good_content), size=len(good_content)
        ),
        "missing.bin": PinnedAsset(
            url=(tmp_path / "does-not-exist.bin").as_uri(), sha256="f" * 64, size=1
        ),
    }
    monkeypatch.setattr(vad_assets, "PINNED_VAD_ASSETS", fixture_assets)

    assert vad_assets.ensure_vad_assets() is False
    assert (static_dir / "good.bin").read_bytes() == good_content
    assert not (static_dir / "missing.bin").exists()
    assert vad_assets.vad_assets_available() is False  # one asset still missing


def test_vad_assets_available_reflects_static_dir_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vad_assets_available() is a pure presence check against the served dir."""
    static_dir = tmp_path / "static_vendor_vad"
    monkeypatch.setattr(vad_assets, "_STATIC_VAD_DIR", static_dir)
    fixture_assets = {
        "a.bin": PinnedAsset(url="file:///dev/null", sha256="0" * 64, size=0),
        "b.bin": PinnedAsset(url="file:///dev/null", sha256="0" * 64, size=0),
    }
    monkeypatch.setattr(vad_assets, "PINNED_VAD_ASSETS", fixture_assets)

    assert vad_assets.vad_assets_available() is False

    static_dir.mkdir()
    (static_dir / "a.bin").write_bytes(b"x")
    assert vad_assets.vad_assets_available() is False  # b.bin still missing

    (static_dir / "b.bin").write_bytes(b"y")
    assert vad_assets.vad_assets_available() is True


def _cdn_reachable(host: str = "cdn.jsdelivr.net", timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def test_pinned_urls_match_pinned_sha256_on_the_real_cdn(tmp_path: Path) -> None:
    """The real pinned jsDelivr URLs still serve exactly the pinned bytes.

    Network-dependent: skipped when the CDN is unreachable (offline dev host,
    sandboxed CI) rather than failing per CLAUDE.md §1.11.
    """
    if not _cdn_reachable():
        pytest.skip(reason="cdn.jsdelivr.net unreachable — offline host/CI, per CLAUDE.md §1.11")

    for name, asset in vad_assets.PINNED_VAD_ASSETS.items():
        dest = tmp_path / name
        vad_assets._download_one(name, asset, dest)
        assert vad_assets.sha256_of(dest) == asset.sha256
        assert dest.stat().st_size == asset.size
