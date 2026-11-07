"""``openral_hal._pinned_files.fetch_pinned_file`` — cache hit and fail-closed digest check.

Real files on disk and a real (unreachable) URL; no doubles. The verified-cache
path must not touch the network, and a digest mismatch must install nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_hal._pinned_files import fetch_pinned_file, is_download_failure, sha256_file

# Port 9 (discard) on localhost: nothing answers, so a download attempt fails fast.
_DEAD_URL = "http://127.0.0.1:9/never-fetched.xml"


def test_verified_cache_is_reused_without_network(tmp_path: Path) -> None:
    dest = tmp_path / "scene.xml"
    dest.write_bytes(b"<mujoco/>")
    digest = hashlib.sha256(b"<mujoco/>").hexdigest()
    fetch_pinned_file(_DEAD_URL, digest, dest, what="test asset")  # no download attempted
    assert sha256_file(dest) == digest


def test_mismatched_cache_refetches_and_fails_closed(tmp_path: Path) -> None:
    dest = tmp_path / "scene.xml"
    dest.write_bytes(b"tampered")
    with pytest.raises(ROSConfigError, match="Could not download pinned test asset"):
        fetch_pinned_file(_DEAD_URL, "0" * 64, dest, what="test asset")
    # The stale file is left as-is (never "verified"); no temp file leaks.
    assert [p.name for p in tmp_path.iterdir()] == ["scene.xml"]


def test_unreachable_host_is_a_download_failure(tmp_path: Path) -> None:
    """A refused connection is classed as "could not download" (a skippable missing dependency)."""
    with pytest.raises(ROSConfigError) as caught:
        fetch_pinned_file(_DEAD_URL, "0" * 64, tmp_path / "x.xml", what="test asset")
    assert is_download_failure(caught.value)


def test_cache_dir_error_is_typed_and_not_a_download_failure(tmp_path: Path) -> None:
    """A cache directory that cannot be created raises ROSConfigError, never a raw OSError.

    It is a local configuration error, so it must not be mistaken for an offline host.
    """
    blocker = tmp_path / "not_a_dir"
    blocker.write_bytes(b"")
    with pytest.raises(ROSConfigError, match="Cannot create the cache directory") as caught:
        fetch_pinned_file(_DEAD_URL, "0" * 64, blocker / "sub" / "x.xml", what="test asset")
    assert not is_download_failure(caught.value)


def test_sha256_mismatch_is_not_a_download_failure() -> None:
    """Integrity failures carry no network cause, so callers never skip them."""
    assert not is_download_failure(ROSConfigError("x failed SHA-256 verification"))


def test_install_failure_is_typed_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """A verified download that cannot be moved into place raises ROSConfigError.

    The source is a real local file (``file://``), so this runs offline; the
    destination is an existing non-empty directory, which ``os.replace`` refuses.
    """
    payload = b"<mujoco/>"
    source = tmp_path / "source.xml"
    source.write_bytes(payload)
    dest = tmp_path / "cache" / "scene.xml"
    dest.mkdir(parents=True)
    (dest / "occupied").write_bytes(b"")
    with pytest.raises(ROSConfigError, match="Cannot install verified test asset") as caught:
        fetch_pinned_file(
            source.as_uri(), hashlib.sha256(payload).hexdigest(), dest, what="test asset"
        )
    assert not is_download_failure(caught.value)
    assert sorted(p.name for p in dest.parent.iterdir()) == ["scene.xml"]


def test_cleanup_failure_preserves_the_typed_error(tmp_path: Path) -> None:
    """When the partial file cannot be removed, the caller still gets ROSConfigError.

    A directory squatting on the per-process temp path makes both the write and the
    cleanup ``unlink`` fail (root-safe, no mocks); the unlink failure is logged and
    must not replace the typed error. It is a local error, not a download failure.
    """
    import os

    payload = b"<mujoco/>"
    source = tmp_path / "source.xml"
    source.write_bytes(payload)
    dest = tmp_path / "scene.xml"
    squatter = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    squatter.mkdir()
    with pytest.raises(ROSConfigError, match="test asset") as caught:
        fetch_pinned_file(
            source.as_uri(), hashlib.sha256(payload).hexdigest(), dest, what="test asset"
        )
    assert isinstance(caught.value.__cause__, IsADirectoryError)
    assert not is_download_failure(caught.value)
    assert squatter.is_dir()  # cleanup failed and was tolerated
    assert not dest.exists()
