"""``openral_hal._pinned_files.fetch_pinned_file`` — cache hit and fail-closed digest check.

Real files on disk and a real (unreachable) URL; no doubles. The verified-cache
path must not touch the network, and a digest mismatch must install nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_hal._pinned_files import fetch_pinned_file, sha256_file

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
