"""The Galaxea A1 URDF and meshes, fetched on demand at a pinned commit.

``robots/galaxea_a1/robot.yaml`` was transcribed from Galaxea's
``A1_URDF_0607_0028.urdf``, which Galaxea publishes in
``userguide-galaxea/robot_descriptions`` (``a1_description/``). That repository
carries no license, so nothing is vendored: this module downloads the URDF and
its nine STL meshes at ``GALAXEA_DESCRIPTIONS_SHA``, each checked against its
SHA-256 (``_pinned_files``), into ``$OPENRAL_CACHE_DIR/galaxea_a1/<sha>/``.

Upstream names its meshes ``package://mobiman/urdf/A1/meshes/<file>`` (a
package of Galaxea's own workspace that is not published), so the URDF
returned is a derived copy beside the original with those refs rewritten to
``../meshes/<file>``, relative to itself.
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_hal._pinned_files import fetch_pinned_file

__all__ = ["GALAXEA_DESCRIPTIONS_SHA", "ensure_a1_urdf"]

#: ``userguide-galaxea/robot_descriptions`` ``galaxea/main`` on 2026-11-12.
GALAXEA_DESCRIPTIONS_SHA: str = "72008ea950a4c9258387d30821a7d8d8a2b588ca"
_RAW = "https://raw.githubusercontent.com/userguide-galaxea/robot_descriptions"
_SUBDIR = "a1_description"
_URDF = "urdf/A1_URDF_0607_0028.urdf"
_UPSTREAM_MESH_PREFIX = "package://mobiman/urdf/A1/meshes/"
# SHA-256 of every file, relative to ``_SUBDIR``, at the pinned commit.
_FILES: dict[str, str] = {
    _URDF: "9c7f3f391d3bd283c406a921565b4d504765cbc4c339c6c8075a5c1803a49c2c",
    "meshes/arm_seg1.STL": "b80652f37e0d2d4b59d5127cbd97b404ddf661cd421d2ea47c100cc991bedcae",
    "meshes/arm_seg2.STL": "98d224d819f1ecbbd19c979e4f18409bd16e277bd752c4bb3fd913182bee41d8",
    "meshes/arm_seg3.STL": "dbfd8fea24de4480195d62096b63c206c002ba122021dbdab71ccfe4ea1a56f0",
    "meshes/arm_seg4.STL": "df203b499f3c2eff565b2078a9b8b0888552c9453b97d7734f82b0689208a6bf",
    "meshes/arm_seg5.STL": "cf71a6682a3a702d92ff6278742549a9fe2620f6edc338c4d5b45621743d2ff6",
    "meshes/arm_seg6.STL": "c0073262d2cc5f950eb66ef4bec90662f30791c4be8ba6b88e0a6b5584f4153f",
    "meshes/base_link.STL": "be3d34b3244d745f527273f2005ca4ae9b08118c7778c4f3fae88a6e361203a0",
    "meshes/gripper1.STL": "e888cc5f0cd77a8e1abef7b436378a38a2d0ee7268cc739fb51f9913d4ce5a1d",
    "meshes/gripper2.STL": "393dfe556f193ff31694e765e01161c00e2a2bb563ed1f5ea2f28d4c00048a47",
}


def ensure_a1_urdf() -> Path:
    """Return the A1 URDF with relative mesh refs, fetching the pinned files if needed.

    Idempotent: verified cached files are reused without network access.

    Raises:
        ROSConfigError: A download fails or a file fails SHA-256 verification.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    root = base / "galaxea_a1" / GALAXEA_DESCRIPTIONS_SHA
    for rel, digest in _FILES.items():
        fetch_pinned_file(
            f"{_RAW}/{GALAXEA_DESCRIPTIONS_SHA}/{_SUBDIR}/{rel}",
            digest,
            root / rel,
            what=f"Galaxea A1 asset {rel} (pin {GALAXEA_DESCRIPTIONS_SHA[:10]})",
            user_agent="OpenRAL-galaxea-a1-assets",
        )
    upstream = root / _URDF
    derived = upstream.with_name("a1.urdf")
    xml = upstream.read_text(encoding="utf-8").replace(_UPSTREAM_MESH_PREFIX, "../meshes/")
    if not derived.is_file() or derived.read_text(encoding="utf-8") != xml:
        tmp = derived.with_name(f".{derived.name}.{os.getpid()}.tmp")
        tmp.write_text(xml, encoding="utf-8")
        os.replace(tmp, derived)
    return derived
