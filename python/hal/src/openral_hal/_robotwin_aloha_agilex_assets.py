"""The RoboTwin 2.0 aloha-agilex URDF and meshes, fetched on demand at a pinned revision.

``robots/aloha_agilex/robot.yaml`` describes the dual-arm AgileX platform that
RoboTwin 2.0 simulates. RoboTwin ships its embodiments in one archive,
``embodiments.zip`` in the ``TianxingChen/RoboTwin2.0`` Hugging Face dataset
(MIT). This module reads only the aloha-agilex URDF
(``embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf``) and the meshes
it references out of that archive at ``ROBOTWIN_ASSETS_REVISION`` (HTTP range
reads, ~98 MB of the 220 MB archive), into
``$OPENRAL_CACHE_DIR/robotwin_aloha_agilex/<rev12>/`` in the archive's own layout,
so the URDF's relative mesh refs resolve unchanged. Every file is checked
against its SHA-256; nothing is vendored.

Two steps, so a caller can tell an unreachable host from bad bytes:
``fetch_aloha_agilex_files`` downloads what is missing, and
``ensure_aloha_agilex_urdf`` verifies every digest and returns the URDF.
"""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_files import sha256_file

__all__ = [
    "ROBOTWIN_ASSETS_REVISION",
    "ensure_aloha_agilex_urdf",
    "fetch_aloha_agilex_files",
]

#: ``TianxingChen/RoboTwin2.0`` dataset commit (MIT).
ROBOTWIN_ASSETS_REVISION: str = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
_ARCHIVE = f"datasets/TianxingChen/RoboTwin2.0@{ROBOTWIN_ASSETS_REVISION}/embodiments.zip"
_MEMBER_DIR = "embodiments/aloha-agilex/urdf/"
_URDF = "arx5_description_isaac.urdf"
# SHA-256 of every file, relative to ``_MEMBER_DIR``, at the pinned revision.
_FILES: dict[str, str] = {
    "arx5_description_isaac.urdf": (
        "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
    ),
    "aloha_maniskill_sim/meshes/back_link7.STL": (
        "baea38e3edf22c80d6cbe813bfdc01439dc51406fa176c1f62732fba380267c5"
    ),
    "aloha_maniskill_sim/meshes/back_link7.dae": (
        "5ef2f06e003b0463868460504951fc864130779cc35e87bb54a8ad7c524bc513"
    ),
    "aloha_maniskill_sim/meshes/base_arm.STL": (
        "3d6524754df3c5b50ea4196c473b447cd7dc0c00afb9a31e7db3bb6d1cd52c60"
    ),
    "aloha_maniskill_sim/meshes/base_arm.dae": (
        "cfe3f44bf9860d72acac2abd21a50422d72efd4de6fac72af44e02b2ea6a52df"
    ),
    "aloha_maniskill_sim/meshes/box1_Link.STL": (
        "e48fbe760dc7c1d98c3e655ca5788dd09a1c1c6544b1ad3f0aeaaaa2c8121917"
    ),
    "aloha_maniskill_sim/meshes/box2_Link.dae": (
        "511af26229e2ec9c9c014c12b04bfda50253b800dbaa129aca69c2c2b81ecc74"
    ),
    "aloha_maniskill_sim/meshes/camera_base_link.STL": (
        "d4cf9f1b01727e55283b7ac1c311eca1e807908e2d2c8e9e43bf7f6936b6e847"
    ),
    "aloha_maniskill_sim/meshes/camera_link1.STL": (
        "e37b71296958e665006034f3432ac4f74b25aea8268dfaa9bfb2164d452ac015"
    ),
    "aloha_maniskill_sim/meshes/camera_link1.dae": (
        "0692aa9f29f5d3aa9c21b81bff671e22128ae9110f3af809196a0d1f9a1b2e02"
    ),
    "aloha_maniskill_sim/meshes/camera_link2.STL": (
        "e37b71296958e665006034f3432ac4f74b25aea8268dfaa9bfb2164d452ac015"
    ),
    "aloha_maniskill_sim/meshes/camera_link2.dae": (
        "0692aa9f29f5d3aa9c21b81bff671e22128ae9110f3af809196a0d1f9a1b2e02"
    ),
    "aloha_maniskill_sim/meshes/castor.dae": (
        "0a61bca6839f37ebebd89021474821abfd908d2b50867f221e3504db4d84a06a"
    ),
    "aloha_maniskill_sim/meshes/castor_joint.dae": (
        "901cad2da533c8fd0f3e3ce25a4d5b5d1c689d5f852640cd7dab041dc647d629"
    ),
    "aloha_maniskill_sim/meshes/d435.dae": (
        "552a84a19ffbce76329911860095f1b16e5f837cc4228d35d6481b178a721f8e"
    ),
    "aloha_maniskill_sim/meshes/link1.STL": (
        "079a59fd0a94237c5042d6c7542b79bdc4c8db396758fdb7860f7b464e3c25ed"
    ),
    "aloha_maniskill_sim/meshes/link1.dae": (
        "49480bb69e5daffdd9092b33d9af6c6b3e6b9a069419e3e76fb9947e3e1753f9"
    ),
    "aloha_maniskill_sim/meshes/link2.STL": (
        "87e1386ed2533e68743f50f448e8cbdc366c2b37f0df3ede1fa2de340666786c"
    ),
    "aloha_maniskill_sim/meshes/link2.dae": (
        "9bb90c7a6c6d9bfab562349bf647f833433d71f2804a7981eba775f31e78723d"
    ),
    "aloha_maniskill_sim/meshes/link3.STL": (
        "76ded14e5f919e7a77ae7501883d55fb349be5c3c16499f59c1de5b552cad842"
    ),
    "aloha_maniskill_sim/meshes/link3.dae": (
        "1eabf99eb30c6cc9bf7c61bf650102fd22d35ac2ce67f36e42d700e3e2442318"
    ),
    "aloha_maniskill_sim/meshes/link4.STL": (
        "d234f4b96822ecf8a2f62219031053f7950d685faa15b4d23f8115c1b434cbbd"
    ),
    "aloha_maniskill_sim/meshes/link4.dae": (
        "3b50cf11a5271fe34aca0c186e623137875fd02c764cb7238c20b898f175f150"
    ),
    "aloha_maniskill_sim/meshes/link5.STL": (
        "dc58ea92599295aaa980e090a034c68397e505b79b681a223cd98ba7f10bab03"
    ),
    "aloha_maniskill_sim/meshes/link5.dae": (
        "c8a96bc00fa9c67723d0a2626bfd015b00a872d7be1588b61f0ae002dd189b13"
    ),
    "aloha_maniskill_sim/meshes/link6.STL": (
        "ffb2de0357066957db00fd7599dd4a7fbc147c8ef5916a492aa9070fdbdf5740"
    ),
    "aloha_maniskill_sim/meshes/link6.dae": (
        "e0c0f316065921fdab8704e5301baaa2b8c50867846a19c489393bc4ab83cbfa"
    ),
    "aloha_maniskill_sim/meshes/link7.STL": (
        "ec3d40b39cccde0075ca3d16603f14f180fce41a46b446473d384def7a1b0be0"
    ),
    "aloha_maniskill_sim/meshes/link7.dae": (
        "cf87ef550d6ab5d71058e7da63ba6ac4c492a9385b3ca0eed37eee0fd0cab97e"
    ),
    "aloha_maniskill_sim/meshes/link8.STL": (
        "29f7a8379cad81797e82e4ba099455cc275ac6ac5bb9e264ce46719fd7d2bb60"
    ),
    "aloha_maniskill_sim/meshes/link8.dae": (
        "e58e8cbc674bb7465ec81a2db6c4b1d08ccf1d63352e245739aa6418d62b0fd7"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link.STL": (
        "bf816a05cd083357ebd4ccc554b0bedf4d1459c60359db425b813a797bfec216"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link.dae": (
        "53c27a6f52d019e02f93748e463448e0c5173d256a9a5772a1b7267047d15c08"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link_no_wheel.dae": (
        "d3a1ee3b34dbc418e888d68148d04fccda589e00d5100fbd3cbc2f17154e0a08"
    ),
    "aloha_maniskill_sim/meshes/tracer_wheel.dae": (
        "42784402bca05e9f406f4e08d8ea1cc7ff57db9fee0e0b0b557670d0ee1d10c1"
    ),
}


def _root() -> Path:
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "robotwin_aloha_agilex" / ROBOTWIN_ASSETS_REVISION[:12]


def fetch_aloha_agilex_files() -> Path:
    """Download every pinned file not already cached; return the cache root.

    Only presence is checked here (``ensure_aloha_agilex_urdf`` verifies), so a
    warm cache needs no network.

    Raises:
        ROSConfigError: ``huggingface_hub`` is missing, or the archive could not
            be read (``.__cause__`` carries the hub / network error).
    """
    root = _root()
    missing = [rel for rel in _FILES if not (root / rel).is_file()]
    if not missing:
        return root
    try:
        from huggingface_hub import HfFileSystem  # noqa: PLC0415  # reason: optional dep
    except ImportError as exc:
        raise ROSConfigError(f"robotwin:aloha_agilex needs huggingface_hub: {exc}") from exc
    (root / ".verified").unlink(missing_ok=True)
    try:
        with (
            HfFileSystem().open(_ARCHIVE, "rb", block_size=8 << 20) as remote,
            zipfile.ZipFile(remote) as archive,
        ):
            for rel in missing:
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with archive.open(_MEMBER_DIR + rel) as src, tmp.open("wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    os.replace(tmp, dest)
                finally:  # a failed read leaves no partial mesh behind
                    tmp.unlink(missing_ok=True)
    except Exception as exc:  # every hub / network / archive read failure is a fetch failure
        raise ROSConfigError(
            f"Failed to read {_MEMBER_DIR} from {_ARCHIVE}: {type(exc).__name__}: {exc}"
        ) from exc
    return root


def ensure_aloha_agilex_urdf() -> Path:
    """Return the aloha-agilex URDF, every file fetched and SHA-256-verified.

    A file that fails verification is deleted, so the next call fetches it again.
    A passed check leaves a ``.verified`` stamp, so a warm cache is not re-hashed
    (~430 MB) on every resolve.

    Raises:
        ROSConfigError: A fetch failed or a file failed SHA-256 verification.
    """
    root = fetch_aloha_agilex_files()
    stamp = root / ".verified"
    if stamp.is_file():
        return root / _URDF
    for rel, digest in _FILES.items():
        path = root / rel
        if sha256_file(path) != digest:
            path.unlink()
            raise ROSConfigError(
                f"robotwin:aloha_agilex {rel} failed SHA-256 verification "
                f"(pin {ROBOTWIN_ASSETS_REVISION[:10]}); removed it from the cache."
            )
    stamp.touch()
    return root / _URDF
