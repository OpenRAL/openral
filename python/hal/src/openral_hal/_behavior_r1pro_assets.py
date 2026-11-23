"""The BEHAVIOR-1K Galaxea R1 Pro URDF, fetched on demand and made loadable.

``robots/r1pro/robot.yaml`` describes the R1 Pro that the 2026 BEHAVIOR-1K
challenge simulates; its joint limits were transcribed from
``behavior-1k/omnigibson-robot-assets`` (a Hugging Face dataset, MIT) at
``BEHAVIOR_ASSETS_REVISION``, ``models/r1pro/urdf/r1pro.urdf``. This module
downloads that URDF and the visual meshes beside it at that exact revision and
writes one derived copy into ``$OPENRAL_CACHE_DIR/behavior_r1pro`` with:

* every mesh ``filename`` made absolute, so the derived file can live outside
  the Hugging Face snapshot; and
* every ``<collision>`` whose mesh is not in the dataset dropped. Upstream
  references ``meshes/collision/*-col-*.obj`` files it does not ship, so no
  loader (MuJoCo, yourdfpy) can open the file as published. The result has
  visual geometry only; collision lowering has no collision meshes to read.

Nothing is vendored: the dataset revision pins the content.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from openral_core.exceptions import ROSConfigError

__all__ = ["BEHAVIOR_ASSETS_REVISION", "ensure_r1pro_urdf"]

#: ``behavior-1k/omnigibson-robot-assets`` commit the r1pro manifest cites.
BEHAVIOR_ASSETS_REVISION: str = "c325bd3b6d1560d3145f3c9c0e885e79e47f180c"
_REPO_ID = "behavior-1k/omnigibson-robot-assets"
_URDF_REL = "models/r1pro/urdf/r1pro.urdf"
_MESH_GLOB = "models/r1pro/urdf/meshes/*"
_COLLISION = re.compile(r"<collision\b.*?</collision>", re.S)
_FILENAME = re.compile(r'filename="([^"]+)"')


def _derived_path() -> Path:
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "behavior_r1pro" / BEHAVIOR_ASSETS_REVISION[:12] / "r1pro.urdf"


def derive_urdf(urdf_xml: str, urdf_dir: Path) -> str:
    """``urdf_xml`` with absolute mesh paths and its unshipped collisions removed.

    Paths are joined, not ``resolve()``-d: a Hugging Face snapshot file is a
    symlink to an extension-less blob, and the loaders pick a mesh decoder by
    the extension.
    """

    def keep(collision: re.Match[str]) -> str:
        files = _FILENAME.findall(collision.group(0))
        return collision.group(0) if all((urdf_dir / f).is_file() for f in files) else ""

    xml = _COLLISION.sub(keep, urdf_xml)

    def absolute(m: re.Match[str]) -> str:
        return m.group(0) if m.group(1).startswith("/") else f'filename="{urdf_dir / m.group(1)}"'

    return _FILENAME.sub(absolute, xml)


def ensure_r1pro_urdf() -> Path:
    """Return the derived R1 Pro URDF, downloading the pinned dataset files if needed.

    Raises:
        ROSConfigError: ``huggingface_hub`` is missing, or the download failed
            (``.__cause__`` carries the hub error).
    """
    out = _derived_path()
    if out.is_file():
        return out
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415  # reason: optional dep
    except ImportError as exc:
        raise ROSConfigError(f"behavior:r1pro needs huggingface_hub: {exc}") from exc
    try:
        root = snapshot_download(
            _REPO_ID,
            repo_type="dataset",
            revision=BEHAVIOR_ASSETS_REVISION,
            allow_patterns=[_URDF_REL, _MESH_GLOB],
        )
    except Exception as exc:  # every hub/network failure is a fetch failure
        raise ROSConfigError(
            f"Failed to fetch {_REPO_ID}@{BEHAVIOR_ASSETS_REVISION[:10]} {_URDF_REL}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    urdf = Path(root) / _URDF_REL
    xml = derive_urdf(urdf.read_text(encoding="utf-8"), urdf.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".r1pro.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(xml)
    os.replace(tmp, out)
    return out
