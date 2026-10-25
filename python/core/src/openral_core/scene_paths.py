"""Resolve a scene YAML's relative file references beside the scene file.

A scene that ships its own assets (a stage USD, prop USDs, declared static geometry)
names them relative to itself, so it runs from any working directory and from any
checkout, this one or another repo's. A relative reference with no file beside the
scene is left as written, for the existing lookups (the repo root, then the cwd).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TypeVar, cast

__all__ = ["anchor_scene_paths"]

_T = TypeVar("_T")


def _anchored(ref: object, scene_dir: Path) -> object:
    if not isinstance(ref, str) or not ref or ":" in ref or Path(ref).is_absolute():
        return ref  # not a plain relative path: `isaac:`, a URL, `file://`, absolute
    beside = scene_dir / ref
    return str(beside.resolve()) if beside.exists() else ref


def anchor_scene_paths(raw: _T, scene_dir: Path) -> _T:
    """``raw`` with its relative file references made absolute where they sit beside the scene.

    ``raw`` is a scene YAML's mapping.

    The references: ``scene.assets_uri``, ``scene.backend_options.objects[].usd``,
    ``runtime.static_geometry_file`` and ``runtime.vision_attachment.segmenter_manifest``.
    ``raw`` is not modified; anything that is not a mapping is returned as is.

    Example:
        >>> import tempfile
        >>> d = Path(tempfile.mkdtemp())
        >>> _ = (d / "stage.usd").write_text("")
        >>> out = anchor_scene_paths({"scene": {"assets_uri": "stage.usd"}}, d)
        >>> out["scene"]["assets_uri"] == str((d / "stage.usd").resolve())
        True
        >>> anchor_scene_paths({"scene": {"assets_uri": "isaac:Isaac/x.usd"}}, d)
        {'scene': {'assets_uri': 'isaac:Isaac/x.usd'}}
    """
    if not isinstance(raw, dict):
        return raw
    out: dict[str, object] = copy.deepcopy(raw)
    scene = out.get("scene")
    if isinstance(scene, dict):
        if "assets_uri" in scene:
            scene["assets_uri"] = _anchored(scene["assets_uri"], scene_dir)
        opts = scene.get("backend_options")
        objects = opts.get("objects") if isinstance(opts, dict) else None
        for obj in objects if isinstance(objects, list) else []:
            if isinstance(obj, dict) and "usd" in obj:
                obj["usd"] = _anchored(obj["usd"], scene_dir)
    runtime = out.get("runtime")
    if isinstance(runtime, dict):
        if "static_geometry_file" in runtime:
            runtime["static_geometry_file"] = _anchored(runtime["static_geometry_file"], scene_dir)
        vision = runtime.get("vision_attachment")
        if isinstance(vision, dict) and "segmenter_manifest" in vision:
            vision["segmenter_manifest"] = _anchored(vision["segmenter_manifest"], scene_dir)
    return cast("_T", out)
