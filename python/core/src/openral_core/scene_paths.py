"""Resolve a scene YAML's relative file references beside the scene file.

A scene that ships its own assets (a stage USD, prop USDs, declared static geometry)
names them relative to itself, so it runs from any working directory and from any
checkout, this one or another repo's. A relative reference with no file beside the
scene is left as written, for the existing lookups (the repo root, then the cwd).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import TypeVar, cast

__all__ = [
    "SCENE_PEERS_ENV",
    "SCENE_ROBOT_ENV",
    "SCENE_ROBOT_MAP_FRAME_ENV",
    "anchor_scene_paths",
    "format_scene_peers",
    "parse_scene_peers",
]

#: Set by ``openral deploy sim`` in each robot graph it starts for a scene with several
#: ``robots`` (openral-pro#99): every loader then reads that robot's own one-robot scene
#: (``geometry.raw_scene_for_robot``), so the graph needs no per-robot scene file.
SCENE_ROBOT_ENV = "OPENRAL_SCENE_ROBOT"
#: Beside it: that robot's ``RobotDescription.map_frame``, the frame its scene's nav goals and
#: place zones are re-expressed in (``"map"`` when unset).
SCENE_ROBOT_MAP_FRAME_ENV = "OPENRAL_SCENE_ROBOT_MAP_FRAME"
#: Beside them: the scene's other robots, a JSON list of ``[name, domain, manifest]``
#: (``format_scene_peers``). Each robot's graph runs a peer producer on it, whose grid its
#: safety kernel checks with no exemption (ADR-0137).
SCENE_PEERS_ENV = "OPENRAL_SCENE_PEERS"
_PEER_FIELDS = 3  # name, ROS domain, manifest path


def format_scene_peers(peers: list[tuple[str, int, str]]) -> str:
    """``(name, ROS domain, manifest path)`` per peer, as ``SCENE_PEERS_ENV`` carries them (JSON).

    Example:
        >>> format_scene_peers([("b", 34, "/r/amr/robot.yaml")])
        '[["b", 34, "/r/amr/robot.yaml"]]'
    """
    return json.dumps([[n, d, m] for n, d, m in peers])


def parse_scene_peers(value: str | None) -> list[tuple[str, int, str]]:
    """The inverse of ``format_scene_peers``; empty for ``None`` or ``""``.

    Raises:
        ValueError: The value is not a JSON list of ``[name, domain, manifest]``.

    Example:
        >>> parse_scene_peers(format_scene_peers([("b", 34, "/srv/openral;lab/robot.yaml")]))
        [('b', 34, '/srv/openral;lab/robot.yaml')]
        >>> parse_scene_peers("{}")
        Traceback (most recent call last):
        ValueError: OPENRAL_SCENE_PEERS is not a list of [name, domain, manifest]: '{}'
    """
    if not value:
        return []
    bad = ValueError(f"{SCENE_PEERS_ENV} is not a list of [name, domain, manifest]: {value!r}")
    try:
        raw = json.loads(value)
    except ValueError as exc:
        raise bad from exc
    entries = raw if isinstance(raw, list) else None
    if entries is None or not all(isinstance(e, list) and len(e) == _PEER_FIELDS for e in entries):
        raise bad
    out: list[tuple[str, int, str]] = []
    for name, domain, manifest in entries:
        if not (isinstance(name, str) and isinstance(domain, int) and isinstance(manifest, str)):
            raise bad
        out.append((name, domain, manifest))
    return out


_T = TypeVar("_T")


def _anchored(ref: object, scene_dir: Path) -> object:
    if not isinstance(ref, str):
        return ref
    rel = ref.removeprefix("file://")  # the Isaac backend's optional local-file scheme
    if not rel or ":" in rel or Path(rel).is_absolute():
        return ref  # not a relative local path: `isaac:`, a URL, absolute
    beside = scene_dir / rel
    return str(beside.resolve()) if beside.is_file() else ref


def anchor_scene_paths(raw: _T, scene_dir: Path) -> _T:
    """``raw`` with its relative file references made absolute where they sit beside the scene.

    ``raw`` is a scene YAML's mapping. With ``$OPENRAL_SCENE_ROBOT`` set and ``robots`` listed,
    the result is that robot's own one-robot scene (``SCENE_ROBOT_ENV``): every loader goes
    through here, so this is where a robot graph of a several-robot scene picks its robot.

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
        >>> anchored = anchor_scene_paths({"scene": {"assets_uri": "file://stage.usd"}}, d)
        >>> anchored == out
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
    robot = os.environ.get(SCENE_ROBOT_ENV)
    if robot and out.get("robots"):
        from openral_core.geometry import (  # noqa: PLC0415  # reason: numpy only when a robot is picked
            raw_scene_for_robot,
        )

        out = raw_scene_for_robot(
            out, robot, map_frame=os.environ.get(SCENE_ROBOT_MAP_FRAME_ENV) or "map"
        )
    return cast("_T", out)
