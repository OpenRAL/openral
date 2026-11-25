"""Resolve a scene YAML's relative file references beside the scene file.

A scene that ships its own assets (a stage USD, prop USDs, declared static geometry)
names them relative to itself, so it runs from any working directory and from any
checkout, this one or another repo's. A relative reference with no file beside the
scene is left as written, for the existing lookups (the repo root, then the cwd).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from openral_core.exceptions import ROSConfigError

__all__ = [
    "SCENE_PEERS_ENV",
    "SCENE_ROBOT_ENV",
    "ScenePeer",
    "anchor_scene_paths",
    "format_scene_peers",
    "parse_scene_peers",
    "require_peer_domains",
]

#: This robot's name in a world it shares with others, set by ``openral deploy fleet``
#: (OpenRAL Pro) on each robot's graph, whose scene is that robot's own one-robot slice. The
#: Isaac sidecar join and the launch's world-pose gate read it.
SCENE_ROBOT_ENV = "OPENRAL_SCENE_ROBOT"
#: Beside it: the scene's other robots, a JSON list of ``{"name", "robot_id", "domain"}``
#: (``format_scene_peers``). Each robot's graph runs a peer producer on it, whose grid its
#: safety kernel checks with no exemption (ADR-0137).
SCENE_PEERS_ENV = "OPENRAL_SCENE_PEERS"


#: The largest ROS 2 domain id on Linux (``ScenePeer.domain``'s bound too).
_MAX_DOMAIN = 232


class ScenePeer(BaseModel):
    """One of a scene's other robots, as ``SCENE_PEERS_ENV`` names it.

    The manifest is looked up from ``robot_id`` where the graph runs, never carried as a path.

    Example:
        >>> ScenePeer(name="b", robot_id="openarm", domain=34).domain
        34
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9]{0,23}$")
    robot_id: str = Field(min_length=1)
    domain: int = Field(ge=0, le=_MAX_DOMAIN)


_PEERS = TypeAdapter(list[ScenePeer])


def format_scene_peers(peers: list[ScenePeer]) -> str:
    """The peers as ``SCENE_PEERS_ENV`` carries them (a JSON list of objects).

    Example:
        >>> format_scene_peers([ScenePeer(name="b", robot_id="amr", domain=34)])
        '[{"name":"b","robot_id":"amr","domain":34}]'
    """
    return _PEERS.dump_json(peers).decode()


def parse_scene_peers(value: str | None) -> list[ScenePeer]:
    """The inverse of ``format_scene_peers``; empty for ``None`` or ``""``.

    Raises:
        ValueError: The value is not a JSON list of valid ``ScenePeer`` objects.

    Example:
        >>> parse_scene_peers(format_scene_peers([ScenePeer(name="b", robot_id="amr", domain=34)]))
        [ScenePeer(name='b', robot_id='amr', domain=34)]
        >>> parse_scene_peers("{}")
        Traceback (most recent call last):
        ValueError: OPENRAL_SCENE_PEERS is not a list of {name, robot_id, domain}: '{}'
    """
    if not value:
        return []
    try:
        return _PEERS.validate_json(value)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise ValueError(
            f"{SCENE_PEERS_ENV} is not a list of {{name, robot_id, domain}}: {value!r}"
        ) from exc


def require_peer_domains(peers: list[ScenePeer], own_domain: str | None) -> None:
    """A robot with peers must export its own ROS domain, distinct from every peer's.

    A hashed default, or one peer's domain, would put two robots on one DDS graph.

    Raises:
        ROSConfigError: Peers are listed and ``own_domain`` is unset, empty, not an integer
            in 0-232 or a peer's, or two peers share a domain.

    Example:
        >>> require_peer_domains([ScenePeer(name="b", robot_id="amr", domain=34)], "30")
        >>> require_peer_domains(
        ...     [ScenePeer(name="b", robot_id="amr", domain=34)], "34"
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        openral_core.exceptions.ROSConfigError: ROS_DOMAIN_ID=34 is shared by ...
    """
    if not peers:
        return
    if not own_domain:
        raise ROSConfigError(
            f"{SCENE_PEERS_ENV} lists peers, so this robot needs its own exported "
            "ROS_DOMAIN_ID (a hashed default would not be distinct from theirs)."
        )
    try:
        own = int(own_domain.strip())
    except ValueError as exc:
        raise ROSConfigError(f"ROS_DOMAIN_ID={own_domain!r} is not an integer.") from exc
    if not 0 <= own <= _MAX_DOMAIN:
        raise ROSConfigError(f"ROS_DOMAIN_ID={own} is outside 0-{_MAX_DOMAIN}.")
    owners = {own: "this robot"}
    names: set[str] = set()
    for peer in peers:
        name = repr(peer.name)
        if name in names:
            raise ROSConfigError(f"{SCENE_PEERS_ENV} lists peer {name} twice.")
        names.add(name)
        if (other := owners.setdefault(peer.domain, name)) != name:
            raise ROSConfigError(
                f"ROS_DOMAIN_ID={peer.domain} is shared by {other} and {name}: "
                "export a distinct ROS_DOMAIN_ID per robot"
            )


_T = TypeVar("_T")


#: The ``runtime`` fields naming a file beside the scene (the CLI's ``_scene_path``).
_RUNTIME_FILES = (
    "static_geometry_file",
    "object_detector_onnx",
    "object_detector_manifest",
    "reward_monitor_manifest",
    "scene_vlm_manifest",
)


def _anchored(ref: object, scene_dir: Path, there: Callable[[Path], bool] = Path.is_file) -> object:
    if not isinstance(ref, str):
        return ref
    rel = ref.removeprefix("file://")  # the Isaac backend's optional local-file scheme
    if not rel or ":" in rel or Path(rel).is_absolute():
        return ref  # not a relative local path: `isaac:`, a URL, absolute
    beside = scene_dir / rel
    return str(beside.resolve()) if there(beside) else ref


def _anchor_hal_and_driver_paths(out: dict[str, object], scene_dir: Path) -> None:
    """``hal.defaults.calibration_dir`` and ``drivers[].args`` ``*_path``, in place."""
    hal = out.get("hal")
    defaults = hal.get("defaults") if isinstance(hal, dict) else None
    if isinstance(defaults, dict) and "calibration_dir" in defaults:
        defaults["calibration_dir"] = _anchored(defaults["calibration_dir"], scene_dir, Path.is_dir)
    drivers = out.get("drivers")
    for driver in drivers if isinstance(drivers, list) else []:
        args = driver.get("args") if isinstance(driver, dict) else None
        if not isinstance(args, dict):
            continue
        for key in [k for k in args if isinstance(k, str) and k.endswith("_path")]:
            args[key] = _anchored(args[key], scene_dir, Path.exists)


def anchor_scene_paths(raw: _T, scene_dir: Path) -> _T:
    """``raw`` with its relative file references made absolute where they sit beside the scene.

    ``raw`` is a scene YAML's mapping.

    The references: ``scene.assets_uri``, ``scene.backend_options.objects[].usd``,
    ``runtime.static_geometry_file``, ``runtime.object_detector_onnx``,
    ``runtime.object_detector_manifest``, ``runtime.reward_monitor_manifest``,
    ``runtime.scene_vlm_manifest`` and ``runtime.vision_attachment.segmenter_manifest`` (each
    a file); ``hal.defaults.calibration_dir`` (a directory); and every ``drivers[].args`` key
    ending in ``_path`` (a file or a directory). So a scene re-written elsewhere (OpenRAL
    Pro's per-robot fleet slices) still points at them. ``raw`` is not modified; anything
    that is not a mapping is returned as is.

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
        for key in _RUNTIME_FILES:
            if key in runtime:
                runtime[key] = _anchored(runtime[key], scene_dir)
        vision = runtime.get("vision_attachment")
        if isinstance(vision, dict) and "segmenter_manifest" in vision:
            vision["segmenter_manifest"] = _anchored(vision["segmenter_manifest"], scene_dir)
    _anchor_hal_and_driver_paths(out, scene_dir)
    return cast("_T", out)
