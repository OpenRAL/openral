"""The single resolver for robot description assets (URDF / MJCF / SRDF).

One grammar replaces ``resolve_urdf_path``, ``resolve_mjcf_uri``, the
plain-path SRDF handling, and ``openral_safety.urdf_lowering._load_urdf_model``.

Grammar (``resolve_asset(ref, kind)``):

* ``rd:<module>`` — import ``robot_descriptions.<module>`` and read its
  ``URDF_PATH`` / ``MJCF_PATH`` / ``SRDF_PATH`` for the requested ``kind``.
  Cache-misses download on first use (with a structlog progress line).
* ``file:<relpath>`` — resolved against the manifest dir, then the repo root.
* ``gym_aloha:<scene>`` · ``openarm:<variant>`` · ``menagerie:<model>`` —
  robot-specific MJCF loaders (sim-only optional deps, lazy-imported).
* ``ros2://robot_description`` — dynamic-detection marker (URDF only); not a
  file, returned as ``None`` for the launch to subscribe at runtime.

``resolve_package_uri`` is the companion for one ROS ``package://<pkg>/<path>``
mesh reference inside a vendored URDF, for the description packages OpenRAL
fetches itself as pinned clones (``openral_hal.ros_package_overlay``).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal

import structlog

_log = structlog.get_logger(__name__)

__all__ = ["AssetFetchError", "AssetKind", "AssetRefError", "resolve_asset", "resolve_package_uri"]

AssetKind = Literal["urdf", "mjcf", "srdf"]

# This file lives at python/core/src/openral_core/assets.py; parents[4] is the
# repo root (the dir containing robots/, python/, docs/). Verified in tests.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROS2_DYNAMIC = "ros2://robot_description"
_RD_ATTR = {"urdf": "URDF_PATH", "mjcf": "MJCF_PATH", "srdf": "SRDF_PATH"}


class AssetRefError(ValueError):
    """A description-asset reference is malformed or cannot be resolved."""


class AssetFetchError(AssetRefError):
    """A well-formed reference whose pinned upstream could not be fetched on this host.

    The one resolution failure an offline host can legitimately skip on; every
    other ``AssetRefError`` is a broken reference.
    """


def resolve_asset(ref: str, kind: AssetKind, *, manifest_dir: Path | None = None) -> Path | None:
    """Resolve an asset reference to a concrete file path.

    Args:
        ref: The asset reference (e.g. ``rd:panda_description``,
            ``file:robot.srdf``, ``ros2://robot_description``).
        kind: Which asset the reference must yield (``urdf`` / ``mjcf`` /
            ``srdf``).
        manifest_dir: Directory the manifest was loaded from; ``file:``
            references resolve against it first.

    Returns:
        The resolved file path, or ``None`` only for the
        ``ros2://robot_description`` dynamic marker (URDF supplied at runtime
        over a topic, not a file).

    Raises:
        AssetRefError: For every other unresolvable or malformed reference.

    Example:
        >>> resolve_asset("ros2://robot_description", "urdf") is None
        True
    """
    if ref == _ROS2_DYNAMIC:
        if kind != "urdf":
            raise AssetRefError(f"{_ROS2_DYNAMIC!r} is only valid for urdf, not {kind}")
        return None
    if ref.startswith("rd:"):
        return _resolve_rd(ref[3:], kind)
    if ref.startswith("file:"):
        return _resolve_file(ref[len("file:") :], manifest_dir)
    if ref.startswith("gym_aloha:"):
        return _resolve_gym_aloha(ref[len("gym_aloha:") :], kind)
    if ref.startswith("openarm:"):
        return _resolve_openarm(ref[len("openarm:") :], kind)
    if ref.startswith("menagerie:"):
        return _resolve_menagerie(ref[len("menagerie:") :], kind)
    raise AssetRefError(
        f"unrecognized asset ref {ref!r}; expected one of: rd:<module>, "
        f"file:<relpath>, gym_aloha:<scene>, openarm:<variant>, "
        f"menagerie:<model>, {_ROS2_DYNAMIC}"
    )


def resolve_package_uri(uri: str) -> Path:
    """Resolve a ROS ``package://<pkg>/<relpath>`` mesh URI to a file on disk.

    Only for the description packages OpenRAL fetches itself as pinned clones
    (``openral_hal.ros_package_overlay.PUBLIC_ROS_PACKAGES`` — today
    ``openarm_description``); a sourced ROS workspace is not consulted, so the
    answer is the same on every host. Lazy import, as ``_resolve_openarm``: the
    fetchers stay off the import path for robots that never need them.

    Args:
        uri: The ``package://`` reference as the URDF spells it.

    Returns:
        The file inside the pinned clone.

    Raises:
        AssetFetchError: The pinned clone could not be fetched (the cause is the
            fetcher's ``ROSConfigError``).
        AssetRefError: A malformed URI, a package OpenRAL does not fetch, or a
            path that is not a file in the clone.

    Example:
        >>> try:
        ...     resolve_package_uri("file:///tmp/not_a_package.stl")
        ... except AssetRefError as exc:
        ...     print(type(exc).__name__)
        AssetRefError
    """
    from openral_core.exceptions import ROSConfigError

    prefix = "package://"
    if not uri.startswith(prefix):
        raise AssetRefError(f"not a package:// URI: {uri!r}")
    pkg, _, rel = uri[len(prefix) :].partition("/")
    if not pkg or not rel:
        raise AssetRefError(f"malformed package URI {uri!r}; expected package://<pkg>/<relpath>")
    try:
        from openral_hal.ros_package_overlay import PUBLIC_ROS_PACKAGES, fetch_public_package
    except ImportError as exc:
        raise AssetRefError(
            f"{uri!r}: resolving package:// refs needs openral_hal ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        root = fetch_public_package(pkg)
    except ROSConfigError as exc:
        raise AssetFetchError(f"{uri!r}: fetching package {pkg!r} failed: {exc}") from exc
    if root is None:
        raise AssetRefError(
            f"{uri!r}: package {pkg!r} is not one OpenRAL fetches "
            f"(known: {sorted(PUBLIC_ROS_PACKAGES)})"
        )
    path = root / rel
    if not path.is_file():
        raise AssetRefError(f"{uri!r}: {path!s} is not a file in the pinned {pkg!r} clone")
    return path


def _resolve_rd(module: str, kind: AssetKind) -> Path:
    """Resolve ``rd:<module>`` via the upstream ``robot_descriptions`` package."""
    _log.info("assets.resolve_rd", module=module, kind=kind, note="downloads on first use")
    try:
        mod = importlib.import_module(f"robot_descriptions.{module}")
    except ImportError as exc:
        raise AssetRefError(
            f"could not import robot_descriptions.{module}: {type(exc).__name__}: {exc}. "
            "Install the sim extras with: just sync --all-packages --group sim"
        ) from exc
    attr = _RD_ATTR[kind]
    if not hasattr(mod, attr):
        if kind == "urdf" and hasattr(mod, "XACRO_PATH"):
            raise AssetRefError(
                f"robot_descriptions.{module} ships only XACRO_PATH (no URDF_PATH). "
                "OpenRAL never expands xacro at runtime — vendor a pre-expanded URDF "
                "with `openral robot vendor-urdf <id>` and point assets.urdf.ref at "
                "file:<id>.urdf."
            )
        raise AssetRefError(f"robot_descriptions.{module} has no {attr}")
    path = Path(str(getattr(mod, attr)))
    if not path.is_file():
        raise AssetRefError(
            f"robot_descriptions.{module}.{attr} is {path!s} but that file does not exist"
        )
    return path


def _resolve_file(relpath: str, manifest_dir: Path | None) -> Path:
    """Resolve ``file:<relpath>`` against the manifest dir, then the repo root."""
    candidates = []
    if manifest_dir is not None:
        candidates.append(manifest_dir / relpath)
    candidates.append(_REPO_ROOT / relpath)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise AssetRefError(f"file:{relpath} not found; tried: {tried}")


def _resolve_gym_aloha(scene: str, kind: AssetKind) -> Path:
    """Resolve ``gym_aloha:<scene>`` to its packaged MJCF (mjcf only)."""
    if kind != "mjcf":
        raise AssetRefError(f"gym_aloha:{scene} is mjcf-only, not {kind}")
    try:
        import gym_aloha  # reason: optional sim-only dep
    except ImportError as exc:
        raise AssetRefError(
            "gym_aloha is not installed. Install the sim extras with: "
            "just sync --all-packages --group sim"
        ) from exc
    path = Path(gym_aloha.__file__).parent / "assets" / f"{scene}.xml"
    if not path.is_file():
        raise AssetRefError(
            f"gym_aloha is installed but the scene MJCF is missing at {path!s}. "
            "Re-install gym-aloha or fix the scene id."
        )
    return path


def _resolve_openarm(variant: str, kind: AssetKind) -> Path:
    """Resolve ``openarm:<variant>`` to a vendored OpenArm MJCF (mjcf only).

    Variants (lazy imports keep the fetchers off the path for robots
    that don't need them):

    * ``bimanual`` — the Enactic OpenArm v2 bimanual MJCF, out-pinned
      past the ``robot_descriptions`` commit by
      ``openral_hal._openarm_v2_assets.ensure_openarm_v2_mjcf``.
    * ``anvil_v2_bimanual`` — the Anvil OpenARM 2.0 bimanual MJCF (v2 plus
      Anvil's J1/J6 range deltas and the wrist support bracket), fetched by
      ``openral_hal._anvil_openarm_v2_assets.ensure_anvil_openarm_v2_mjcf``.
    """
    if kind != "mjcf":
        raise AssetRefError(f"openarm:{variant} is mjcf-only, not {kind}")
    if variant == "bimanual":
        from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

        return Path(ensure_openarm_v2_mjcf())
    if variant == "anvil_v2_bimanual":
        from openral_hal._anvil_openarm_v2_assets import ensure_anvil_openarm_v2_mjcf

        return Path(ensure_anvil_openarm_v2_mjcf())
    raise AssetRefError(
        f"unknown openarm variant {variant!r}; expected 'bimanual' or 'anvil_v2_bimanual'"
    )


def _resolve_menagerie(model: str, kind: AssetKind) -> Path:
    """Resolve ``menagerie:<model>`` — not yet wired (YAGNI)."""
    raise AssetRefError(f"menagerie:{model} not yet wired — widowx is sim-only via SimplerEnv")
