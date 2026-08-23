"""Built-in lookup indices used by the assembler.

Two small registries:

- :func:`canonical_robot_path` — given a ``bh_robot_type`` produced by
  ``openral_cli.autodetect`` (USB VID/PID match, SocketCAN interface-name
  match, or DDS topic-prefix inference), return the path to the canonical
  ``robots/<name>/robot.yaml``.
  Used by the assembler so that **standard rigs** (SO-100, ALOHA, Unitree
  G1, …) get their canonical ``RobotDescription`` directly via
  ``RobotDescription.from_yaml(...)``, with detected sensors and compute
  spliced on top — the assembler never re-synthesises a known robot.

- :func:`signature_for_realsense` / :func:`signature_for_v4l2` /
  :func:`signature_for_usb_uvc` — convenience helpers that build the
  matching :class:`openral_sensors.SensorSignature` for each probe
  output kind.  Centralised here so probes don't import the catalog and
  the assembler doesn't grow ad-hoc helpers.
"""

from __future__ import annotations

from pathlib import Path

from openral_sensors import SensorSignature

__all__ = [
    "canonical_robot_path",
    "signature_for_realsense",
    "signature_for_usb_uvc",
    "signature_for_v4l2",
]


# The workspace root is found by walking up from this module, so the index
# works whether the package is consumed from a checkout or from an installed
# wheel sitting next to a `robots/` tree.
#
# This was previously a hard-coded ``parents[5]``, which overshot the root by
# one level (this file is 4 deep: ``python/detect/src/openral_detect/``, and
# ``openral_core.assets`` uses ``parents[4]`` at the identical depth). The
# index therefore never resolved from the package at all — every successful
# lookup was coming from the CWD fallback, so running ``openral detect``
# from anywhere but the repo root silently produced an empty scaffold for a
# robot the probes had already identified. An upward search is used instead
# of a fixed index so moving this module cannot silently break it again.
#
# Both markers are required: a bare `robots/` directory is a plausible name
# for unrelated user content, while `robots/` beside `python/` is this
# workspace's shape.
_WORKSPACE_MARKERS: tuple[str, ...] = ("robots", "python")
_MAX_ROOT_SEARCH_DEPTH = 6


def _discover_workspace_root() -> Path | None:
    """Nearest ancestor of this module that looks like the OpenRAL workspace."""
    here = Path(__file__).resolve()
    for ancestor in here.parents[:_MAX_ROOT_SEARCH_DEPTH]:
        if all((ancestor / marker).is_dir() for marker in _WORKSPACE_MARKERS):
            return ancestor
    return None


_PACKAGE_WORKSPACE_ROOT: Path | None = _discover_workspace_root()


def _repo_root_candidates() -> tuple[Path, ...]:
    """Roots to search for ``robots/<name>/robot.yaml``, best first.

    The CWD is read per call rather than captured at import, so a caller that
    changes directory (or a test that uses ``monkeypatch.chdir``) sees the
    directory it is actually in.
    """
    if _PACKAGE_WORKSPACE_ROOT is None:
        return (Path.cwd(),)
    return (_PACKAGE_WORKSPACE_ROOT, Path.cwd())


_OPENRAL_ROBOT_TYPE_TO_DIR: dict[str, str] = {
    # The SO-101 shares the SO-100's Feetech USB controller (identical VID/PID)
    # and the same `SO100FollowerHAL`, so USB auto-detection cannot tell them
    # apart. The SO-101 is the current revision, so a bare plug-in defaults to
    # it (see `openral_cli.autodetect._VID_PID_TABLE`); the older SO-100 is
    # selected explicitly with `openral detect --robot so100`.
    "so101": "so101_follower",
    "so100": "so100_follower",
    "aloha": "aloha_bimanual",
    # The CAN probe reports the bare family slug `openarm` (the interface-name
    # prefix a udev rule pins); the committed manifest directory is `openarm`
    # and its `RobotDescription.name` is `openarm_v2`. Alias the versioned name
    # so `openral detect --robot openarm_v2` resolves the same manifest.
    "openarm_v2": "openarm",
    # Future entries land here as new HAL adapters publish a canonical
    # `robots/<name>/robot.yaml`:
    # "unitree_g1": "unitree_g1",
    # "lekiwi": "lekiwi",
    # "ros2_control": "",  # generic — no canonical yaml
}


def canonical_robot_path(bh_robot_type: str) -> Path | None:
    """Resolve a ``bh_robot_type`` slug to a committed ``robots/<name>/robot.yaml``.

    Resolution is two-step: the slug is first translated through
    :data:`_OPENRAL_ROBOT_TYPE_TO_DIR` (``"so100"`` → ``"so100_follower"``);
    if it is not a known alias the slug is tried **verbatim** as a
    ``robots/<slug>/`` directory name. The second step lets an explicit
    ``openral detect --robot <name>`` override target any committed robot by
    its canonical directory name (e.g. ``"so101_follower"``) without first
    teaching the VID/PID table about it.

    Args:
        bh_robot_type: Slug as produced by
            ``openral_cli.autodetect.match_known_devices`` (USB VID/PID),
            ``match_can_interfaces`` (SocketCAN interface name), or
            ``infer_robot_from_topics`` (DDS topic-prefix), or a canonical
            ``robots/<name>`` directory name passed via an operator override.

    Returns:
        Path to the canonical manifest if the slug resolves (via alias or
        directly) **and** the file exists on disk; ``None`` otherwise (e.g.
        for ``"unknown"``, for an unfamiliar slug, or when the workspace tree
        is absent at runtime).

    Example:
        >>> from openral_detect.registry import canonical_robot_path
        >>> p = canonical_robot_path("so100")
        >>> p is None or p.name == "robot.yaml"
        True
        >>> # Canonical directory name resolves directly (no alias needed):
        >>> q = canonical_robot_path("so101_follower")
        >>> q is None or q.parent.name == "so101_follower"
        True
    """
    sub = _OPENRAL_ROBOT_TYPE_TO_DIR.get(bh_robot_type, bh_robot_type)
    if not sub:
        return None
    for root in _repo_root_candidates():
        candidate = root / "robots" / sub / "robot.yaml"
        if candidate.is_file():
            return candidate
    return None


def signature_for_realsense(model_id: str) -> SensorSignature:
    """Build a catalog signature for a probed RealSense ``model_id``."""
    return SensorSignature(kind="realsense", value=model_id.upper())


def signature_for_v4l2(name: str) -> SensorSignature:
    """Build a catalog signature for a V4L2 product name."""
    return SensorSignature(kind="v4l2_name", value=name)


def signature_for_usb_uvc(vid: int, pid: int) -> SensorSignature:
    """Build a USB UVC signature in the canonical ``"0xVVVV:0xPPPP"`` form."""
    return SensorSignature(kind="usb_uvc", value=f"0x{vid:04x}:0x{pid:04x}")
