"""Built-in lookup indices used by the assembler.

Two small registries:

- :func:`canonical_robot_path` — given a ``bh_robot_type`` produced by
  ``openral_cli.autodetect`` (USB VID/PID match or DDS topic-prefix
  inference), return the path to the canonical ``robots/<name>/robot.yaml``.
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


# Resolved against the workspace root at import time so the index works
# whether the package is consumed from a checkout or from an installed wheel
# next to a `robots/` tree.
_REPO_ROOT_CANDIDATES: tuple[Path, ...] = (
    # Editable install from the workspace.
    Path(__file__).resolve().parents[5],
    # Fallback: the user's CWD.
    Path.cwd(),
)


_OPENRAL_ROBOT_TYPE_TO_DIR: dict[str, str] = {
    "so100": "so100_follower",
    "aloha": "aloha_bimanual",
    # Future entries land here as new HAL adapters publish a canonical
    # `robots/<name>/robot.yaml`:
    # "unitree_g1": "unitree_g1",
    # "lekiwi": "lekiwi",
    # "ros2_control": "",  # generic — no canonical yaml
}


def canonical_robot_path(bh_robot_type: str) -> Path | None:
    """Resolve a ``bh_robot_type`` slug to a committed ``robots/<name>/robot.yaml``.

    Args:
        bh_robot_type: Slug as produced by
            ``openral_cli.autodetect.match_known_devices`` (USB VID/PID)
            or ``infer_robot_from_topics`` (DDS topic-prefix).

    Returns:
        Path to the canonical manifest if both the slug is known **and**
        the file exists on disk; ``None`` otherwise (e.g. for ``"unknown"``,
        for an unfamiliar slug, or when the workspace tree is absent at
        runtime).

    Example:
        >>> from openral_detect.registry import canonical_robot_path
        >>> p = canonical_robot_path("so100")
        >>> p is None or p.name == "robot.yaml"
        True
    """
    sub = _OPENRAL_ROBOT_TYPE_TO_DIR.get(bh_robot_type)
    if sub is None:
        return None
    for root in _REPO_ROOT_CANDIDATES:
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
