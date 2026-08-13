"""Shared camera-id → image-topic resolution for the perception nodes.

Every node in this package is deliberately **camera-agnostic**: none of them
bakes in a camera name. They all take the same pair of parameters — a
``cameras`` string array of ``"id=topic"`` entries, plus a single-camera
``primary_camera`` / ``image_topic`` fallback — and all resolved the pair with
their own private copy of the same eight lines. This is that resolution, once
(CLAUDE.md §1.13).

Pure and ROS-free: it takes the already-extracted parameter values, so it is
unit-testable without rclpy.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["resolve_camera_topics"]


def resolve_camera_topics(
    entries: Sequence[str],
    *,
    primary_camera: str,
    image_topic: str,
) -> dict[str, str]:
    """Resolve the ``cameras`` parameter into an ordered camera-id → topic map.

    Args:
        entries: ``"id=topic"`` strings from the ``cameras`` parameter. Empty
            strings and entries missing either half are skipped — rclpy's
            default for an empty string array is ``[""]``, so a node that was
            never given cameras arrives here with one blank entry.
        primary_camera: Id to file the single-camera fallback under. Blank falls
            back to ``"default"``, matching the nodes' declared default.
        image_topic: Topic for the single-camera fallback.

    Returns:
        Camera id → topic, in declaration order. The first key is the primary
        camera (dicts preserve insertion order), which is what the nodes use
        when a request leaves its ``camera`` field empty.

    Example:
        >>> resolve_camera_topics(
        ...     ["wrist=/openral/cameras/wrist/image", "top=/openral/cameras/top/image"],
        ...     primary_camera="wrist",
        ...     image_topic="/unused",
        ... )
        {'wrist': '/openral/cameras/wrist/image', 'top': '/openral/cameras/top/image'}
        >>> resolve_camera_topics([""], primary_camera="", image_topic="/cam/image")
        {'default': '/cam/image'}
    """
    cameras: dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        cid, _, topic = entry.partition("=")
        if cid and topic:
            cameras[cid] = topic
    if not cameras:
        cameras[primary_camera or "default"] = image_topic
    return cameras
