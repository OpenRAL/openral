"""The perception nodes' shared camera-id → topic resolution.

Four nodes take the same ``cameras`` / ``primary_camera`` / ``image_topic``
parameter trio, and until this helper landed each carried its own copy of the
resolution. These tests pin the behaviour the copies shared, including the two
edge cases that only show up through rclpy: an unset string array arrives as
``[""]``, and an unset ``primary_camera`` arrives as ``""``.
"""

from __future__ import annotations

from openral_perception_ros.camera_topics import resolve_camera_topics


def test_entries_resolve_in_declaration_order() -> None:
    """Insertion order matters: the first key is the primary camera."""
    cameras = resolve_camera_topics(
        ["wrist=/openral/cameras/wrist/image", "top=/openral/cameras/top/image"],
        primary_camera="ignored_when_entries_exist",
        image_topic="/ignored",
    )
    assert list(cameras) == ["wrist", "top"]
    assert cameras["top"] == "/openral/cameras/top/image"


def test_an_unset_camera_array_falls_back_to_the_single_topic() -> None:
    """rclpy hands an unset string array through as [""], not []."""
    cameras = resolve_camera_topics(
        [""], primary_camera="wrist", image_topic="/openral/cameras/wrist/image"
    )
    assert cameras == {"wrist": "/openral/cameras/wrist/image"}


def test_an_unset_primary_camera_becomes_default() -> None:
    """The fallback id matches every node's declared `primary_camera` default."""
    cameras = resolve_camera_topics([], primary_camera="", image_topic="/cam/image")
    assert cameras == {"default": "/cam/image"}


def test_malformed_entries_are_skipped_not_guessed() -> None:
    """An entry missing an id or a topic contributes nothing."""
    cameras = resolve_camera_topics(
        ["=/no/id", "no_topic=", "", "wrist=/openral/cameras/wrist/image"],
        primary_camera="wrist",
        image_topic="/unused",
    )
    assert cameras == {"wrist": "/openral/cameras/wrist/image"}


def test_a_repeated_id_keeps_the_last_topic() -> None:
    """Later entries win, matching plain dict assignment — no silent merge."""
    cameras = resolve_camera_topics(
        ["wrist=/first", "wrist=/second"], primary_camera="wrist", image_topic="/unused"
    )
    assert cameras == {"wrist": "/second"}
