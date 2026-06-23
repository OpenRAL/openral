"""Lifecycle entrypoint import test for the generic scene-attached HAL package."""

from __future__ import annotations


def test_lifecycle_entrypoint_imports() -> None:
    from openral_hal_scene_attached.lifecycle_node import main

    assert callable(main)
