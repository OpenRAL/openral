"""Shared helpers for ``tests/hil/``."""

from __future__ import annotations


def _can_links_up(can_links: tuple[str, ...]) -> bool:
    """True if every link in *can_links* is up (per ``enumerate_can_interfaces``)."""
    from openral_cli.autodetect import enumerate_can_interfaces

    up = {i.name for i in enumerate_can_interfaces() if i.is_up}
    return set(can_links) <= up
