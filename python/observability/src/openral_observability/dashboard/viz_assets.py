"""The 3D view the dashboard serves at ``/viz/``: a pinned Lichtblick web build.

Lichtblick is the MPL-2.0 fork of Foxglove Studio (ADR-0129 §5): it speaks the
``foxglove_bridge`` protocol, which the dashboard proxies at ``/ws``. The web build is
fetched on the first ``/viz/`` request (``ensure_viz_assets``), pinned by sha256 and
size, and extracted without its source maps (``ui_assets.ensure_bundle``).

MPL-2.0 §3.2: the build is served as distributed, and ``NOTICE.txt`` next to it says
where its source is.
"""

from __future__ import annotations

from pathlib import Path

from openral_observability.dashboard.ui_assets import PinnedAsset, ensure_bundle

__all__ = ["LICHTBLICK", "ensure_viz_assets"]

LICHTBLICK = PinnedAsset(
    url="https://github.com/Lichtblick-Suite/lichtblick/releases/download/v1.30.0/lichtblick-web.tar.gz",
    sha256="f73b6bd0b4d2857b7311eb780279494334690e203de43440b54bc75994fd16c6",
    size=40636655,
)
"""Lichtblick v1.30.0 (2026-10-22), ``lichtblick-web.tar.gz``."""

_NOTICE = (
    "Lichtblick v1.30.0 (https://github.com/Lichtblick-Suite/lichtblick), a fork of\n"
    "Foxglove Studio, is licensed under the Mozilla Public License 2.0\n"
    "(https://mozilla.org/MPL/2.0/). This is its unmodified web build; its source is at\n"
    "https://github.com/Lichtblick-Suite/lichtblick/tree/v1.30.0\n"
)


def ensure_viz_assets() -> Path | None:
    """The Lichtblick build directory (with ``NOTICE.txt``), fetching it once; ``None`` offline."""
    root = ensure_bundle("lichtblick", LICHTBLICK)
    if root is not None and not (root / "NOTICE.txt").is_file():
        (root / "NOTICE.txt").write_text(_NOTICE, encoding="utf-8")
    return root
