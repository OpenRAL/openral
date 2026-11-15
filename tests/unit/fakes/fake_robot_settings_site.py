"""Network-boundary double for openral.com's ``GET /api/robot/settings`` (ADR-0129, ADR-0132).

``settings_site`` stands in for ``openral_cli.robot._open``: it records each request's
URL, ``Authorization`` and ``User-Agent`` and answers ``{"settings": ...}`` the way the
site does; ``apply_account_settings`` runs unmodified on top of it.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable


class SettingsResponse:
    """The site's answer, as the ``urlopen`` context manager the CLI reads it from."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> SettingsResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


Opener = Callable[[urllib.request.Request, float], SettingsResponse]


def settings_site(
    settings: dict[str, str], seen: list[tuple[str, str, str]] | None = None
) -> Opener:
    """An opener answering ``settings``; each request is appended to ``seen``."""

    def site(req: urllib.request.Request, timeout: float) -> SettingsResponse:
        if seen is not None:
            seen.append(
                (
                    req.full_url,
                    str(req.get_header("Authorization")),
                    str(req.get_header("User-agent")),
                )
            )
        return SettingsResponse(json.dumps({"settings": settings}).encode())

    return site
