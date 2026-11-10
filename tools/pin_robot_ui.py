"""Pin the robot UI the dashboard serves to one ``robot-ui.yml`` build (ADR-0129).

``robot-ui.yml`` runs this after publishing a ``robot-ui-<sha>`` pre-release, then
opens the pin bump as a PR. The pin stays a sha256 committed in code: the UI runs
with the local operator's full control, so it is never trusted on a release alone.

Usage: python tools/pin_robot_ui.py <version> <url> <sha256> <size> [ui_assets.py]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

UI_ASSETS = Path("python/observability/src/openral_observability/dashboard/ui_assets.py")

_PIN = re.compile(
    r'(ROBOT_UI: PinnedAsset \| None = PinnedAsset\(\n)    url="[^"]*",\n'
    r'    sha256="[0-9a-f]{64}",\n    size=\d+,\n(\)\n"""[^"]*?)``robot-ui-[^`]+``([^"]*""")',
)


def pin(text: str, version: str, url: str, sha256: str, size: int) -> str:
    """``ui_assets.py`` source ``text`` with ``ROBOT_UI`` set to the given build.

    Raises:
        ValueError: A malformed argument, or no pin block to rewrite.

    Example:
        >>> src = UI_ASSETS.read_text(encoding="utf-8")
        >>> out = pin(src, "robot-ui-abc1234", "https://x/y.tar.gz", "0" * 64, 7)
        >>> 'sha256="' + "0" * 64 + '"' in out and "``robot-ui-abc1234``" in out
        True
    """
    if not re.fullmatch(r"[0-9a-f]{64}", sha256) or size <= 0:
        raise ValueError("sha256 must be 64 hex characters and size positive")
    if not re.fullmatch(r"robot-ui-[A-Za-z0-9._-]+", version) or not url.startswith("https://"):
        raise ValueError("version must be a robot-ui-<id> tag and url an https URL")
    new, n = _PIN.subn(
        lambda m: (
            f'{m[1]}    url="{url}",\n    sha256="{sha256}",\n    size={size},\n'
            f"{m[2]}``{version}``{m[3]}"
        ),
        text,
    )
    if n != 1:
        raise ValueError("no ROBOT_UI pin block found")
    return new


def main(argv: list[str]) -> int:
    """Rewrite ``ui_assets.py`` in place; prints nothing."""
    version, url, sha256, size = argv[:4]
    path = Path(argv[4]) if len(argv) > 4 else UI_ASSETS
    path.write_text(pin(path.read_text(encoding="utf-8"), version, url, sha256, int(size)), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
