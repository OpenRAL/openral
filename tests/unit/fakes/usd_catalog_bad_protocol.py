"""Process-boundary fake that writes an invalid USD reader response."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    Path(sys.argv[2]).write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    main()
