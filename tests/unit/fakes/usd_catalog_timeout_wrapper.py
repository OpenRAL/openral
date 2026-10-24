#!/usr/bin/env python3
"""Process-boundary fake with a child that outlives the configured timeout."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    child_code = (
        "import os, time; from pathlib import Path; "
        "Path(os.environ['OPENRAL_USD_READER_CHILD_STARTED']).touch(); "
        "time.sleep(2); "
        "Path(os.environ['OPENRAL_USD_READER_CHILD_SURVIVED']).touch()"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    child.wait()


if __name__ == "__main__":
    main()
