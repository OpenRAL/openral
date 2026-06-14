#!/usr/bin/env python3
"""Write ~/.local/bin/openral so the CLI is reachable from any terminal.

Called by `just install-cli` (and transitively by `just quickstart`).
Idempotent — safe to re-run after moving the repo or upgrading Python.

The generated wrapper:
  1. Sources the ROS 2 distro overlay (/opt/ros/*/setup.bash) if present.
  2. Sources the colcon workspace overlay (<repo>/install/setup.bash) if built.
  3. exec-replaces itself with .venv/bin/openral, forwarding all args.
     So `openral` (no args) drops into the REPL and `openral <cmd>` is
     one-shot, matching the behaviour `just openral` used to provide.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_BIN = Path.home() / ".local" / "bin"
WRAPPER = LOCAL_BIN / "openral"

# Marker used to detect our own PATH injection; must not appear elsewhere in rc files.
_PATH_MARKER = "# openral-install-cli"
_PATH_SNIPPET = f'\n{_PATH_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n'

# The wrapper is plain bash — no Python templating inside it, so we use a
# raw template string and substitute the single token __REPO__ explicitly.
_WRAPPER_TEMPLATE = r"""#!/usr/bin/env bash
# OpenRAL CLI launcher — written by `just install-cli` / `just quickstart`.
# Re-run `just install-cli` if you move the repo.
set -euo pipefail

_OPENRAL_DIR="__REPO__"

# ROS 2 distro overlay (non-fatal: pure-Python commands work without it;
# ROS 2 topic/node/action commands need it).
_ROS_SETUP=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -1 || true)
if [ -z "$_ROS_SETUP" ]; then
    echo "WARNING: no /opt/ros/*/setup.bash found — ROS 2 features will be unavailable." >&2
    echo "         Run \`just bootstrap\` inside $_OPENRAL_DIR to install ROS 2." >&2
fi
[ -n "$_ROS_SETUP" ] && source "$_ROS_SETUP"

# Colcon workspace overlay (non-fatal for the same reason).
if [ -f "$_OPENRAL_DIR/install/setup.bash" ]; then
    source "$_OPENRAL_DIR/install/setup.bash"
else
    echo "WARNING: workspace overlay missing — run \`just ros2-build\` inside $_OPENRAL_DIR." >&2
fi

_VENV_BIN="$_OPENRAL_DIR/.venv/bin/openral"
if [ ! -x "$_VENV_BIN" ]; then
    echo "ERROR: $_VENV_BIN not found." >&2
    echo "       Run \`just sync --all-packages\` inside $_OPENRAL_DIR" >&2
    exit 1
fi

exec "$_VENV_BIN" "$@"
"""


def _write_wrapper() -> None:
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    content = _WRAPPER_TEMPLATE.replace("__REPO__", str(REPO))
    WRAPPER.write_text(content)
    WRAPPER.chmod(WRAPPER.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"==> wrote {WRAPPER}")


def _ensure_path() -> None:
    """Patch ~/.bashrc / ~/.zshrc to export ~/.local/bin when it's missing from $PATH."""
    path_dirs = os.environ.get("PATH", "").split(":")
    local_bin_str = str(LOCAL_BIN)

    if local_bin_str in path_dirs:
        print(f"==> {LOCAL_BIN} already on $PATH — no shell config changes needed")
        return

    patched: list[str] = []
    for rc_name in (".bashrc", ".zshrc"):
        rc = Path.home() / rc_name
        if not rc.exists():
            continue
        text = rc.read_text()
        # Skip if we already injected our snippet, or if the user already has .local/bin.
        if _PATH_MARKER in text or ".local/bin" in text:
            continue
        rc.write_text(text + _PATH_SNIPPET)
        patched.append(f"~/{rc_name}")

    if patched:
        files = " and ".join(patched)
        print(f"==> added ~/.local/bin to PATH in {files}")
        print('    To activate now:  export PATH="$HOME/.local/bin:$PATH"')
        print("    Or open a new terminal.")
    else:
        print(f"==> NOTE: {LOCAL_BIN} is not on the current $PATH.")
        print('    To activate now:  export PATH="$HOME/.local/bin:$PATH"')
        print("    Then add that line to ~/.bashrc (or ~/.zshrc) for future sessions.")


def main() -> None:
    """Write the ~/.local/bin/openral wrapper and ensure ~/.local/bin is on $PATH."""
    _write_wrapper()
    _ensure_path()
    print(
        "==> `openral` is now available from any terminal "
        "(new shell, or after running the export above)"
    )


if __name__ == "__main__":
    main()
