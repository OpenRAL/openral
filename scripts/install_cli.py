#!/usr/bin/env python3
"""Write ~/.local/bin/openral so the CLI is reachable from any terminal.

Called by `just install-cli` (and transitively by `just quickstart`).
Idempotent — safe to re-run after moving the repo or upgrading Python.

The generated wrapper:
  1. Resolves which checkout to drive (see below).
  2. Sources the ROS 2 distro overlay (/opt/ros/*/setup.bash) if present.
  3. Sources the colcon workspace overlay (<repo>/install/setup.bash) if built.
  4. exec-replaces itself with .venv/bin/openral, forwarding all args.
     So `openral` (no args) drops into the REPL and `openral <cmd>` is
     one-shot, matching the behaviour `just openral` used to provide.

Repo-root resolution is a *provenance* control, not a convenience. The wrapper
bakes in the checkout that generated it, so running `openral` from a second
checkout (a git worktree used for validation) used to silently execute the
first checkout's venv, colcon overlay and ``robots/`` manifests — a validation
run on a DGX Spark was attributed to the wrong branch with nothing in the log
to show it. The generated wrapper therefore:

  * honours ``OPENRAL_REPO_ROOT`` as an explicit override, validating that the
    named tree really has an executable ``.venv/bin/openral`` before exec'ing
    it (hard error otherwise — never a silent fall back to the baked path), and
    printing the root it settled on to stderr so every log records it;
  * warns (non-fatally) when the cwd sits inside a *different* OpenRAL checkout
    than the baked-in one, which is the shape of the Spark incident.

The single-checkout default is unchanged and stays silent.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Install target. Overridable via ``OPENRAL_CLI_BIN_DIR`` so tests can write a
# throwaway wrapper under ``tmp_path`` instead of clobbering the real
# ``~/.local/bin/openral`` a developer is actively using.
_BIN_DIR_ENV = "OPENRAL_CLI_BIN_DIR"
LOCAL_BIN = Path(os.environ.get(_BIN_DIR_ENV) or (Path.home() / ".local" / "bin"))
WRAPPER = LOCAL_BIN / "openral"

# Marker used to detect our own PATH injection; must not appear elsewhere in rc files.
_PATH_MARKER = "# openral-install-cli"
_PATH_SNIPPET = f'\n{_PATH_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n'

# The wrapper is plain bash — no Python templating inside it, so we use a
# raw template string and substitute the single token __REPO__ explicitly.
#
# Strict mode (`set -euo pipefail`) guards the wrapper's *own* logic, but the
# ROS 2 distro and colcon workspace overlays are ament-generated and are NOT
# nounset/errexit safe: e.g. ``/opt/ros/<distro>/setup.bash`` line 8 reads
# ``$AMENT_TRACE_SETUP_FILES`` (and others read ``$COLCON_TRACE``) with no
# default, so sourcing them under ``set -u`` aborts at
# "AMENT_TRACE_SETUP_FILES: unbound variable" before we ever ``exec`` the CLI —
# the wrapper exits and the user never reaches the REPL. ``_source_overlay``
# drops nounset+errexit just around the ``source`` and restores them after.
# Sourcing inside a function still mutates the global/exported environment
# (no ``local``), so PATH / AMENT_PREFIX_PATH / PYTHONPATH survive the ``exec``.
_WRAPPER_TEMPLATE = r"""#!/usr/bin/env bash
# OpenRAL CLI launcher — written by `just install-cli` / `just quickstart`.
# Re-run `just install-cli` if you move the repo.
#
# Which checkout does this run? In order:
#   1. $OPENRAL_REPO_ROOT, when set — validated, announced on stderr.
#   2. The baked-in checkout below (the one that generated this file).
# A cwd inside a DIFFERENT OpenRAL checkout is warned about, not overridden:
# a launcher must never quietly execute a tree the caller didn't mean.
set -euo pipefail

_OPENRAL_DIR="__REPO__"
_OPENRAL_BAKED_DIR="$_OPENRAL_DIR"

if [ -n "${OPENRAL_REPO_ROOT:-}" ]; then
    _OPENRAL_OVERRIDE="$OPENRAL_REPO_ROOT"
    # Normalise so the announced root is absolute and symlink-free.
    if [ -d "$_OPENRAL_OVERRIDE" ]; then
        _OPENRAL_OVERRIDE=$(cd "$_OPENRAL_OVERRIDE" && pwd -P)
    fi
    if [ ! -x "$_OPENRAL_OVERRIDE/.venv/bin/openral" ]; then
        echo "ERROR: OPENRAL_REPO_ROOT=$OPENRAL_REPO_ROOT has no executable .venv/bin/openral." >&2
        echo "       Run \`just sync --all-packages\` inside that checkout, or unset" >&2
        echo "       OPENRAL_REPO_ROOT to use the installed default ($_OPENRAL_BAKED_DIR)." >&2
        exit 1
    fi
    _OPENRAL_DIR="$_OPENRAL_OVERRIDE"
    # One line, so a captured log always names the tree that actually ran.
    echo "openral: repo root $_OPENRAL_DIR" \
         "(OPENRAL_REPO_ROOT override; installed default $_OPENRAL_BAKED_DIR)" >&2
else
    # Provenance guard: is the cwd inside a different OpenRAL checkout? If so
    # the baked-in tree still wins (unchanged behaviour) but we say so loudly,
    # because that mismatch silently misattributes validation runs.
    _OPENRAL_CWD_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [ -n "$_OPENRAL_CWD_ROOT" ] && [ -d "$_OPENRAL_CWD_ROOT" ]; then
        _OPENRAL_CWD_ROOT=$(cd "$_OPENRAL_CWD_ROOT" && pwd -P)
        if [ "$_OPENRAL_CWD_ROOT" != "$_OPENRAL_DIR" ] \
           && [ -x "$_OPENRAL_CWD_ROOT/.venv/bin/openral" ]; then
            echo "WARNING: cwd is inside the OpenRAL checkout $_OPENRAL_CWD_ROOT," >&2
            echo "         but this launcher is baked to $_OPENRAL_DIR — running the latter's" >&2
            echo "         venv, overlay and robots/ manifests. To run THIS checkout instead:" >&2
            echo "           export OPENRAL_REPO_ROOT=$_OPENRAL_CWD_ROOT" >&2
            echo "         (or re-run \`just install-cli\` from it to re-bake the default)." >&2
        fi
    fi
fi

# Source an ament-generated overlay that is not `set -u`/`set -e` safe.
# Disable nounset+errexit only around the `source`, then restore them.
_source_overlay() {
    set +u +e
    # shellcheck disable=SC1090
    source "$1"
    set -u -e
}

# ROS 2 distro overlay (non-fatal: pure-Python commands work without it;
# ROS 2 topic/node/action commands need it).
_ROS_SETUP=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -1 || true)
if [ -z "$_ROS_SETUP" ]; then
    echo "WARNING: no /opt/ros/*/setup.bash found — ROS 2 features will be unavailable." >&2
    echo "         Run \`just bootstrap\` inside $_OPENRAL_DIR to install ROS 2." >&2
fi
[ -n "$_ROS_SETUP" ] && _source_overlay "$_ROS_SETUP"

# Colcon workspace overlay (non-fatal for the same reason).
if [ -f "$_OPENRAL_DIR/install/setup.bash" ]; then
    _source_overlay "$_OPENRAL_DIR/install/setup.bash"
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


def render_wrapper(repo: Path) -> str:
    """Return the ``openral`` launcher bash with ``repo`` baked in as the repo dir.

    ``repo`` is the *default* checkout only: the rendered wrapper lets
    ``OPENRAL_REPO_ROOT`` override it at run time (see the module docstring).

    Args:
        repo: Absolute path to the OpenRAL checkout the wrapper should drive
            unless ``OPENRAL_REPO_ROOT`` says otherwise.

    Returns:
        The complete bash source of the wrapper, ready to write to disk.

    Example:
        >>> from pathlib import Path
        >>> script = render_wrapper(Path("/opt/openral"))
        >>> "set -euo pipefail" in script
        True
        >>> '_OPENRAL_DIR="/opt/openral"' in script
        True
        >>> "OPENRAL_REPO_ROOT" in script  # run-time override honoured
        True
        >>> "__REPO__" in script  # token fully substituted
        False
    """
    return _WRAPPER_TEMPLATE.replace("__REPO__", str(repo))


def _write_wrapper() -> None:
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    content = render_wrapper(REPO)
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
