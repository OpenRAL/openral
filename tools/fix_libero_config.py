"""Detect and repair a stale ``~/.libero/config.yaml``.

LIBERO installs persist a YAML file at ``$LIBERO_CONFIG_PATH/config.yaml``
(default ``~/.libero/config.yaml``) that pins absolute filesystem paths
to the ``libero`` package's data dirs (assets / bddl_files / benchmark_root /
datasets / init_states). The paths are computed **once**, the first time
LIBERO is imported from any environment, and never refreshed. Switching
to a different virtual environment, repo clone, or workspace path leaves
the file pointing at a directory that no longer exists; the next
``openral sim libero`` invocation crashes deep inside
``lerobot.envs.libero.get_task_init_states`` with a confusing
``FileNotFoundError`` for ``<stale-path>/init_files/<task>.pruned_init``.

This script:

* locates the **currently active** ``libero`` package via ``import libero``,
* computes the canonical config that pairs with it,
* compares to ``$LIBERO_CONFIG_PATH/config.yaml`` (default ``~/.libero``),
* rewrites the file only when stale (or absent).

It is invoked from the ``_ensure-libero-config`` private recipe in the
``Justfile`` so every ``just sim-*-libero`` (and the ``sim-pi05-robocasa``
recipe, which loads LIBERO via ``lerobot`` transitively) sees a config
that matches the active venv.

Idempotent: re-runs are no-ops when the file already matches.

Documented in ``docs/reference/vla_compatibility.md`` under the LIBERO
section.

Usage::

    uv run python tools/fix_libero_config.py
    uv run python tools/fix_libero_config.py --verbose
    uv run python tools/fix_libero_config.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _expected_config(libero_pkg_dir: Path) -> dict[str, str]:
    """Return the canonical ``~/.libero/config.yaml`` payload for ``libero_pkg_dir``.

    Mirrors the dict layout LIBERO writes on first import — five keys,
    all absolute paths under the ``libero/libero/`` subtree (which is
    itself a sibling of a ``../datasets`` directory the upstream package
    places relative to the install).
    """
    base = libero_pkg_dir / "libero"
    return {
        "assets": str(base / "assets"),
        "bddl_files": str(base / "bddl_files"),
        "benchmark_root": str(base),
        "datasets": str(base / ".." / "datasets"),
        "init_states": str(base / "init_files"),
    }


def _parse_yaml_map(text: str) -> dict[str, str]:
    """Parse the flat ``key: value`` map LIBERO writes — no PyYAML dep needed."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        out[key.strip()] = value.strip()
    return out


def _render_yaml(payload: dict[str, str]) -> str:
    """Render the same flat ``key: value`` layout LIBERO writes."""
    return "".join(f"{key}: {value}\n" for key, value in payload.items())


def _locate_active_libero() -> Path:
    """Import ``libero`` and return its package directory.

    Raises ``RuntimeError`` with a clear message when LIBERO is not
    installed — the caller (the Justfile recipe) treats this as a no-op
    rather than failing the whole sim run, because the libero recipes
    install the libero group before invoking ``openral sim run`` anyway.
    """
    try:
        libero = importlib.import_module("libero")
    except ImportError as exc:
        raise RuntimeError(
            "libero is not importable in the active environment. "
            "Run `uv sync --group libero` first or invoke this script "
            "from inside the libero recipe."
        ) from exc

    pkg_file = libero.__file__
    if pkg_file is None:
        raise RuntimeError(
            "libero.__file__ is None — the package is a namespace package "
            "with no on-disk location; expected a regular package."
        )
    return Path(pkg_file).parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print the resolved paths even when no-op."
    )
    args = parser.parse_args()

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    config_file = config_dir / "config.yaml"

    try:
        libero_pkg_dir = _locate_active_libero()
    except RuntimeError as exc:
        # Don't fail the just recipe when libero isn't installed yet —
        # the recipe's `uv sync --group libero` will install it, and the
        # next invocation will repair the config.
        print(f"fix_libero_config: skipped ({exc})", file=sys.stderr)
        return 0

    expected = _expected_config(libero_pkg_dir)

    if config_file.exists():
        current = _parse_yaml_map(config_file.read_text())
        if current == expected:
            if args.verbose:
                print(
                    f"fix_libero_config: {config_file} already matches active libero at {libero_pkg_dir}"
                )
            return 0
        # Show the diff so contributors see the repair.
        stale_keys = [k for k, v in current.items() if expected.get(k) != v]
        print(
            f"fix_libero_config: rewriting {config_file} "
            f"(stale keys: {', '.join(stale_keys) or '<none>'}; "
            f"now pointing at {libero_pkg_dir})",
            file=sys.stderr,
        )
    else:
        print(
            f"fix_libero_config: creating {config_file} (active libero at {libero_pkg_dir})",
            file=sys.stderr,
        )

    if args.dry_run:
        print(_render_yaml(expected))
        return 0

    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(_render_yaml(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
