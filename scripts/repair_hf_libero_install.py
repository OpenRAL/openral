#!/usr/bin/env python3
"""Repair a malformed ``hf-libero`` install so ``uv sync`` can uninstall it.

``hf-libero==0.1.3``'s PyPI sdist (built with ``uv_build``, same for future
releases on that backend) ships both ``hf_libero-<ver>.dist-info/`` (PEP 376
wheel metadata) and ``hf_libero-<ver>.egg-info`` as a plain FILE
byte-identical to ``dist-info/METADATA``, at the site-packages root. uv's
uninstall sees the ``.egg-info`` sibling, falls back to the distutils path,
and demands an ``installed-files.txt`` manifest the wheel never produced::

    error: Unable to uninstall `hf-libero==0.1.3`.
    distutils-installed distributions do not include the metadata
    required to uninstall safely.

Removes the spurious ``.egg-info`` file and strips its RECORD line so uv
treats the install as a plain wheel. Idempotent; no-op with exit 0 if
nothing is broken. Invoked by the ``just sync`` recipe after every
``uv sync``.

Usage::

    uv run python scripts/repair_hf_libero_install.py [--venv PATH]

Exits 1 only on unexpected I/O errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_site_packages(venv: Path) -> Path | None:
    """Return ``<venv>/lib/pythonX.Y/site-packages`` or ``None`` if absent."""
    lib = venv / "lib"
    if not lib.is_dir():
        return None
    for py_dir in sorted(lib.glob("python*")):
        sp = py_dir / "site-packages"
        if sp.is_dir():
            return sp
    return None


def repair(site_packages: Path) -> int:
    """Repair every ``hf_libero-*.dist-info`` install under ``site_packages``.

    Returns the number of installs repaired. Deletes the spurious
    ``hf_libero-<ver>.egg-info`` FILE (a real setuptools ``.egg-info`` is a
    directory and is left alone) and strips its ``RECORD`` line.
    """
    repaired = 0
    for dist_info in site_packages.glob("hf_libero-*.dist-info"):
        if not dist_info.is_dir():
            continue
        # Derive the egg-info sibling path from the dist-info name.
        # ``hf_libero-0.1.3.dist-info`` → ``hf_libero-0.1.3.egg-info``.
        egg_info = dist_info.with_name(dist_info.name.removesuffix(".dist-info") + ".egg-info")
        record_path = dist_info / "RECORD"

        cleaned_egg = False
        if egg_info.is_file():
            egg_info.unlink()
            cleaned_egg = True
        elif egg_info.is_dir():
            # Real setuptools .egg-info directory — leave it alone. The
            # bug we're fixing only manifests as a FILE shape.
            print(
                f"[repair_hf_libero_install] {egg_info} is a directory "
                f"(real setuptools metadata); leaving untouched.",
                file=sys.stderr,
            )

        cleaned_record = False
        if record_path.is_file():
            lines = record_path.read_text().splitlines()
            bogus_prefix = f"{egg_info.name},"
            kept = [ln for ln in lines if not ln.startswith(bogus_prefix)]
            if len(kept) != len(lines):
                # Preserve trailing newline if the original had one.
                trailing_nl = "\n" if record_path.read_text().endswith("\n") else ""
                record_path.write_text("\n".join(kept) + trailing_nl)
                cleaned_record = True

        if cleaned_egg or cleaned_record:
            repaired += 1
            print(
                f"[repair_hf_libero_install] repaired {dist_info.name}: "
                f"removed_egg_info_file={cleaned_egg} "
                f"stripped_record_line={cleaned_record}"
            )
    return repaired


def main() -> int:
    """CLI entry: parse ``--venv`` and run ``repair`` against its site-packages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv"),
        help="Path to the uv-managed virtual environment (default: ./.venv).",
    )
    args = parser.parse_args()

    site_packages = find_site_packages(args.venv)
    if site_packages is None:
        # Not an error: `just sync` runs this unconditionally and .venv may
        # not exist yet.
        return 0

    try:
        repair(site_packages)
    except OSError as exc:
        print(f"[repair_hf_libero_install] unexpected I/O error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
