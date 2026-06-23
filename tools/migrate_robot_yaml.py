#!/usr/bin/env python3
"""Migrate ``robot.yaml`` manifests from schema_version 0.1 → 0.2.

Schema 0.2 (ADR-0069) replaces the single ``compute:`` slot with three
deployment-tier slots: ``compute_edge``, ``compute_local``, ``compute_cloud``.

Usage::

    # Dry-run (prints what would change):
    python tools/migrate_robot_yaml.py robots/*/robot.yaml

    # In-place update:
    python tools/migrate_robot_yaml.py --write robots/*/robot.yaml

    # Single file to stdout:
    python tools/migrate_robot_yaml.py robots/so100_follower/robot.yaml

If a YAML has no ``compute:`` key (the common case — most manifests never had
it), the tool appends ``schema_version: '0.2'`` as a trailing line so the rest
of the file (comments, ordering, formatting) is untouched.

When ``compute:`` IS present (i.e. the file was written by ``openral detect``
between the ComputeSpec extraction and ADR-0069 implementation), ``ruamel.yaml``
is used for comment-preserving round-trip parsing, and the slot is moved into
``compute_edge`` or ``compute_local`` based on the ``gpu_probe.backend`` value.

The tier assignment logic mirrors :func:`openral_detect.assemble._enrich_compute`:

- ``onboard_compute.gpu_probe.backend == "jtop"``  →  ``compute_edge``
- any other backend (``nvml``, ``metal``, ``none``, …)  →  ``compute_local``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

_SCHEMA_MARKER = "schema_version: '0.2'"


def _migrate_text_append(text: str) -> tuple[str, bool]:
    """Fast path: no ``compute:`` key — just append ``schema_version`` if absent."""
    if "schema_version:" in text:
        return text, False
    new_text = text.rstrip("\n") + "\n" + _SCHEMA_MARKER + "\n"
    return new_text, True


def _migrate_with_ruamel(text: str) -> tuple[str, bool]:
    """Slow path: ``compute:`` key present — use ruamel.yaml for round-trip safety."""
    try:
        from ruamel.yaml import YAML  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ruamel.yaml is required for compute: migration: pip install ruamel.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    import io

    ry = YAML()
    ry.preserve_quotes = True
    data = ry.load(text)
    if not isinstance(data, dict):
        return text, False

    compute = data.pop("compute", _SENTINEL := object())
    if compute is _SENTINEL:
        # key disappeared between check and here — shouldn't happen
        return text, False

    changed = True
    if compute is not None:
        onboard = data.get("onboard_compute") or {}
        probe = onboard.get("gpu_probe") or {} if isinstance(onboard, dict) else {}
        backend = probe.get("backend", "") if isinstance(probe, dict) else ""
        slot = "compute_edge" if backend == "jtop" else "compute_local"
        if slot not in data:
            data[slot] = compute

    if data.get("schema_version") != "0.2":
        data["schema_version"] = "0.2"

    buf = io.StringIO()
    ry.dump(data, buf)
    return buf.getvalue(), changed


def migrate_text(text: str) -> tuple[str, bool]:
    """Return (new_text, changed).  Chooses fast vs slow path automatically."""
    if "compute:" not in text:
        return _migrate_text_append(text)
    return _migrate_with_ruamel(text)


def migrate_file(path: Path, *, write: bool, verbose: bool = True) -> bool:
    """Migrate *path* and optionally write back.

    Returns ``True`` if the file was (or would be) changed.
    """
    raw = path.read_text(encoding="utf-8")
    new_text, changed = migrate_text(raw)
    if not changed:
        if verbose:
            print(f"  OK    {path}  (already up to date)")
        return False

    if write:
        path.write_text(new_text, encoding="utf-8")
        if verbose:
            print(f"  WROTE {path}")
    else:
        if verbose:
            print(f"  WOULD {path}  (pass --write to apply)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path, help="robot.yaml files to migrate")
    parser.add_argument(
        "--write", action="store_true", help="Write changes back to disk (default: dry-run)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    changed_count = 0
    for p in args.files:
        if not p.exists():
            print(f"  ERROR {p}  (file not found)", file=sys.stderr)
            continue
        if migrate_file(p, write=args.write, verbose=not args.quiet):
            changed_count += 1

    mode = "written" if args.write else "would change (dry-run)"
    print(f"\n{changed_count}/{len(args.files)} file(s) {mode}.")
    if not args.write and changed_count:
        print("Re-run with --write to apply changes.")


if __name__ == "__main__":
    main()
