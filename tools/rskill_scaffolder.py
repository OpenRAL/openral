"""tools/rskill_scaffolder.py — Standalone CLI wrapper for ``ral skill new``.

Scaffolds a new local rSkill package from ``rskills/template/`` and
rewrites the manifest placeholders for ``name`` / ``license`` /
``embodiment_tags``. Identical behavior to ``ral skill new``; this
wrapper exists so power users can scaffold without installing the
``openral-cli`` distribution.

Usage
-----
::

    uv run python tools/rskill_scaffolder.py pi05-pick-cube \\
        --owner your-org --license apache-2.0 \\
        --embodiment-tag franka_panda \\
        --out-dir rskills/pi05-pick-cube
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

# Make the in-tree workspace packages importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("cli", "core", "skill", "observability"):
    sys.path.insert(0, str(_REPO_ROOT / "python" / _pkg / "src"))

from openral_cli._rskill_scaffolder import scaffold_rskill  # noqa: E402
from openral_core.schemas import EmbodimentTag, RSkillLicensePosture  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rskill_scaffolder",
        description="Scaffold a new local rSkill package from rskills/template/.",
    )
    parser.add_argument(
        "rskill_id",
        metavar="ID",
        help="Local skill id, convention <policy>-<task> e.g. pi05-pick-cube.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to rskills/<ID>/ under the repo root.",
    )
    parser.add_argument(
        "--owner",
        default="your-org",
        help="HF Hub owner segment for the manifest 'name' field.",
    )
    parser.add_argument(
        "--license",
        dest="license_",
        default="apache-2.0",
        choices=[v.value for v in RSkillLicensePosture],
        help="License posture (default: apache-2.0).",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="franka_panda",
        choices=list(get_args(EmbodimentTag)),
        help="Canonical embodiment tag (default: franka_panda).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination directory instead of refusing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parse_args(argv)
    out_dir = args.out_dir or (_REPO_ROOT / "skills" / args.rskill_id)
    try:
        result = scaffold_rskill(
            args.rskill_id,
            out_dir=out_dir,
            owner=args.owner,
            license_=RSkillLicensePosture(args.license_),
            embodiment_tag=args.embodiment_tag,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"scaffolded: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
