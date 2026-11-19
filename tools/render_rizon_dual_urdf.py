#!/usr/bin/env python3
"""Render ``robots/rizon_dual/rizon_dual.urdf`` from ``openral_hal_rizon``'s xacro (the only way).

Three steps over plain ``xacro`` output (``ros2_control:=false``, the committed mounting):

1. Mesh refs ``package://flexiv_description/`` → ``rd:rizon4_description:`` (the
   ``robot_descriptions`` Flexiv clone, byte-identical for Rizon4s), so the committed file
   holds no host path and needs no ROS package to resolve.
2. Each link's bare ``origin``/``mass``/``inertia`` wrapped in ``<inertial>``:
   flexiv_description jazzy-v1.9's ``get_inertias`` emits them unwrapped, which URDF
   parsers ignore.
3. The provenance header.

Needs ``xacro`` and the pinned ``flexiv_description`` on the ament path (source
``~/flexiv_ws/install/setup.bash`` from ``provision_flexiv.sh``). Standard library only.
After a re-render, re-lower the collision model (robots/rizon_dual/README.md).

Usage::

    python3 tools/render_rizon_dual_urdf.py           # rewrite the committed URDF
    python3 tools/render_rizon_dual_urdf.py --check   # exit 1 if it would change
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "packages" / "openral_hal_rizon"
XACRO = HERE / "urdf" / "rizon_dual.urdf.xacro"
MOUNTING = HERE / "config" / "rizon_dual_mounting.yaml"
OUT = REPO / "robots" / "rizon_dual" / "rizon_dual.urdf"

HEADER = """<?xml version="1.0" ?>
<!-- Rendered from packages/openral_hal_rizon/urdf/rizon_dual.urdf.xacro (ros2_control:=false)
     against flexiv_description jazzy-v1.9 (892f5a1) and the reference rig mounting
     (config/rizon_dual_mounting.yaml). Meshes resolve through robot_descriptions'
     flexiv_description clone (rd:rizon4_description), byte-identical for Rizon4s.
     Re-render after changing either; do not edit by hand. -->
"""

_BARE_INERTIAL = re.compile(r"\n(\s*)(<origin [^>]*/>)\n\s*(<mass [^>]*/>)\n\s*(<inertia [^>]*/>)")


def portable(xacro_xml: str) -> str:
    """Steps 1-3 over ``xacro`` output (everything before ``<robot`` is xacro's own prolog)."""
    body = xacro_xml[xacro_xml.index("<robot") :]
    body = body.replace("package://flexiv_description/", "rd:rizon4_description:")

    def wrap(m: re.Match[str]) -> str:
        ind = m[1]
        inner = "\n".join(f"{ind}  {tag}" for tag in (m[2], m[3], m[4]))
        return f"\n{ind}<inertial>\n{inner}\n{ind}</inertial>"

    return HEADER + _BARE_INERTIAL.sub(wrap, body)


def render() -> str:
    """Run ``xacro`` on the package xacro with the committed mounting."""
    out = subprocess.run(
        ["xacro", str(XACRO), "ros2_control:=false", f"mounting_file:={MOUNTING}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return portable(out.stdout)


def main() -> int:
    """Write (or ``--check``) the committed URDF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the render differs")
    args = parser.parse_args()
    urdf = render()
    if args.check:
        same = OUT.read_text(encoding="utf-8") == urdf
        print(f"{OUT.relative_to(REPO)}: {'up to date' if same else 'DIFFERS from the render'}")
        return 0 if same else 1
    OUT.write_text(urdf, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
