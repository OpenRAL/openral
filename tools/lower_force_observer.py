"""Print a robot's ``force_observer:`` manifest block, lowered from its MJCF (issue #373).

The block feeds the safety kernel's external joint-torque observer. Every inertial
is re-expressed in the manifest's link frames, and the lowering refuses when the
MJCF and the manifest disagree about a frame
(``openral_safety.dynamics_lowering.lower_force_observer_from_mjcf``).

Usage::

    uv run python tools/lower_force_observer.py robots/openarm/robot.yaml > block.yaml

Paste the output over the manifest's ``force_observer:`` block. A manifest whose
block drifts from this output fails ``tests/sim/safety/test_force_observer_lowering.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from openral_core import RobotDescription
from openral_safety.dynamics_lowering import lower_force_observer_from_mjcf


class _FlowTuples(yaml.SafeDumper):
    """Dump short numeric lists inline (``[x, y, z]``) as the manifests do."""


def _represent_list(dumper: yaml.SafeDumper, data: list[object]) -> yaml.Node:
    flow = all(isinstance(v, int | float) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_FlowTuples.add_representer(list, _represent_list)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the lowered block for one manifest."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path, help="robots/<id>/robot.yaml")
    args = ap.parse_args(argv)
    robot = RobotDescription.from_yaml(str(args.manifest))
    spec = lower_force_observer_from_mjcf(robot, manifest_dir=args.manifest.parent)
    block = {"force_observer": spec.model_dump(mode="json")}
    sys.stdout.write(yaml.dump(block, Dumper=_FlowTuples, sort_keys=False, width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
