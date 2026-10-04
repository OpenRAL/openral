#!/usr/bin/env python
"""The real Isaac sidecar wire, minus Omniverse Kit — a process-boundary double.

Runs the real ``tools/isaac_sidecar.py`` argument parsing and ZMQ ``_serve``
loop, standing in only for the Kit-built scene (which needs an RTX GPU and the
~50 GB Isaac venv). Lets a unit test drive the openral-side factory through a
real subprocess spawn + identity-checked ping handshake, and read back the exact
argv the factory launched the sidecar with (written to the JSON file named by
``OPENRAL_TEST_ISAAC_ARGV_OUT``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from isaac_sidecar import _parse_args, _serve


class _NoKitScene:
    """Answers the sidecar contract with the robot spec's action width.

    ``step`` appends every action it receives (NaN kept, as JSON ``null``) to the
    file named by ``OPENRAL_TEST_ISAAC_ACTIONS_OUT`` when set — the exact vector
    Isaac would have applied.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def _obs(self) -> dict[str, Any]:
        return {"images": {}, "state": np.zeros(0, dtype=np.float32), "task": ""}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        out = os.environ.get("OPENRAL_TEST_ISAAC_ACTIONS_OUT")
        if out:
            row = [None if np.isnan(v) else float(v) for v in np.asarray(action).reshape(-1)]
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return {
            "observation": self._obs(),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
            "sim_time_ns": 0,
        }

    def sim_time_ns(self) -> int:
        return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    Path(os.environ["OPENRAL_TEST_ISAAC_ARGV_OUT"]).write_text(json.dumps(argv))
    action_dim = 0
    if args.robot_spec:
        with open(args.robot_spec, encoding="utf-8") as fh:
            action_dim = int(json.load(fh)["action"]["dim"])
    return _serve(
        _NoKitScene(action_dim),
        host=args.host,
        port=args.port,
        sim_app=None,
        task=args.task,
        layout=args.layout,
        environment=args.environment_usd or "",
        spawn=list(args.spawn_pose),
        robot=args.robot,
        objects=args.objects_json or "",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
