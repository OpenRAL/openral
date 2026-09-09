"""How fast was the arm actually moving when the kernel stopped it?

This settles the trade `PLAN.md`'s 25 → 15 mm lever turns on. A finer world grid
buys **8.66 mm** of static quantisation (21.65 → 12.99 mm half-diagonal) and pays
in map **age**: the wire measurement puts a 2.80 MB grid at +15 ms of staleness
at the median and +50–60 ms at p99, against 0.61 MB today. Age is millimetres
too — `speed × staleness` — so the two are directly comparable and the break-even
is a speed.

Every other term on this lever has been measured. This is the last one, and
measuring it is the whole point: on this lever the estimate was wrong by 32×
once already.

**What is measured.** The end-effector's linear speed at the instant of the stop,
from the `robot_joint_state` the round's own
`sim.estop_ground_truth_snapshot` recorded — real positions, real velocities,
pushed through the real Panda model's body Jacobian (`mj_jacBody`), not a
finite difference and not an estimate from joint limits.

`link7` is the body: the payload attaches there, so its linear velocity is the
carried object's, and the carried payload is 71 % of all stops.

**Why the stop instant is the right sample.** It is the moment the kernel reads
the map and decides. A faster or slower arm elsewhere in the trajectory does not
change what that decision cost.

Run::

    uv run python tools/stop_ee_speed.py outputs/validation-matrix/<round>...
    uv run python tools/stop_ee_speed.py --json outputs/validation-matrix/*

Needs MuJoCo and the robosuite Panda assets (the `lowering` group); reads
recorded artifacts only, no simulator rollout and no GPU.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The body the payload attaches to, so its linear velocity is the payload's.
EE_BODY = "link7"

#: Quantisation recovered by 25 -> 15 mm, in metres: the difference of the two
#: cells' half body-diagonals. What the staleness cost is weighed against.
QUANTISATION_GAIN_M = (0.025 - 0.015) * (3.0**0.5) / 2.0


#: Mobile-base DOFs, which the arm-only Panda model does not carry.
_BASE_JOINTS = ("base_x", "base_y", "base_yaw")


class StopSpeed(NamedTuple):
    """One stop's end-effector speed, with the class that decides how it counts."""

    round_id: str
    stop_class: str
    ee_speed_mps: float
    base_speed_mps: float = 0.0

    @property
    def world_speed_mps(self) -> float:
        """Worst-case world speed: arm and base contributions taken as aligned."""
        return self.ee_speed_mps + self.base_speed_mps

    @property
    def is_carry(self) -> bool:
        """Carry-phase stops are the ones staleness can cost anything at."""
        return self.stop_class == "attached_payload"


def _panda_model() -> tuple[Any, Any]:
    """The real Panda model, sourced exactly as the tight-geometry generator does."""
    import mujoco

    spec = importlib.util.spec_from_file_location(
        "generate_tight_geometry", REPO_ROOT / "tools" / "generate_tight_geometry.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = "could not load tools/generate_tight_geometry.py"
        raise RuntimeError(msg)
    gen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen
    spec.loader.exec_module(gen)
    xml, _ = gen._sources_for("panda_mobile")
    model = mujoco.MjModel.from_xml_path(str(xml))
    return model, mujoco.MjData(model)


def collect(round_dirs: list[Path]) -> list[StopSpeed]:
    """End-effector speed at every stop these rounds recorded a joint state for."""
    import mujoco
    import numpy as np

    model, data = _panda_model()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if body < 0:
        msg = f"body {EE_BODY!r} not in the Panda model"
        raise ValueError(msg)

    out: list[StopSpeed] = []
    for round_dir in round_dirs:
        for log in sorted(glob.glob(str(round_dir / "*" / "run_deploy.log"))):
            for line in Path(log).read_text(errors="replace").splitlines():
                if "sim.estop_ground_truth_snapshot" not in line:
                    continue
                match = re.search(r"\{.*\}", line)
                if match is None:
                    continue
                snapshot = json.loads(match.group(0))
                state = snapshot.get("robot_joint_state") or {}
                names = state.get("name") or []
                positions = state.get("position") or []
                velocities = state.get("velocity") or []
                if not velocities:
                    break
                mujoco.mj_resetData(model, data)
                # The snapshot names arm joints `panda_jointN`; the Panda model
                # names them `jointN`. Match on the index rather than on a
                # prefix, so neither spelling silently matches nothing — which
                # reads as a perfectly stationary arm and is the one failure
                # this measurement must not have.
                matched = 0
                base_speed = 0.0
                for name, pos, vel in zip(names, positions, velocities, strict=False):
                    if name in _BASE_JOINTS:
                        # The mobile base is not in the arm model. Its planar
                        # speed is reported separately rather than dropped: a
                        # driving base moves the end effector through the world
                        # even with the arm frozen.
                        if name in ("base_x", "base_y"):
                            base_speed += float(vel) ** 2
                        continue
                    index = re.search(r"joint(\d+)$", name)
                    if index is None:
                        continue
                    joint = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index.group(1)}"
                    )
                    if joint < 0:
                        continue
                    data.qpos[model.jnt_qposadr[joint]] = pos
                    data.qvel[model.jnt_dofadr[joint]] = vel
                    matched += 1
                if matched == 0:
                    msg = (
                        f"{log}: no arm joint in the snapshot matched the Panda model; "
                        "refusing to report 0 m/s, which would read as a stationary arm"
                    )
                    raise ValueError(msg)
                mujoco.mj_forward(model, data)
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, None, body)
                out.append(
                    StopSpeed(
                        round_id=round_dir.name,
                        stop_class=str(snapshot.get("stop_class", "?")),
                        ee_speed_mps=float(np.linalg.norm(jacp @ data.qvel)),
                        base_speed_mps=base_speed**0.5,
                    )
                )
                break
    return out


def summarise(stops: list[StopSpeed]) -> dict[str, Any]:
    """Per-class speeds, and the millimetres each costs at a given staleness."""
    carry = [s.world_speed_mps for s in stops if s.is_carry]
    other = [s.world_speed_mps for s in stops if not s.is_carry]
    summary: dict[str, Any] = {
        "stops": len(stops),
        "quantisation_gain_mm": QUANTISATION_GAIN_M * 1e3,
    }
    for label, group in (("carry_phase", carry), ("other", other)):
        if not group:
            continue
        summary[label] = {
            "n": len(group),
            "median_mps": statistics.median(group),
            "max_mps": max(group),
        }
    return summary


def render(stops: list[StopSpeed], summary: dict[str, Any]) -> str:
    lines = ["End-effector speed at the stop, and what map staleness costs there", ""]
    if not stops:
        return "\n".join([*lines, "  No stop carried a recorded joint state."])
    lines.append(f"  {'round':<40} {'stop class':<18} {'arm':>8} {'base':>8} {'world':>8}")
    for s in stops:
        lines.append(
            f"  {s.round_id[-38:]:<40} {s.stop_class:<18} {s.ee_speed_mps:>8.4f} "
            f"{s.base_speed_mps:>8.4f} {s.world_speed_mps:>8.4f} m/s"
        )
    gain = summary["quantisation_gain_mm"]
    lines += ["", f"  quantisation gain from 25 -> 15 mm: {gain:.2f} mm", ""]
    for label in ("carry_phase", "other"):
        block = summary.get(label)
        if not block:
            continue
        lines.append(
            f"  {label:<12} n={block['n']:<3} median {block['median_mps']:.4f} m/s, "
            f"max {block['max_mps']:.4f} m/s"
        )
        # The trade, at the two staleness figures the wire probe measured.
        for stale_label, stale_s in (("median +15 ms", 0.015), ("p99 +55 ms", 0.055)):
            for which, speed in (("median", block["median_mps"]), ("max", block["max_mps"])):
                cost = speed * stale_s * 1e3
                lines.append(
                    f"      {stale_label:<14} {which:<6} costs {cost:6.2f} mm "
                    f"-> net {gain - cost:+6.2f} mm"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rounds", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    stops = collect([p for p in args.rounds if p.is_dir()])
    summary = summarise(stops)
    print(json.dumps(summary, indent=2) if args.json else render(stops, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
