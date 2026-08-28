"""Which RoboCasa tasks make the base drive while an object is carried?

[Issue #108](https://github.com/OpenRAL/openral/issues/108) is blocked on one
missing scene. `openral_nav2_bringup` publishes a payload-grown Nav2 footprint
and filters the payload's and chassis's own lidar returns out of the scan
(PR #143), and neither has ever run against a base that actually *translates*
while carrying — so a working footprint and a decorative one look identical.
Its README specifies the scene that would close #108 in four criteria, the
first being **> ~1.0 m of required base displacement while an object is held**.

This tool answers "does any RoboCasa task already do that?" by **building the
env and measuring it** — not by reading the source. That distinction is the
whole point, and it is a correction:

* A first pass classified tasks statically, on the premise that RoboCasa's
  `ref=` keyword bounds a fixture pair to 0.10 m ("if specified, will search for
  fixture close to ref (within 0.10m)", `Kitchen.get_fixture`). **The docstring
  does not describe the code.** The body keeps candidates within 0.10 m *of the
  nearest one* — `[f for f, d in zip(cand, dists) if d - min_dist < 0.10]` —
  which is a tie-break among near-equidistant fixtures, not a proximity bound.
  A `ref=`'d fixture is the *nearest of its type*, and in a large kitchen that
  can be metres away.
* Linkage also need not involve the pair that matters. `DeliverStraw` registers
  `dining_counter` with `ref=self.stool` — a third fixture — which says nothing
  about how far it is from the drawer the base starts at.

Both errors point the same way (they under-count qualifying tasks), and both
vanish once the number is measured rather than inferred. What is measured here
is the distance from the **base's start pose** to each object the task
manipulates: a task qualifies when some non-distractor object sits further than
`REQUIRED_BASE_TRANSLATION_M` from where the base begins, because the arm
cannot bridge that and the base must drive.

Needs RoboCasa provisioned (a GPU host; the env build is CPU MuJoCo but the
install is the sim stack). Run::

    uv run python tools/robocasa_carry_survey.py --seeds 1 2 3
    uv run python tools/robocasa_carry_survey.py --tasks DeliverStraw --json

The result it recorded is in
`docs/reference/robocasa-carry-survey.md`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

#: Criterion 1 of the scene that would close #108 — `openral_nav2_bringup`
#: README, "What is still open". Below this the arm bridges the gap from one
#: base pose and `NavigateToPose` never runs.
REQUIRED_BASE_TRANSLATION_M = 1.0

#: RoboCasa names clutter objects `distr*` — and, in six places, `dstr*`. They
#: are never manipulated, so a distractor on a far counter must not make a task
#: look like a carry. BOTH spellings are needed: with only `distr`,
#: `ArrangeBreadBasket` and `PanTransfer` reported `dstr_dining*` as their
#: furthest object and read as cross-kitchen carries when they are not.
DISTRACTOR_PREFIXES = ("distr", "dstr")


class SurveyError(RuntimeError):
    """RoboCasa is unavailable, or its task registry could not be read."""


@dataclass(frozen=True)
class CarryMeasurement:
    """One (task, seed) reset, measured."""

    task: str
    seed: int
    layout: int
    style: int
    #: Straight-line distance from the base start pose to the furthest
    #: non-distractor object, in metres.
    furthest_object_m: float
    furthest_object: str
    #: Distance to the nearest one — a task needs something graspable at the
    #: start too, or the base drives before it is ever carrying anything.
    nearest_object_m: float
    nearest_object: str
    lang: str

    @property
    def requires_base_translation(self) -> bool:
        return self.furthest_object_m > REQUIRED_BASE_TRANSLATION_M


def robocasa_root(explicit: str | None = None) -> Path:
    """Locate the RoboCasa source tree (for the task registry)."""
    if explicit:
        root = Path(explicit)
    else:
        try:
            import robocasa  # reason: optional, probed not required

            root = Path(robocasa.__file__).parent
        except ImportError:
            cache = os.environ.get("OPENRAL_CACHE_HOME") or str(Path.home() / ".cache" / "openral")
            root = Path(cache) / "repos" / "robocasa-kitchen" / "robocasa"
    if not (root / "environments" / "kitchen").is_dir():
        raise SurveyError(
            f"no RoboCasa source tree at {root}. Pass --robocasa-root, or provision it: "
            'python -c "from openral_sim._deps import ensure_backend_deps as e; '
            "e('robocasa_kitchen')\""
        )
    return root


def read_target50(root: Path) -> list[str]:
    """The 50 task names XR-1 RoboCasa365 reports against.

    XR-1's model card (`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`) pins
    `task set: target50` — 50 tasks x 50 episodes, base seed 7, 57.28%. RoboCasa
    defines it in `dataset_registry.py` as the three target splits concatenated.
    Parsed with `ast` so the registry can be read without importing RoboCasa.
    """
    src = (root / "utils" / "dataset_registry.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "TARGET_TASKS" for t in node.targets):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            break
        sets: dict[str, list[str]] = {
            kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg
        }
        return [t for k in ("atomic_seen", "composite_seen", "composite_unseen") for t in sets[k]]
    raise SurveyError(f"no TARGET_TASKS dict in {root / 'utils' / 'dataset_registry.py'}")


def measure(task: str, seed: int) -> CarryMeasurement:
    """Build `task` at `seed`, reset, and measure object distances from the base.

    Raises:
        SurveyError: When the env cannot be built or carries no object.
    """
    import mujoco  # reason: heavy optional dep
    import numpy as np
    import robocasa  # noqa: F401  # reason: import IS the env registration
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller = load_composite_controller_config(controller="BASIC", robot="PandaMobile")
    env = robosuite.make(
        env_name=task,
        robots=["PandaMobile"],
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=600,
        ignore_done=True,
        seed=seed,
    )
    try:
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base")
        base_xy = np.array(data.xpos[base_id])[:2]
        dists: dict[str, float] = {}
        for name, obj in env.objects.items():
            if name.startswith(DISTRACTOR_PREFIXES):
                continue
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            dists[name] = float(np.linalg.norm(np.array(data.xpos[body])[:2] - base_xy))
        if not dists:
            raise SurveyError(f"{task} at seed {seed} registers no manipulable object")
        far = max(dists, key=lambda k: dists[k])
        near = min(dists, key=lambda k: dists[k])
        return CarryMeasurement(
            task=task,
            seed=seed,
            layout=int(env.layout_id),
            style=int(env.style_id),
            furthest_object_m=round(dists[far], 3),
            furthest_object=far,
            nearest_object_m=round(dists[near], 3),
            nearest_object=near,
            lang=str(env.get_ep_meta()["lang"]),
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None, help="Tasks to measure (default: target50)."
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3], help="Seeds per task.")
    parser.add_argument("--robocasa-root", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        tasks = args.tasks or read_target50(robocasa_root(args.robocasa_root))
    except SurveyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows: list[CarryMeasurement] = []
    failures: dict[str, str] = {}
    for task in tasks:
        for seed in args.seeds:
            try:
                rows.append(measure(task, seed))
            except Exception as exc:  # reason: one unbuildable task must not sink the sweep; the reason is reported
                failures[f"{task}@{seed}"] = f"{type(exc).__name__}: {exc}"[:200]

    # "candidate", not "qualifies": distance alone cannot tell a carried object
    # from a fixed one the arm reaches (a fridge `door_obj` at 1.06 m is a reach
    # target, not a payload). Each candidate's `_check_success` still has to be
    # read to confirm the object must be MOVED there.
    candidates = sorted(
        {r.task for r in rows if r.requires_base_translation},
        key=lambda t: -max(r.furthest_object_m for r in rows if r.task == t),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "seeds": args.seeds,
                    "candidates": candidates,
                    "measurements": [asdict(r) for r in rows],
                    "failures": failures,
                },
                indent=2,
            )
        )
        return 0

    print(f"{'task':30s} {'seed':>4s} {'lay/sty':>8s} {'nearest':>8s} {'furthest':>9s}  carries?")
    for r in rows:
        mark = "YES" if r.requires_base_translation else "-"
        print(
            f"{r.task:30s} {r.seed:4d} {r.layout:4d}/{r.style:<3d} "
            f"{r.nearest_object_m:8.2f} {r.furthest_object_m:9.2f}  {mark}"
        )
    print(
        f"\n{len(candidates)} of {len({r.task for r in rows})} measured task(s) place an object "
        f"more than {REQUIRED_BASE_TRANSLATION_M} m from the base start pose.\n"
        "These are CANDIDATES: read each one's `_check_success` to confirm the object must be "
        "carried there rather than merely reached."
    )
    for task in candidates:
        best = max((r for r in rows if r.task == task), key=lambda r: r.furthest_object_m)
        print(
            f"  {task:28s} {best.furthest_object_m:.2f} m ({best.furthest_object}, seed {best.seed})"
        )
    if failures:
        print(f"\n{len(failures)} (task, seed) build(s) failed:")
        for key, why in sorted(failures.items()):
            print(f"  {key}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
