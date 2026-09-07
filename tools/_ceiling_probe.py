#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The policy's success ceiling with the world-voxel gate off.

Ceiling measurement, not a validation round: bypasses
``tools/validation_matrix.py`` (which refuses ``no_enable_octomap_kernel_check``
/ ``_SAFETY_KNOB_PATTERNS`` — a validation round must never silently disable
the collision check). A CLI flag on one simulated process; writes no scene
file, launch default or manifest.

Answers: of the runs the kernel stops, how many would have succeeded anyway?
120 runs at 5-10% completion (gate on), median true clearance 20.1 mm at the
stop — the ceiling could be 12% or 60%, and that number decides whether the
collision programme continues (``PLAN.md`` §4). External analogue: PACS
(arXiv:2511.06385 Table I, unfiltered 0.70 vs binary-filtered 0.04).

One worker = one scene/gate/N rounds, launched in parallel by
``tools/ceiling_battery.sh``. Each needs its own ``ROS_DOMAIN_ID``
(``SimSensorBridge`` uses ``count_publishers``, see ``tests/sim/conftest.py``)
and its own XR-1 sidecar port: the sidecar's resettable policy object
(``tools/_xr1_server.py`` ``reset`` endpoint) is not safe to share across
concurrent clients.

Launch mirrors the harness (``materialise_scene``, readiness gate, dispatch
tool, deadline) so the gate flag is the only difference between arms.

Usage (normally via the battery script):
    python tools/_ceiling_probe.py --scene baguette --gate off --rounds 10 \\
        --domain 61 --rskill rskills/xr1-robocasa365-w0 --out outputs/ceiling
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `validation_matrix` is a sibling script in tools/, imported by path via the
# sys.path insert above rather than as an installed package, so mypy has no stub.
import validation_matrix as vm  # type: ignore[import-not-found]  # reason: see above

#: Deadline per scene run. The harness's own default; kept identical so a
#: gate-off run is not given more time to succeed than a gate-on one.
DEADLINE_S = 420.0


def _success_from_log(deploy_log: Path) -> tuple[bool | None, str]:
    """Read ``sim.task_success_final`` out of a finished run's deploy log.

    Returns ``(succeeded, detail)``. ``None`` means the line was never printed
    — the run did not reach the point of reporting, which is a harness-class
    outcome and must not be counted as a failure of the policy.
    """
    try:
        text = deploy_log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"unreadable deploy log: {exc}"
    payload: dict[str, object] | None = None
    for line in text.splitlines():
        if "sim.task_success_final" not in line:
            continue
        brace = line.find("{")
        if brace < 0:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(line[brace:])
    if payload is None:
        return None, "no sim.task_success_final line"
    ever = bool(payload.get("ever_succeeded") or payload.get("success"))
    return ever, json.dumps(payload, sort_keys=True)


def run_one(
    # `validation_matrix.SceneSpec`, from an untyped sibling script imported by
    # path, so its real type is unavailable to mypy.
    spec: Any,
    gate: str,
    seed: int,
    run_dir: Path,
    rskill: str,
) -> dict[str, object]:
    """Launch one scene end to end and return its record."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = "run"
    deploy_log = run_dir / f"{stem}_deploy.log"
    goal_log = run_dir / f"{stem}_goal.log"

    config_path, config_file = vm.materialise_scene(spec, seed, run_dir)
    launcher = vm.resolve_launcher()
    gate_flag = (
        "--enable-octomap-kernel-check" if gate == "on" else "--no-enable-octomap-kernel-check"
    )
    argv = [
        str(launcher),
        "deploy",
        "sim",
        "--config",
        str(config_file),
        "--enable-slam",
        "--enable-nav2",
        "--enable-octomap",
        gate_flag,
        "--no-object-detector",
        "--no-enable-scene-vlm",
        "--no-dashboard",
        "--hal",
        "viewer_enabled=false",
    ]
    env = vm._launch_env(run_dir, stem)
    started = time.time()
    with deploy_log.open("wb") as sink:
        proc = subprocess.Popen(
            argv, cwd=vm.REPO_ROOT, env=env, stdout=sink, stderr=sink, start_new_session=True
        )
        try:
            vm.wait_for_dds_transport_ready(deploy_log, proc, timeout_s=300.0)
            # The helper's signature differs across the commits this probe is
            # pointed at (older ones take no `env` and inherit ROS_DOMAIN_ID
            # from the environment, which the per-worker export already sets).
            if "env" in inspect.signature(vm._wait_for_action_server).parameters:
                server_up = vm._wait_for_action_server(proc, timeout_s=600, env=env)
            else:
                server_up = vm._wait_for_action_server(proc, 600)
            if not server_up:
                return {
                    "outcome": "harness-error",
                    "detail": "/openral/execute_rskill never appeared",
                    "success": None,
                    "wall_s": round(time.time() - started, 1),
                    "config": config_path,
                }
            time.sleep(5.0)
            # A dispatch that overruns its own timeout must cost ONE round, not
            # the worker. Under 8-way contention four workers died here with an
            # uncaught TimeoutExpired, two of them before recording a single
            # round.
            dispatch_timed_out = False
            try:
                with goal_log.open("wb") as goal_sink:
                    subprocess.run(
                        [
                            str(vm.REPO_ROOT / ".venv" / "bin" / "python"),
                            str(vm.REPO_ROOT / "tools" / "_validation_matrix_dispatch.py"),
                            "--deadline-s",
                            str(DEADLINE_S),
                            "--rskill-id",
                            rskill,
                            "--prompt",
                            spec.prompt,
                        ],
                        cwd=vm.REPO_ROOT,
                        env=env,
                        stdout=goal_sink,
                        stderr=goal_sink,
                        check=False,
                        timeout=DEADLINE_S + 600,
                    )
            except subprocess.TimeoutExpired:
                dispatch_timed_out = True
            time.sleep(5.0)
        finally:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), 15)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=60)
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), 9)

    success, detail = _success_from_log(deploy_log)
    return {
        "outcome": "ran" if success is not None else "harness-error",
        "success": success,
        "detail": detail,
        "wall_s": round(time.time() - started, 1),
        "config": config_path,
        "dispatch_timed_out": dispatch_timed_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gate", required=True, choices=("on", "off"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rskill", required=True, help="rSkill dir carrying this worker's port.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    spec = next(s for s in vm.MATRIX if s.key == args.scene)
    out = args.out / f"{args.scene}-{args.gate}"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for n in range(1, args.rounds + 1):
        rec = run_one(spec, args.gate, args.seed, out / f"r{n:02d}", args.rskill)
        rec |= {"round": n, "scene": args.scene, "gate": args.gate}
        records.append(rec)
        (out / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(
            f"[{args.scene}/{args.gate}] r{n:02d} success={rec['success']} "
            f"{rec['wall_s']}s {rec['outcome']}",
            flush=True,
        )
    ran = [r for r in records if r["success"] is not None]
    print(
        f"[{args.scene}/{args.gate}] DONE {sum(1 for r in ran if r['success'])}/{len(ran)}"
        f" ({len(records) - len(ran)} harness-error)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
