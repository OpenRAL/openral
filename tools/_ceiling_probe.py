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
import re
import signal
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


def _chunks_from_goal_log(goal_log: Path) -> int | None:
    """Action chunks the policy delivered inside the deadline, or ``None``.

    The single number that says whether a run was given a fair trial: a healthy
    run delivers 1.2-1.8 chunks/s, while one starved by host load or killed by
    a Nav2 teardown delivers ~0.04/s and cannot reach a grasp whatever the
    policy does (issue #256).
    """
    try:
        text = goal_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    seen = re.findall(r'"latest_chunk":\s*(\d+)', text)
    return int(seen[-1]) if seen else None


def _bond_teardown(deploy_log: Path) -> str:
    """Whether Nav2 tore its own stack down early in this run.

    Delegates to the harness's detector so the probe and
    ``tools/validation_matrix.py`` agree on what a voided run looks like.
    """
    try:
        lines = deploy_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return str(vm._nav2_bond_teardown(lines))


def _record(
    *,
    outcome: str,
    success: bool | None,
    detail: str,
    started: float,
    config_path: str,
    dispatch_timed_out: bool,
    load_start: list[float],
    goal_log: Path,
    deploy_log: Path,
    attempts: list[str] | None = None,
) -> dict[str, object]:
    """One round's record, in the one shape every exit path returns.

    Built here rather than at each ``return`` so a run that ends early cannot
    quietly omit the fields issue #256 added: the host load, the delivered
    chunk count and the Nav2 teardown flag are least dispensable on exactly the
    runs that fail before dispatch.
    """
    return {
        "outcome": outcome,
        "success": success,
        "detail": detail,
        # Absolute, not just a duration: an A/B whose arms are supposed to run
        # concurrently can only be shown to have done so from wall-clock
        # windows. Reconstructing them from directory mtimes afterwards is how
        # the 2026-09-10 pairing failure was caught, a run too late.
        "started_at": round(started, 1),
        "wall_s": round(time.time() - started, 1),
        "config": config_path,
        "dispatch_timed_out": dispatch_timed_out,
        # Issue #256: without these, a run starved by host load is
        # indistinguishable in the record from one the policy genuinely failed.
        "load_start": load_start,
        "load_end": [round(v, 2) for v in os.getloadavg()],
        "chunks": _chunks_from_goal_log(goal_log),
        "nav2_bond_teardown": _bond_teardown(deploy_log) or None,
        # Every dispatch the graph was not assembled for, in order (#263).
        # Empty when the run dispatched first time; a trailing `GAVE UP:` entry
        # means the graph never assembled, so the round is not a policy result.
        "dispatch_not_ready_attempts": list(attempts or ()),
    }


def _reap_domain(domain: str, sig: int) -> int:
    """Signal every ROS node left on `domain`, returning how many were hit.

    `ros2 launch` starts the octomap nodes in their own session, so the
    `killpg` in `run_one` never reaches them: every round leaked an
    `octomap_server_node` + `octomap_voxel_bridge` pair. The wasted CPU and
    RSS are the small half of the damage — each orphan also keeps its
    Fast-DDS `/dev/shm` segments and its `fastrtps_port<N>_el` lock file
    open, so a later round on the same domain fails `open_and_lock_file`,
    the policy is handed 0 chunks, and the round buckets `harness-error`
    with no line in any log naming the cause.

    Measured on q-laptop 2026-09-10: 46 orphans surviving up to 23.7 h
    across the ceiling battery and the first resolution A/B — 30 % of a
    core, 826 MB, 253 stale shm segments. A single round launched alone at
    load 1.2 still died with `latest_chunk: 0`.

    `ROS_DOMAIN_ID` is the ownership key, and `--ros-args` is what separates
    a node from this probe and the worker shell, which share the domain and
    must survive the sweep. SIGTERM first: Fast-DDS unlinks its own segments
    on a graceful exit and leaks them on SIGKILL.
    """
    hit = 0
    needle = f"ROS_DOMAIN_ID={domain}\0".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if needle not in (entry / "environ").read_bytes():
                continue
            if b"--ros-args" not in (entry / "cmdline").read_bytes():
                continue
        except OSError:  # reason: the process exited between iterdir and the read
            continue
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(int(entry.name), sig)
            hit += 1
    return hit


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

    load_start = [round(v, 2) for v in os.getloadavg()]
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
                # Same shape as the normal return: a run that never got its
                # action server is exactly the one whose host load matters, so
                # it must not be the record that omits it.
                return _record(
                    outcome="harness-error",
                    success=None,
                    detail="/openral/execute_rskill never appeared",
                    started=started,
                    config_path=config_path,
                    dispatch_timed_out=False,
                    load_start=load_start,
                    goal_log=goal_log,
                    deploy_log=deploy_log,
                )
            time.sleep(5.0)
            # A dispatch that overruns its own timeout must cost ONE round, not
            # the worker. Under 8-way contention four workers died here with an
            # uncaught TimeoutExpired, two of them before recording a single
            # round.
            dispatch_timed_out = False
            # The action server answering does not mean the graph is assembled:
            # the TF tree can still be two disjoint trees and a declared camera
            # can still have published nothing. The fixed 5 s above covered that
            # on an idle host and did not under contention (#263) — 10 of 18
            # runs in the first resolution A/B died at 0.4 s with
            # `ConnectivityException`, each scored as a policy failure. So
            # re-dispatch while the ONLY thing wrong is that the graph is not up
            # yet, bounded, and record every attempt: a retry invisible in the
            # artifacts is the hidden fallback CLAUDE.md §1.4 forbids.
            attempts: list[str] = []
            ready_deadline = time.time() + vm.DISPATCH_READY_TIMEOUT_S
            while True:
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
                    break
                not_ready = vm.dispatch_not_ready_reason(goal_log)
                if not not_ready:
                    break
                if time.time() >= ready_deadline:
                    attempts.append(f"GAVE UP: {not_ready}")
                    break
                attempts.append(not_ready)
                time.sleep(vm.DISPATCH_RETRY_INTERVAL_S)
            time.sleep(5.0)
        finally:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), 15)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=60)
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), 9)
            domain = env.get("ROS_DOMAIN_ID", "")
            if domain and _reap_domain(domain, signal.SIGTERM):
                time.sleep(5.0)
                _reap_domain(domain, signal.SIGKILL)

    success, detail = _success_from_log(deploy_log)
    return _record(
        outcome="ran" if success is not None else "harness-error",
        success=success,
        detail=detail,
        started=started,
        config_path=config_path,
        dispatch_timed_out=dispatch_timed_out,
        load_start=load_start,
        goal_log=goal_log,
        deploy_log=deploy_log,
        attempts=attempts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gate", required=True, choices=("on", "off"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rskill", required=True, help="rSkill dir carrying this worker's port.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help=(
            "Number the first round N instead of 1 and append to any existing records.json. "
            "Lets a caller drive rounds one at a time — which is how the resolution A/B "
            "alternates its two arms round by round on a host that fits only one graph."
        ),
    )
    args = parser.parse_args(argv)

    spec = next(s for s in vm.MATRIX if s.key == args.scene)
    out = args.out / f"{args.scene}-{args.gate}"
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "records.json"
    records: list[dict[str, object]] = (
        json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else []
    )
    for n in range(args.start_round, args.start_round + args.rounds):
        rec = run_one(spec, args.gate, args.seed, out / f"r{n:02d}", args.rskill)
        rec |= {"round": n, "scene": args.scene, "gate": args.gate}
        records.append(rec)
        ledger.write_text(json.dumps(records, indent=1), encoding="utf-8")
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
