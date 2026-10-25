"""Run a batch of ``deploy sim`` episodes on one Isaac sidecar.

``--mode persistent`` boots Kit and builds the scene once: every episode runs a FRESH ROS
graph (kernel, e-stop latch, tracker, world state, octomap, HAL, runner: a new process
each, so nothing a previous episode left in them survives) against the one kept-alive
sidecar, which the new HAL adopts and resets with the episode's seed (``world.reset``).
``--mode fresh`` is the baseline: the sidecar is booted and ended with every episode.

Between episodes the harness checks the domain is quiet and no graph process of this
domain is left, and that the sidecar still answers. If anything leaked it ends the
sidecar too and says so (``restarts`` in ``batch.json``), so the next episode starts
from a cold stack. Nothing here clears an e-stop: a latched e-stop dies with the
kernel process.

Run (a sourced ROS 2 overlay and the scene's rSkill/prompt)::

    python tools/deploy_sim_batch.py --scene scenes/deploy/isaac_leisaac_pick_orange.yaml \\
        --out /tmp/batch --mode persistent --seeds 1,2,3,4 --sidecar-port 41300 \\
        --rskill-id OpenRAL/rskill-smolvla-so101-leisaac_pick_orange-bf16 \\
        --prompt "Grab orange and place into plate" --deadline-s 90

``--record`` also bags each episode's kernel inputs and verdicts (``kernel_replay_diff``'s
``REPLAY_TOPICS``, into ``<episode>/bag``) and dumps the kernel's parameters into
``<episode>/kernel_params.yaml``: the input of ``tools/kernel_replay_diff.py diff`` (ADR-0121).

``--end-on-grade`` ends a goal through ``/openral/rskill/complete`` (the completion check's own
service) once the HAL's sim grader has held ``complete`` for ``--grade-settle-s`` of sim time,
and records ``ended_by`` (``grade`` or ``goal``): a run is then scored like ``openral sim run``,
at its first settled completion, instead of manipulating on until its deadline. Sim ground
truth ends the goal; the safety stack and the kernel are untouched.

``python tools/deploy_sim_batch.py compare <fresh_dir> <persistent_dir>`` prints the
per-episode outcome, stop time and wall time side by side.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
READY_MARKERS = ("HAL activated at",)
PRELOAD_MARKER = "rskill_runner.preload_done"


def episode_scene(
    scene: dict[str, Any], *, seed: int, keep_alive: bool, port: int
) -> dict[str, Any]:
    """The scene dict one episode launches: its reset seed, the sidecar port and keep-alive.

    Example:
        >>> s = episode_scene(
        ...     {"scene": {"backend_options": {}}}, seed=3, keep_alive=True, port=41300
        ... )
        >>> s["seed"], s["scene"]["backend_options"]
        (3, {'port': 41300, 'keep_alive': True})
    """
    out = copy.deepcopy(scene)
    out["seed"] = seed
    opts = out.setdefault("scene", {}).setdefault("backend_options", {})
    opts["port"] = port
    opts["keep_alive"] = keep_alive
    return out


#: The HAL's grade-change line (``SimAttachedHAL._observe_placement_grade``), JSON variant.
_GRADE_EVENT = "sim.placement_grade {"


def grade_events(text: str) -> list[tuple[str, float]]:
    r"""The ``(grade, sim_time_s)`` of each grade-change line in a slice of ``deploy.log``.

    Example:
        >>> grade_events(
        ...     '[hal] sim.placement_grade {"grade": "complete", "sim_time_s": 60.15}\n'
        ...     '[hal] sim.placement_grade_final {"grade": "fail", "sim_time_s": 70.0}\n'
        ... )
        [('complete', 60.15)]
    """
    out = []
    for line in text.splitlines():
        at = line.find(_GRADE_EVENT)
        if at < 0:
            continue
        with contextlib.suppress(ValueError, KeyError, TypeError):
            data = json.loads(line[at + len(_GRADE_EVENT) - 1 :])
            out.append((str(data["grade"]), float(data["sim_time_s"])))
    return out


class GradeSettle:
    """When a goal may end on the sim grader: ``complete`` held for ``settle_s`` of sim time.

    A transient pass does not count: Spark r2 master_dr seed 2 graded ``complete`` for 0.1 s
    while the banana leaned on the bowl's rim, then ``fail``.

    Example:
        >>> g = GradeSettle(settle_s=1.0)
        >>> g.feed("complete", 60.15)
        >>> g.due(60.5)
        False
        >>> g.feed("fail", 60.28)
        >>> g.due(62.0)
        False
        >>> g.feed("complete", 63.0)
        >>> g.due(64.1)
        True
    """

    def __init__(self, settle_s: float) -> None:
        self.settle_s = settle_s
        self._since: float | None = None

    def feed(self, grade: str, sim_time_s: float) -> None:
        """One grade change (``grade_events``)."""
        self._since = sim_time_s if grade == "complete" else None

    def due(self, now_sim_s: float) -> bool:
        """Whether ``complete`` has held ``settle_s`` by sim time ``now_sim_s``."""
        return self._since is not None and now_sim_s - self._since >= self.settle_s


class _SimGraph:
    """The harness's own ROS participant: the sim clock, and the goal's ``complete`` service."""

    def __init__(self, env: Mapping[str, str]) -> None:
        import rclpy  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
        from rosgraph_msgs.msg import (
            Clock,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
        )
        from std_srvs.srv import (
            Trigger,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
        )

        for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "RMW_IMPLEMENTATION"):
            if env.get(key):
                os.environ.setdefault(key, env[key])  # read once, at the first participant
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx, domain_id=int(env.get("ROS_DOMAIN_ID", "0")))
        self._node = rclpy.create_node("deploy_sim_batch_grade", context=self._ctx)
        self._exec = rclpy.executors.SingleThreadedExecutor(context=self._ctx)
        self._exec.add_node(self._node)
        self.now_s = -math.inf
        self._node.create_subscription(Clock, "/clock", self._on_clock, 10)
        self._trigger_cls = Trigger
        self._complete = self._node.create_client(Trigger, "/openral/rskill/complete")

    def _on_clock(self, msg: Any) -> None:
        self.now_s = msg.clock.sec + msg.clock.nanosec * 1e-9

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._exec.spin_once(timeout_sec=0.05)

    def complete(self, timeout_s: float = 10.0) -> bool:
        """End the running goal as a success; ``True`` if the runner accepted."""
        if not self._complete.wait_for_service(timeout_sec=timeout_s):
            return False
        future = self._complete.call_async(self._trigger_cls.Request())
        self._exec.spin_until_future_complete(future, timeout_sec=timeout_s)
        result = future.result()
        return bool(result is not None and result.success)

    def close(self) -> None:
        import rclpy  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay

        self._node.destroy_node()
        rclpy.shutdown(context=self._ctx)


def _dispatch_goal(args: argparse.Namespace, env: Mapping[str, str], log: Path) -> tuple[str, str]:
    """Run one goal; with ``--end-on-grade``, end it on a settled grade. Returns its output
    and ``ended_by`` (``grade``: the harness completed it; ``goal``: it ended on its own)."""
    cmd = [
        sys.executable,
        str(REPO / "tools/_validation_matrix_dispatch.py"),
        "--deadline-s",
        str(args.deadline_s),
        "--rskill-id",
        args.rskill_id,
        "--prompt",
        args.prompt,
    ]
    if not args.end_on_grade:
        res = subprocess.run(cmd, env=dict(env), capture_output=True, text=True, check=False)
        return res.stdout + res.stderr, "goal"
    offset = log.stat().st_size if log.exists() else 0
    dispatch = subprocess.Popen(
        cmd, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    settle = GradeSettle(args.grade_settle_s)
    ended_by, tail = "goal", ""
    graph: _SimGraph | None = None
    try:
        graph = _SimGraph(env)
        while dispatch.poll() is None:
            # Spin first, then read: a grade logged during the spin (a `fail` after the
            # `complete`) is fed before `due()` can complete the goal on the stale grade.
            graph.spin(0.5)
            with log.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read().decode(errors="replace")
            offset += len(chunk.encode())
            text, _, tail = (tail + chunk).rpartition("\n")
            for grade, sim_s in grade_events(text):
                settle.feed(grade, sim_s)
            if ended_by == "goal" and settle.due(graph.now_s) and graph.complete():
                ended_by = "grade"
    # Any monitor failure is recorded in ended_by; the caller still tears the episode down.
    except Exception as exc:
        dispatch.kill()
        out, _ = dispatch.communicate()
        return out, f"monitor_failed: {exc!r}"
    finally:
        if graph is not None:
            graph.close()
    out, _ = dispatch.communicate()
    return out, ended_by


def sidecar_pids(port: int) -> list[int]:
    """Pids of this user's Isaac sidecars serving ``port`` (the ``--port`` of their argv)."""
    needle = ["--port", str(port)]
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any("isaac_sidecar.py" in a for a in argv) and any(
                argv[i : i + 2] == needle for i in range(len(argv))
            ):
                pids.append(int(entry.name))
    return pids


def sidecar_call(port: int, endpoint: str, timeout_ms: int = 4000) -> dict[str, Any] | None:
    """One request to the sidecar on ``port``; ``None`` when it does not answer."""
    import msgpack
    import numpy as np
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send(msgpack.packb({"endpoint": endpoint, "data": {}}, use_bin_type=True))

        def hook(obj: dict[str, Any]) -> Any:
            if "__ndarray__" in obj:
                import io

                return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
            return obj

        return msgpack.unpackb(sock.recv(), object_hook=hook, raw=False)  # type: ignore[no-any-return]
    except zmq.ZMQError:
        return None
    finally:
        sock.close()


def graph_leaks(domain: str) -> list[str]:
    """Command lines of deploy-graph processes of this user still alive on ROS ``domain``."""
    from openral_cli.deploy_sim import _cmdline_is_openral_graph_process

    leaks = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        with contextlib.suppress(OSError):
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            env = (entry / "environ").read_bytes().split(b"\0")
            if _cmdline_is_openral_graph_process(cmd) and f"ROS_DOMAIN_ID={domain}".encode() in env:
                leaks.append(f"{entry.name}: {cmd[:160]}")
    return leaks


def domain_nodes(env: dict[str, str]) -> list[str]:
    """ROS node names visible on the domain (``ros2 node list`` without the daemon)."""
    out = subprocess.run(
        ["ros2", "node", "list", "--no-daemon"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return [n for n in out.stdout.split() if n.startswith("/")]


def _wait_for(
    log: Path, markers: tuple[str, ...], proc: subprocess.Popen[bytes], timeout_s: float
) -> float | None:
    """Seconds until every marker is in ``log``; ``None`` if the launch died or timed out."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        text = log.read_text(errors="replace") if log.exists() else ""
        if all(m in text for m in markers):
            return time.monotonic() - t0
        if proc.poll() is not None:
            return None
        time.sleep(1.0)
    return None


def _dump_kernel_params(env: Mapping[str, str], tries: int = 3) -> str:
    """``ros2 param dump`` of the kernel, without the CLI daemon (whose graph view can be
    stale), retried: the first discovery spin can miss a node that is up."""
    out = ""
    for _ in range(tries):
        res = subprocess.run(
            ["ros2", "param", "dump", "/openral_safety_kernel", "--no-daemon", "--spin-time", "10"],
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout or res.stderr
        if "ros__parameters" in out:
            break
    return out


def _stop_group(proc: subprocess.Popen[bytes], grace_s: float) -> str:
    """SIGINT the launch's process group (what Ctrl-C does), SIGKILL after ``grace_s``."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return "graceful"
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "killed"


def _end_sidecar(port: int) -> None:
    sidecar_call(port, "close", timeout_ms=3000)
    for _ in range(20):
        if not sidecar_pids(port):
            return
        time.sleep(0.5)
    for pid in sidecar_pids(port):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def batch_env(environ: Mapping[str, str]) -> tuple[dict[str, str], str]:
    """The episodes' environment and ROS domain, the domain written into it.

    It also carries the deploy graph's Fast DDS profile (``apply_fastdds_large_data_profile``).

    ``graph_leaks`` matches ``ROS_DOMAIN_ID=<domain>`` in each process's environ, so the
    children must carry the variable even when the caller's shell left it unset.

    Example:
        >>> env, domain = batch_env({"PATH": "/bin"})
        >>> domain, env["ROS_DOMAIN_ID"], env["FASTRTPS_DEFAULT_PROFILES_FILE"][-22:]
        ('0', '0', 'fastdds_large_data.xml')
    """
    env = dict(environ)
    domain = env.setdefault("ROS_DOMAIN_ID", "0")
    from openral_cli.deploy_sim import apply_fastdds_large_data_profile

    # The graph's own transport profile, so the recorder and the param dump see it as it is.
    apply_fastdds_large_data_profile(env)
    return env, domain


def run_batch(args: argparse.Namespace) -> int:  # reason: one linear episode loop
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load(Path(args.scene).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    env, domain = batch_env(os.environ)
    episodes: list[dict[str, Any]] = []
    restarts: list[str] = []
    # Before the try: its finally ends the port's sidecar, which must be one this batch owns.
    if sidecar_pids(args.sidecar_port):
        print(f"refusing: a sidecar is already serving port {args.sidecar_port}", file=sys.stderr)
        return 2
    try:
        for i, seed in enumerate(seeds):
            last = i == len(seeds) - 1
            ep = out / f"ep{i}_seed{seed}"
            ep.mkdir()
            keep = args.mode == "persistent" and not last
            quiet_nodes = domain_nodes(env)
            if quiet_nodes:
                print(
                    f"domain {domain} is not quiet before episode {i}: {quiet_nodes}",
                    file=sys.stderr,
                )
                return 3
            scene_path = ep / "scene.yaml"
            scene_path.write_text(
                yaml.safe_dump(
                    episode_scene(template, seed=seed, keep_alive=keep, port=args.sidecar_port)
                )
            )
            adopted = bool(sidecar_pids(args.sidecar_port))
            t0 = time.monotonic()
            log = ep / "deploy.log"
            with log.open("wb") as fh:
                proc = subprocess.Popen(
                    [
                        str(REPO / ".venv/bin/openral"),
                        "deploy",
                        "sim",
                        "--config",
                        str(scene_path),
                        *args.deploy_arg,
                    ],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=REPO,
                    start_new_session=True,
                )
            markers = READY_MARKERS + ((PRELOAD_MARKER,) if args.expect_preload else ())
            boot_s = _wait_for(log, markers, proc, args.boot_timeout_s)
            rec: dict[str, Any] = {
                "episode": i,
                "seed": seed,
                "sidecar_adopted": adopted,
                "boot_s": boot_s,
            }
            goal: dict[str, Any] = {}
            recorder = None
            if boot_s is not None and args.record:
                from kernel_replay_diff import (  # type: ignore[import-not-found,unused-ignore]  # reason: sibling tool, on sys.path[0] when run as a script
                    record_command,
                )

                recorder = subprocess.Popen(
                    record_command(str(ep / "bag")),
                    stdout=(ep / "record.log").open("wb"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                dump = _dump_kernel_params(env)
                if "ros__parameters" in dump:
                    (ep / "kernel_params.yaml").write_text(dump)
                else:
                    (ep / "kernel_params.err").write_text(dump)
                    rec["param_dump"] = "failed"
            if boot_s is None:
                rec["failure"] = "stack did not come up"
            else:
                time.sleep(args.rest_s)
                truth = sidecar_call(args.sidecar_port, "scene_ground_truth")
                (ep / "ground_truth.json").write_text(
                    json.dumps(truth, default=lambda o: o.tolist())
                )
                t_goal = time.monotonic()
                output, rec["ended_by"] = _dispatch_goal(args, env, log)
                (ep / "goal.log").write_text(output)
                with contextlib.suppress(ValueError, IndexError):
                    goal = json.loads(output.strip().splitlines()[-1])
                rec["goal_s"] = time.monotonic() - t_goal
            if recorder is not None:
                rec["recorder"] = _stop_group(recorder, 20.0)
            t_down = time.monotonic()
            rec["teardown"] = _stop_group(proc, args.grace_s)
            rec["teardown_s"] = time.monotonic() - t_down
            rec["goal"] = goal
            rec["total_s"] = time.monotonic() - t0
            leaks = graph_leaks(domain)
            sidecar_ok = (not keep) or sidecar_call(args.sidecar_port, "ping") is not None
            # The last persistent episode adopts the kept-alive sidecar and does not end it
            # (an adopted sidecar is not its child): the batch ends it below, which is no leak.
            stray_sidecar = (
                not keep
                and not (last and args.mode == "persistent")
                and bool(sidecar_pids(args.sidecar_port))
            )
            if leaks or not sidecar_ok or stray_sidecar:
                why = f"episode {i}: leaked={leaks} sidecar_ok={sidecar_ok}"
                restarts.append(why)
                print(f"STACK RESTART ({why})", flush=True)
                for line in leaks:
                    with contextlib.suppress(ValueError, ProcessLookupError):
                        os.kill(int(line.split(":")[0]), signal.SIGKILL)
                _end_sidecar(args.sidecar_port)
            rec["leaks"] = leaks
            rec["sidecar_pids_after"] = sidecar_pids(args.sidecar_port)
            episodes.append(rec)
            (out / "batch.json").write_text(
                json.dumps(
                    {"mode": args.mode, "episodes": episodes, "restarts": restarts}, indent=1
                )
            )
            print(
                json.dumps(
                    {k: rec[k] for k in ("episode", "seed", "sidecar_adopted", "boot_s", "total_s")}
                ),
                flush=True,
            )
    finally:
        _end_sidecar(args.sidecar_port)
    return 0


def _max_abs_diff(a: Any, b: Any) -> float:
    """Largest absolute difference between the numbers of two JSON-like trees (inf if shapes differ).

    Example:
        >>> _max_abs_diff({"x": [1.0, 2.0], "n": "a"}, {"x": [1.5, 2.0], "n": "a"})
        0.5
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((_max_abs_diff(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def compare(a: Path, b: Path) -> int:
    """Per-episode outcome, stop time and wall time of two batches, side by side."""
    ra, rb = (json.loads((d / "batch.json").read_text()) for d in (a, b))
    print(f"{'ep':>2} {'seed':>4} | {a.name:^34} | {b.name:^34}")
    for ea, eb in zip(ra["episodes"], rb["episodes"], strict=False):

        def cell(e: dict[str, Any]) -> str:
            g = e.get("goal", {})
            return f"ok={g.get('success')!s:5} {str(g.get('failure_reason') or '')[:12]:12} boot {e.get('boot_s') or 0:5.1f} tot {e['total_s']:6.1f}"

        print(f"{ea['episode']:>2} {ea['seed']:>4} | {cell(ea):34} | {cell(eb):34}")
    print("initial scene (ground truth right after boot), max |difference| per episode:")
    for ea, eb in zip(ra["episodes"], rb["episodes"], strict=False):
        pa, pb = (
            d / f"ep{e['episode']}_seed{e['seed']}" / "ground_truth.json"
            for d, e in ((a, ea), (b, eb))
        )
        if not (pa.exists() and pb.exists()):  # an episode that did not boot has none
            print(f"  ep{ea['episode']}: n/a (no ground truth)")
            continue
        diff = _max_abs_diff(json.loads(pa.read_text()), json.loads(pb.read_text()))
        print(f"  ep{ea['episode']}: {diff:.4f}")
    saved = [
        ea["total_s"] - eb["total_s"]
        for ea, eb in zip(ra["episodes"], rb["episodes"], strict=False)
    ]
    print("wall saved per episode (s):", [round(s, 1) for s in saved])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``run`` (default) or ``compare A B``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"]:
        return compare(Path(argv[1]), Path(argv[2]))
    restore_sigint()
    return run_batch(parser().parse_args(argv))


def restore_sigint() -> bool:
    """Catch SIGINT again when this process inherited it ignored; ``True`` if it was.

    A harness started as a background job of a non-interactive shell inherits SIGINT
    ignored, and every child inherits that too. ``_stop_group``'s SIGINT then reaches no
    one: the launch is SIGKILLed after its grace, and ``ros2 bag record`` dies without
    finalizing its bag. A caught signal resets to its default across ``exec``, so this
    gives every later child its SIGINT back.
    """
    if signal.getsignal(signal.SIGINT) is not signal.SIG_IGN:
        return False
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return True


def parser() -> argparse.ArgumentParser:
    """The ``run`` command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("persistent", "fresh"), default="persistent")
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--sidecar-port", type=int, required=True)
    p.add_argument("--rskill-id", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--deadline-s", type=float, default=90.0)
    p.add_argument("--rest-s", type=float, default=20.0)
    p.add_argument("--boot-timeout-s", type=float, default=420.0)
    p.add_argument("--grace-s", type=float, default=60.0)
    p.add_argument("--expect-preload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--end-on-grade",
        action="store_true",
        help="end a goal once the sim grader holds complete for --grade-settle-s (sim time)",
    )
    p.add_argument("--grade-settle-s", type=float, default=1.0)
    p.add_argument(
        "--record",
        action="store_true",
        help="bag the kernel's replay topics and dump its parameters per episode (ADR-0121)",
    )
    p.add_argument(
        "--deploy-arg",
        action="append",
        default=[],
        help="extra `openral deploy sim` argument, repeatable",
    )
    return p


if __name__ == "__main__":
    sys.exit(main())
