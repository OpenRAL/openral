"""Replay one recorded deploy into two safety kernels and diff their verdicts (ADR-0121 parity).

Step 3 of ADR-0121 removes #349 / K2 / K5 / K7 / OT-10 from the open kernel once an OpenRAL Pro
exemption provider reproduces them. This is the deterministic half of that parity gate:

* **A**: the kernel exactly as the run configured it (the in-tree rules on).
* **B**: the same parameters with every moved in-tree rule off and the provider loaded
  (``--provider``), its ``contact_exemptions.*`` parameters mirroring the in-tree ones
  (:func:`provider_params`).

Both kernels run as real ``safety_kernel_node`` processes on two private ROS domains with
``use_sim_time``. The recorded ``/clock`` and every kernel input are published to both in
recorded order. After each ``/openral/candidate_action`` the tool waits for each kernel's verdict:
a published ``/openral/safe_action`` (its ``flat``) or none within ``--verdict-timeout-s``
(refused). It reports every candidate whose verdicts differ.

``REPLAY_TOPICS`` is the single list of what a run must record for this to work. The Spark parity
harness imports it, so the recorder and the replay cannot drift.

Usage::

    # what to record during a deploy (prints a ros2 bag record command)
    python tools/kernel_replay_diff.py record-cmd /data/run_01/replay_bag
    # after the run: dump the live kernel's parameters next to the bag
    ros2 param dump /openral_safety_kernel > /data/run_01/kernel_params.yaml
    # offline, anywhere with ROS 2 Jazzy and both packages built:
    python tools/kernel_replay_diff.py diff --bag /data/run_01/replay_bag \\
        --params /data/run_01/kernel_params.yaml \\
        --provider openral_pro_safety_exemptions/ContactExemptions --out diff.json

Not part of the runtime; CI-adjacent parity tooling only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Everything the kernel consumes, plus its verdict topics (recorded for adjudication).
REPLAY_TOPICS: tuple[str, ...] = (
    "/clock",
    "/joint_states",
    "/openral/candidate_action",
    "/openral/world_voxels",
    "/openral/world_state_fast",
    "/openral/estop",
    "/openral/safe_action",
    "/openral/safety_status",
    "/openral/failure/safety",
)

#: Kernel inputs the replay publishes (the rest of REPLAY_TOPICS are outputs).
_INPUTS: tuple[str, ...] = (
    "/clock",
    "/joint_states",
    "/openral/candidate_action",
    "/openral/world_voxels",
    "/openral/world_state_fast",
    "/openral/estop",
)

#: In-tree rule switches ADR-0121 moves to the provider, and their "off" value.
_MOVED_RULES_OFF: dict[str, object] = {
    "workspace_contact_enabled": False,
    "workspace_contact_brake_enabled": False,
    "workspace_contact_hold_max_consecutive": 0,
    "workspace_explained_cells_enabled": False,
    "tracked_grasp_exemption_enabled": False,
    "tracked_release_jaw_joints": [],
    "tracked_release_jaw_open_sign": [],
    "payload_own_cells_enabled": False,
    "pinch_payload_own_cells_enabled": False,
}

#: Parameter prefixes the provider mirrors under ``contact_exemptions.``.
_MIRRORED_PREFIXES: tuple[str, ...] = (
    "workspace_contact_",
    "workspace_explained_",
    "tracked_grasp_",
    "tracked_release_",
    "payload_own_",
    "pinch_payload_own_",
)

_NODE = "openral_safety_kernel"


def record_command(out_dir: str) -> list[str]:
    """The ``ros2 bag record`` command a parity run uses.

    Args:
        out_dir: The bag directory to create.

    Returns:
        The argv, MCAP storage, exactly ``REPLAY_TOPICS``.

    Example:
        >>> record_command("/tmp/b")[:5]
        ['ros2', 'bag', 'record', '--storage', 'mcap']
    """
    return ["ros2", "bag", "record", "--storage", "mcap", "-o", out_dir, *REPLAY_TOPICS]


def _kernel_params(dump: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ros__parameters`` block of a ``ros2 param dump`` of the kernel."""
    for key, value in dump.items():
        if key.strip("/") == _NODE and isinstance(value, Mapping):
            params = value.get("ros__parameters")
            if isinstance(params, Mapping):
                return dict(params)
    raise ValueError(f"no /{_NODE} ros__parameters block in the parameter dump")


def provider_params(params: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """Kernel B's parameters: the moved in-tree rules off, the provider on, values mirrored.

    The hand-speed measurement parameters (``workspace_contact_control_period_s``,
    ``workspace_contact_jaw_joints``) stay, because the open kernel still measures for the
    provider. The provider's mask is the in-tree workspace's hand links.

    Args:
        params: Kernel A's ``ros__parameters``.
        provider: The pluginlib class, e.g. ``openral_pro_safety_exemptions/ContactExemptions``.

    Returns:
        Kernel B's ``ros__parameters``.

    Raises:
        ValueError: A has none of the moved rules on, so there is nothing to compare.

    Example:
        >>> b = provider_params(
        ...     {"workspace_contact_enabled": True, "workspace_contact_links": ["l7"]}, "p/P"
        ... )
        >>> b["workspace_contact_enabled"], b["exemption_provider"], b["exemption_provider_links"]
        (False, 'p/P', ['l7'])
        >>> b["contact_exemptions.workspace_contact_enabled"]
        True
    """
    if not any(
        params.get(k) not in (None, False, 0, [], "") for k in _MOVED_RULES_OFF if k in params
    ):
        raise ValueError("kernel A runs none of the ADR-0121 moved rules: nothing to compare")
    out = dict(params)
    for key, value in params.items():
        if key.startswith(_MIRRORED_PREFIXES):
            out[f"contact_exemptions.{key}"] = value
    out.update(_MOVED_RULES_OFF)
    links = params.get("workspace_contact_links") or params.get("grasp_contact_links") or []
    out["exemption_provider"] = provider
    out["exemption_provider_links"] = list(links)
    return out


@dataclass
class Verdict:
    """One kernel's answer to one candidate chunk."""

    passed: bool
    flat: list[float] = field(default_factory=list)
    latched: bool = False


@dataclass
class Diff:
    """A candidate whose two verdicts differ."""

    index: int
    stamp_ns: int
    a: Verdict
    b: Verdict


def verdicts_equal(a: Verdict, b: Verdict, tol: float = 1e-9) -> bool:
    """Same pass/refuse, same latch, and (when passed) the same published rows.

    Example:
        >>> verdicts_equal(Verdict(True, [0.1]), Verdict(True, [0.1 + 1e-12]))
        True
        >>> verdicts_equal(Verdict(True, [0.1]), Verdict(False))
        False
    """
    if a.passed != b.passed or a.latched != b.latched or len(a.flat) != len(b.flat):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a.flat, b.flat, strict=True))


def summarize(diffs: list[Diff], n_candidates: int) -> dict[str, Any]:
    """The report: counts plus every differing candidate.

    Example:
        >>> summarize([], 3)["equal"]
        3
    """
    return {
        "candidates": n_candidates,
        "equal": n_candidates - len(diffs),
        "different": len(diffs),
        "a_only_passes": sum(1 for d in diffs if d.a.passed and not d.b.passed),
        "b_only_passes": sum(1 for d in diffs if d.b.passed and not d.a.passed),
        "diffs": [asdict(d) for d in diffs],
    }


def outside_noise(report: Mapping[str, Any], self_checks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Candidate indices where A and B differ but no A-vs-A run did.

    The ADR-0121 tier-2 criterion: A-vs-B differences must fall inside the replay's own
    nondeterminism, which ``--self-check`` measures by replaying A against itself.

    Example:
        >>> outside_noise({"diffs": [{"index": 3}, {"index": 9}]}, [{"diffs": [{"index": 3}]}])
        [9]
    """
    noise = {d["index"] for r in self_checks for d in r["diffs"]}
    return sorted(d["index"] for d in report["diffs"] if d["index"] not in noise)


# ── the live replay (needs ROS 2 Jazzy and the kernel package sourced) ─────────


def _read_bag(bag: Path) -> Iterator[tuple[str, bytes, int]]:
    import rosbag2_py  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay, absent from some venvs

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(_INPUTS)))
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        yield str(topic), bytes(data), int(stamp)


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def params_yaml(params: Mapping[str, Any]) -> str:
    """The kernel's ``--params-file`` text for ``params``.

    Empty lists are omitted: an empty YAML list carries no type and rclcpp refuses it, and
    every kernel array parameter but the (never empty) force-observer gravity defaults to
    empty, so omitting one is its default. No anchors or aliases: a mirrored
    ``contact_exemptions.<x>`` list is the same object as ``<x>``, and rcl refuses an
    aliased file ("Will not support aliasing").

    Example:
        >>> jaws = ["finger_joint"]
        >>> text = params_yaml({"a": jaws, "contact_exemptions.a": jaws, "e": []})
        >>> "&" in text or "*" in text, "e:" in text
        (False, False)
    """
    typed = {k: v for k, v in params.items() if v != []}
    return yaml.dump({f"/{_NODE}": {"ros__parameters": typed}}, Dumper=_NoAliasDumper)


def _spawn_kernel(params: Mapping[str, Any], domain: int, workdir: Path) -> subprocess.Popen[bytes]:
    path = workdir / f"kernel_{domain}.yaml"
    path.write_text(params_yaml(params))
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    log = (workdir / f"kernel_{domain}.log").open("wb")
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "openral_safety_kernel",
            "safety_kernel_node",
            "--ros-args",
            "--params-file",
            str(path),
            "-p",
            "use_sim_time:=true",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        # Its own process group: `ros2 run` forks the node, and terminating the wrapper alone
        # leaves the kernel running (and answering on the domain) after the replay.
        start_new_session=True,
    )


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the kernel's whole process group, SIGKILL it if it lingers.

    Waiting on ``proc`` only reaps the ``ros2 run`` wrapper; the kernel it spawned is a
    separate group member, so the group itself is polled until it is empty.
    """
    import signal

    def group_gone(within_s: float) -> bool:
        deadline = time.monotonic() + within_s
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return True
            proc.poll()
            time.sleep(0.05)
        return False

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if not group_gone(10.0):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        group_gone(10.0)
    proc.wait(timeout=10)


class _Side:
    """One kernel's ROS context: publishers for its inputs, subscriptions on its verdicts."""

    def __init__(self, domain: int, types: Mapping[str, Any]) -> None:
        import rclpy  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay, absent from some venvs
        from openral_msgs.msg import (
            ActionChunk,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
        )
        from rclpy.qos import (  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import (
            Empty,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
        )

        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = rclpy.create_node(f"kernel_replay_{domain}", context=self.ctx)
        self.executor = rclpy.executors.SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pubs = {
            t: self.node.create_publisher(types[t], t, best if t == "/joint_states" else reliable)
            for t in _INPUTS
        }
        self.safe: list[list[float]] = []
        self.latched = False
        self.node.create_subscription(
            ActionChunk,
            "/openral/safe_action",
            lambda m: self.safe.append([float(x) for x in m.flat]),
            reliable,
        )
        self.node.create_subscription(Empty, "/openral/estop", lambda _m: self._latch(), reliable)

    def _latch(self) -> None:
        self.latched = True

    def activate(self, timeout_s: float) -> None:
        """CONFIGURE then ACTIVATE, each confirmed by ``get_state``.

        A Jazzy ``change_state`` reply can be dropped while the transition lands
        (``tools/lifecycle_autostart.py``), so the FSM state, not the reply, is the evidence.
        """
        from lifecycle_msgs.srv import (  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
            ChangeState,
            GetState,
        )

        change = self.node.create_client(ChangeState, f"/{_NODE}/change_state")
        get = self.node.create_client(GetState, f"/{_NODE}/get_state")
        if not change.wait_for_service(timeout_sec=timeout_s) or not get.wait_for_service(
            timeout_sec=timeout_s
        ):
            raise RuntimeError("the kernel's lifecycle services never appeared")

        def state() -> str:
            future = get.call_async(GetState.Request())
            self.executor.spin_until_future_complete(future, timeout_sec=5.0)
            result = future.result()
            return str(result.current_state.label) if result is not None else ""

        for transition, goal in ((1, "inactive"), (3, "active")):  # CONFIGURE, ACTIVATE
            for _attempt in range(3):
                if state() == goal:
                    break
                req = ChangeState.Request()
                req.transition.id = transition
                future = change.call_async(req)
                self.executor.spin_until_future_complete(future, timeout_sec=timeout_s / 3)
            if state() != goal:
                raise RuntimeError(f"the kernel never reached '{goal}'")

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.002)

    def close(self) -> None:
        import rclpy  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay

        self.node.destroy_node()
        rclpy.shutdown(context=self.ctx)


#: The deploy graph's Fast DDS profile (``openral_cli.deploy_sim.FASTDDS_LARGE_DATA_PROFILE``):
#: under the default 512 KiB shared-memory segment a multi-MB voxel grid is mostly lost.
_FASTDDS_PROFILE = (
    Path(__file__).resolve().parent.parent / "python/cli/src/openral_cli/fastdds_large_data.xml"
)


def replay(
    bag: Path,
    params_a: Mapping[str, Any],
    params_b: Mapping[str, Any],
    workdir: Path,
    verdict_timeout_s: float = 0.3,
    settle_s: float = 0.02,
) -> dict[str, Any]:
    """Run the two-kernel replay and return :func:`summarize`'s report.

    Delivery is made deterministic, as far as the kernel's KEEP_LAST(1) inputs allow: the
    deploy graph's Fast DDS profile (set before the first participant, inherited by both
    kernels), every reliable input acknowledged by both kernels before the next message, and
    ``settle_s`` of spinning before each candidate so its inputs' callbacks run first (the
    executor does not order callbacks across subscriptions). ``--self-check`` measures what
    nondeterminism is left.
    """
    os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(_FASTDDS_PROFILE))
    from rclpy.duration import (
        Duration,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
    )
    from rclpy.serialization import (
        deserialize_message,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
    )
    from rosidl_runtime_py.utilities import (
        get_message,  # type: ignore[import-not-found,unused-ignore]  # reason: ROS 2 overlay
    )

    type_names = {
        "/clock": "rosgraph_msgs/msg/Clock",
        "/joint_states": "sensor_msgs/msg/JointState",
        "/openral/candidate_action": "openral_msgs/msg/ActionChunk",
        "/openral/world_voxels": "openral_msgs/msg/OccupancyVoxels",
        "/openral/world_state_fast": "openral_msgs/msg/WorldStateStamped",
        "/openral/estop": "std_msgs/msg/Empty",
        "/openral/safe_action": "openral_msgs/msg/ActionChunk",
    }
    types = {t: get_message(n) for t, n in type_names.items()}
    workdir.mkdir(parents=True, exist_ok=True)
    procs = [_spawn_kernel(params_a, 71, workdir), _spawn_kernel(params_b, 72, workdir)]
    sides = [_Side(71, types), _Side(72, types)]
    diffs: list[Diff] = []
    n = 0
    try:
        for side in sides:
            side.activate(timeout_s=60.0)
        for topic, data, stamp in _read_bag(bag):
            msg = deserialize_message(data, types[topic])
            if topic == "/openral/candidate_action":
                for side in sides:
                    side.spin(settle_s)
            for side in sides:
                side.pubs[topic].publish(msg)
            if topic != "/openral/candidate_action":
                for side in sides:
                    if topic != "/joint_states":  # the reliable inputs: both kernels have it
                        side.pubs[topic].wait_for_all_acked(Duration(seconds=2.0))
                    side.spin(0.003)
                continue
            before = [len(s.safe) for s in sides]
            end = time.monotonic() + verdict_timeout_s
            while time.monotonic() < end and any(
                len(s.safe) == b for s, b in zip(sides, before, strict=True)
            ):
                for side in sides:
                    side.spin(0.005)
            got = [
                Verdict(len(s.safe) > b, s.safe[-1] if len(s.safe) > b else [], s.latched)
                for s, b in zip(sides, before, strict=True)
            ]
            if not verdicts_equal(got[0], got[1]):
                diffs.append(Diff(n, stamp, got[0], got[1]))
            n += 1
    finally:
        for side in sides:
            side.close()
        for p in procs:
            _stop(p)
    return summarize(diffs, n)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``record-cmd`` or ``diff``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record-cmd", help="print the ros2 bag record command")
    rec.add_argument("out_dir")
    diff = sub.add_parser("diff", help="replay a bag into kernels A and B")
    diff.add_argument("--bag", type=Path, required=True)
    diff.add_argument("--params", type=Path, required=True, help="ros2 param dump of kernel A")
    diff.add_argument("--provider", required=True)
    diff.add_argument("--out", type=Path, required=True)
    diff.add_argument("--verdict-timeout-s", type=float, default=0.3)
    diff.add_argument("--settle-s", type=float, default=0.02)
    diff.add_argument(
        "--self-check",
        type=int,
        default=0,
        metavar="K",
        help="first replay A against itself K times; pass iff every A-vs-B diff is in that noise",
    )
    args = parser.parse_args(argv)
    if args.cmd == "record-cmd":
        print(" ".join(record_command(args.out_dir)))
        return 0
    params_a = _kernel_params(yaml.safe_load(args.params.read_text()))
    params_b = provider_params(params_a, args.provider)
    workdir = args.out.parent
    workdir.mkdir(parents=True, exist_ok=True)
    checks = [
        replay(args.bag, params_a, dict(params_a), workdir, args.verdict_timeout_s, args.settle_s)
        for _ in range(args.self_check)
    ]
    report = replay(args.bag, params_a, params_b, workdir, args.verdict_timeout_s, args.settle_s)
    report["self_check"] = [{"different": c["different"], "diffs": c["diffs"]} for c in checks]
    report["outside_noise"] = outside_noise(report, checks)
    args.out.write_text(json.dumps(report, indent=2))
    print(
        f"candidates={report['candidates']} equal={report['equal']} "
        f"different={report['different']} a_only={report['a_only_passes']} "
        f"b_only={report['b_only_passes']} self_check={[c['different'] for c in checks]} "
        f"outside_noise={report['outside_noise']}"
    )
    return 0 if not report["outside_noise"] else 1


if __name__ == "__main__":
    sys.exit(main())
