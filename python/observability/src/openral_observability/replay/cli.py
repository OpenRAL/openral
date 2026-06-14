"""Backing logic for ``openral replay`` and ``openral record``.

The CLI surface lives in :mod:`openral_cli.main`; this module contains
the heavy lifting so the CLI module stays import-cheap. Separating it
also makes the logic unit-testable without spawning ``openral`` through
typer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from openral_observability.replay.bag_reader import read_bag
from openral_observability.replay.correlator import (
    TimelineEntry,
    build_timeline,
    list_bag_trace_ids,
)
from openral_observability.replay.trace_query import DashboardTraceClient, TraceQueryError

__all__ = [
    "RECORD_PROFILES",
    "ProfileName",
    "ReplayResult",
    "build_record_command",
    "run_replay",
]

ProfileName = Literal["slim", "full"]


# ADR-0018 §F7 — recorded topic sets. Slim is the default and is what a
# 24 h rollout can sustain on a single SSD; full adds the high-rate
# state + every camera + every event topic and needs a sized disk.
#
# Per-namespace patterns use ``rosbag2``'s ``--regex`` flag so the user
# does not have to enumerate every sensor / failure subtopic. Bare
# topics that always exist are listed verbatim.
RECORD_PROFILES: Final[dict[str, dict[str, list[str]]]] = {
    "slim": {
        "topics": [
            "/openral/candidate_action",
            "/openral/safe_action",
            "/openral/estop",
            "/openral/human_estop",
            "/openral/world_state_slow",
            "/openral/prompt",
            "/diagnostics",
        ],
        "regex": [
            r"/openral/failure/.*",
            # One compressed image stream per camera.
            r"/openral/sensors/[^/]+/compressed",
        ],
    },
    "full": {
        "topics": [
            "/openral/candidate_action",
            "/openral/safe_action",
            "/openral/estop",
            "/openral/human_estop",
            "/openral/world_state_fast",
            "/openral/world_state_slow",
            "/openral/prompt",
            "/joint_states",
            "/tf",
            "/tf_static",
            "/diagnostics",
        ],
        "regex": [
            r"/openral/failure/.*",
            r"/openral/perception/.*",
            r"/openral/sensors/.*",
        ],
    },
}


def build_record_command(
    *,
    profile: ProfileName,
    output_dir: Path,
    storage: str = "mcap",
    extra_topics: Iterable[str] = (),
    extra_regex: Iterable[str] = (),
) -> list[str]:
    """Compose the ``ros2 bag record`` argv for ``profile``.

    Returned as a list of strings — the caller is responsible for
    ``subprocess.Popen`` / ``check_call``. Splitting the argv build from
    the invocation lets tests assert on the exact command line without
    spawning ROS 2.

    Args:
        profile: ``slim`` (default for ``openral record``) or ``full``.
        output_dir: ``-o`` directory for ``ros2 bag record``. Created
            on demand by the recorder.
        storage: rosbag2 storage backend; ``mcap`` (the openral default)
            interoperates with :mod:`openral_dataset.bag` and
            :mod:`openral_observability.replay.bag_reader`.
        extra_topics: Verbatim topics to append to the profile's list.
        extra_regex: Additional regex patterns to OR into ``--regex``.

    Returns:
        argv list, ready for ``subprocess``.

    Raises:
        ValueError: When ``profile`` is not in :data:`RECORD_PROFILES`.
    """
    if profile not in RECORD_PROFILES:
        msg = f"unknown record profile {profile!r}; expected one of {sorted(RECORD_PROFILES)}"
        raise ValueError(msg)
    spec = RECORD_PROFILES[profile]
    topics = list(spec["topics"]) + [t for t in extra_topics if t]
    regexes = list(spec["regex"]) + [r for r in extra_regex if r]

    cmd: list[str] = ["ros2", "bag", "record", "-s", storage, "-o", str(output_dir)]
    if regexes:
        # rosbag2's --regex takes one combined pattern; we OR the parts.
        combined = "|".join(f"(?:{r})" for r in regexes)
        cmd.extend(["--regex", combined])
    cmd.extend(topics)
    return cmd


@dataclass(frozen=True)
class ReplayResult:
    """Output of :func:`run_replay` — both summary + the joined timeline.

    Attributes:
        trace_id: The trace_id used for the join. ``None`` when the bag
            had no ADR-0018-typed messages and the dashboard had no
            indexed traces.
        bag_trace_ids: Distinct trace_ids discovered in the bag with
            counts. Useful when the user did not pass ``--trace`` and
            the function had to auto-pick.
        timeline: Chronological list of :class:`TimelineEntry`.
        bag_path: The mcap file actually read.
    """

    trace_id: str | None
    bag_trace_ids: list[dict[str, Any]]
    timeline: list[TimelineEntry]
    bag_path: Path

    def to_json(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for ``json.dumps``."""
        return {
            "bag_path": str(self.bag_path),
            "trace_id": self.trace_id,
            "bag_trace_ids": self.bag_trace_ids,
            "timeline": [e.to_json() for e in self.timeline],
        }


def run_replay(
    *,
    bag_path: Path,
    trace_id: str | None,
    dashboard_url: str | None,
) -> ReplayResult:
    """Read ``bag_path``, fetch matching spans, return the joined timeline.

    Args:
        bag_path: Path to an mcap file or a rosbag2 directory.
        trace_id: When given, filter both bag and spans down to this
            trace. When ``None``, pick the most-frequent trace_id in the
            bag (or the dashboard's most-recent indexed trace if the
            bag has none).
        dashboard_url: Base URL of the ADR-0017 dashboard. When
            ``None``, spans are skipped — the timeline is bag-only.
    """
    bag_messages = list(read_bag(bag_path))
    bag_trace_ids = list_bag_trace_ids(bag_messages)

    if trace_id is None and bag_trace_ids:
        trace_id = bag_trace_ids[0]["trace_id"]

    spans: list[dict[str, Any]] = []
    if dashboard_url is not None:
        client = DashboardTraceClient(base_url=dashboard_url)
        if trace_id is None:
            try:
                recent = client.list_traces()
            except TraceQueryError:
                recent = []
            if recent:
                trace_id = str(recent[0]["trace_id"])
        if trace_id is not None:
            try:
                spans = client.get_spans(trace_id)
            except TraceQueryError:
                spans = []

    timeline = build_timeline(bag_messages, spans, trace_id=trace_id)
    return ReplayResult(
        trace_id=trace_id,
        bag_trace_ids=bag_trace_ids,
        timeline=timeline,
        bag_path=bag_path,
    )


def run_record(
    *,
    profile: ProfileName,
    output_dir: Path,
    storage: str = "mcap",
    extra_topics: Iterable[str] = (),
    extra_regex: Iterable[str] = (),
    dry_run: bool = False,
) -> tuple[list[str], subprocess.CompletedProcess[bytes] | None]:
    """Invoke ``ros2 bag record`` with the chosen profile.

    Returns ``(argv, completed)``. ``completed`` is ``None`` when
    ``dry_run`` is set (no subprocess fork) so callers can validate the
    composed command without needing ROS 2 sourced.

    SIGINT and SIGTERM received by the parent are forwarded to the
    child as **SIGINT** — ``rosbag2 record`` only flushes
    ``metadata.yaml`` on a clean Ctrl-C. The default
    :func:`subprocess.run` signal handling would send SIGTERM through,
    which kills the recorder hard and leaves a 0-byte bag.

    Raises:
        FileNotFoundError: When ``dry_run`` is False and ``ros2`` is
            not on PATH.
    """
    argv = build_record_command(
        profile=profile,
        output_dir=output_dir,
        storage=storage,
        extra_topics=extra_topics,
        extra_regex=extra_regex,
    )
    if dry_run:
        return argv, None
    if shutil.which("ros2") is None:
        msg = "ros2 executable not found on PATH; source install/setup.bash before `openral record`"
        raise FileNotFoundError(msg)

    import os
    import signal
    import time as _time

    # argv is built from a frozen profile dict; only opt-in extras can flow in.
    proc = subprocess.Popen(argv, start_new_session=True)

    def _forward(signum: int, _frame: object) -> None:
        # Always send SIGINT to the child group — rosbag2 needs Ctrl-C
        # semantics to flush metadata.yaml. Without start_new_session
        # + os.killpg the child would die hard on SIGTERM.
        import contextlib

        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)

    prev_int = signal.signal(signal.SIGINT, _forward)
    prev_term = signal.signal(signal.SIGTERM, _forward)
    try:
        proc.wait()
        # Give rosbag2 a beat to finish writing metadata.yaml — empirically
        # ~1.5 s is enough on a laptop SSD; we cap at 5 s.
        for _ in range(50):
            if (output_dir / "metadata.yaml").exists():
                break
            _time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
    completed: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(
        args=argv, returncode=proc.returncode
    )
    return argv, completed


def write_timeline(result: ReplayResult, out_path: Path) -> None:
    """Persist a :class:`ReplayResult` as pretty-printed JSON to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=False), encoding="utf-8")
