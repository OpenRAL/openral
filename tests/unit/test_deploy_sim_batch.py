"""``tools/deploy_sim_batch.py``: the harness's guards (#421 review)."""

from __future__ import annotations

import importlib.util
import json
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "deploy_sim_batch.py"
_SCENE = _REPO / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_sim_batch_under_test", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_resolved_domain_is_written_into_the_episode_env() -> None:
    """``graph_leaks`` greps ``ROS_DOMAIN_ID=<d>`` in each child's environ: an unset
    variable made every leak invisible."""
    env, domain = _tool().batch_env({"PATH": "/bin"})
    assert domain == "0" and env["ROS_DOMAIN_ID"] == "0"
    env, domain = _tool().batch_env({"ROS_DOMAIN_ID": "42"})
    assert domain == "42" and env["ROS_DOMAIN_ID"] == "42"


def test_refusing_a_running_sidecar_leaves_it_running(tmp_path: Path) -> None:
    """The refusal used to return inside the try, whose finally ended the foreign sidecar."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # A process whose argv is what `sidecar_pids` matches (the sidecar's own command line).
    foreign = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "isaac_sidecar.py",
            "--port",
            str(port),
        ]
    )
    try:
        tool = _tool()
        args = tool.parser().parse_args(
            [
                "--scene", str(_SCENE), "--out", str(tmp_path / "batch"),
                "--sidecar-port", str(port), "--rskill-id", "x", "--prompt", "p",
            ]
        )  # fmt: skip
        assert tool.sidecar_pids(port) == [foreign.pid]
        assert tool.run_batch(args) == 2
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait()


def test_compare_reports_an_episode_that_never_booted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed boot writes no ground_truth.json; compare used to raise FileNotFoundError."""
    for name, boot in (("a", None), ("b", 12.0)):
        d = tmp_path / name
        (d / "ep0_seed1").mkdir(parents=True)
        ep: dict[str, Any] = {"episode": 0, "seed": 1, "boot_s": boot, "total_s": 30.0, "goal": {}}
        (d / "batch.json").write_text(json.dumps({"episodes": [ep]}))
    (tmp_path / "b" / "ep0_seed1" / "ground_truth.json").write_text("{}")
    assert _tool().compare(tmp_path / "a", tmp_path / "b") == 0
    out = capsys.readouterr().out
    assert "ep0: n/a (no ground truth)" in out and "wall saved per episode" in out


def test_expect_preload_can_be_turned_off() -> None:
    base = [
        "--scene",
        "s",
        "--out",
        "o",
        "--sidecar-port",
        "1",
        "--rskill-id",
        "r",
        "--prompt",
        "p",
    ]
    p = _tool().parser()
    assert p.parse_args(base).expect_preload is True
    assert p.parse_args([*base, "--no-expect-preload"]).expect_preload is False


def test_record_is_opt_in_and_bags_exactly_the_replay_topics() -> None:
    """``--record`` feeds ``kernel_replay_diff diff`` (ADR-0121); off by default."""
    tool = _tool()
    base = ["--scene", "s", "--out", "o", "--sidecar-port", "1", "--rskill-id", "r"]
    assert not tool.parser().parse_args([*base, "--prompt", "p"]).record
    assert tool.parser().parse_args([*base, "--prompt", "p", "--record"]).record
    sys.path.insert(0, str(_TOOL.parent))
    try:
        from kernel_replay_diff import (  # type: ignore[import-not-found,unused-ignore]  # reason: sibling tool
            REPLAY_TOPICS,
            record_command,
        )
    finally:
        sys.path.remove(str(_TOOL.parent))
    argv = record_command("/tmp/x/bag")
    assert argv[-len(REPLAY_TOPICS) :] == list(REPLAY_TOPICS)
    assert "/openral/candidate_action" in REPLAY_TOPICS


def test_an_inherited_ignored_sigint_is_caught_again_for_the_children() -> None:
    """Spark 2026-10-25: a background harness SIGKILLed every unfinalized bag recorder."""
    code = (
        "import signal, subprocess, sys;"
        f"sys.path.insert(0, {str(_TOOL.parent)!r});"
        "import deploy_sim_batch as d;"
        "print(d.restore_sigint(), d.restore_sigint());"
        "print(subprocess.run(['grep', 'SigIgn', '/proc/self/status'],"
        " capture_output=True, text=True).stdout.split()[1])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[:2] == ["True", "False"]
    assert int(out[2], 16) & (1 << (signal.SIGINT - 1)) == 0, "the child's SIGINT is not ignored"


# Spark r2 master_dr seed 2, verbatim: the banana leaned on the bowl's rim for 0.13 s.
_GRADE_LINES = (
    '[lifecycle_node.py-7] sim.placement_grade {"goal_active": true, "grade": "complete", '
    '"previous": "fail", "prompt": "pick up the banana and place it in the bowl", "scene_id": '
    '"isaac_sim", "score": 1.0, "sim_time_ns": 60150003137, "sim_time_s": 60.15, "step": 877, '
    '"task_id": "isaac_sim/_hal_deploy_noop"}\n'
    "[lifecycle_node.py-7] 2026-10-25T07:27:12.624576Z [info     ] sim.placement_grade            "
    "goal_active=True grade=fail previous=complete sim_time_s=60.283\n"
    '[lifecycle_node.py-7] sim.placement_grade {"goal_active": true, "grade": "fail", '
    '"previous": "complete", "sim_time_s": 60.283}\n'
)


def test_a_transient_complete_never_ends_the_goal() -> None:
    """--end-on-grade needs complete held for the settle time; a rim-lean pass is not success."""
    tool = _tool()
    events = tool.grade_events(_GRADE_LINES)
    assert events == [("complete", 60.15), ("fail", 60.283)], "only the JSON lines, in order"
    settle = tool.GradeSettle(settle_s=1.0)
    for grade, t in events:
        settle.feed(grade, t)
    assert not settle.due(75.0)
    settle.feed("complete", 80.0)
    assert not settle.due(80.9)
    assert settle.due(81.0)


def test_end_on_grade_is_opt_in() -> None:
    tool = _tool()
    base = [
        "--scene",
        "s",
        "--out",
        "o",
        "--sidecar-port",
        "1",
        "--rskill-id",
        "r",
        "--prompt",
        "p",
    ]
    args = tool.parser().parse_args(base)
    assert not args.end_on_grade and args.grade_settle_s == 1.0
    assert tool.parser().parse_args([*base, "--end-on-grade"]).end_on_grade
