"""``tools/deploy_sim_batch.py``: the harness's guards (#421 review)."""

from __future__ import annotations

import importlib.util
import json
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
