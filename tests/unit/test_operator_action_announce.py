"""Operator actions are announced pseudonymously (issue #468).

`openral estop reset` (real rclpy) and the dashboard (a fake `ros2` executable at the process
boundary) announce on /openral/operator_action before they act; the operator's name or IP
never appears, only an HMAC under the deployer's key.
"""

from __future__ import annotations

import asyncio
import json
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from openral_core import pseudonymous_actor_id


def test_cli_announces_the_reset_with_a_pseudonym_only(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = pytest.importorskip("rclpy")
    msgs = pytest.importorskip("openral_msgs.msg")
    from openral_cli.estop import OPERATOR_ACTION_TOPIC, _announce_reset

    monkeypatch.setenv("OPENRAL_ACTOR_KEY", "deployer-key")
    rclpy.init()
    try:
        node = rclpy.create_node("estop_announce_test")
        got: list[Any] = []
        node.create_subscription(msgs.OperatorAction, OPERATOR_ACTION_TOPIC, got.append, 10)
        _announce_reset(node, "alice", discovery_wait_s=2.0)
        deadline = time.monotonic() + 3.0
        while not got and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        (msg,) = got
        assert (msg.action, msg.channel_label) == ("estop_reset", "cli")
        assert msg.actor_id == pseudonymous_actor_id("alice", "deployer-key")
        assert "alice" not in str(msg)
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_dashboard_announces_param_set_without_the_client_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from openral_observability.dashboard import app as dashboard

    argv_file = tmp_path / "argv.json"
    fake = tmp_path / "ros2"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr("shutil.which", lambda _: str(fake))
    monkeypatch.setenv("OPENRAL_ACTOR_KEY", "deployer-key")
    asyncio.run(dashboard._announce_operator_action("param_set", "10.0.0.7", "/n:temperature"))
    argv = json.loads(argv_file.read_text())
    assert argv[:2] == ["topic", "pub"] and "/openral/operator_action" in argv
    payload = json.loads(argv[-1])
    assert payload["actor_id"] == pseudonymous_actor_id("10.0.0.7", "deployer-key")
    assert payload["detail"] == "/n:temperature"
    assert "10.0.0.7" not in argv[-1]
