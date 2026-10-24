"""The startup orphan reap does not announce the DDS transport ready (#421 review).

``DDS_TRANSPORT_READY_MARKER`` tells the validation matrix's evidence monitor that the
reap and the SHM purge have run and ``ros2 launch`` is next. The reap built its env with
``_prepare_launch_env``, which purges and prints the marker, so the marker appeared
BEFORE the reap and a monitor could attach to the leftover graph.
"""

from __future__ import annotations

import pytest
from openral_cli import deploy_sim


def test_the_orphan_reap_prints_no_transport_ready_marker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENRAL_SKIP_ORPHAN_REAP", raising=False)
    # A domain no graph of this host runs on: the sweep is scoped by domain + checkout.
    monkeypatch.setenv("ROS_DOMAIN_ID", "231")
    deploy_sim._reap_orphans_with_log("real")
    assert deploy_sim.DDS_TRANSPORT_READY_MARKER not in capsys.readouterr().out
