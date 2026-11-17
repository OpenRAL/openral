"""Nav2's bring-up waits long enough for its action servers on a loaded host.

``wait_for_service_timeout`` is 30000 ms. At 1000 ms, bt_navigator aborted the whole
Nav2 bring-up when the planner's compute_path_to_pose server came up a second late
(Spark, Isaac + pi0.5, 2026-11-15), and at 10000 ms when its "spin" client took over
10 s to discover an active behavior_server (2026-11-16): every later NavigateToPose
goal was rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CONFIG = Path(__file__).resolve().parents[2] / "packages/openral_nav2_bringup/config"


@pytest.mark.parametrize("name", ["nav2_panda_mobile.yaml", "nav2_visual.yaml"])
def test_bt_navigator_waits_thirty_seconds_for_its_servers(name: str) -> None:
    params = yaml.safe_load((_CONFIG / name).read_text())["bt_navigator"]["ros__parameters"]
    assert params["wait_for_service_timeout"] >= 30_000
