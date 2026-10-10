"""Megabyte best-effort clouds cross the real robot self-filter under the deploy's Fast DDS.

The Isaac deploy on Spark (2026-10-04) published its 2.8 MB base-frame depth cloud at
3.3 Hz and the robot self-filter emitted 1.9 Hz, with 2 s gaps; the grasp-target leg, which
needs a self-filtered cloud within 0.1 s of each depth frame, refused most captures. Cause:
Jazzy's Fast DDS gives each participant a 512 KiB shared-memory segment, and a same-host
BEST_EFFORT sample larger than that is lost almost every time (480 KB 40/40, 540 KB 3/40,
720 KB and up 0/40 on the reference laptop). ``openral deploy`` now exports a profile with
a 16 MiB segment (``apply_fastdds_large_data_profile``; ``docs/reference/dds-large-messages.md``).

This runs both lossy hops of the deploy graph with real processes and real messages: a
publisher with the sim sensor bridge's depth-cloud QoS (BEST_EFFORT, KEEP_LAST 5) →
the real colcon-built ``robot_self_filter`` (``cloud_in`` SensorDataQoS KEEP_LAST 2), posed
by the real OpenArm collision model the deploy launch builds (``_self_filter_params``) →
a subscriber with the HAL vision bridge's kept-cloud QoS (BEST_EFFORT, KEEP_LAST 1). The
cloud lies clear of the arms, so the filter keeps it whole and its output is as large as
its input. Asserted: ≥ 95 % of the clouds come out of the filter, and ``publish()`` never
blocks the producer.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + a sourced overlay with
``openral_octomap_bridge`` built. ``scripts/ros_live_tests.sh`` is the runner.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import pytest

_LIVE = os.environ.get("OPENRAL_TEST_ROS_LIVE") == "1" and bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live-ROS test — set OPENRAL_TEST_ROS_LIVE=1 with ROS 2 sourced "
    "(scripts/ros_live_tests.sh does both).",
)
if _LIVE:
    pytest.importorskip("rclpy")
    for _dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(_dep, reason=f"{_dep} is a module-level import of the launch file")

from openral_cli.deploy_sim import apply_fastdds_large_data_profile  # noqa: E402
from openral_core import RobotDescription  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    isolated_domain_id,
    terminate_kernel,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _REPO / "tools" / "cloud_transport_probe.py"
_LAUNCH = _REPO / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
#: An Isaac 512x384 depth camera's cloud as measured on Spark: ~2.8 MB of xyz float32.
_POINTS = 233_000
_MIN_DELIVERY = 0.95
#: A producer thread must never stall on a cloud; measured p99 ~4 ms on the laptop.
_MAX_PUBLISH_MS = 50.0
#: Three participants map the profile's segment each (resident in /dev/shm), plus slack.
_SHM_NEEDED_BYTES = 4 * 16 * 1024 * 1024


def _self_filter_params() -> tuple[dict[str, Any], str, list[str]]:
    """The OpenArm self-filter's parameters, built by the deploy launch's own builder."""
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", _LAUNCH)
    assert spec is not None and spec.loader is not None
    launch = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch"] = launch
    spec.loader.exec_module(launch)
    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    params: dict[str, Any] = launch._self_filter_params(
        collision_params_from_description(description),
        description,
        "/probe/joint_states",
        0.02,
    )
    root = str(params["collision_link_names"][0])
    return params, root, [str(n) for n in params["collision_joint_names"]]


def _json_line(text: str, role: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith("{") and f'"role": "{role}"' in line and '"ready"' not in line:
            result: dict[str, Any] = json.loads(line)
            return result
    raise AssertionError(f"no {role} result in probe output:\n{text}")


@pytest.mark.parametrize("hz", [4.0, 10.0])
def test_megabyte_clouds_cross_the_real_self_filter(hz: float, tmp_path: pathlib.Path) -> None:
    if shutil.which("ros2") is None:
        pytest.skip("ros2 not on PATH; source install/setup.bash first")
    prefix = subprocess.run(
        ["ros2", "pkg", "prefix", "openral_octomap_bridge"], capture_output=True, check=False
    )
    if prefix.returncode != 0:
        pytest.skip("openral_octomap_bridge is not built in the sourced overlay")
    if shutil.disk_usage("/dev/shm").free < _SHM_NEEDED_BYTES:
        pytest.skip("/dev/shm too small for three Fast DDS participants with 16 MiB segments")

    import yaml

    params, root, joints = _self_filter_params()
    params_file = tmp_path / "self_filter.yaml"
    params_file.write_text(
        yaml.safe_dump({"/**": {"ros__parameters": {k: v for k, v in params.items() if v != []}}}),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "ROS_DOMAIN_ID": str(isolated_domain_id()),
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }
    env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    apply_fastdds_large_data_profile(env)
    count = int(10 * hz)

    filter_log = tmp_path / "self_filter.log"
    with filter_log.open("wb") as log:
        self_filter = subprocess.Popen(
            [
                "ros2", "run", "openral_octomap_bridge", "robot_self_filter",
                "--ros-args", "--params-file", str(params_file),
                "-r", "cloud_in:=/probe/cloud", "-r", "cloud_out:=/probe/kept",
            ],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
    sub = subprocess.Popen(
        [
            sys.executable, str(_PROBE), "sub", "--topic", "/probe/kept",
            "--count", str(count), "--timeout-s", str(count / hz + 30.0),
        ],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )  # fmt: skip
    try:
        assert sub.stdout is not None
        while '"ready": true' not in (line := sub.stdout.readline()):
            assert line, "the probe subscriber exited before it was ready"
        pub = subprocess.run(
            [
                sys.executable, str(_PROBE), "pub", "--topic", "/probe/cloud",
                "--count", str(count), "--hz", str(hz), "--points", str(_POINTS),
                "--frame", root, "--joint-topic", "/probe/joint_states", "--joint", *joints,
            ],
            env=env, capture_output=True, text=True, timeout=count / hz + 60.0, check=False,
        )  # fmt: skip
        sub_out, _ = sub.communicate(timeout=60.0)
    finally:
        terminate_kernel(sub)
        terminate_kernel(self_filter)
    log_text = filter_log.read_text(encoding="utf-8", errors="replace")
    sent = _json_line(pub.stdout + pub.stderr, "pub")
    got = _json_line(sub_out, "sub")
    context = f"pub={sent}\nsub={got}\nself-filter log:\n{log_text[-3000:]}"

    assert sent["fastdds_profile"].endswith("fastdds_large_data.xml"), context
    assert sent["bytes"] > 2_500_000, context
    # The filter kept the whole cloud, so the second hop carried a megabyte sample too.
    assert got["bytes_max"] == sent["bytes"], context
    assert got["ratio"] >= _MIN_DELIVERY, context
    assert sent["publish_ms_p99"] < _MAX_PUBLISH_MS, context
