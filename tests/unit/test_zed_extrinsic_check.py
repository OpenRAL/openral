"""``tools/zed_extrinsic_check.py`` recovers a head-camera extrinsic from a real rosbag2 bag.

The bag is written here with the real ``rosbag2_py`` writer and real ``sensor_msgs`` /
``tf2_msgs`` messages: a table plus two flat markers, seen from a known TRUE mount pose
through a ZED-style internal transform (``zed_camera_link -> zed_left_camera_frame``). The
manifest under test is the committed ``robots/openarm/robot.yaml`` with its ``head_zed`` pose
swapped: the robot manifest is the only place a robot sensor's mount lives.

What is pinned: the true pose passes; a wrong pose fails and its suggestion recovers the
true one; and ``verify`` refuses a missing report,
a report for a different pose, and a report checked at looser criteria — including the
committed default state, where no calibration report exists yet — and the run script refuses
to launch without a verified report.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — rosbag2_py / tf2_ros need a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
_REPORT = _REPO_ROOT / "robots" / "openarm" / "calibration" / "head_zed_extrinsic.json"
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import zed_extrinsic_check as zc  # noqa: E402

_TRUE_POSE = (0.02, -0.01, 0.22, 0.01, 0.80, 0.02)
_INTERNAL_XYZ = (0.0, 0.03, 0.015)  # zed_camera_link -> left camera, identity rotation
_TABLE_Z = -0.20
_ROI = (0.25, 0.75, -0.35, 0.35)
_MARKERS = ((0.45, 0.15), (0.55, -0.15))
_CLOUD_TOPIC = "/zed/zed_node/point_cloud/cloud_registered"


def _world_points() -> np.ndarray:
    """Table (5 mm grid) plus two 8 cm square boards whose tops sit 20 mm above it."""
    xs, ys = np.meshgrid(np.arange(0.25, 0.75, 0.005), np.arange(-0.35, 0.35, 0.005))
    table = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, _TABLE_Z)], axis=1)
    boards = []
    for mx, my in _MARKERS:
        bx, by = np.meshgrid(np.arange(-0.04, 0.04, 0.004), np.arange(-0.04, 0.04, 0.004))
        boards.append(
            np.stack([bx.ravel() + mx, by.ravel() + my, np.full(bx.size, _TABLE_Z + 0.02)], 1)
        )
    pts = np.concatenate([table, *boards])
    return pts + np.random.default_rng(0).normal(0.0, 0.001, pts.shape)


def _write_bag(bag: Path) -> None:
    import rosbag2_py
    from geometry_msgs.msg import TransformStamped
    from rclpy.serialization import serialize_message
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage

    x, y, z, roll, pitch, yaw = _TRUE_POSE
    rot = zc._rpy_to_matrix(roll, pitch, yaw)
    cam = (_world_points() - np.array([x, y, z])) @ rot  # base -> zed_camera_link
    cam -= np.array(_INTERNAL_XYZ)  # -> zed_left_camera_frame

    def tf(parent: str, child: str, xyz: tuple[float, float, float]) -> TransformStamped:
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        t.transform.rotation.w = 1.0
        return t

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (name, typ) in enumerate(
        (("/tf_static", "tf2_msgs/msg/TFMessage"), (_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2"))
    ):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, typ, "cdr"))
    static_tfs = TFMessage(
        transforms=[tf("zed_camera_link", "zed_left_camera_frame", _INTERNAL_XYZ)]
    )
    writer.write("/tf_static", serialize_message(static_tfs), 0)
    header = Header(frame_id="zed_left_camera_frame")
    for k in range(3):
        writer.write(
            _CLOUD_TOPIC, serialize_message(create_cloud_xyz32(header, cam)), (k + 1) * 10**8
        )
    del writer


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("zed") / "bag"
    _write_bag(path)
    return path


def _robot_with_pose(tmp_path: Path, pose: list[float]) -> Path:
    data = yaml.safe_load(_ROBOT.read_text(encoding="utf-8"))
    (entry,) = [s for s in data["sensors"] if s["name"] == "head_zed"]
    entry["static_transform_xyz_rpy"] = pose
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _check(robot: Path, bag: Path, out: Path) -> int:
    argv = ["check", "--robot", str(robot), "--bag", str(bag), "--cloud-topic", _CLOUD_TOPIC]
    argv += ["--table-z", str(_TABLE_Z), "--table-roi", *map(str, _ROI)]
    argv += ["--stride", "1", "--out", str(out)]
    for mx, my in _MARKERS:
        argv += ["--marker", str(mx), str(my)]
    return zc.main(argv)


def test_the_true_pose_passes_and_verifies(bag: Path, tmp_path: Path) -> None:
    robot, out = _robot_with_pose(tmp_path, list(_TRUE_POSE)), tmp_path / "r.json"
    assert _check(robot, bag, out) == 0
    res = json.loads(out.read_text())["residuals"]
    assert res["tilt_deg"] < 0.1
    assert abs(res["height_err_m"]) < 0.002
    assert all(m["error_m"] < 0.003 for m in res["markers"])
    assert zc.main(["verify", "--robot", str(robot), "--report", str(out)]) == 0


def test_a_wrong_pose_fails_and_the_suggestion_recovers_the_true_one(
    bag: Path, tmp_path: Path
) -> None:
    x, y, z, roll, pitch, yaw = _TRUE_POSE
    wrong = [x + 0.012, y - 0.008, z + 0.015, roll - 0.02, pitch + math.radians(2.0), yaw + 0.025]
    robot, out = _robot_with_pose(tmp_path, wrong), tmp_path / "r.json"
    assert _check(robot, bag, out) == 1
    report = json.loads(out.read_text())
    assert report["failures"]
    suggested = report["suggested_static_transform_xyz_rpy"]
    assert np.allclose(suggested[:3], _TRUE_POSE[:3], atol=0.003), suggested
    assert np.allclose(suggested[3:], _TRUE_POSE[3:], atol=math.radians(0.3)), suggested

    fixed = _robot_with_pose(tmp_path, suggested)
    assert _check(fixed, bag, tmp_path / "fixed.json") == 0


def test_verify_refuses_a_report_for_another_pose_or_looser_criteria(
    bag: Path, tmp_path: Path
) -> None:
    robot, out = _robot_with_pose(tmp_path, list(_TRUE_POSE)), tmp_path / "r.json"
    assert _check(robot, bag, out) == 0

    moved = tmp_path / "moved"
    moved.mkdir()
    edited = _robot_with_pose(moved, [*_TRUE_POSE[:5], _TRUE_POSE[5] + 0.001])
    assert zc.main(["verify", "--robot", str(edited), "--report", str(out)]) == 1

    report = json.loads(out.read_text())
    report["criteria"]["max_height_err_m"] = 0.05
    loose = tmp_path / "loose.json"
    loose.write_text(json.dumps(report))
    assert zc.main(["verify", "--robot", str(robot), "--report", str(loose)]) == 1

    assert zc.main(["verify", "--robot", str(robot), "--report", str(tmp_path / "no")]) == 1


def test_the_committed_manifest_is_refused_until_a_calibration_is_committed() -> None:
    """Fail-closed by default: the shipped pose is a placeholder with no passing report."""
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    rc = zc.main(["verify", "--robot", str(_ROBOT), "--report", str(_REPORT)])
    assert rc == 1


def test_the_run_script_refuses_without_a_verified_extrinsic() -> None:
    """Both gates exported, still refused: no report verifies the manifest's pose.

    The verify step runs before the interactive-terminal check, so this reaches it with
    stdin closed; were the report to verify, the terminal check would still refuse.
    """
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    env = {**os.environ, "OPENRAL_OPENARM_ALLOW_MOTION": "1", "OPENRAL_OPENARM_ATTENDED": "1"}
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "head_zed extrinsic not verified for the manifest's pose" in proc.stderr


def test_the_run_script_refuses_without_both_motion_gates() -> None:
    """The first guard: no gate exported, nothing past it runs."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENRAL_OPENARM_ALLOW_MOTION", "OPENRAL_OPENARM_ATTENDED")
    }
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2
    assert "OPENRAL_OPENARM_ALLOW_MOTION" in proc.stderr


def test_nan_cannot_open_the_gate(bag: Path, tmp_path: Path) -> None:
    """NaN compares False against every limit, so it must never read as a pass.

    ``check`` refuses a non-finite limit at the command line, and ``verify`` re-derives
    the verdict from the stored residuals against its own limits, so a report edited to
    ``passed: true`` with NaN criteria or residuals is refused.
    """
    robot, out = _robot_with_pose(tmp_path, list(_TRUE_POSE)), tmp_path / "r.json"
    argv = ["check", "--robot", str(robot), "--bag", str(bag), "--cloud-topic", _CLOUD_TOPIC]
    argv += ["--table-z", str(_TABLE_Z), "--table-roi", *map(str, _ROI), "--max-tilt-deg", "nan"]
    with pytest.raises(SystemExit):
        zc.main(argv)

    assert _check(robot, bag, out) == 0
    verify = ["verify", "--robot", str(robot), "--report", str(out)]
    good = json.loads(out.read_text())

    for mutate in (
        lambda r: r["criteria"].__setitem__("max_tilt_deg", float("nan")),
        lambda r: r["residuals"].__setitem__("tilt_deg", float("nan")),
        lambda r: r["residuals"]["markers"][0].__setitem__("error_m", float("nan")),
        lambda r: r["residuals"].__setitem__("height_err_m", 1.0),  # stored verdict still true
    ):
        report = json.loads(json.dumps(good))
        mutate(report)
        assert report["passed"] is True
        out.write_text(json.dumps(report))
        assert zc.main(verify) == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--config", "scenes/deploy/openarm_tabletop.yaml"],
        ["--no-enable-octomap-kernel-check"],
        ["--hal", "viewer_enabled=false"],
    ],
)
def test_the_run_script_refuses_args_that_could_change_the_verified_graph(
    extra: list[str],
) -> None:
    """Only observability flags pass through; the scene and safety posture are fixed."""
    script = _REPO_ROOT / "tools" / "openarm_world_voxel_run.sh"
    proc = subprocess.run(
        ["bash", str(script), *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2
    assert "is not allowed here" in proc.stderr


def test_the_run_script_lets_observability_flags_through_to_the_gates() -> None:
    """An allowed flag passes the allow-list and stops at the first gate instead."""
    script = _REPO_ROOT / "tools" / "openarm_world_voxel_run.sh"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENRAL_OPENARM_ALLOW_MOTION", "OPENRAL_OPENARM_ATTENDED")
    }
    proc = subprocess.run(
        ["bash", str(script), "--foxglove", "--dataset-out", "/tmp/x"],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2
    assert "OPENRAL_OPENARM_ALLOW_MOTION" in proc.stderr
