"""``tools/depth_extrinsic_check.py`` recovers a depth-camera extrinsic from a real rosbag2 bag.

The bag is written here with the real ``rosbag2_py`` writer and real ``sensor_msgs`` /
``tf2_msgs`` messages: a table plus two flat markers, seen from a known TRUE mount pose
through a driver-style internal transform (``<frame_id> -> <frame_id>_cloud``). The manifest
under test is a committed ``robots/<id>/robot.yaml`` with the sensor's pose swapped, and, for
``--unit``, a unit overlay carrying the calibrated pose (a report never crosses units).

OpenArm's ``head_zed`` is parented to the base frame. G1's ``head`` (on ``torso_link``, behind
the waist joints) and SO-101's ``wrist`` (on ``gripper``) are not: their bags carry the
``base_frame -> parent_frame`` chain on ``/tf`` the way robot_state_publisher records it, and
both manifests are turned into depth cameras here, because in-tree they are RGB-only — which
the tool must refuse.

What is pinned: the true pose passes; a wrong pose fails and its suggestion (in
``parent_frame``) recovers the true one; a parent that moves mid-recording is refused;
``verify`` refuses a missing report, a report for a different pose, and a report checked at
looser criteria — including the committed default state, where no calibration report exists
yet — and the run script refuses to launch without a verified report.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
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
_REPORT = _REPO_ROOT / "robots" / "openarm" / "calibration" / "thor" / "head_zed_extrinsic.json"
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import depth_extrinsic_check as zc  # noqa: E402

_TRUE_POSE = (0.02, -0.01, 0.22, 0.01, 0.80, 0.02)
_INTERNAL_XYZ = (0.0, 0.03, 0.015)  # <frame_id> -> <frame_id>_cloud, identity rotation
_TABLE_Z = -0.20
_ROI = (0.25, 0.75, -0.35, 0.35)
_MARKERS = ((0.45, 0.15), (0.55, -0.15))
_CLOUD_TOPIC = "/camera/depth/points"

Chain = tuple[tuple[str, str, tuple[float, ...]], ...]  # (parent, child, xyz + rpy)


@dataclass(frozen=True)
class _Rig:
    """One robot's depth camera: manifest, sensor, and the recorded base -> parent chain."""

    robot: str
    sensor: str
    frame_id: str
    true_pose: tuple[float, ...]
    chain: Chain = ()  # empty: parent_frame IS the base frame


_OPENARM = _Rig("openarm", "head_zed", "zed_camera_link", _TRUE_POSE)
# Waist yawed and the torso pitched forward: torso_link is neither fixed nor z-up.
_G1 = _Rig(
    "g1",
    "head",
    "head_camera",
    (0.08, 0.0, 0.42, 0.0, 0.95, 0.0),
    (
        ("pelvis", "waist_yaw_link", (0.0, 0.0, 0.0, 0.0, 0.0, 0.3)),
        ("waist_yaw_link", "torso_link", (0.0039, 0.0, 0.054, 0.05, 0.25, 0.0)),
    ),
)
# The arm reaching forward with the gripper pitched down at the table.
_SO101 = _Rig(
    "so101_follower",
    "wrist",
    "wrist_camera",
    (0.0, 0.06, 0.035, 3.1, -0.2, 0.1),
    (
        ("base", "shoulder", (0.0, 0.0, 0.10, 0.0, 0.0, 0.2)),
        ("shoulder", "gripper", (0.30, 0.0, 0.25, 0.1, 1.3, -0.1)),
    ),
)


def _matrix(xyz_rpy: tuple[float, ...]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = zc._rpy_to_matrix(*xyz_rpy[3:])
    m[:3, 3] = xyz_rpy[:3]
    return m


def _quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


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


def _write_bag(bag: Path, rig: _Rig = _OPENARM, *, sway_rad: float = 0.0) -> None:
    """Clouds of the table seen from ``rig.true_pose``, plus the rig's TF.

    ``sway_rad`` turns the last chain link a little more at every ``/tf`` sample: a
    parent that moves while recording.
    """
    import rosbag2_py
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped
    from rclpy.serialization import serialize_message
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage

    base_parent = np.eye(4)
    for _, _, xyz_rpy in rig.chain:
        base_parent = base_parent @ _matrix(xyz_rpy)
    cam_in_base = base_parent @ _matrix(rig.true_pose)
    rot, trans = cam_in_base[:3, :3], cam_in_base[:3, 3]
    cam = (_world_points() - trans) @ rot  # base -> <frame_id>
    cam -= np.array(_INTERNAL_XYZ)  # -> <frame_id>_cloud

    def tf(parent: str, child: str, xyz_rpy: tuple[float, ...], ns: int) -> TransformStamped:
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.header.stamp = Time(sec=ns // 10**9, nanosec=ns % 10**9)
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz_rpy[
            :3
        ]
        q = _quat(*xyz_rpy[3:])
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = q
        return t

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (name, typ) in enumerate(
        (
            ("/tf_static", "tf2_msgs/msg/TFMessage"),
            ("/tf", "tf2_msgs/msg/TFMessage"),
            (_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2"),
        )
    ):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, typ, "cdr"))
    internal = (*_INTERNAL_XYZ, 0.0, 0.0, 0.0)
    static = TFMessage(transforms=[tf(rig.frame_id, f"{rig.frame_id}_cloud", internal, 0)])
    writer.write("/tf_static", serialize_message(static), 0)
    # robot_state_publisher's /tf at 20 Hz around the clouds (joint links are dynamic).
    for k in range(11):
        ns = k * 5 * 10**7
        links = [
            tf(p, c, (*xyz_rpy[:4], xyz_rpy[4] + (k * sway_rad if j else 0.0), xyz_rpy[5]), ns)
            for j, (p, c, xyz_rpy) in enumerate(rig.chain)
        ]
        if links:
            writer.write("/tf", serialize_message(TFMessage(transforms=links)), ns)
    for k in range(3):
        ns = (k + 1) * 10**8
        header = Header(frame_id=f"{rig.frame_id}_cloud", stamp=Time(nanosec=ns))
        writer.write(_CLOUD_TOPIC, serialize_message(create_cloud_xyz32(header, cam)), ns)
    del writer


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("zed") / "bag"
    _write_bag(path)
    return path


def _robot_with_pose(tmp_path: Path, pose: list[float], rig: _Rig = _OPENARM) -> Path:
    """The committed manifest with ``rig.sensor`` at ``pose`` (made a depth camera if RGB)."""
    robot = _REPO_ROOT / "robots" / rig.robot / "robot.yaml"
    data = yaml.safe_load(robot.read_text(encoding="utf-8"))
    (entry,) = [s for s in data["sensors"] if s["name"] == rig.sensor]
    entry["modality"] = "depth"
    entry["static_transform_xyz_rpy"] = pose
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _check(robot: Path, bag: Path, out: Path, sensor: str = "head_zed") -> int:
    argv = ["check", "--robot", str(robot), "--sensor", sensor, "--bag", str(bag)]
    argv += ["--cloud-topic", _CLOUD_TOPIC, "--table-z", str(_TABLE_Z)]
    argv += ["--table-roi", *map(str, _ROI), "--stride", "1", "--out", str(out)]
    for mx, my in _MARKERS:
        argv += ["--marker", str(mx), str(my)]
    return zc.main(argv)


def _verify(robot: Path, report: Path, sensor: str = "head_zed", unit: str | None = None) -> int:
    argv = ["verify", "--robot", str(robot), "--sensor", sensor, "--report", str(report)]
    return zc.main([*argv, "--unit", unit] if unit else argv)


def test_the_true_pose_passes_and_verifies(bag: Path, tmp_path: Path) -> None:
    robot, out = _robot_with_pose(tmp_path, list(_TRUE_POSE)), tmp_path / "r.json"
    assert _check(robot, bag, out) == 0
    res = json.loads(out.read_text())["residuals"]
    assert res["tilt_deg"] < 0.1
    assert abs(res["height_err_m"]) < 0.002
    assert all(m["error_m"] < 0.003 for m in res["markers"])
    assert _verify(robot, out) == 0


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
    assert _verify(edited, out) == 1

    report = json.loads(out.read_text())
    report["criteria"]["max_height_err_m"] = 0.05
    loose = tmp_path / "loose.json"
    loose.write_text(json.dumps(report))
    assert _verify(robot, loose) == 1

    assert _verify(robot, tmp_path / "no") == 1


def test_the_committed_manifest_is_refused_until_a_calibration_is_committed() -> None:
    """Fail-closed by default: the shipped pose is a placeholder with no passing report."""
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    rc = _verify(_ROBOT, _REPORT, unit="thor")
    assert rc == 1


def test_the_run_script_refuses_without_a_verified_extrinsic() -> None:
    """Both gates exported, still refused: no report verifies the manifest's pose.

    The verify step runs before the interactive-terminal check, so this reaches it with
    stdin closed; were the report to verify, the terminal check would still refuse.
    """
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    env = {
        **os.environ,
        "OPENRAL_OPENARM_ALLOW_MOTION": "1",
        "OPENRAL_OPENARM_ATTENDED": "1",
        "OPENRAL_ROBOT_UNIT": "thor",
    }
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "head_zed extrinsic not verified for unit thor's pose" in proc.stderr


def test_the_run_script_refuses_without_a_robot_unit() -> None:
    """Both motion gates set but no unit: the ZED pose is per cell, so nothing is verified."""
    env = {k: v for k, v in os.environ.items() if k != "OPENRAL_ROBOT_UNIT"}
    env |= {"OPENRAL_OPENARM_ALLOW_MOTION": "1", "OPENRAL_OPENARM_ATTENDED": "1"}
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2
    assert "OPENRAL_ROBOT_UNIT is not set" in proc.stderr


def test_check_and_verify_measure_the_unit_pose(bag: Path, tmp_path: Path) -> None:
    """``--unit`` checks the pose that unit publishes, and a report never crosses units.

    The manifest's nominal pose is wrong; unit ``cell_a``'s overlay carries the true one,
    unit ``cell_b`` has no calibrated pose (so it publishes the wrong nominal).
    """
    wrong = [p + d for p, d in zip(_TRUE_POSE, (0.03, 0.0, 0.0, 0.0, 0.08, 0.0), strict=True)]
    robot_dir = tmp_path / "openarm"
    robot_dir.mkdir()
    robot = robot_dir / "robot.yaml"
    robot.write_text(_robot_with_pose(tmp_path, wrong).read_text(), encoding="utf-8")
    (robot_dir / "units").mkdir()
    for unit, pose in (("cell_a", list(_TRUE_POSE)), ("cell_b", None)):
        entry: dict[str, object] = {"name": "head_zed"}
        if pose is not None:
            entry["static_transform_xyz_rpy"] = pose
        doc = {"robot_id": "openarm", "unit": unit, "sensors": [entry]}
        (robot_dir / "units" / f"{unit}.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    out = tmp_path / "a.json"
    argv = ["check", "--robot", str(robot), "--sensor", "head_zed", "--unit", "cell_a"]
    argv += ["--bag", str(bag)]
    argv += ["--cloud-topic", _CLOUD_TOPIC, "--table-z", str(_TABLE_Z)]
    argv += ["--table-roi", *map(str, _ROI), "--stride", "1", "--out", str(out)]
    for mx, my in _MARKERS:
        argv += ["--marker", str(mx), str(my)]
    assert zc.main(argv) == 0
    assert json.loads(out.read_text())["unit"] == "cell_a"

    verify = ["verify", "--robot", str(robot), "--sensor", "head_zed", "--report", str(out)]
    assert zc.main([*verify, "--unit", "cell_a"]) == 0
    assert zc.main(verify) == 2  # the robot ships units/: no unit-less check
    assert zc.main([*verify, "--unit", "cell_b"]) == 1
    assert zc.main([*verify, "--unit", "no_such_unit"]) == 2


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
    argv = ["check", "--robot", str(robot), "--sensor", "head_zed", "--bag", str(bag)]
    argv += ["--cloud-topic", _CLOUD_TOPIC, "--table-z", str(_TABLE_Z)]
    argv += ["--table-roi", *map(str, _ROI), "--max-tilt-deg", "nan"]
    with pytest.raises(SystemExit):
        zc.main(argv)

    assert _check(robot, bag, out) == 0
    verify = ["verify", "--robot", str(robot), "--sensor", "head_zed", "--report", str(out)]
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


def test_the_run_script_refuses_when_deploy_would_load_another_manifest(
    tmp_path: Path,
) -> None:
    """The gate must verify the manifest `deploy run` loads. With OPENRAL_ROBOTS_DIR pointing
    elsewhere, deploy would publish that copy's head_zed pose, so the wrapper refuses."""
    other = tmp_path / "robots" / "openarm"
    other.mkdir(parents=True)
    (other / "robot.yaml").write_text(_ROBOT.read_text(encoding="utf-8"), encoding="utf-8")
    env = {
        **os.environ,
        "OPENRAL_OPENARM_ALLOW_MOTION": "1",
        "OPENRAL_OPENARM_ATTENDED": "1",
        "OPENRAL_ROBOTS_DIR": str(tmp_path / "robots"),
        "OPENRAL_ROBOT_UNIT": "thor",
    }
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "openral deploy run would load" in proc.stderr


@pytest.mark.parametrize("rig", [_G1, _SO101], ids=lambda r: r.robot)
def test_a_non_base_parent_is_resolved_through_the_recorded_tf(tmp_path: Path, rig: _Rig) -> None:
    """G1's torso-mounted head and SO-101's gripper-mounted wrist: the table and markers
    are in the base frame, the mount is in parent_frame, the bag's /tf joins them."""
    bag = tmp_path / "bag"
    _write_bag(bag, rig)
    robot, out = _robot_with_pose(tmp_path, list(rig.true_pose), rig), tmp_path / "r.json"
    assert _check(robot, bag, out, rig.sensor) == 0
    report = json.loads(out.read_text())
    assert report["base_frame"] != report["parent_frame"]
    assert _verify(robot, out, rig.sensor) == 0

    x, y, z, roll, pitch, yaw = rig.true_pose
    wrong = [x + 0.012, y - 0.008, z + 0.015, roll - 0.02, pitch + math.radians(2.0), yaw + 0.025]
    moved = tmp_path / "wrong"
    moved.mkdir()
    robot = _robot_with_pose(moved, wrong, rig)
    assert _check(robot, bag, out, rig.sensor) == 1
    suggested = json.loads(out.read_text())["suggested_static_transform_xyz_rpy"]
    # The suggestion is a parent_frame pose — what the manifest holds — not a base pose.
    assert np.allclose(suggested[:3], rig.true_pose[:3], atol=0.003), suggested
    fixed = _robot_with_pose(tmp_path, suggested, rig)
    assert _check(fixed, bag, tmp_path / "fixed.json", rig.sensor) == 0


def test_a_parent_that_moves_while_recording_is_refused(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    _write_bag(bag, _SO101, sway_rad=0.01)
    robot = _robot_with_pose(tmp_path, list(_SO101.true_pose), _SO101)
    assert _check(robot, bag, tmp_path / "r.json", _SO101.sensor) == 2


def test_a_bag_without_the_parent_chain_is_refused(bag: Path, tmp_path: Path) -> None:
    """The OpenArm bag has no pelvis -> torso_link chain: no silent z-up assumption."""
    robot = _robot_with_pose(tmp_path, list(_G1.true_pose), _G1)
    assert _check(robot, bag, tmp_path / "r.json", _G1.sensor) == 2


@pytest.mark.parametrize(("robot", "sensor"), [("g1", "head"), ("so101_follower", "wrist")])
def test_rgb_only_cameras_are_refused_clearly(
    robot: str, sensor: str, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _REPO_ROOT / "robots" / robot / "robot.yaml"
    assert zc.main(["verify", "--robot", str(manifest), "--sensor", sensor]) == 2
    assert "covers depth cameras only" in capsys.readouterr().err


def test_sensor_is_required() -> None:
    with pytest.raises(SystemExit):
        zc.main(["verify", "--robot", str(_ROBOT)])
