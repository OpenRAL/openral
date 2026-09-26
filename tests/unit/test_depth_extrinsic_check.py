"""``tools/depth_extrinsic_check.py`` fits a depth camera's mount to the robot it looks at.

Captures are simulated from the robot's REAL model (its MJCF meshes, posed by MuJoCo at
each recorded joint configuration) seen through the manifest camera's intrinsics from a
known TRUE mount, with range noise: exactly what ``tools/depth_extrinsic_capture.py``
records on a rig, minus the rig. The manifest under test is a committed
``robots/<id>/robot.yaml`` with the sensor's pose swapped and, for ``--unit``, a unit
overlay carrying it (a report never crosses units).

OpenArm's ``head_zed`` is parented to the base frame. G1's ``head`` sits on ``torso_link``
behind the waist joints, which move between poses here: its mount is fitted in the parent
frame through forward kinematics, the path every non-base camera takes.

What is pinned: the true mount passes and verifies; a wrong one fails and the fitted
suggestion recovers the truth; poses that cannot show every axis (the arms hanging) are
refused however well they fit; ``verify`` refuses a missing report, another pose, looser
criteria and NaN; the committed default state (no report yet) is refused; and the run
script refuses to launch without a verified report.
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

pytest.importorskip("mujoco", reason="the robot fit poses the robot's MJCF meshes (sim group)")

from openral_core import RobotDescription

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
_REPORT = _REPO_ROOT / "robots" / "openarm" / "calibration" / "thor" / "head_zed_extrinsic.json"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.depth_extrinsic_capture as capture_tool  # noqa: E402
import tools.depth_extrinsic_check as zc  # noqa: E402
from tools._extrinsic_fit import (  # noqa: E402
    FitPoseFile,
    PoseCapture,
    RobotSurface,
    simulate_capture,
    write_capture,
    xyzrpy_to_matrix,
)

#: The OpenArm head camera's true mount in the tests (near the nominal one).
_TRUE_POSE = [-0.021, -0.003, 0.214, -0.0123, 1.1823, 0.006]
#: A mount displaced from the truth by 12 mm / 0.7 deg: what a nominal guess looks like.
_WRONG_POSE = [-0.009, -0.008, 0.226, -0.0023, 1.1923, 0.0136]
_OPENARM_POSES = FitPoseFile.load(
    _REPO_ROOT / "robots/openarm/calibration/extrinsic_fit_poses.yaml"
)


def _robot_with_pose(
    tmp_path: Path, pose: list[float], robot: str = "openarm", sensor: str = "head_zed"
) -> Path:
    """The committed manifest in ``tmp_path/<robot>/`` with ``sensor`` at ``pose`` (made depth)."""
    src = _REPO_ROOT / "robots" / robot / "robot.yaml"
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    (entry,) = [s for s in data["sensors"] if s["name"] == sensor]
    entry["modality"] = "depth"
    entry["static_transform_xyz_rpy"] = pose
    out_dir = tmp_path / robot
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "robot.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    for extra in ("openarm.srdf", "openarm.urdf", "patches"):
        link = out_dir / extra
        if (src.parent / extra).exists() and not link.exists():
            link.symlink_to(src.parent / extra)
    return path


def _capture(
    tmp_path: Path,
    robot_yaml: Path,
    true_pose: list[float],
    targets: list[tuple[str, dict[str, float]]],
    sensor: str = "head_zed",
    name: str = "capture",
) -> Path:
    """A capture directory as ``depth_extrinsic_capture.py`` writes it, seen from ``true_pose``."""
    desc = RobotDescription.from_yaml(str(robot_yaml))
    (spec,) = [s for s in desc.sensors if s.name == sensor]
    robot = RobotSurface(desc, robot_yaml.parent)
    rng = np.random.default_rng(3)
    caps = [
        PoseCapture(
            n,
            simulate_capture(
                robot, spec, xyzrpy_to_matrix(true_pose), q, rng=rng, noise_m=0.003, max_points=3000
            ),
            q,
        )
        for n, q in targets
    ]
    out = tmp_path / name
    write_capture(out, {"sensor": spec.name, "frame_id": spec.frame_id}, caps)
    return out


def _openarm_targets() -> list[tuple[str, dict[str, float]]]:
    return _OPENARM_POSES.targets("openarm", RobotDescription.from_yaml(str(_ROBOT)))


def _check(robot: Path, capture: Path, out: Path, sensor: str = "head_zed") -> int:
    argv = ["check", "--robot", str(robot), "--sensor", sensor, "--capture", str(capture)]
    return zc.main([*argv, "--out", str(out)])


def _verify(robot: Path, report: Path, sensor: str = "head_zed", unit: str | None = None) -> int:
    argv = ["verify", "--robot", str(robot), "--sensor", sensor, "--report", str(report)]
    return zc.main([*argv, "--unit", unit] if unit else argv)


@pytest.fixture(scope="module")
def openarm_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("openarm")
    robot = _robot_with_pose(tmp, _TRUE_POSE)
    return _capture(tmp, robot, _TRUE_POSE, _openarm_targets())


@pytest.fixture(scope="module")
def passing_report(
    tmp_path_factory: pytest.TempPathFactory, openarm_capture: Path
) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("pass")
    robot, out = _robot_with_pose(tmp, _TRUE_POSE), tmp / "r.json"
    assert _check(robot, openarm_capture, out) == 0
    return robot, out


@pytest.mark.slow
def test_the_true_mount_passes_and_verifies(passing_report: tuple[Path, Path]) -> None:
    robot, out = passing_report
    report = json.loads(out.read_text())
    assert report["method"] == "robot_fit" and report["passed"] is True
    assert report["residuals"]["poses"] == len(_OPENARM_POSES.poses)
    assert _verify(robot, out) == 0


@pytest.mark.slow
def test_a_wrong_mount_fails_and_the_suggestion_recovers_the_true_one(
    openarm_capture: Path, tmp_path: Path
) -> None:
    robot, out = _robot_with_pose(tmp_path, _WRONG_POSE), tmp_path / "r.json"
    assert _check(robot, openarm_capture, out) == 1
    suggested = json.loads(out.read_text())["suggested_static_transform_xyz_rpy"]
    assert np.allclose(suggested[:3], _TRUE_POSE[:3], atol=0.003), suggested
    assert np.allclose(suggested[3:], _TRUE_POSE[3:], atol=math.radians(0.3)), suggested
    fixed_dir = tmp_path / "fixed"
    fixed = _robot_with_pose(fixed_dir, suggested)
    assert _check(fixed, openarm_capture, tmp_path / "fixed.json") == 0


@pytest.mark.slow
def test_poses_that_hide_an_axis_are_refused_however_well_they_fit(tmp_path: Path) -> None:
    """The arms hanging straight down, recorded four times: a close fit, and a useless one.

    Measured on both OpenArm cells (2026-09-26): from hanging arms the camera's height could
    slide 2 cm without the residual noticing. The gate must refuse that, not certify it.
    """
    robot = _robot_with_pose(tmp_path, _TRUE_POSE)
    rest = {j.name: 0.0 for j in RobotDescription.from_yaml(str(robot)).joints}
    capture = _capture(tmp_path, robot, _TRUE_POSE, [(f"rest{i}", rest) for i in range(4)])
    out = tmp_path / "r.json"
    assert _check(robot, capture, out) == 1
    failures = json.loads(out.read_text())["failures"]
    # Which axis hides first depends on the noise; that one does is the point.
    assert any("unobservable" in f for f in failures), failures


@pytest.mark.slow
def test_verify_refuses_a_report_for_another_pose_or_looser_criteria(
    passing_report: tuple[Path, Path], tmp_path: Path
) -> None:
    robot, out = passing_report
    edited = _robot_with_pose(tmp_path, [*_TRUE_POSE[:5], _TRUE_POSE[5] + 0.001])
    assert _verify(edited, out) == 1

    report = json.loads(out.read_text())
    for key, value in (
        ("max_height_err_m", 0.05),
        ("min_fit_poses", 2),
        ("max_axis_unrecovered", 0.9),
    ):
        loose = json.loads(json.dumps(report))
        loose["criteria"][key] = value
        path = tmp_path / f"loose_{key}.json"
        path.write_text(json.dumps(loose))
        assert _verify(robot, path) == 1, key
    assert _verify(robot, tmp_path / "no") == 1


@pytest.mark.slow
def test_nan_cannot_open_the_gate(passing_report: tuple[Path, Path], tmp_path: Path) -> None:
    """``verify`` re-derives the verdict from the stored residuals: an edited report that
    still says ``passed: true`` is refused for NaN or out-of-limit values."""
    robot, out = passing_report
    good = json.loads(out.read_text())
    for mutate in (
        lambda r: r["criteria"].__setitem__("max_tilt_deg", float("nan")),
        lambda r: r["residuals"]["mount_error"].__setitem__("height_m", float("nan")),
        lambda r: r["residuals"]["axis_unrecovered"].__setitem__("yaw", float("nan")),
        lambda r: r["residuals"].__setitem__("median_residual_m", 1.0),
        lambda r: r["residuals"].__setitem__("poses", 3),
    ):
        report = json.loads(json.dumps(good))
        mutate(report)
        assert report["passed"] is True
        path = tmp_path / "edited.json"
        path.write_text(json.dumps(report))
        assert _verify(robot, path) == 1


@pytest.mark.slow
def test_check_and_verify_measure_the_unit_pose(openarm_capture: Path, tmp_path: Path) -> None:
    """``--unit`` checks the pose that unit publishes, and a report never crosses units.

    The manifest's nominal pose is wrong; unit ``cell_a`` carries the true one, unit
    ``cell_b`` has no pose (so it publishes the wrong nominal).
    """
    robot = _robot_with_pose(tmp_path, _WRONG_POSE)
    units = robot.parent / "units"
    units.mkdir()
    for unit, pose in (("cell_a", _TRUE_POSE), ("cell_b", None)):
        entry: dict[str, object] = {"name": "head_zed"}
        if pose is not None:
            entry["static_transform_xyz_rpy"] = pose
        doc = {"robot_id": "openarm", "unit": unit, "sensors": [entry]}
        (units / f"{unit}.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    out = tmp_path / "a.json"
    argv = ["check", "--robot", str(robot), "--sensor", "head_zed", "--unit", "cell_a"]
    assert zc.main([*argv, "--capture", str(openarm_capture), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["unit"] == "cell_a"

    verify = ["verify", "--robot", str(robot), "--sensor", "head_zed", "--report", str(out)]
    assert zc.main([*verify, "--unit", "cell_a"]) == 0
    assert zc.main(verify) == 2  # the robot ships units/: no unit-less check
    assert zc.main([*verify, "--unit", "cell_b"]) == 1
    assert zc.main([*verify, "--unit", "no_such_unit"]) == 2


@pytest.mark.slow
def test_a_capture_of_another_camera_is_refused(openarm_capture: Path, tmp_path: Path) -> None:
    meta = json.loads((openarm_capture / "capture.json").read_text())
    other = tmp_path / "other"
    other.mkdir()
    for f in openarm_capture.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    (other / "capture.json").write_text(json.dumps({**meta, "frame_id": "wrist_cam"}))
    robot = _robot_with_pose(tmp_path, _TRUE_POSE)
    assert _check(robot, other, tmp_path / "r.json") == 2


@pytest.mark.slow
def test_a_camera_on_a_moving_link_is_fitted_through_forward_kinematics(tmp_path: Path) -> None:
    """G1's head on ``torso_link``, with the waist turning and pitching between poses: the
    mount is fitted in the PARENT frame, so the suggestion is what the manifest holds."""
    true_pose = [0.06, 0.0, 0.40, 0.0, 0.75, 0.0]
    robot = _robot_with_pose(tmp_path, true_pose, "g1", "head")
    desc = RobotDescription.from_yaml(str(robot))
    zero = {j.name: 0.0 for j in desc.joints}

    arms = {
        "left_shoulder_pitch_joint": -1.0,
        "right_shoulder_pitch_joint": -1.0,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
    }

    def pose(*layers: dict[str, float]) -> dict[str, float]:
        out = dict(zero)
        for layer in layers:
            out.update(layer)
        return out

    targets = [
        ("arms_forward", pose(arms)),
        ("waist_yawed", pose(arms, {"waist_yaw_joint": 0.4})),
        ("waist_pitched", pose(arms, {"waist_pitch_joint": 0.25, "left_elbow_joint": 1.2})),
        ("asymmetric", pose(arms, {"waist_yaw_joint": -0.3, "right_shoulder_roll_joint": -0.4})),
        (
            "reach_up",
            pose(
                {
                    "left_shoulder_pitch_joint": -1.4,
                    "right_shoulder_pitch_joint": -0.6,
                    "left_elbow_joint": 0.3,
                    "right_elbow_joint": 1.3,
                }
            ),
        ),
    ]
    capture = _capture(tmp_path, robot, true_pose, targets, sensor="head")
    wrong = [
        p + d for p, d in zip(true_pose, (0.012, -0.008, 0.01, 0.01, -0.012, 0.01), strict=True)
    ]
    wrong_robot = _robot_with_pose(tmp_path / "wrong", wrong, "g1", "head")
    out = tmp_path / "r.json"
    assert _check(wrong_robot, capture, out, "head") == 1
    report = json.loads(out.read_text())
    assert report["parent_frame"] == "torso_link" != report["base_frame"]
    suggested = report["suggested_static_transform_xyz_rpy"]
    assert np.allclose(suggested[:3], true_pose[:3], atol=0.004), suggested
    fixed = _robot_with_pose(tmp_path / "fixed", suggested, "g1", "head")
    assert _check(fixed, capture, tmp_path / "fixed.json", "head") == 0, json.loads(
        (tmp_path / "fixed.json").read_text()
    )["failures"]


@pytest.mark.slow
def test_the_committed_openarm_poses_pass_plan() -> None:
    """The committed pose file clears the offline plan for both OpenArm cells' mounts."""
    argv = ["plan", "--robot", str(_ROBOT), "--sensor", "head_zed", "--unit", "thor"]
    assert zc.main(argv) == 0


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(robot_id="g1"), "for robot 'g1'"),
        (lambda d: d["poses"][1]["joints"].update(no_such_joint=0.1), "unknown joints"),
        (lambda d: d["poses"][1]["joints"].update(left_joint4=9.0), "outside"),
        (lambda d: d.update(poses=d["poses"][:3]), "needs 4"),
    ],
)
def test_a_bad_pose_file_is_refused(edit: object, message: str) -> None:
    doc = yaml.safe_load(
        (_REPO_ROOT / "robots/openarm/calibration/extrinsic_fit_poses.yaml").read_text()
    )
    edit(doc)  # type: ignore[operator]
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match=message):
        FitPoseFile.model_validate(doc).targets("openarm", RobotDescription.from_yaml(str(_ROBOT)))


def test_the_capture_tool_refuses_without_both_gates_or_a_terminal() -> None:
    assert capture_tool.refuse_reason({}, True) is not None
    assert capture_tool.refuse_reason({capture_tool.GATES[0]: "1"}, True) is not None
    assert capture_tool.refuse_reason(dict.fromkeys(capture_tool.GATES, "1"), False) is not None
    assert capture_tool.refuse_reason(dict.fromkeys(capture_tool.GATES, "1"), True) is None
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "depth_extrinsic_capture.py"),
            "--robot",
            str(_ROBOT),
            "--sensor",
            "head_zed",
            "--unit",
            "thor",
            "--cloud-topic",
            "/x",
            "--out",
            "/tmp/never",
        ],
        env={k: v for k, v in os.environ.items() if k not in capture_tool.GATES},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 2 and "REFUSED" in proc.stderr


def test_the_committed_manifest_is_refused_until_a_calibration_is_committed() -> None:
    """Fail-closed by default: no report verifies the shipped pose yet."""
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    assert _verify(_ROBOT, _REPORT, unit="thor") == 1


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


_NO_GATES = {
    k: v
    for k, v in os.environ.items()
    if k not in ("OPENRAL_OPENARM_ALLOW_MOTION", "OPENRAL_OPENARM_ATTENDED")
}


def test_the_run_script_refuses_without_a_verified_extrinsic() -> None:
    """Both gates exported, still refused: no report verifies the manifest's pose."""
    if _REPORT.exists():
        pytest.skip("a calibration report is committed; the default-refusal state is past")
    env = {
        **os.environ,
        "OPENRAL_OPENARM_ALLOW_MOTION": "1",
        "OPENRAL_OPENARM_ATTENDED": "1",
        "OPENRAL_ROBOT_UNIT": "thor",
    }
    proc = _run_script(env)
    assert proc.returncode == 2, proc.stderr
    assert "head_zed extrinsic not verified for unit thor's pose" in proc.stderr


def test_the_run_script_refuses_without_a_robot_unit() -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPENRAL_ROBOT_UNIT"}
    env |= {"OPENRAL_OPENARM_ALLOW_MOTION": "1", "OPENRAL_OPENARM_ATTENDED": "1"}
    proc = _run_script(env)
    assert proc.returncode == 2
    assert "OPENRAL_ROBOT_UNIT is not set" in proc.stderr


def test_the_run_script_refuses_without_both_motion_gates() -> None:
    proc = _run_script(_NO_GATES)
    assert proc.returncode == 2
    assert "OPENRAL_OPENARM_ALLOW_MOTION" in proc.stderr


@pytest.mark.parametrize(
    "extra",
    [
        ["--config", "scenes/deploy/openarm_tabletop.yaml"],
        ["--no-enable-octomap-kernel-check"],
        ["--hal", "viewer_enabled=false"],
    ],
)
def test_the_run_script_refuses_args_that_could_change_the_verified_graph(extra: list[str]) -> None:
    """Only observability flags pass through; the scene and safety posture are fixed."""
    proc = _run_script(dict(os.environ), *extra)
    assert proc.returncode == 2
    assert "is not allowed here" in proc.stderr


def test_the_run_script_lets_observability_flags_through_to_the_gates() -> None:
    proc = _run_script(_NO_GATES, "--foxglove", "--dataset-out", "/tmp/x")
    assert proc.returncode == 2
    assert "OPENRAL_OPENARM_ALLOW_MOTION" in proc.stderr


def test_the_run_script_refuses_when_deploy_would_load_another_manifest(tmp_path: Path) -> None:
    """With OPENRAL_ROBOTS_DIR pointing elsewhere, deploy would publish that copy's pose."""
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
    proc = _run_script(env)
    assert proc.returncode == 2, proc.stderr
    assert "openral deploy run would load" in proc.stderr


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


def test_a_robot_without_a_model_is_refused(tmp_path: Path) -> None:
    """No MJCF and no URDF: nothing to fit to, said plainly."""
    robot = _robot_with_pose(tmp_path, _TRUE_POSE)
    data = yaml.safe_load(robot.read_text())
    data["assets"] = {k: v for k, v in data["assets"].items() if k not in ("mjcf", "urdf")}
    robot.write_text(yaml.safe_dump(data))
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="neither an MJCF nor a URDF"):
        RobotSurface(RobotDescription.from_yaml(str(robot)), robot.parent)
