"""``tools/openarm_world_voxel_run.sh``: the guarded launcher refuses in the documented order.

The script moves real arms once it gets past its gates, so every test here must stop at a
refusal: the motion gates, the unit, the manifest `deploy run` would load, the declared
head_zed mount, and finally the interactive terminal (stdin is never a TTY under pytest).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"

_NO_GATES = {
    k: v
    for k, v in os.environ.items()
    if k not in ("OPENRAL_OPENARM_ALLOW_MOTION", "OPENRAL_OPENARM_ATTENDED")
}


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_REPO_ROOT / "tools" / "openarm_world_voxel_run.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _gated(unit: str) -> dict[str, str]:
    return {
        **os.environ,
        "OPENRAL_OPENARM_ALLOW_MOTION": "1",
        "OPENRAL_OPENARM_ATTENDED": "1",
        "OPENRAL_ROBOT_UNIT": unit,
    }


def _skip_unless_ros_and_openral() -> None:
    """The gates after the unit one need a sourced ROS 2 and ``openral`` on PATH."""
    if (
        "ROS_DISTRO" not in os.environ
        or subprocess.run(
            ["bash", "-c", "command -v openral"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("needs a sourced ROS 2 and openral on PATH to reach this gate")


def test_the_run_script_refuses_without_both_motion_gates() -> None:
    proc = _run_script(_NO_GATES)
    assert proc.returncode == 2
    assert "OPENRAL_OPENARM_ALLOW_MOTION" in proc.stderr


def test_the_run_script_refuses_without_a_robot_unit() -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPENRAL_ROBOT_UNIT"}
    env |= {"OPENRAL_OPENARM_ALLOW_MOTION": "1", "OPENRAL_OPENARM_ATTENDED": "1"}
    proc = _run_script(env)
    assert proc.returncode == 2
    assert "OPENRAL_ROBOT_UNIT is not set" in proc.stderr


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
    _skip_unless_ros_and_openral()
    other = tmp_path / "robots" / "openarm"
    other.mkdir(parents=True)
    (other / "robot.yaml").write_text(_ROBOT.read_text(encoding="utf-8"), encoding="utf-8")
    proc = _run_script({**_gated("thor"), "OPENRAL_ROBOTS_DIR": str(tmp_path / "robots")})
    assert proc.returncode == 2, proc.stderr
    assert "openral deploy run would load" in proc.stderr


def test_thors_declared_mount_passes_the_gate_and_orins_undeclared_one_does_not() -> None:
    """Thor's overlay declares head_zed's mount, so the script gets past the mount gate and
    stops at the last refusal before motion (no interactive terminal); Orin's does not yet."""
    _skip_unless_ros_and_openral()
    proc = _run_script(_gated("thor"))
    assert proc.returncode == 2, proc.stderr
    assert "not an interactive terminal" in proc.stderr, proc.stderr
    proc = _run_script(_gated("orin"))
    assert proc.returncode == 2, proc.stderr
    assert "head_zed's mount is not declared for unit orin" in proc.stderr, proc.stderr


def test_the_run_script_refuses_a_unit_that_does_not_exist() -> None:
    """A unit nobody wrote declares nothing: the mount gate refuses before the prompt."""
    _skip_unless_ros_and_openral()
    proc = _run_script(_gated("no_such_cell"))
    assert proc.returncode == 2, proc.stderr
    assert "head_zed's mount is not declared for unit no_such_cell" in proc.stderr


def test_the_run_script_refuses_a_scene_outside_scenes_deploy(tmp_path: Path) -> None:
    """A --scene copy must live under scenes/deploy/; checked before any gate."""
    outside = tmp_path / "openarm_real_world_voxels.yaml"
    outside.write_text(_SCENE.read_text(encoding="utf-8"), encoding="utf-8")
    for scene in (str(outside), str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml")):
        proc = _run_script(_NO_GATES, "--scene", scene)
        assert proc.returncode == 2
        assert "--scene must be an existing .yaml file under" in proc.stderr


def _local_scene(name: str, doc: dict[str, object]) -> Path:
    path = _REPO_ROOT / "scenes" / "deploy" / f"_test_{name}_{os.getpid()}.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _committed_scene() -> dict[str, object]:
    doc = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"robot_id": "so101"}, "robot_id is 'so101'"),
        ({"runtime": {"enable_octomap_kernel_check": False}}, "is not true"),
        ({"robot_unit": "orin"}, "robot_unit 'orin' is not OPENRAL_ROBOT_UNIT 'thor'"),
    ],
)
def test_the_run_script_refuses_a_local_scene_that_weakens_the_graph(
    patch: dict[str, object], reason: str
) -> None:
    """A copied scene runs the same gates and must stay OpenArm, kernel check on, same unit."""
    _skip_unless_ros_and_openral()
    doc = _committed_scene()
    for key, value in patch.items():
        if isinstance(value, dict):
            doc[key] = {**doc[key], **value}  # type: ignore[dict-item]
        else:
            doc[key] = value
    path = _local_scene("weak", doc)
    try:
        proc = _run_script(_gated("thor"), "--scene", str(path))
    finally:
        path.unlink()
    assert proc.returncode == 2, proc.stderr
    assert "is not an OpenArm world-voxel scene" in proc.stderr, proc.stderr
    assert reason in proc.stderr, proc.stderr


def test_a_local_scene_with_a_grasp_declaration_reaches_the_last_gate() -> None:
    """Runbook step 4's copy (committed scene + grasp_declaration) passes every check up to
    the interactive-terminal refusal, i.e. it gets the same gates as the committed scene."""
    _skip_unless_ros_and_openral()
    doc = _committed_scene()
    doc["grasp_declaration"] = {
        "target_id": "cell:restock_item",
        "contact_links": ["openarm_left_finger_pair"],
        "timeout_s": 70.0,
        "stamp_ns": 0,
        "search_box": {
            "frame_id": "openarm_base",
            "pose": {
                "xyz": [0.37, -0.15, -0.24],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame_id": "openarm_base",
            },
            "half_extents": [0.06, 0.08, 0.08],
        },
    }
    path = _local_scene("grasp", doc)
    try:
        proc = _run_script(_gated("thor"), "--scene", str(path))
    finally:
        path.unlink()
    assert proc.returncode == 2, proc.stderr
    assert "not an interactive terminal" in proc.stderr, proc.stderr
