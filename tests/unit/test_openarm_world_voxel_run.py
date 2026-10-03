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
_AUTONOMOUS = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_autonomous.yaml"

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


def test_the_run_script_refuses_a_scene_outside_scenes_deploy_local(tmp_path: Path) -> None:
    """A --scene copy must live under the gitignored scenes/deploy/local/; checked before any
    gate. A tracked scenes/deploy/ file is refused too: copies stay out of the registry."""
    outside = tmp_path / "openarm_real_world_voxels.yaml"
    outside.write_text(_SCENE.read_text(encoding="utf-8"), encoding="utf-8")
    for scene in (str(outside), str(_ROBOT), str(_SCENE)):
        proc = _run_script(_NO_GATES, "--scene", scene)
        assert proc.returncode == 2
        assert "--scene must be an existing .yaml file under" in proc.stderr
        assert "scenes/deploy/local/" in proc.stderr


def _local_scene(name: str, doc: dict[str, object]) -> Path:
    """Write into the gitignored dir the wrapper accepts; callers unlink in ``finally``."""
    local = _REPO_ROOT / "scenes" / "deploy" / "local"
    local.mkdir(exist_ok=True)
    path = local / f"_test_{name}_{os.getpid()}.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _committed_scene(path: Path = _SCENE) -> dict[str, object]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _merge(doc: dict[str, object], patch: dict[str, object]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            _merge(doc[key], value)  # type: ignore[arg-type]
        else:
            doc[key] = value


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"robot_id": "so101"}, "robot_id is 'so101'"),
        ({"runtime": {"enable_octomap_kernel_check": False}}, "is not true"),
        ({"robot_unit": "orin"}, "robot_unit 'orin' is not OPENRAL_ROBOT_UNIT 'thor'"),
        ({"runtime": {"enable_octomap": False}}, "at runtime.enable_octomap"),
        (
            {"runtime": {"robot_self_filter_padding_m": 0.10}},
            "at runtime.robot_self_filter_padding_m",
        ),
        ({"runtime": {"grasp_allowance_enabled": True}}, "at runtime.grasp_allowance_enabled"),
        (
            {"extra_allowed_collision_pairs": [["openarm_left_link7", "openarm_right_link7"]]},
            "at extra_allowed_collision_pairs",
        ),
        (
            {
                "safety": {
                    "workspace_box_min_xyz": [-5.0, -5.0, -5.0],
                    "workspace_box_max_xyz": [5.0, 5.0, 5.0],
                }
            },
            "at safety",
        ),
        ({"drivers": []}, "at drivers"),
        ({"hal": {"defaults": {"viewer_enabled": False}}}, "at hal"),
        ({"scene": {"id": "openarm_other"}}, "at scene.id"),
    ],
)
def test_the_run_script_refuses_a_local_scene_that_weakens_the_graph(
    patch: dict[str, object], reason: str
) -> None:
    """A copied scene must parse to the committed scene bar grasp_declaration; anything else
    it changes (octomap, self-filter padding, allowances, envelope, drivers, HAL) is refused."""
    _skip_unless_ros_and_openral()
    doc = _committed_scene()
    _merge(doc, patch)
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


def test_autonomous_refuses_without_a_reasoner_model() -> None:
    """--autonomous selects the committed autonomous scene, whose reasoner needs a planner."""
    env = {k: v for k, v in _gated("thor").items() if k != "OPENRAL_REASONER_MODEL"}
    proc = _run_script(env, "--autonomous")
    assert proc.returncode == 2, proc.stderr
    assert "OPENRAL_REASONER_MODEL is not set" in proc.stderr


def test_autonomous_reaches_the_last_gate() -> None:
    """The committed autonomous scene passes every check up to the interactive terminal."""
    _skip_unless_ros_and_openral()
    proc = _run_script(
        {**_gated("thor"), "OPENRAL_REASONER_MODEL": "claude-opus-4-8"}, "--autonomous"
    )
    assert proc.returncode == 2, proc.stderr
    assert "not an interactive terminal" in proc.stderr, proc.stderr


@pytest.mark.parametrize("flags", [["--autonomous"], []])
def test_a_local_copy_is_judged_against_the_selected_committed_scene(flags: list[str]) -> None:
    """A copy of the autonomous scene passes only with --autonomous; without it, it is judged
    against the voxel scene and its reasoner/detector/memory legs are refused by key path."""
    _skip_unless_ros_and_openral()
    path = _local_scene("auto", _committed_scene(_AUTONOMOUS))
    env = {**_gated("thor"), "OPENRAL_REASONER_MODEL": "claude-opus-4-8"}
    try:
        proc = _run_script(env, "--scene", str(path), *flags)
    finally:
        path.unlink()
    assert proc.returncode == 2, proc.stderr
    if flags:
        assert "not an interactive terminal" in proc.stderr, proc.stderr
    else:
        assert "at runtime.enable_reasoner" in proc.stderr, proc.stderr


def test_the_autonomous_scene_differs_from_the_voxel_scene_only_in_its_autonomous_legs() -> None:
    """Same cell posture (drivers, octomap, self-filter, envelope, vision leg off) — only the
    reasoner, the pinned open-vocab detector and spatial-memory ingest are added."""
    voxel, auto = _committed_scene(), _committed_scene(_AUTONOMOUS)
    v_rt, a_rt = voxel.pop("runtime"), auto.pop("runtime")
    assert isinstance(v_rt, dict) and isinstance(a_rt, dict)
    assert auto.pop("scene") == {"id": "openarm_real_autonomous"}
    voxel.pop("scene")
    assert auto == voxel
    added = {
        "enable_reasoner": True,
        "enable_object_detector": True,
        "object_detector_manifest": "../../rskills/omdet-turbo-indoor/rskill.yaml",
        "spatial_memory_ingest": True,
    }
    assert {k: v for k, v in a_rt.items() if v_rt.get(k) != v} == added
    assert set(v_rt) <= set(a_rt)
    assert a_rt["vision_attachment"]["enabled"] is False
    assert "preload_rskill_id" not in a_rt and "grasp_declaration" not in auto
