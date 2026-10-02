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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"

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
    other = tmp_path / "robots" / "openarm"
    other.mkdir(parents=True)
    (other / "robot.yaml").write_text(_ROBOT.read_text(encoding="utf-8"), encoding="utf-8")
    proc = _run_script({**_gated("thor"), "OPENRAL_ROBOTS_DIR": str(tmp_path / "robots")})
    assert proc.returncode == 2, proc.stderr
    assert "openral deploy run would load" in proc.stderr


def test_thors_declared_mount_passes_the_gate_and_orins_undeclared_one_does_not() -> None:
    """Thor's overlay declares head_zed's mount, so the script gets past the mount gate and
    stops at the last refusal before motion (no interactive terminal); Orin's does not yet."""
    if (
        "ROS_DISTRO" not in os.environ
        or subprocess.run(
            ["bash", "-c", "command -v openral"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("needs a sourced ROS 2 and openral on PATH to reach the mount gate")
    proc = _run_script(_gated("thor"))
    assert proc.returncode == 2, proc.stderr
    assert "not an interactive terminal" in proc.stderr, proc.stderr
    proc = _run_script(_gated("orin"))
    assert proc.returncode == 2, proc.stderr
    assert "head_zed's mount is not declared for unit orin" in proc.stderr, proc.stderr


def test_the_run_script_refuses_a_unit_that_does_not_exist() -> None:
    """A unit nobody wrote declares nothing: the mount gate refuses before the prompt."""
    if (
        "ROS_DISTRO" not in os.environ
        or subprocess.run(
            ["bash", "-c", "command -v openral"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("needs a sourced ROS 2 and openral on PATH to reach the mount gate")
    proc = _run_script(_gated("no_such_cell"))
    assert proc.returncode == 2, proc.stderr
    assert "head_zed's mount is not declared for unit no_such_cell" in proc.stderr
