"""The clock refusal on a real robot with peers (openral#765, HZ-0137-11)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import ScenePeer
from openral_core.clock_sync import ChronyTracking, chrony_tracking, require_synced_clock
from openral_core.exceptions import ROSConfigError
from openral_core.scene_paths import SCENE_PEERS_ENV, format_scene_peers

# The 14 fields of `chronyc -c tracking`: offset is index 4, root delay 10, root dispersion 11.
_CSV = (
    "A9FEA97B,169.254.169.123,4,1764000000.0,-0.000012345,0.000001,0.000002,-1.5,0.001,0.02,"
    "0.000456,0.000123,64.0,Normal\n"
)


def _run(
    stdout: str = "", code: int = 0, raises: BaseException | None = None
) -> Callable[..., subprocess.CompletedProcess[str]]:
    # Process-boundary double for subprocess.run (CLAUDE.md 1.11).
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, code, stdout, "")

    return run


def test_tracking_parses_offset_delay_dispersion_and_leap() -> None:
    assert chrony_tracking(_run(_CSV)) == ChronyTracking(-1.2345e-05, 4.56e-4, 1.23e-4, "Normal")


@pytest.mark.parametrize(
    "run",
    [
        _run(raises=FileNotFoundError()),
        _run(code=1),
        _run(raises=subprocess.TimeoutExpired("chronyc", 5)),
        _run("A,1.2.3.4,4,1.0,0.0,Normal\n"),  # too few fields
    ],
)
def test_tracking_none_when_unavailable(
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    assert chrony_tracking(run) is None


def test_synced_passes() -> None:
    require_synced_clock(ChronyTracking(0.0, 0.0, 0.0, "Normal"))
    # 4 + 3 + 6/2 = 10 ms: at the limit.
    require_synced_clock(ChronyTracking(-0.004, 0.006, 0.003, "Normal"))


_NAN = float("nan")


@pytest.mark.parametrize(
    ("tracking", "match"),
    [
        (ChronyTracking(0.011, 0.0, 0.0, "Normal"), "11.0 ms"),
        # A small offset, but the reference is far away: 1 + 5 + 10/2 = 11 ms worst case.
        (ChronyTracking(0.001, 0.010, 0.005, "Normal"), "11.0 ms"),
        (ChronyTracking(0.0, 0.0, 0.0105, "Normal"), "10.5 ms"),
        (ChronyTracking(_NAN, 0.0, 0.0, "Normal"), "nan ms"),
        (ChronyTracking(0.0, _NAN, 0.0, "Normal"), "nan ms"),
        (ChronyTracking(0.0, 0.0, _NAN, "Normal"), "nan ms"),
        (ChronyTracking(float("inf"), 0.0, 0.0, "Normal"), "inf ms"),
        (ChronyTracking(0.0, 0.0, 0.0, "Not synchronised"), "leap status"),
        (None, "chrony"),
    ],
)
def test_unsynced_refuses(tracking: ChronyTracking | None, match: str) -> None:
    with pytest.raises(ROSConfigError, match=match):
        require_synced_clock(tracking)


_PEERS = [ScenePeer(name="b", robot_id="openarm", domain=34)]


def _fake_chronyc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offset: float) -> Path:
    """A real ``chronyc`` script first on PATH (a process-boundary double); returns its marker."""
    marker = tmp_path / "called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "chronyc"
    exe.write_text(
        f"#!/bin/sh\ntouch {marker}\n"
        f"echo 'A,1.2.3.4,4,1.0,{offset},0.0,0.0,0.0,0.0,0.0,0.0001,0.0001,64.0,Normal'\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv(SCENE_PEERS_ENV, format_scene_peers(_PEERS))
    return marker


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offset: float, hal_mode: str) -> Path:
    """``resolve_launch_invocation`` with the fake ``chronyc``; returns its marker."""
    marker = _fake_chronyc(monkeypatch, tmp_path, offset)
    repo = Path(__file__).resolve().parents[2]
    try:
        resolve_launch_invocation(
            config=repo / "scenes/deploy/so101_box.yaml",
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode=hal_mode,
        )
    except ROSConfigError as exc:
        assert "clock" not in str(exc) or offset > 0.01, exc
        if offset > 0.01:
            raise
    return marker


def test_real_with_peers_and_a_bad_clock_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ROSConfigError, match="clock"):
        _wire(monkeypatch, tmp_path, 0.5, "real")


def test_real_with_peers_and_a_good_clock_is_not_refused_on_the_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _wire(monkeypatch, tmp_path, 0.0001, "real").exists()


def test_sim_with_peers_never_calls_chronyc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert not _wire(monkeypatch, tmp_path, 0.5, "sim").exists()


_LAUNCH = (
    Path(__file__).resolve().parents[2] / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
)


def _launch_module() -> Any:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_clock_sync", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_clock_sync"] = module
    spec.loader.exec_module(module)
    return module


def test_a_direct_real_launch_with_peers_and_a_bad_clock_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``ros2 launch`` without the CLI still runs the clock check on real hardware."""
    mod = _launch_module()
    _fake_chronyc(monkeypatch, tmp_path, 0.5)
    with pytest.raises(ROSConfigError, match="clock"):
        mod._scene_peers(object(), "real")  # a world pose is there; the clock refuses


def test_a_direct_launch_checks_the_clock_only_on_real_hardware_with_peers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _launch_module()
    marker = _fake_chronyc(monkeypatch, tmp_path, 0.5)
    assert mod._scene_peers(object(), "sim") == _PEERS
    assert not marker.exists()
    monkeypatch.delenv(SCENE_PEERS_ENV)
    assert mod._scene_peers(None, "real") == []
    assert not marker.exists()
    (tmp_path / "ok").mkdir()
    good = _fake_chronyc(monkeypatch, tmp_path / "ok", 0.0001)
    assert mod._scene_peers(object(), "real") == _PEERS
    assert good.exists()
