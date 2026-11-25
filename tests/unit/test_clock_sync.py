"""The clock refusal on a real robot with peers (openral#765, HZ-0137-11)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openral_cli._clock_sync import chrony_tracking, require_synced_clock
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import ScenePeer
from openral_core.exceptions import ROSConfigError
from openral_core.scene_paths import SCENE_PEERS_ENV, format_scene_peers

_CSV = (
    "A9FEA97B,169.254.169.123,4,1764000000.0,-0.000012345,0.0,0.0,0.0,0.0,0.0,0.0,0.0,64.0,Normal\n"
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


def test_tracking_parses_offset_and_leap() -> None:
    assert chrony_tracking(_run(_CSV)) == (-1.2345e-05, "Normal")


@pytest.mark.parametrize(
    "run",
    [
        _run(raises=FileNotFoundError()),
        _run(code=1),
        _run(raises=subprocess.TimeoutExpired("chronyc", 5)),
    ],
)
def test_tracking_none_when_unavailable(
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    assert chrony_tracking(run) is None


def test_synced_passes() -> None:
    require_synced_clock((0.0, "Normal"))


@pytest.mark.parametrize(
    ("tracking", "match"),
    [
        ((0.011, "Normal"), "11"),
        ((float("nan"), "Normal"), "off chrony"),
        ((float("inf"), "Normal"), "off chrony"),
        ((0.0, "Not synchronised"), "leap status"),
        (None, "chrony"),
    ],
)
def test_unsynced_refuses(tracking: tuple[float, str] | None, match: str) -> None:
    with pytest.raises(ROSConfigError, match=match):
        require_synced_clock(tracking)


_PEERS = [ScenePeer(name="b", robot_id="openarm", domain=34)]


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offset: float, hal_mode: str) -> Path:
    """A real ``chronyc`` script first on PATH (a process-boundary double); returns its marker."""
    marker = tmp_path / "called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "chronyc"
    exe.write_text(
        f"#!/bin/sh\ntouch {marker}\n"
        f"echo 'A,1.2.3.4,4,1.0,{offset},0.0,0.0,0.0,0.0,0.0,0.0,0.0,64.0,Normal'\n"
    )
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv(SCENE_PEERS_ENV, format_scene_peers(_PEERS))
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
