"""Nav2 launches on the HAL autostart helper's clean exit, not on a transition_event.

Hermetic: drives ``_after_hal_active``'s handler with real ``ProcessExited`` events.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    path = Path(__file__).resolve().parents[1] / "launch" / "deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_nav2_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("returncode", "launched"), [(0, True), (1, False)])
def test_nav2_include_follows_the_hal_autostart_exit(returncode: int, launched: bool) -> None:
    from launch import LaunchContext
    from launch.actions import ExecuteProcess, LogInfo
    from launch.event_handlers import OnProcessExit
    from launch.events.process import ProcessExited

    mod = _import_launch_module()
    autostart = ExecuteProcess(cmd=["true"])
    nav2 = LogInfo(msg="nav2")
    handler = mod._after_hal_active(autostart, nav2).event_handler
    assert isinstance(handler, OnProcessExit)
    event = ProcessExited(
        action=autostart,
        name="hal_autostart",
        cmd=["true"],
        cwd=None,
        env=None,
        pid=1,
        returncode=returncode,
    )
    assert handler.matches(event)
    entities = handler.handle(event, LaunchContext())
    assert (nav2 in (entities or [])) is launched
