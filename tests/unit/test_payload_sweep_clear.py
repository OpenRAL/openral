"""The octomap bridge's payload octree clearing is opt-in and sim-only (PR #417 review).

``DeployRuntime.payload_sweep_clear_enabled`` (default off) is the only way the launch
hands the bridge octomap_server's ``~/clear_bbox``; ``deploy_e2e`` on a non-sim HAL and
``openral deploy run`` refuse it (it removes world occupancy; Safety-WG review owed).
Real launch file, real CLI resolver, real scene fixtures (CLAUDE.md §1.11).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import DeployRuntime, DeployScene
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
_CELL = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_OPTED_IN = (
    _REPO / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml",
    _REPO / "scenes" / "deploy" / "isaac_arena_droid_banana_in_bowl.yaml",
)


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_sweep", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_sweep"] = module
    spec.loader.exec_module(module)
    return module


def test_it_is_off_by_default() -> None:
    assert DeployRuntime().payload_sweep_clear_enabled is False


@pytest.mark.parametrize("scene", _OPTED_IN, ids=lambda p: p.stem)
def test_the_pick_orange_and_droid_sim_scenes_opt_in(scene: Path) -> None:
    runtime = DeployScene.from_yaml(str(scene)).runtime
    assert runtime is not None and runtime.payload_sweep_clear_enabled


def test_the_launch_passes_the_service_only_when_enabled_in_sim(launch_module: Any) -> None:
    derive = launch_module._payload_sweep_clear_params
    assert derive(enabled=False, hal_mode="sim") == {}
    assert derive(enabled=False, hal_mode="real") == {}
    assert derive(enabled=True, hal_mode="sim") == {
        "payload_sweep_clear_service": "/openral_octomap_server/clear_bbox"
    }


@pytest.mark.parametrize("hal_mode", ["real", "mock", ""])
def test_the_launch_refuses_it_outside_sim(launch_module: Any, hal_mode: str) -> None:
    with pytest.raises(ROSConfigError, match="sim-only until Safety-WG review"):
        launch_module._payload_sweep_clear_params(enabled=True, hal_mode=hal_mode)


def _cell_scene(tmp_path: Path, runtime: dict[str, object]) -> Path:
    data = yaml.safe_load(_CELL.read_text(encoding="utf-8"))
    data["runtime"].update(runtime)
    scene = tmp_path / "cell.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return scene


def _invoke(scene: Path, hal_mode: str) -> Any:
    from openral_cli.deploy_sim import resolve_launch_invocation

    return resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
        hal_mode=hal_mode,
    )


def test_deploy_sim_forwards_it_and_deploy_run_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    off = _invoke(_cell_scene(tmp_path, {}), "sim").argv_template
    assert not any(a.startswith("payload_sweep_clear") for a in off), "off: argv unchanged"
    scene = _cell_scene(tmp_path, {"payload_sweep_clear_enabled": True})
    assert "payload_sweep_clear_enabled:=true" in _invoke(scene, "sim").argv_template
    with pytest.raises(ROSConfigError, match="sim-only until Safety-WG review"):
        _invoke(scene, "real")
