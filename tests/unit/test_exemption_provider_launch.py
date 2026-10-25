"""The launch's safety-kernel exemption-provider params (ADR-0121).

The kernel enforces every bound itself (``cpp/openral_safety_kernel/test/
test_exemption_provider_kernel.cpp``); this pins the launch half: no provider → no params
(the kernel exactly as today), a provider → its class plus the manifest's hand links as the
kernel's relaxable-link mask, and an off-sim request refused at launch. Real OpenArm and
DROID manifests, the real launch file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol

import pytest
import yaml
from openral_cli.deploy_sim import LaunchInvocation, resolve_launch_invocation
from openral_core import DeployRuntime, RobotDescription
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
_OPENARM = _REPO / "robots" / "openarm" / "robot.yaml"
_DROID = _REPO / "robots" / "droid_franka_robotiq" / "robot.yaml"
_PROVIDER = "openral_pro_safety_exemptions/ContactExemptions"
_CELL = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"


class _LaunchModule(Protocol):
    """The slice of ``deploy_e2e.launch.py`` these tests read."""

    _PROVIDER_MOVED_RULES_OFF: dict[str, object]
    _PROVIDER_MIRRORED_PREFIXES: tuple[str, ...]
    _PROVIDER_ONLY_RULES: tuple[str, ...]
    _PROVIDER_INPUTS: tuple[str, ...]

    def _exemption_provider_kernel_params(
        self,
        description: RobotDescription,
        *,
        provider: str,
        hal_mode: str,
        wrist_links: int = 1,
    ) -> dict[str, object]: ...

    def _hand_rules_to_provider(self, kernel_params: dict[str, object]) -> dict[str, object]: ...

    def _object_tracking(
        self,
        mode: str,
        *,
        hal_mode: str,
        vision_attachment_enabled: bool,
        enable_octomap: bool,
        self_filter_runs: bool,
        grounding_wait_s: str,
        cameras: str,
        depth_cameras: list[str],
        exemption_provider: str = "",
    ) -> tuple[str, tuple[str, ...]]: ...


@pytest.fixture(scope="module")
def launch_module() -> _LaunchModule:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_provider", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_provider"] = module
    spec.loader.exec_module(module)
    return module


def test_no_provider_adds_no_kernel_params(launch_module: _LaunchModule) -> None:
    robot = RobotDescription.from_yaml(str(_OPENARM))
    derive = launch_module._exemption_provider_kernel_params
    assert derive(robot, provider="", hal_mode="sim") == {}
    assert derive(robot, provider="", hal_mode="real") == {}


@pytest.mark.parametrize("manifest", [_OPENARM, _DROID])
def test_a_provider_gets_the_manifest_hand_links_as_its_mask(
    launch_module: _LaunchModule, manifest: Path
) -> None:
    robot = RobotDescription.from_yaml(str(manifest))
    params = launch_module._exemption_provider_kernel_params(
        robot, provider=_PROVIDER, hal_mode="sim"
    )
    assert params == {
        "exemption_provider": _PROVIDER,
        "exemption_provider_links": list(robot.hand_links()),
    }
    assert params["exemption_provider_links"], "a gripper robot always has hand links"
    wide = launch_module._exemption_provider_kernel_params(
        robot, provider=_PROVIDER, hal_mode="sim", wrist_links=2
    )
    assert wide["exemption_provider_links"] == list(robot.hand_links(wrist_links=2))


@pytest.mark.parametrize("mode", ["real", "", "SIM"])
def test_a_provider_off_sim_is_refused_at_launch(launch_module: _LaunchModule, mode: str) -> None:
    robot = RobotDescription.from_yaml(str(_OPENARM))
    with pytest.raises(ROSConfigError, match="sim-only"):
        launch_module._exemption_provider_kernel_params(robot, provider=_PROVIDER, hal_mode=mode)


def _cell_invocation(tmp_path: Path, runtime: dict[str, object], hal_mode: str) -> LaunchInvocation:

    data = yaml.safe_load(_CELL.read_text(encoding="utf-8"))
    data["runtime"].update(runtime)
    scene = tmp_path / "cell.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
        hal_mode=hal_mode,
    )


def test_deploy_sim_forwards_the_runtime_provider_and_deploy_run_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DeployRuntime.exemption_provider`` is how ``openral deploy sim`` selects one."""
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    assert DeployRuntime().exemption_provider == ""
    off = _cell_invocation(tmp_path, {}, "sim").argv_template
    assert not any(a.startswith("exemption_provider") for a in off), "none: argv unchanged"
    on = _cell_invocation(tmp_path, {"exemption_provider": _PROVIDER}, "sim").argv_template
    assert f"exemption_provider:={_PROVIDER}" in on
    with pytest.raises(ROSConfigError, match="sim-only"):
        _cell_invocation(tmp_path, {"exemption_provider": _PROVIDER}, "real")


#: Kernel A as the launch computes it for `object_tracking: on` (every rule's value present).
_ALL_RULES: dict[str, object] = {
    "workspace_contact_enabled": True,
    "workspace_contact_brake_enabled": True,
    "workspace_contact_max_speed_mps": 0.5,
    "workspace_contact_control_period_s": 0.0667,
    "tracked_release_timeout_s": 3.0,
    "tracked_grasp_exemption_enabled": True,
    "tracked_grasp_extra_contact_links": ["fixed_jaw"],
    "tracked_release_jaw_joints": ["gripper"],
    "tracked_release_jaw_open_sign": [1.0],
    "tracked_release_gap": 0.01,
    "tracked_release_dwell_s": 0.3,
    "workspace_explained_cells_enabled": True,
    "payload_own_cells_enabled": True,
    "pinch_payload_own_cells_enabled": True,
    "static_points_file": "static_cells.txt",
    "grasp_allowance_enabled": True,
}


def test_without_a_provider_the_removed_rules_are_never_emitted(
    launch_module: _LaunchModule,
) -> None:
    """ADR-0121 step 3: the open kernel has no K2 / K5-release / K7 / OT-10, so their params
    and the inputs only a provider reads are dropped; the box-based rules stay as they are."""
    out = launch_module._hand_rules_to_provider(dict(_ALL_RULES))
    for key in (*launch_module._PROVIDER_ONLY_RULES, *launch_module._PROVIDER_INPUTS):
        assert key not in out, key
    assert not [k for k in out if k.startswith("contact_exemptions.")]
    for key in (
        "workspace_contact_enabled",
        "workspace_contact_brake_enabled",
        "workspace_contact_max_speed_mps",
        "tracked_release_timeout_s",
        "grasp_allowance_enabled",
    ):
        assert out[key] == _ALL_RULES[key], key


def test_a_provider_takes_over_the_moved_rules_never_both(launch_module: _LaunchModule) -> None:
    """With a provider the #349 switches are off and every rule is mirrored for it (ADR-0121);
    the removed rules reach it only as ``contact_exemptions.*``, never as kernel params."""
    b = launch_module._hand_rules_to_provider({**_ALL_RULES, "exemption_provider": _PROVIDER})
    for key, off in launch_module._PROVIDER_MOVED_RULES_OFF.items():
        assert b.get(key, []) == off, key
    for key in launch_module._PROVIDER_ONLY_RULES:
        assert key not in b, key
        assert b[f"contact_exemptions.{key}"] == _ALL_RULES[key], key
    # launch_ros refuses an empty list (no ROS type): Spark 2026-10-25, every provider launch died.
    assert not [k for k, v in b.items() if isinstance(v, list | tuple) and not v]
    assert b["contact_exemptions.workspace_contact_enabled"] is True
    assert b["workspace_contact_control_period_s"] == 0.0667, "the measurement stays open"
    assert b["tracked_release_timeout_s"] == 3.0, "the kernel's freeze window stays open"
    assert b["tracked_grasp_extra_contact_links"] == ["fixed_jaw"], "hand geometry, provider-read"
    assert b["static_points_file"] == "static_cells.txt", "the provider's static cells"
    assert b["grasp_allowance_enabled"] is True, "ADR-0115 stays open"


def _track(launch_module: _LaunchModule, mode: str, provider: str) -> tuple[str, tuple[str, ...]]:
    return launch_module._object_tracking(
        mode,
        hal_mode="sim",
        vision_attachment_enabled=True,
        enable_octomap=True,
        self_filter_runs=True,
        grounding_wait_s="8",
        cameras="",
        depth_cameras=["top_depth"],
        exemption_provider=provider,
    )


def test_tracking_on_without_a_provider_is_refused_at_launch(launch_module: _LaunchModule) -> None:
    """In open, `on` stands the HAL region producer down and no grasp exemption is left."""
    with pytest.raises(ROSConfigError, match="ADR-0121"):
        _track(launch_module, "on", "")
    assert _track(launch_module, "on", _PROVIDER) == ("on", ("top_depth",))
    for mode in ("shadow", "bind"):
        assert _track(launch_module, mode, "") == (mode, ("top_depth",))


def test_deploy_runtime_refuses_tracking_on_without_a_provider() -> None:
    """The same refusal at scene load: the committed PickOrange twin, switched to `on`."""
    from openral_core import DeployScene
    from pydantic import ValidationError

    scene = _REPO / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"
    runtime = DeployScene.from_yaml(str(scene)).runtime
    assert runtime is not None and runtime.object_tracking == "shadow"
    base = runtime.model_dump()
    with pytest.raises(ValidationError, match="ADR-0121"):
        DeployRuntime.model_validate({**base, "object_tracking": "on"})
    on = DeployRuntime.model_validate(
        {**base, "object_tracking": "on", "exemption_provider": _PROVIDER}
    )
    assert on.object_tracking == "on"


def test_the_launch_and_the_replay_tool_move_the_same_rules(launch_module: _LaunchModule) -> None:
    """``tools/kernel_replay_diff.py``'s kernel B is the launch's provider configuration."""
    spec = importlib.util.spec_from_file_location(
        "kernel_replay_diff_pin", _REPO / "tools" / "kernel_replay_diff.py"
    )
    assert spec is not None and spec.loader is not None
    krd = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = krd
    spec.loader.exec_module(krd)
    assert krd._MOVED_RULES_OFF == launch_module._PROVIDER_MOVED_RULES_OFF
    assert krd._MIRRORED_PREFIXES == launch_module._PROVIDER_MIRRORED_PREFIXES
    assert krd._PROVIDER_ONLY_RULES == launch_module._PROVIDER_ONLY_RULES
