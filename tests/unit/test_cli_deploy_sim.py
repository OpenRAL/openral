"""Unit tests for ``openral deploy sim``.

No mocks (CLAUDE.md §1.11). The CLI is exercised via Typer's
``CliRunner`` against the real openarm DeployScene config with
``--dry-run`` so the launch is never shelled out. The end-to-end
``ros2 launch`` smoke test is gated on the presence of ``ros2`` +
``OPENRAL_DEPLOY_SIM_SMOKE=1``; without both it skips per CLAUDE.md
§1.11.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from openral_cli import deploy_sim
from openral_cli.deploy_sim import (
    _ROBOT_HAL_REGISTRY,
    _cmdline_is_openral_graph_process,
    _preflight_palette_deps,
    _prepare_launch_env,
    _run_launch,
    _scan_params_from_description,
    _terminate_launch_group,
    assert_ros2_packages_discoverable,
    resolve_launch_invocation,
    run_launch_invocation,
)
from openral_cli.main import app
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENARM_CONFIG = _REPO_ROOT / "scenes" / "deploy" / "openarm_tabletop.yaml"
_PANDA_MOBILE_CONFIG = _REPO_ROOT / "scenes" / "deploy" / "robocasa_pnp.yaml"
_SO101_CONFIG = _REPO_ROOT / "scenes" / "deploy" / "so101_box.yaml"


def test_bh_deploy_sim_help_renders() -> None:
    """``openral deploy sim --help`` lists every primary flag, no --rskill."""
    runner = CliRunner()
    result = runner.invoke(app, ["deploy", "sim", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--config", "--robot", "--dashboard-port", "--hal", "--dry-run"):
        assert flag in result.output, f"{flag} missing from help"
    # --rskill was removed: the reasoner picks the active rSkill
    # dynamically from rskills/ at on_configure.
    assert "--rskill" not in result.output


def test_bh_deploy_sim_dry_run_openarm() -> None:
    """Dry-run dispatch against the in-tree openarm DeployScene config."""
    assert _OPENARM_CONFIG.is_file(), f"missing fixture: {_OPENARM_CONFIG}"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["deploy", "sim", "--config", str(_OPENARM_CONFIG), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "robot=openarm" in flat
    assert "manifest.name=openarm_v2" in flat
    assert "hal_package=openral_hal_openarm" in flat
    assert "hal_node_name=openral_hal_openarm" in flat
    assert "sim_e2e.launch.py" in flat
    assert "robots/openarm/robot.yaml" in flat
    # Envelope is synthesised at launch time from robot.yaml — never a file.
    assert "synthesised at launch time" in flat
    # And the CLI's argv no longer carries an envelope_file:= arg.
    assert "envelope_file:=" not in flat


def test_bh_deploy_sim_resolve_openarm_invocation() -> None:
    """Resolution returns the right argv template for openarm; no envelope file."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.robot_id == "openarm"
    assert invocation.robot_manifest_name == "openarm_v2"
    assert invocation.robot_yaml == _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    assert invocation.hal.package == "openral_hal_openarm"
    assert invocation.hal.executable == "lifecycle_node.py"
    assert invocation.hal.node_name == "openral_hal_openarm"
    assert invocation.hal.supported_robot_names == frozenset({"openarm_v2", "openarm"})
    # issue #191 Phase 3b — openarm is manifest-driven now: robot_yaml + hal_mode
    # are injected; scene params moved to the manifest's scene_defaults.composition
    # and HAL kwargs to hal.parameters. `bare_twin_sim=True` suppresses the
    # `sim_env_yaml` scene-attach (openarm composes its own MJCF). Only the viewer
    # toggle remains a node param.
    assert invocation.hal_params == {
        "viewer_enabled": True,
        "robot_yaml": str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"),
        "hal_mode": "sim",
    }
    assert "sim_env_yaml" not in invocation.hal_params
    assert invocation.reset_to_pose_service == "/openral/openarm/reset_to_pose"
    # ADR-0053 — MoveIt approach is opt-in; empty default keeps the legacy snap
    # and is NOT forwarded as a launch arg (ros2 launch rejects empty name:=).
    assert invocation.approach_skill_id == ""
    joined = " ".join(invocation.argv_template)
    assert "approach_skill_id:=" not in joined
    assert joined.startswith("ros2 launch openral_rskill_ros sim_e2e.launch.py")
    assert "envelope_file:=" not in joined  # no file path of any kind
    assert "HAL_PARAMS_FILE_PLACEHOLDER" in joined
    assert "hal_package:=openral_hal_openarm" in joined
    # ADR-0025: default is enable_slam=false; the launch arg is still
    # forwarded so the OpaqueFunction can read it.
    assert "enable_slam:=false" in joined
    assert invocation.enable_slam is False


def test_deploy_sim_object_detector_manifest_selects_vlm() -> None:
    """A detector manifest auto-enables the leg and is forwarded (ADR-0037 amendment).

    Passing ``--object-detector-manifest`` for the LocateAnything VLM rSkill must
    auto-enable the object-detection leg (no ONNX file needed) and forward both the
    resolved manifest path and the open-vocab query into the launch argv.
    """
    manifest = _REPO_ROOT / "rskills" / "locateanything-3b-nf4" / "rskill.yaml"
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        object_detector_manifest=str(manifest),
        object_detector_query="red mug",
    )
    assert invocation.enable_object_detector is True
    assert invocation.object_detector_manifest == str(manifest.resolve())
    assert invocation.object_detector_query == "red mug"
    joined = " ".join(invocation.argv_template)
    assert f"object_detector_manifest:={manifest.resolve()}" in joined
    assert "object_detector_query:=red mug" in joined
    assert "enable_object_detector:=true" in joined


def test_deploy_sim_no_detector_emits_no_empty_launch_args(tmp_path: Path) -> None:
    """With no usable backend, the leg downgrades off and emits no empty args.

    Regression: ``ros2 launch`` rejects ``object_detector_manifest:=`` (empty
    value), so the optional detector overrides must be omitted entirely when
    unset rather than forwarded blank — otherwise the whole graph aborts at
    launch. (Surfaced bringing up robocasa deploy-sim without a detector.)

    The detector is on by default (ADR-0035), but auto-downgrades to off when no
    backend is available. An explicit ``--object-detector-onnx`` selects the
    RT-DETR path; pointing it at a guaranteed-absent file (and supplying no
    manifest) reproduces the no-weights condition deterministically on every
    host — neither omdet nor RT-DETR can build, so the leg downgrades off.
    """
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        object_detector_onnx=tmp_path / "absent-rtdetr.onnx",
        # no object_detector_manifest / query → RT-DETR path, weights absent → off
    )
    assert invocation.enable_object_detector is False
    # Every forwarded arg is a well-formed ``name:=value`` with a non-empty value.
    for arg in invocation.argv_template:
        if ":=" in arg:
            name, _, value = arg.partition(":=")
            assert value != "", f"empty launch arg {name!r} would abort ros2 launch"
    joined = " ".join(invocation.argv_template)
    assert "object_detector_manifest:=" not in joined
    assert "object_detector_query:=" not in joined
    assert "enable_object_detector:=false" in joined


def test_deploy_sim_default_detector_is_omdet_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default detector = open-vocab omdet-turbo-indoor when its deps import.

    No ``--object-detector-*`` override + omdet runtime deps present → the leg is
    on and resolves to the omdet-turbo-indoor manifest (grounds arbitrary
    indoor/kitchen objects, unlike the fixed COCO-80 of RT-DETR).
    """
    monkeypatch.setattr(deploy_sim, "_omdet_runtime_available", lambda: True)
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    omdet = _REPO_ROOT / "rskills" / "omdet-turbo-indoor" / "rskill.yaml"
    assert invocation.enable_object_detector is True
    assert invocation.object_detector_manifest == str(omdet.resolve())
    joined = " ".join(invocation.argv_template)
    assert f"object_detector_manifest:={omdet.resolve()}" in joined
    assert "enable_object_detector:=true" in joined


def test_deploy_sim_default_detector_falls_back_to_rtdetr_when_omdet_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """omdet deps absent → graceful fallback to the in-tree RT-DETR COCO ONNX.

    The leg stays on (the ONNX ships in-tree), no manifest is forwarded, and the
    onnx arg points at rskills/rtdetr-coco-r18/model.onnx.
    """
    monkeypatch.setattr(deploy_sim, "_omdet_runtime_available", lambda: False)
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    rtdetr = _REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "model.onnx"
    assert invocation.enable_object_detector is True
    assert invocation.object_detector_manifest == ""
    joined = " ".join(invocation.argv_template)
    assert "object_detector_manifest:=" not in joined
    assert "enable_object_detector:=true" in joined
    assert f"object_detector_onnx:={rtdetr}" in joined


def test_deploy_sim_default_locator_is_omdet_turbo_locator_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0056 — default on-demand locator is omdet-turbo-locator when omdet deps import."""
    monkeypatch.setattr(deploy_sim, "_omdet_runtime_available", lambda: True)
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    locator = _REPO_ROOT / "rskills" / "omdet-turbo-locator" / "rskill.yaml"
    assert invocation.object_detector_locators == (str(locator.resolve()),)
    joined = " ".join(invocation.argv_template)
    assert f"object_detector_locators:={locator.resolve()}" in joined


def test_deploy_sim_no_locator_when_omdet_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """omdet deps absent → no default on-demand locator (RT-DETR continuous still on)."""
    monkeypatch.setattr(deploy_sim, "_omdet_runtime_available", lambda: False)
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.object_detector_locators == ()
    assert "object_detector_locators:=" not in " ".join(invocation.argv_template)


def test_deploy_sim_explicit_locator_alias_resolves_to_manifest() -> None:
    """An explicit --object-detector-locator alias resolves to its in-tree manifest."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        object_detector_locators=["locateanything-3b-nf4"],
    )
    locator = _REPO_ROOT / "rskills" / "locateanything-3b-nf4" / "rskill.yaml"
    assert invocation.object_detector_locators == (str(locator.resolve()),)
    assert f"object_detector_locators:={locator.resolve()}" in " ".join(invocation.argv_template)


def test_deploy_sim_no_locators_when_detector_disabled() -> None:
    """--no-object-detector → no continuous detector AND no on-demand locators."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_object_detector=False,
        object_detector_locators=["omdet-turbo-locator"],
    )
    assert invocation.object_detector_locators == ()
    assert "object_detector_locators:=" not in " ".join(invocation.argv_template)


def test_deploy_sim_no_object_detector_flag_disables() -> None:
    """``--no-object-detector`` (enable_object_detector=False) turns the leg off."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_object_detector=False,
    )
    assert invocation.enable_object_detector is False
    joined = " ".join(invocation.argv_template)
    assert "enable_object_detector:=false" in joined
    assert "object_detector_manifest:=" not in joined
    for arg in invocation.argv_template:
        if ":=" in arg:
            name, _, value = arg.partition(":=")
            assert value != "", f"empty launch arg {name!r} would abort ros2 launch"


def test_bh_deploy_sim_so101_manifest_driven_bare_twin() -> None:
    """so101 resolves to the shared so100 node, now manifest-driven (issue #191).

    `openral deploy sim` is a digital twin, so the so100/so101 node builds a bare
    `MujocoArmHAL.from_description` from the robot manifest (its `sim.mjcf_uri`)
    rather than opening the Feetech serial bus. After the Phase 2 migration the
    CLI forwards the resolved `robots/so101_follower/robot.yaml` as `robot_yaml`
    + `hal_mode="sim"` (manifest-driven node); `bare_twin_sim=True` keeps it a
    bare twin (no `sim_env_yaml` scene-attach). The SAME node serves both so100
    and so101 from their own MJCF, and the scene YAML never has to define the
    robot simulation.
    """
    invocation = resolve_launch_invocation(
        config=_SO101_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.robot_id == "so101_follower"
    # so101 reuses the so100 ROS lifecycle node (no separate so101 package).
    assert invocation.hal.package == "openral_hal_so100"
    assert invocation.hal.manifest_driven is True
    assert invocation.hal.bare_twin_sim is True
    assert invocation.hal_params["robot_yaml"] == str(
        _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"
    )
    assert invocation.hal_params["hal_mode"] == "sim"
    # Bare twin → no scene-attach and no legacy sim_robot_yaml param.
    assert "sim_env_yaml" not in invocation.hal_params
    assert "sim_robot_yaml" not in invocation.hal_params


def test_bh_deploy_sim_robot_yaml_override_wins() -> None:
    """An explicit `--hal robot_yaml=…` overrides the injected manifest default."""
    invocation = resolve_launch_invocation(
        config=_SO101_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides={"robot_yaml": "/custom/robot.yaml"},
    )
    assert invocation.hal_params["robot_yaml"] == "/custom/robot.yaml"


def test_bh_deploy_sim_hal_executables_have_main_entrypoint() -> None:
    """Every registry HAL node script actually calls ``main()`` when executed.

    Regression guard: ament symlink-installs each ``lifecycle_node.py`` and runs
    it directly as ``__main__``. The so100 node shipped WITHOUT an
    ``if __name__ == "__main__": main()`` guard, so executing it defined the
    node class but never started it — the HAL silently exited 0 and `openral deploy
    sim` came up with no /joint_states (every other node up, HAL absent). Assert
    the guard is present on every registry HAL's executable so a missing
    entry-point can never silently no-op again.
    """
    seen_packages: set[str] = set()
    for hal in _ROBOT_HAL_REGISTRY.values():
        if hal.package in seen_packages:
            continue
        seen_packages.add(hal.package)
        node = _REPO_ROOT / "packages" / hal.package / hal.package / hal.executable
        assert node.is_file(), f"HAL executable not found: {node}"
        assert 'if __name__ == "__main__":' in node.read_text(), (
            f'{node} lacks an `if __name__ == "__main__": main()` guard — '
            "ament runs it as __main__, so without the guard the node never "
            "starts (silent exit 0)."
        )


def test_bh_deploy_sim_enable_slam_forwards_launch_arg_and_flag() -> None:
    """ADR-0025 — --enable-slam toggles the launch arg and the dataclass field."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_slam=True,
    )
    assert invocation.enable_slam is True
    joined = " ".join(invocation.argv_template)
    assert "enable_slam:=true" in joined


def test_bh_deploy_sim_forwards_hal_mode_sim() -> None:
    """ADR-0036 — ``deploy sim`` forwards ``hal_mode:=sim`` so the reasoner's
    action-mode palette gate admits the scene's robosuite-OSC cartesian skills.
    """
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    # Default deploy-sim path: hal_mode defaults to "sim".
    assert invocation.hal_mode == "sim"
    assert "hal_mode:=sim" in " ".join(invocation.argv_template)


def test_bh_deploy_run_forwards_hal_mode_real() -> None:
    """ADR-0036 — ``deploy run`` (``hal_mode="real"``) shells the SAME launch
    with ``hal_mode:=real`` so the reasoner admits only the robot's declared
    ``supported_control_modes``.

    Uses ur5e: a manifest-driven HAL with a real-hardware backend
    (``hal.real``), so real mode resolves instead of raising
    ROSCapabilityMismatch for a sim-only robot.
    """
    invocation = resolve_launch_invocation(
        config=None,
        robot_override="ur5e",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        hal_mode="real",
    )
    assert invocation.hal_mode == "real"
    assert "hal_mode:=real" in " ".join(invocation.argv_template)


def test_deploy_sim_octomap_auto_off_without_depth_sensor() -> None:
    """ADR-0030 — openarm has no depth SensorSpec → octomap auto-disabled."""
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.enable_octomap is False
    assert "enable_octomap:=false" in " ".join(invocation.argv_template)


def test_deploy_sim_octomap_auto_on_with_depth_sensor() -> None:
    """ADR-0030 — panda_mobile declares a depth SensorSpec → octomap auto-on."""
    invocation = resolve_launch_invocation(
        config=_PANDA_MOBILE_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.enable_octomap is True
    assert "enable_octomap:=true" in " ".join(invocation.argv_template)


def test_deploy_sim_octomap_explicit_override_wins() -> None:
    """ADR-0030 — ``--no-enable-octomap`` overrides the depth-sensor auto-on."""
    invocation = resolve_launch_invocation(
        config=_PANDA_MOBILE_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_octomap=False,
    )
    assert invocation.enable_octomap is False
    assert "enable_octomap:=false" in " ".join(invocation.argv_template)


def test_bh_deploy_sim_hal_override_wins() -> None:
    """--hal key=value overrides the per-robot default node param."""
    # issue #191 Phase 3b — openarm's scene params moved to the manifest, so
    # `viewer_enabled` is the remaining overridable node param. The default is
    # True; an explicit override must win.
    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides={"viewer_enabled": False},
    )
    assert invocation.hal_params["viewer_enabled"] is False


def test_bh_deploy_sim_robot_registry_covers_known_robots() -> None:
    """Every robot in _ROBOT_HAL_REGISTRY has a real robot.yaml and matching name."""
    for robot_id, hal in _ROBOT_HAL_REGISTRY.items():
        manifest = _REPO_ROOT / "robots" / robot_id / "robot.yaml"
        assert manifest.is_file(), f"registry references missing manifest: {manifest}"
        description = RobotDescription.from_yaml(str(manifest))
        # e2e contract is satisfied.
        description.validate_for_e2e_pipeline()
        # HAL spec accepts this robot's manifest name.
        assert description.name in hal.supported_robot_names, (
            f"registry: robot_id={robot_id!r} -> {hal.package!r} declares "
            f"supported_robot_names={sorted(hal.supported_robot_names)}, "
            f"but manifest's name={description.name!r}"
        )


def test_bh_deploy_sim_registry_hal_packages_exist_on_disk() -> None:
    """Every registry ``hal.package`` is a real ROS package under ``packages/``.

    Regression guard: the prior coverage test only checked the
    ``robots/<id>/robot.yaml`` manifest and the manifest-name match — it
    never verified the HAL *package* itself ships. That gap let the
    ``so101_follower`` entry point at a ``openral_hal_so101`` ROS package
    that was never created (only ``openral_hal_so100`` exists; the SO-101
    reuses the SO-100 Feetech serial driver). ``openral deploy sim`` then
    failed its ``assert_ros2_packages_discoverable`` preflight with a
    misleading "overlay not sourced / build stale" message. A registry
    entry pointing at a package that does not exist on disk can never be
    built by ``just ros2-build``, so assert the directory + package.xml
    are present.
    """
    packages_root = _REPO_ROOT / "packages"
    for robot_id, hal in _ROBOT_HAL_REGISTRY.items():
        pkg_dir = packages_root / hal.package
        manifest = pkg_dir / "package.xml"
        assert manifest.is_file(), (
            f"registry: robot_id={robot_id!r} -> package={hal.package!r}, "
            f"but {manifest} does not exist. A HAL package that is not on "
            "disk cannot be built or discovered by `openral deploy sim`."
        )
        # The package.xml's <name> must match the registry package name,
        # else `ros2 run <package>` resolves to nothing at launch time.
        assert f"<name>{hal.package}</name>" in manifest.read_text(), (
            f"registry: package={hal.package!r} dir exists but its "
            f"package.xml declares a different <name>."
        )


def test_bh_deploy_sim_hal_robot_mismatch_fails(tmp_path: Path) -> None:
    """A robot.yaml whose `name:` is not in the HAL's supported set fails loud.

    Catches the case where someone adds a new robot directory but the
    HAL registry entry was copied from a different robot — the wrong
    HAL would otherwise silently boot against the wrong manifest.
    """
    # Synthesise a DeployScene whose ``robot_id`` resolves to "openarm",
    # whose manifest name ("openarm_v2") will then mismatch the HAL's
    # repointed ``supported_robot_names``. ``deploy sim --config`` is
    # strict DeployScene (ADR-0041), so no ``task:`` block is included.
    scene_yaml = tmp_path / "scene.yaml"
    scene_yaml.write_text(
        "robot_id: openarm\nscene:\n  id: noop/zero\n  backend: mujoco\n  cameras: []\n"
    )
    # Repoint _ROBOT_HAL_REGISTRY["openarm"].supported_robot_names so
    # the manifest's "openarm_v2" no longer matches — verifies the
    # assertion fires.
    import openral_cli.deploy_sim as ds

    original = ds._ROBOT_HAL_REGISTRY["openarm"]
    try:
        ds._ROBOT_HAL_REGISTRY["openarm"] = ds._HalSpec(
            package=original.package,
            executable=original.executable,
            node_name=original.node_name,
            supported_robot_names=frozenset({"not_openarm_v2"}),
            default_params=original.default_params,
        )
        with pytest.raises(ROSConfigError) as ei:
            resolve_launch_invocation(
                config=scene_yaml,
                robot_override=None,
                dashboard_port=4318,
                reset_to_pose_service=None,
                hal_param_overrides=None,
            )
    finally:
        ds._ROBOT_HAL_REGISTRY["openarm"] = original
    assert "HAL/robot mismatch" in str(ei.value)
    assert "openarm_v2" in str(ei.value)
    assert "not_openarm_v2" in str(ei.value)


def test_bh_deploy_sim_unknown_robot_fails() -> None:
    """Unsupported robot_id raises ROSConfigError with the supported set."""
    with pytest.raises(ROSConfigError) as ei:
        resolve_launch_invocation(
            config=_OPENARM_CONFIG,
            robot_override="nonexistent",
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_param_overrides=None,
        )
    assert "no HAL entry" in str(ei.value)
    assert "openarm" in str(ei.value)


def test_bh_deploy_sim_missing_config_fails() -> None:
    """A missing --config path exits non-zero with a clear typer error."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["deploy", "sim", "--config", "/tmp/does-not-exist.yaml", "--dry-run"],
    )
    assert result.exit_code != 0
    lower = result.output.lower()
    assert "invalid value" in lower
    assert "config" in lower


def test_bh_deploy_sim_missing_robot_in_yaml_fails(tmp_path: Path) -> None:
    """DeployScene without robot_id + no --robot override is rejected.

    ``noop/zero`` is not a fixed-robot scene (no ``SCENES.fixed_robot``
    fallback), so the resolver must report ``robot_id is undefined``.
    """
    bare = tmp_path / "free_axis_scene.yaml"
    bare.write_text("scene:\n  id: noop/zero\n  backend: mujoco\n  cameras: []\n")
    with pytest.raises(ROSConfigError) as ei:
        resolve_launch_invocation(
            config=bare,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_param_overrides=None,
        )
    assert "robot_id is undefined" in str(ei.value)


def test_bh_preflight_palette_deps_silent_when_no_capability_match(tmp_path: Path) -> None:
    """A repo with rskills/ but no capability-matching skills returns silently.

    Uses ``g1`` from the in-tree registry (humanoid: 0 skills match in the
    current registry per ``build_tool_palette``). The preflight should
    not gate on capability misses — that's the reasoner's job to
    surface at on_configure.
    """
    g1_yaml = _REPO_ROOT / "robots" / "g1" / "robot.yaml"
    if not g1_yaml.is_file():
        pytest.skip(f"missing fixture: {g1_yaml}")
    # No raise / no exit — silent return.
    _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=g1_yaml)


def test_bh_preflight_palette_deps_returns_silent_when_no_rskills_dir(tmp_path: Path) -> None:
    """An empty repo root (no ``rskills/``) is a clean no-op, not an error."""
    (tmp_path / "robots").mkdir()
    (tmp_path / "rskills").mkdir()  # exists but empty
    fake_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    _preflight_palette_deps(repo_root=tmp_path, robot_yaml=fake_yaml)


def test_bh_preflight_palette_deps_drops_blocked_non_tty_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """openarm + missing pi05 extras + non-TTY → drop-and-proceed with the hint.

    Real fixture: ``robots/openarm/robot.yaml`` matches two in-tree
    rSkills via ``build_tool_palette`` — ``rskill-pi05-openarm-vision-nf4``
    (``model_family=pi05``, imports ``transformers`` gated behind ``uv sync
    --group sim``) and the family-less ``rskill-moveit-joints``
    (``kind: ros_action``, no policy extras, always importable). Because a
    dispatchable skill survives a pi05 miss, the advisory preflight does
    NOT hard-fail: it drops the blocked pi05 skill and proceeds, printing
    the install hint for the dropped skill. The empty-palette hard-fail
    path (every matching skill blocked → ``typer.Exit(1)``) is covered by
    ``test_bh_preflight_install_cmd_uses_just_sync_all_packages``, which
    force-misses every family. The test asserts on the structured
    proceed-path output (no install attempt) — this is what CI / scripts
    see.

    The printed install command MUST go through ``just sync
    --all-packages --group <X>`` rather than bare ``uv sync --group
    <X>``. Without ``--all-packages``, uv tears down every workspace
    member (openral-core, openral-cli, ...) — and the next ROS launch
    then fails with the exact ``No module named 'openral_core'`` the
    preflight is meant to prevent. ``just sync`` also wraps the
    hf-libero distutils-uninstall repair around the sync.
    """
    from openral_sim.policy_deps import can_import_policy_family

    openarm_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    if not openarm_yaml.is_file():
        pytest.skip(f"missing fixture: {openarm_yaml}")
    ok, _ = can_import_policy_family("pi05")
    if ok:
        pytest.skip(
            "pi05 extras already installed in this venv; this test "
            "asserts the missing-extras branch — install the extras "
            "to exercise the happy path elsewhere"
        )

    # Force non-interactive without touching the underlying TTY of the
    # test runner. sys.stdin.isatty + sys.stdout.isatty both False →
    # preflight skips the typer.confirm prompt entirely.
    import sys as _sys

    # The pi05 miss blocks rskill-pi05-openarm-vision-nf4, but the
    # family-less rskill-moveit-joints stays dispatchable → palette is
    # non-empty → the advisory preflight drops the blocked skill and
    # proceeds (no Exit). Disable auto-install (default=1) so the test
    # exercises the warn-and-proceed path without calling `just sync`.
    monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "0")
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)

    # A surviving dispatchable skill means the preflight returns cleanly.
    _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=openarm_yaml)

    out = capsys.readouterr().out
    assert "proceeding" in out and "dropped from the reasoner palette" in out, (
        "preflight should drop the blocked pi05 skill and proceed (the "
        "family-less rskill-moveit-joints keeps the palette non-empty); "
        f"got:\n{out}"
    )
    assert "just sync --all-packages" in out, (
        "preflight install command must go through `just sync "
        "--all-packages` so workspace members survive AND the hf-libero "
        "uninstall trap is repaired; got:\n"
        f"{out}"
    )
    # pi05 maps to (sim, libero); both groups must appear.
    assert "--group libero" in out and "--group sim" in out, out


def test_bh_preflight_install_cmd_uses_just_sync_all_packages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forced-miss preflight prints ``just sync --all-packages --group <X>``.

    Companion to ``test_bh_preflight_palette_deps_blocks_non_tty_when_extras_missing``
    that does NOT skip when the extras happen to already be installed:
    monkeypatch ``openral_sim.policy_deps.can_import_policy_family`` to
    always say "missing" so the test reliably exercises the
    install-command print path and asserts the regression invariants:

    * goes through ``just sync --all-packages`` (NOT bare ``uv sync``),
      so workspace members survive AND the hf-libero distutils trap is
      repaired before+after;
    * the pi05 family expands to BOTH ``--group libero`` and
      ``--group sim`` (per ``_FAMILY_INSTALL_GROUPS['pi05']``);
    * the pre-fix command shape ``uv sync --group ...`` never appears.
    """
    import sys as _sys

    import typer
    from openral_sim import policy_deps as _pd

    openarm_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    if not openarm_yaml.is_file():
        pytest.skip(f"missing fixture: {openarm_yaml}")

    monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "0")
    monkeypatch.setattr(
        _pd, "can_import_policy_family", lambda _family: (False, "forced miss for test")
    )
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)

    with pytest.raises(typer.Exit) as ei:
        _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=openarm_yaml)
    assert ei.value.exit_code == 1

    out = capsys.readouterr().out
    # The line printed by the non-TTY branch is
    # ``  {shlex.join(install_cmd)}`` (two-space indent). Locate it
    # precisely so the regression guard isn't fooled by per-skill hint
    # strings elsewhere in the output that still read
    # ``uv sync --group sim ...`` (pre-existing wording in
    # ``openral_sim.policy_deps._FAMILY_INSTALL_HINTS``).
    install_lines = [
        ln.strip()
        for ln in out.splitlines()
        if ln.startswith("  ") and ("sync" in ln) and ("--group" in ln)
    ]
    # First matching line is from the per-skill hint; the install_cmd
    # line is the one that begins with the binary name.
    cmd_lines = [ln for ln in install_lines if ln.startswith(("just ", "uv "))]
    assert cmd_lines, f"no install-command line found in:\n{out}"
    cmd_line = cmd_lines[-1]
    assert cmd_line.startswith("just sync --all-packages "), (
        "preflight install command must be `just sync --all-packages "
        "--group <X>`; got:\n"
        f"{cmd_line}"
    )
    assert "--group libero" in cmd_line and "--group sim" in cmd_line, cmd_line
    # Installing a policy group must be ADDITIVE — `--inexact` preserves
    # packages from sibling groups already in the venv (the omdet detector's
    # `timm`, robosuite, rldx's pyzmq/msgpack). Without it, `uv sync --group
    # rldx` is exact-match and silently uninstalls `timm`, so the OmDet-Turbo
    # detector then ImportErrors on every frame and /openral/perception/objects
    # stays empty (issue #12). Mirrors the robocasa AUTO_INSTALL plan, which
    # already uses --inexact for exactly this reason.
    assert "--inexact" in cmd_line, (
        "preflight install command must pass --inexact so installing one group "
        "does not uninstall another run-critical group's packages (e.g. the "
        f"omdet detector's timm); got:\n{cmd_line}"
    )


def test_bh_preflight_accept_propagates_auto_install_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting the preflight install sets ``OPENRAL_AUTO_INSTALL_DEPS=1``.

    The operator's "yes, install" must propagate to the launched graph: the scene
    backend's ``on_configure`` asset/dep install (openral_sim._assets, gated on
    that env var) would otherwise re-prompt and block before the viewer opens
    (ADR-0034 step-4 fix). ``run_launch_invocation`` copies ``os.environ`` into the
    launch env, so setting it here carries the single consent answer downstream.
    """
    import contextlib
    import sys as _sys

    import typer
    from openral_sim import policy_deps as _pd

    # _preflight_palette_deps sets OPENRAL_AUTO_INSTALL_DEPS=1 directly in the
    # real os.environ (production side-effect, asserted below). Register it with
    # monkeypatch up front so teardown restores the pre-test state and the var
    # does not leak into later tests (e.g. the typer.confirm interleave test,
    # whose prompt path that var would silently bypass).
    monkeypatch.delenv("OPENRAL_AUTO_INSTALL_DEPS", raising=False)

    openarm_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    if not openarm_yaml.is_file():
        pytest.skip(f"missing fixture: {openarm_yaml}")

    # No pre-set consent; interactive TTY; user accepts; the install subprocess
    # "succeeds" (process-boundary fake). monkeypatch.delenv restores the absent
    # var on teardown, so this never leaks into other tests.
    monkeypatch.delenv("OPENRAL_AUTO_INSTALL_DEPS", raising=False)
    monkeypatch.setattr(
        _pd, "can_import_policy_family", lambda _family: (False, "forced miss for test")
    )
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **_k: subprocess.CompletedProcess(a[0] if a else [], 0),
    )

    # All matched skills stay blocked after the faked install → preflight raises
    # typer.Exit; the consent env var must have been set BEFORE that.
    with contextlib.suppress(typer.Exit):
        _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=openarm_yaml)

    assert os.environ.get("OPENRAL_AUTO_INSTALL_DEPS") == "1"


def test_bh_preflight_warns_and_proceeds_when_some_skills_importable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial miss drops the blocked skill(s) and PROCEEDS (no exit).

    The preflight is advisory: the reasoner already drops unimportable
    rSkills at on_configure and runs the rest. franka_panda's palette is
    robot-WIDE (matches act / molmoact2 / pi05 / rldx / smolvla / xvla),
    so blocking just one family must NOT take the whole launch down — the
    importable remainder is still dispatchable. Regression guard for the
    pre-fix behaviour where ANY missing extra hard-exited(1).
    """
    import sys as _sys

    from openral_sim import policy_deps as _pd

    franka_yaml = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
    if not franka_yaml.is_file():
        pytest.skip(f"missing fixture: {franka_yaml}")

    # Block ONLY the rldx family; every other family is reported
    # importable so the palette keeps a dispatchable remainder.
    monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "0")
    monkeypatch.setattr(
        _pd,
        "can_import_policy_family",
        lambda family: (False, "blocked for test") if family == "rldx" else (True, None),
    )
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)

    # Must NOT raise — blocked rldx skill(s) are dropped, the rest proceed.
    _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=franka_yaml)

    out = capsys.readouterr().out
    assert "proceeding" in out, f"expected a warn-and-proceed message; got:\n{out}"
    assert "will be\ndropped" in out or "will be dropped" in out.replace("\n", " "), out
    # The rldx install hint is still surfaced so the operator can enable it.
    assert "--group rldx" in out, out


def test_bh_preflight_auto_installs_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OPENRAL_AUTO_INSTALL_DEPS=1 installs missing extras + continues.

    Non-TTY (CI / background) path: the env var is the same unattended-
    consent signal openral_sim._assets / _deps honour. We fake the
    ``just sync`` subprocess (process boundary, CLAUDE.md §1.11) and flip
    the import probe to "importable" once it runs, then assert the
    preflight re-probes clean and returns without exiting.
    """
    import sys as _sys
    import types

    from openral_cli import deploy_sim as _ds
    from openral_sim import policy_deps as _pd

    openarm_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    if not openarm_yaml.is_file():
        pytest.skip(f"missing fixture: {openarm_yaml}")

    state = {"installed": False}
    calls: list[list[str]] = []

    def fake_can_import(_family: str) -> tuple[bool, str | None]:
        return (True, None) if state["installed"] else (False, "missing before install")

    def fake_run(cmd: list[str], *, check: bool = False, cwd: str | None = None) -> object:
        calls.append(cmd)
        state["installed"] = True  # the install "succeeded"
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "1")
    monkeypatch.setattr(_pd, "can_import_policy_family", fake_can_import)
    monkeypatch.setattr(_ds.subprocess, "run", fake_run)
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)

    # Must NOT raise — install runs, re-probe clears the block.
    _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=openarm_yaml)

    assert calls, "auto-install should have shelled out to `just sync`"
    assert calls[0][:3] == ["just", "sync", "--all-packages"], calls[0]
    out = capsys.readouterr().out
    assert "extras installed" in out, out


def test_bh_preflight_auto_install_failure_exits_with_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed auto-install surfaces the non-zero exit, never drops silently."""
    import sys as _sys
    import types

    import typer
    from openral_cli import deploy_sim as _ds
    from openral_sim import policy_deps as _pd

    openarm_yaml = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
    if not openarm_yaml.is_file():
        pytest.skip(f"missing fixture: {openarm_yaml}")

    def fake_run(cmd: list[str], *, check: bool = False, cwd: str | None = None) -> object:
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "1")
    monkeypatch.setattr(
        _pd, "can_import_policy_family", lambda _family: (False, "forced miss for test")
    )
    monkeypatch.setattr(_ds.subprocess, "run", fake_run)
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)

    with pytest.raises(typer.Exit) as ei:
        _preflight_palette_deps(repo_root=_REPO_ROOT, robot_yaml=openarm_yaml)
    assert ei.value.exit_code == 7


def test_bh_assert_ros2_packages_discoverable_all_present() -> None:
    """When every package resolves, the assertion is a no-op."""
    fake = {
        "openral_rskill_ros": "/ws/install/openral_rskill_ros",
        "openral_hal_franka": "/ws/install/openral_hal_franka",
    }
    assert_ros2_packages_discoverable(
        ["openral_rskill_ros", "openral_hal_franka"],
        prefix_lookup=fake.get,
    )


def test_bh_assert_ros2_packages_discoverable_overlay_not_sourced() -> None:
    """Missing packages → ROSConfigError naming each and telling the user to source the overlay.

    Reproduces the operator failure seen with ``openral deploy sim --robot
    franka_panda``: ``ros2`` is on PATH but ``install/setup.bash`` was
    never sourced, so ``ros2 launch`` searches only ``/opt/ros/jazzy``
    and the OpenRAL packages are missing.
    """
    with pytest.raises(ROSConfigError) as ei:
        assert_ros2_packages_discoverable(
            ["openral_rskill_ros", "openral_hal_franka"],
            prefix_lookup=lambda _pkg: None,
        )
    msg = str(ei.value)
    assert "openral_rskill_ros" in msg
    assert "openral_hal_franka" in msg
    assert "just ros2-build" in msg
    assert "source install/setup.bash" in msg
    # Disambiguate the user's "is it called rskill now?" guess.
    assert "not ``rskill``" in msg


def test_bh_assert_ros2_packages_discoverable_partial_only_lists_missing() -> None:
    """A stale build where only the HAL is missing reports just the HAL."""
    fake = {"openral_rskill_ros": "/ws/install/openral_rskill_ros"}
    with pytest.raises(ROSConfigError) as ei:
        assert_ros2_packages_discoverable(
            ["openral_rskill_ros", "openral_hal_franka"],
            prefix_lookup=fake.get,
        )
    msg = str(ei.value)
    assert "openral_hal_franka" in msg
    assert "openral_rskill_ros" not in msg.split("ROS package(s):", 1)[1].split(".", 1)[0]


@pytest.mark.skipif(
    shutil.which("ros2") is None or os.environ.get("OPENRAL_DEPLOY_SIM_SMOKE") != "1",
    reason=(
        "ros2 not on PATH or OPENRAL_DEPLOY_SIM_SMOKE=1 not set. The live "
        "smoke test additionally needs ``just ros2-build`` + ``source "
        "install/setup.bash``."
    ),
)
def test_bh_deploy_sim_live_launch_openarm() -> None:
    """End-to-end: actually invoke ``ros2 launch`` against the openarm graph."""
    import subprocess
    import sysconfig
    import tempfile

    import yaml as _yaml

    invocation = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides={"viewer_enabled": False},
    )
    hal_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        _yaml.safe_dump({"/**": {"ros__parameters": invocation.hal_params}}, hal_tmp)
        hal_tmp.close()
        argv = [
            arg.replace("HAL_PARAMS_FILE_PLACEHOLDER", hal_tmp.name)
            for arg in invocation.argv_template
        ]
        # Mirror ``deploy_sim_command``: ros2 launch runs under the
        # system Python, so export OPENRAL_VENV_SITE + PYTHONPATH so
        # the launch's deferred openral_core / openral_safety imports
        # resolve against the workspace venv.
        env = os.environ.copy()
        venv_site = sysconfig.get_paths()["purelib"]
        env["OPENRAL_VENV_SITE"] = venv_site
        env["PYTHONPATH"] = (
            f"{venv_site}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else venv_site
        )
        # The launch graph has no natural exit (lifecycle nodes tick
        # indefinitely until SIGINT/SIGTERM); we deliberately time it
        # out at 20s and treat that as "launch came up healthy". The
        # critical signal — that the safety kernel loaded from ROS
        # params — is captured in the partial stdout via subprocess's
        # output= buffer on TimeoutExpired.
        try:
            completed = subprocess.run(argv, timeout=20, check=False, capture_output=True, env=env)
        except subprocess.TimeoutExpired as exc:
            output = (exc.output or b"").decode(errors="replace")
            assert "envelope loaded from ROS params" in output, output
            return
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    finally:
        Path(hal_tmp.name).unlink(missing_ok=True)


def test_bh_prepare_launch_env_defaults_expandable_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_prepare_launch_env (shared by `deploy sim` AND `deploy run`) defaults the
    expandable-segments CUDA allocator so the runtime_node's VLA load doesn't
    fragment-OOM on a tight 8 GiB GPU (ADR-0034). ``setdefault`` → an operator
    override wins. Both env-var spellings are set (cross-torch-version).

    Regression guard: the fix originally lived only in run_launch_invocation, but
    `deploy sim` shells through deploy_sim_command's own inline env build — so the
    var never reached the runtime_node. Both paths now route through this helper.
    """
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = _prepare_launch_env()
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["OPENRAL_VENV_SITE"]  # venv site is exported for editable .pth resolution

    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.9")
    assert _prepare_launch_env()["PYTORCH_ALLOC_CONF"] == "garbage_collection_threshold:0.9"


def test_bh_run_launch_invocation_sets_expandable_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_launch_invocation defaults PYTORCH_ALLOC_CONF=expandable_segments:True.

    The runtime_node loads VLA weights on the GPU; on a tight 8 GiB card the
    default CUDA allocator fragments and OOMs at the forward pass even for an
    NF4 model that otherwise fits (molmoact2-libero-nf4 peaks ~7.6 GiB).
    expandable_segments recovers the fragmented headroom (ADR-0034 OOM fix).
    ``setdefault`` so an operator override wins. Both env-var spellings are set
    (PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF) for cross-torch-version safety.
    """
    import openral_cli.deploy_sim as _ds

    inv = resolve_launch_invocation(
        config=_OPENARM_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides={"viewer_enabled": False},
    )
    captured: dict[str, dict[str, str]] = {}

    def _fake_run_launch(argv: list[str], env: dict[str, str], **_kw: object) -> int:
        captured["env"] = env
        return 0

    monkeypatch.setattr(_ds, "_run_launch", _fake_run_launch)

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert run_launch_invocation(inv, run_preflight=False) == 0
    assert captured["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    # An explicit operator setting must win (setdefault, not overwrite).
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.9")
    assert run_launch_invocation(inv, run_preflight=False) == 0
    assert captured["env"]["PYTORCH_ALLOC_CONF"] == "garbage_collection_threshold:0.9"


# ── Launch process-group teardown (ADR-0027 orphan-reap hardening) ──────────
#
# These exercise the real teardown path with real child processes (no
# mocks, CLAUDE.md §1.11): the bug they guard against is deploy_sim's
# launch tree orphaning onto /tf_static + the GPU when the CLI exits.


def _pgid_alive(pgid: int) -> bool:
    """True while any process in ``pgid`` is still alive (signal 0 probe)."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_terminate_launch_group_reaps_whole_session_tree() -> None:
    """``_terminate_launch_group`` SIGKILLs a child tree that ignores SIGINT.

    Spawns a session leader (``start_new_session=True``) that traps
    SIGINT and forks a grandchild — the shape of ``ros2 launch`` + a node
    that doesn't shut down cleanly. With a short grace the helper must
    escalate to SIGKILL and leave nothing in the group alive.
    """
    # Session leader ignores SIGINT, spawns a child, then sleeps. Mirrors
    # a launch tree where graceful SIGINT does NOT bring the tree down.
    script = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    pgid = os.getpgid(proc.pid)
    try:
        assert _pgid_alive(pgid)
        # grace_s small: SIGINT is ignored, so the helper must escalate.
        _terminate_launch_group(proc, grace_s=1.0)
        # Give the kernel a beat to reap after SIGKILL.
        for _ in range(50):
            if not _pgid_alive(pgid):
                break
            time.sleep(0.1)
        assert not _pgid_alive(pgid), "launch group survived teardown"
    finally:
        with __import__("contextlib").suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)


def test_terminate_launch_group_graceful_sigint_path() -> None:
    """A SIGINT-respecting tree exits within grace without needing SIGKILL."""
    # Default SIGINT handling: the process dies on the forwarded SIGINT.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    try:
        _terminate_launch_group(proc, grace_s=10.0)
        assert proc.poll() is not None, "child should have exited on SIGINT"
        assert not _pgid_alive(pgid)
    finally:
        with __import__("contextlib").suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)


def test_run_launch_returns_exit_code_and_leaves_no_orphans() -> None:
    """``_run_launch`` runs a real command, returns its code, reaps the group."""
    rc = _run_launch(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        dict(os.environ),
        grace_s=5.0,
    )
    assert rc == 7


def test_orphan_needles_cover_tf_publishers_and_sidecar() -> None:
    """The reaper matches the three process types that used to leak.

    Regression guard for the rldx-rc365 bug: the static_transform_publisher
    (z=0.4 /tf_static poisoning), the URDF robot_state_publisher, and the
    rldx GPU sidecar were absent from the needle set, so they survived
    every startup reap. Sample cmdlines are the real argv signatures
    observed live via ``/proc/<pid>/cmdline``.
    """
    static_tf = (
        "/opt/ros/jazzy/lib/tf2_ros/static_transform_publisher --x 0.0 --y 0.0 "
        "--z 0.4 --frame-id base_link --child-frame-id panda_link0 "
        "--ros-args -r __node:=static_base_link_to_panda_link0"
    )
    rsp = (
        "/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher "
        "--ros-args -r __node:=robot_state_publisher --params-file /tmp/launch_params_x"
    )
    sidecar = (
        "/home/u/.cache/openral/rldx-sidecar/source/.venv/bin/python "
        "/home/u/.cache/openral/rldx-sidecar/boot_server.py"
    )
    assert _cmdline_is_openral_graph_process(static_tf)
    assert _cmdline_is_openral_graph_process(rsp)
    assert _cmdline_is_openral_graph_process(sidecar)
    # An unrelated user process must NOT match.
    assert not _cmdline_is_openral_graph_process("/usr/bin/python3 -m http.server 8000")
    # A non-openral static_transform_publisher (different executable path
    # prefix is still tf2_ros, so this WOULD match) — but an unrelated
    # editor / shell must not.
    assert not _cmdline_is_openral_graph_process("vim /etc/hosts")


def test_scan_params_derived_from_robot_yaml_lidar() -> None:
    """ADR-0025 single source — deploy_sim maps the panda_mobile
    robot.yaml ``lidar_2d`` sensor onto the HAL ``scan_*`` ROS params
    instead of hardcoding a scan envelope in ``_ROBOT_HAL_REGISTRY``."""
    description = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    )
    lidar = description.lidar_sensor
    assert lidar is not None, "panda_mobile robot.yaml must declare a lidar_2d sensor"
    assert _scan_params_from_description(description) == {
        "scan_publish_rate_hz": lidar.rate_hz,
        "scan_n_beams": lidar.n_channels,
        "scan_max_range_m": lidar.range_max_m,
        "scan_min_range_m": lidar.range_min_m,
    }
    # The registry no longer hardcodes a scan envelope (single source).
    assert "scan_n_beams" not in _ROBOT_HAL_REGISTRY["panda_mobile"].default_params


def test_scan_params_empty_for_robot_without_lidar() -> None:
    """The derivation is generic + a no-op: a manipulator with no
    ``lidar_2d`` sensor yields no ``scan_*`` params."""
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    assert description.lidar_sensor is None
    assert _scan_params_from_description(description) == {}


def test_nav2_param_overrides_from_robot_yaml() -> None:
    """ADR-0025 — Nav2 robot_radius + inflation_radius + motion_model derive
    from the panda_mobile robot.yaml (footprint_radius + base_kinematics) so
    the Nav2 bringup needs no hand-vendored per-robot param values."""
    description = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    )
    assert description.footprint_radius == pytest.approx(0.35)
    assert description.base_kinematics == "omni"
    # inflation_radius = footprint_radius (0.35) + NAV2_INFLATION_CLEARANCE_M
    # (0.05) = 0.40, kept >= the costmap-discretised circumscribed radius.
    assert description.nav2_param_overrides() == {
        "robot_radius": "0.35",
        "inflation_radius": "0.400",
        "motion_model": "Omni",
    }


def test_nav2_param_overrides_empty_for_fixed_base_arm() -> None:
    """A fixed-base arm declares no mobile-base props -> no Nav2 overrides."""
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    assert description.footprint_radius is None
    assert description.base_kinematics is None
    assert description.nav2_param_overrides() == {}


def test_footprint_polygon_accepts_valid_and_rejects_too_few_points() -> None:
    """footprint_polygon is optional; when set it needs >= 3 base-frame XY points."""
    base = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    ).model_dump()

    ok = RobotDescription.model_validate(
        {**base, "footprint_polygon": [[0.35, 0.25], [-0.35, 0.25], [-0.35, -0.25]]}
    )
    assert ok.footprint_polygon == [(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25)]

    with pytest.raises(ValidationError):
        RobotDescription.model_validate({**base, "footprint_polygon": [[0.0, 0.0], [1.0, 1.0]]})

    with pytest.raises(ValidationError):
        RobotDescription.model_validate({**base, "footprint_polygon": []})


def test_footprint_polygon_rejects_non_finite_vertices() -> None:
    """NaN/inf vertices are rejected (they would silently render nothing)."""
    base = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    ).model_dump()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            RobotDescription.model_validate(
                {**base, "footprint_polygon": [[0.0, 0.0], [1.0, 0.0], [0.0, bad]]}
            )


def test_footprint_polygon_defaults_none_for_fixed_base_arm() -> None:
    """A fixed-base arm declares no footprint polygon."""
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    assert description.footprint_polygon is None


def test_panda_mobile_declares_real_footprint_polygon() -> None:
    """panda_mobile carries the OmronMobileBase collision box as a 0.70x0.50 m
    rectangle (half-extents 0.35 x 0.25), centered on base_link."""
    description = RobotDescription.from_yaml(
        str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")
    )
    assert description.footprint_polygon == [
        (0.35, 0.25),
        (-0.35, 0.25),
        (-0.35, -0.25),
        (0.35, -0.25),
    ]
