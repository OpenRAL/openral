"""Unit tests for ``resolve_launch_invocation(hal_mode="real")`` — the
``openral deploy run`` resolution contract.

These pin the *resolution layer* (which argv + HAL params the real-mode launch
gets) without running ``ros2 launch`` — the live launch is HIL-verified on a
robot host. Real mode must: build from a bare ``robot_id`` (no sim scene
config), forward ``hal_mode="real"`` to manifest-driven nodes, NOT inject the
sim digital-twin (so the so100/so101 node opens its serial bus), and fail fast
for a simulation-only robot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_cli.deploy_sim import LaunchInvocation, resolve_launch_invocation
from openral_core import SensorSpec
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError


def _resolve(robot_id: str, mode: str) -> LaunchInvocation:
    return resolve_launch_invocation(
        config=None,
        robot_override=robot_id,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode=mode,
    )


class TestRealModeResolution:
    def test_galaxea_a1_scene_resolves_to_real_lifecycle_package(self) -> None:
        config = Path("scenes/deploy/galaxea_a1_bench.yaml")
        inv = resolve_launch_invocation(
            config=config,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
        )
        assert inv.hal.package == "openral_hal_node"
        assert inv.hal.executable == "lifecycle_node.py"
        assert inv.hal_params["hal_mode"] == "real"
        assert str(inv.hal_params["robot_yaml"]).endswith("robots/galaxea_a1/robot.yaml")
        assert "enable_reasoner:=false" in inv.argv_template

    def test_direct_rskill_scene_rejects_initial_task(self) -> None:
        with pytest.raises(ROSConfigError, match=r"runtime\.enable_reasoner=true"):
            resolve_launch_invocation(
                config=Path("scenes/deploy/galaxea_a1_bench.yaml"),
                robot_override=None,
                dashboard_port=4318,
                reset_to_pose_service=None,
                hal_mode="real",
                initial_task_prompt="pick the mango",
            )

    def test_manifest_robot_real_forwards_hal_mode(self) -> None:
        inv = _resolve("franka_panda", "real")
        assert inv.hal_params["hal_mode"] == "real"
        assert "sim_robot_yaml" not in inv.hal_params  # no sim twin in real mode

    def test_so100_real_opens_serial_no_twin(self) -> None:
        inv = _resolve("so100_follower", "real")
        # so100 is now manifest-driven (issue #191): real mode forwards
        # hal_mode="real"; build_hal constructs SO100FollowerHAL with port /
        # calibrate_on_connect from the manifest's hal.parameters. No sim twin.
        assert inv.hal_params["hal_mode"] == "real"
        assert "sim_robot_yaml" not in inv.hal_params
        assert "sim_env_yaml" not in inv.hal_params

    def test_sim_only_robot_real_fails_fast(self) -> None:
        with pytest.raises(ROSCapabilityMismatch, match="g1"):
            _resolve("g1", "real")

    def test_no_robot_and_no_config_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="robot_id is undefined"):
            resolve_launch_invocation(
                config=None,
                robot_override=None,
                dashboard_port=4318,
                reset_to_pose_service=None,
                hal_mode="real",
            )

    def test_real_mode_config_forwards_workcell_json(self, tmp_path) -> None:
        config = tmp_path / "deploy.yaml"
        config.write_text(
            "scene:\n"
            "  id: so101_box\n"
            "  backend: mujoco\n"
            "robot_id: so100_follower\n"
            "safety:\n"
            "  max_force_n: 5.0\n",
            encoding="utf-8",
        )
        inv = resolve_launch_invocation(
            config=config,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
        )
        assert any(arg.startswith("workcell_json:=") for arg in inv.argv_template)

    def test_scene_hal_binding_feeds_hal_params(self, tmp_path) -> None:
        """A DeployScene ``hal:`` binding lands in hal_params so
        ``deploy run --config <scene>`` needs no ``--hal`` (port + calibration)."""
        (tmp_path / "calibration").mkdir()
        (tmp_path / "calibration" / "so_follower.json").write_text("{}", encoding="utf-8")
        config = tmp_path / "deploy.yaml"
        config.write_text(
            "scene:\n  id: so101_bench\n"
            "robot_id: so101_follower\nrobot_unit: bench_laptop\n"
            "hal:\n"
            "  defaults:\n"
            "    port: /dev/ttyACM0\n"
            "    id: so_follower\n"
            "    calibration_dir: calibration\n"
            "    calibrate_on_connect: false\n",
            encoding="utf-8",
        )
        inv = resolve_launch_invocation(
            config=config,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
        )
        assert inv.hal_params["port"] == "/dev/ttyACM0"
        assert inv.hal_params["id"] == "so_follower"
        # A relative calibration_dir resolves against the scene file's dir.
        assert inv.hal_params["calibration_dir"] == str((tmp_path / "calibration").resolve())

    def test_cli_hal_override_beats_scene_binding(self, tmp_path) -> None:
        """Precedence: ``--hal`` > scene ``hal`` > ``robot.yaml`` defaults."""
        config = tmp_path / "deploy.yaml"
        config.write_text(
            "scene:\n  id: so101_bench\n"
            "robot_id: so101_follower\nrobot_unit: bench_laptop\n"
            "hal:\n  defaults:\n    port: /dev/ttyACM0\n",
            encoding="utf-8",
        )
        inv = resolve_launch_invocation(
            config=config,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
            hal_param_overrides={"port": "/dev/ttyUSB9"},
        )
        assert inv.hal_params["port"] == "/dev/ttyUSB9"


class TestSimModeUnchanged:
    def test_so100_sim_builds_bare_twin(self) -> None:
        inv = _resolve("so100_follower", "sim")
        # issue #191 Phase 2 — manifest-driven + bare_twin_sim: sim forwards
        # robot_yaml + hal_mode="sim" and builds a bare MujocoArmHAL twin (no
        # scene-attach), replacing the legacy sim_robot_yaml injection.
        assert inv.hal_params["hal_mode"] == "sim"
        assert inv.hal_params["robot_yaml"].endswith("so100_follower/robot.yaml")
        assert "sim_robot_yaml" not in inv.hal_params
        assert "sim_env_yaml" not in inv.hal_params

    def test_manifest_robot_sim_forwards_sim_mode(self) -> None:
        inv = _resolve("franka_panda", "sim")
        assert inv.hal_params["hal_mode"] == "sim"


def _passing_report(spec: SensorSpec, base_frame: str) -> dict[str, object]:
    """A ``tools/depth_extrinsic_check.py check`` report that clears ``spec``'s pose."""
    from openral_core.depth_extrinsic import (
        MAX_HEIGHT_ERR_M,
        MAX_MARKER_ERR_M,
        MAX_TILT_DEG,
        MIN_MARKERS,
    )

    markers = [
        {"expected_xy": [0.4, y], "measured_xy": [0.4, y], "error_m": 0.002} for y in (-1, 1)
    ]
    return {
        "sensor": spec.name,
        "parent_frame": spec.parent_frame,
        "frame_id": spec.frame_id,
        "base_frame": base_frame,
        "static_transform_xyz_rpy": list(spec.static_transform_xyz_rpy or ()),
        "criteria": {
            "max_tilt_deg": MAX_TILT_DEG,
            "max_height_err_m": MAX_HEIGHT_ERR_M,
            "max_marker_err_m": MAX_MARKER_ERR_M,
            "min_markers": MIN_MARKERS,
        },
        "residuals": {"tilt_deg": 0.1, "height_err_m": 0.001, "markers": markers},
        "passed": True,
        "failures": [],
    }


class TestDepthExtrinsicPreflight:
    """A real deploy with the world-voxel check on refuses an unverified depth extrinsic.

    Galaxea A1 is a real-HAL robot whose manifest ``wrist`` camera (on ``arm_seg6``) is
    RGB in-tree; it is made a depth camera here, in a copy served via
    ``OPENRAL_ROBOTS_DIR``, which is how ``deploy run`` resolves manifests.
    """

    @pytest.fixture
    def robot_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        import yaml

        data = yaml.safe_load(Path("robots/galaxea_a1/robot.yaml").read_text(encoding="utf-8"))
        (wrist,) = [s for s in data["sensors"] if s["name"] == "wrist"]
        wrist["modality"] = "depth"
        wrist["static_transform_xyz_rpy"] = [0.05, 0.0, 0.03, 0.0, 0.4, 0.0]
        out = tmp_path / "robots" / "galaxea_a1"
        out.mkdir(parents=True)
        (out / "robot.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))
        return out

    def _write_report(self, robot_dir: Path, **edit: object) -> None:
        import json

        from openral_core import RobotDescription
        from openral_core.depth_extrinsic import extrinsic_report_path

        desc = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
        (spec,) = [s for s in desc.sensors if s.name == "wrist"]
        report = {**_passing_report(spec, desc.base_frame), **edit}
        path = extrinsic_report_path(robot_dir / "robot.yaml", "wrist")
        path.parent.mkdir()
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_refuses_without_a_report(self, robot_dir: Path) -> None:
        with pytest.raises(ROSConfigError, match=r"wrist: no extrinsic report"):
            _resolve("galaxea_a1", "real")

    def test_launches_with_a_verified_report(self, robot_dir: Path) -> None:
        self._write_report(robot_dir)
        inv = _resolve("galaxea_a1", "real")
        assert "enable_octomap_kernel_check:=true" in inv.argv_template

    def test_refuses_a_stale_report(self, robot_dir: Path) -> None:
        self._write_report(robot_dir, static_transform_xyz_rpy=[0.05, 0.0, 0.03, 0.0, 0.41, 0.0])
        with pytest.raises(ROSConfigError, match=r"wrist: manifest pose .* != checked pose"):
            _resolve("galaxea_a1", "real")

    def test_refuses_a_failed_report(self, robot_dir: Path) -> None:
        self._write_report(robot_dir, passed=False, failures=["tilt"])
        with pytest.raises(ROSConfigError, match=r"wrist: report did not pass"):
            _resolve("galaxea_a1", "real")

    def test_not_gated_when_the_world_voxel_check_is_off(self, robot_dir: Path) -> None:
        inv = resolve_launch_invocation(
            config=None,
            robot_override="galaxea_a1",
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
            enable_octomap_kernel_check=False,
        )
        assert "enable_octomap_kernel_check:=false" in inv.argv_template

    def test_the_committed_openarm_refuses_until_calibrated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """head_zed is a depth camera with no committed report: bare `deploy run` refuses."""
        if Path("robots/openarm/calibration/thor/head_zed_extrinsic.json").exists():
            pytest.skip("a head_zed calibration report is committed")
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
        with pytest.raises(ROSConfigError, match=r"head_zed: no extrinsic report .*/thor/"):
            _resolve("openarm", "real")

    def test_a_unit_is_gated_on_its_own_pose_and_report(
        self, robot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``units/``, the preflight checks the selected unit's overlaid pose against
        ``calibration/<unit>/``: the manifest-pose report at the unit-less path clears
        nothing, and a report must name the unit it was measured on."""
        import json

        import yaml
        from openral_core import RobotDescription
        from openral_core.depth_extrinsic import extrinsic_report_path

        unit_pose = [0.06, 0.0, 0.03, 0.0, 0.45, 0.0]
        (robot_dir / "units").mkdir()
        doc = {
            "robot_id": "galaxea_a1",
            "unit": "cell_a",
            "sensors": [{"name": "wrist", "static_transform_xyz_rpy": unit_pose}],
        }
        (robot_dir / "units" / "cell_a.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        self._write_report(robot_dir)  # the manifest's nominal pose, unit-less path
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "cell_a")
        with pytest.raises(ROSConfigError, match=r"wrist: no extrinsic report .*/cell_a/"):
            _resolve("galaxea_a1", "real")

        desc = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
        (spec,) = [s for s in desc.sensors if s.name == "wrist"]
        spec = spec.model_copy(update={"static_transform_xyz_rpy": tuple(unit_pose)})
        path = extrinsic_report_path(robot_dir / "robot.yaml", "wrist", "cell_a")
        path.parent.mkdir()
        report = _passing_report(spec, desc.base_frame)
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(ROSConfigError, match=r"wrist: report is for unit None, not 'cell_a'"):
            _resolve("galaxea_a1", "real")

        path.write_text(json.dumps({**report, "unit": "cell_a"}), encoding="utf-8")
        inv = _resolve("galaxea_a1", "real")
        assert "enable_octomap_kernel_check:=true" in inv.argv_template
