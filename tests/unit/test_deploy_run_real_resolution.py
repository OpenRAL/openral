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
from typing import Any

import pytest
from openral_cli.deploy_sim import LaunchInvocation, resolve_launch_invocation
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


#: A real depth driver's cloud. A real deploy with a depth camera and nothing pinned is
#: refused before the extrinsic gate (no in-tree node publishes a real cloud), so these
#: tests pin one the way a real scene's ``runtime.octomap_cloud_topic`` does.
_DRIVER_CLOUD = "/camera/depth/color/points"


def _resolve_real(robot_id: str, **kwargs: Any) -> LaunchInvocation:
    """``deploy run`` of ``robot_id`` through a minimal scene that pins the driver cloud."""
    import tempfile

    import yaml

    scene = {
        "scene": {"id": "extrinsic_preflight"},
        "robot_id": robot_id,
        "runtime": {"enable_octomap": True, "octomap_cloud_topic": _DRIVER_CLOUD},
    }
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "scene.yaml"
        config.write_text(yaml.safe_dump(scene), encoding="utf-8")
        return resolve_launch_invocation(
            config=config,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
            **kwargs,
        )


class TestDepthExtrinsicPreflight:
    """A real deploy with the world-voxel check on refuses an undeclared depth mount.

    OpenRAL does not measure a depth camera's extrinsic; the operator does, and declares
    it in the unit overlay. Galaxea A1 is a real-HAL robot whose manifest ``wrist``
    camera (on ``arm_seg6``) is RGB in-tree; it is made a depth camera here, in a copy
    served via ``OPENRAL_ROBOTS_DIR``, which is how ``deploy run`` resolves manifests.
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

    @staticmethod
    def _write_unit(robot_dir: Path, unit: str, sensors: list[dict[str, object]]) -> None:
        import yaml

        (robot_dir / "units").mkdir(exist_ok=True)
        doc = {"robot_id": "galaxea_a1", "unit": unit, "sensors": sensors}
        (robot_dir / "units" / f"{unit}.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

    def test_refuses_the_manifests_nominal_mount(self, robot_dir: Path) -> None:
        """No ``units/``: the only mount is the manifest's nominal one, which proves nothing."""
        with pytest.raises(ROSConfigError, match=r"wrist: mount is the manifest's nominal"):
            _resolve_real("galaxea_a1")

    def test_refuses_a_unit_that_declares_no_mount(
        self, robot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A unit overlay that only binds the camera still leaves its pose nominal."""
        self._write_unit(robot_dir, "cell_a", [{"name": "wrist"}])
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "cell_a")
        with pytest.raises(ROSConfigError, match=r"wrist: mount is the manifest's nominal"):
            _resolve_real("galaxea_a1")

    def test_launches_with_the_units_declared_mount(
        self, robot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_unit(
            robot_dir,
            "cell_a",
            [{"name": "wrist", "static_transform_xyz_rpy": [0.06, 0.0, 0.03, 0.0, 0.45, 0.0]}],
        )
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "cell_a")
        inv = _resolve_real("galaxea_a1")
        assert "enable_octomap_kernel_check:=true" in inv.argv_template

    def test_not_gated_when_the_world_voxel_check_is_off(self, robot_dir: Path) -> None:
        inv = _resolve_real("galaxea_a1", enable_octomap_kernel_check=False)
        assert "enable_octomap_kernel_check:=false" in inv.argv_template

    def test_the_committed_openarm_units_gate_on_their_declared_mount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thor's overlay declares head_zed's calibrated mount, so bare `deploy run` with that
        unit passes the gate (its correctness is the operator's); Orin's does not yet, so the
        same launch refuses there."""
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
        inv = _resolve_real("openarm")
        assert "enable_octomap_kernel_check:=true" in inv.argv_template
        monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "orin")
        with pytest.raises(ROSConfigError, match=r"head_zed: mount is the manifest's nominal"):
            _resolve_real("openarm")
