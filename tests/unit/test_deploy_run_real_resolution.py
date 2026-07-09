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

import pytest
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core.exceptions import ROSCapabilityMismatch


def _resolve(robot_id: str, mode: str) -> object:
    return resolve_launch_invocation(
        config=None,
        robot_override=robot_id,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode=mode,
    )


class TestRealModeResolution:
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
        from openral_core.exceptions import ROSConfigError

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
            "robot_id: so101_follower\n"
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
            "robot_id: so101_follower\n"
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
