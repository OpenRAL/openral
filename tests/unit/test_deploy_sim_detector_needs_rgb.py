"""``deploy sim`` turns the implicit object detector off on a robot with no RGB camera.

The detector reads one of the robot's own RGB cameras and the launch refuses a
robot without one; the default must not make ``deploy sim`` fail for UR5e & co.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import RobotDescription

_REPO = Path(__file__).resolve().parents[2]


def _scene(tmp_path: Path) -> Path:
    cfg = tmp_path / "ur5e_tabletop.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "scene": {"id": "tabletop_push", "backend": "mujoco"},
                "robot_id": "ur5e",
                "safety": {
                    "workspace_box_min_xyz": [-0.5, -0.5, 0.0],
                    "workspace_box_max_xyz": [0.5, 0.5, 0.8],
                },
            }
        )
    )
    return cfg


def _invoke(cfg: Path, **kw: object):  # type: ignore[no-untyped-def]  # reason: test helper
    return resolve_launch_invocation(
        config=cfg,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        # A real detector manifest, so the missing-backend downgrade never masks the RGB rule.
        object_detector_manifest=str(_REPO / "rskills/omdet-turbo-indoor/rskill.yaml"),
        **kw,
    )


def test_ur5e_declares_no_rgb_sensor() -> None:
    robot = RobotDescription.from_yaml(str(_REPO / "robots/ur5e/robot.yaml"))
    assert not any(s.modality == "rgb" for s in robot.sensors)


def test_implicit_detector_is_disabled(tmp_path: Path) -> None:
    assert _invoke(_scene(tmp_path)).enable_object_detector is False


def test_detector_stays_on_for_a_robot_with_rgb() -> None:
    """Control: the same default on SO-101 (RGB cameras) keeps the leg on."""
    assert _invoke(_REPO / "scenes/deploy/so101_box.yaml").enable_object_detector is True
