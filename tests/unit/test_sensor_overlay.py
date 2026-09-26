"""Per-unit / per-host sensor overlays (``robots/<id>/units/<unit>.yaml``).

A robot manifest owns a sensor's identity (name, modality, frames); a ``RobotUnit`` overlay
owns what differs per physical unit or host (device path, driver topic, calibrated mount,
intrinsics). Exercised on the real SO-101 and OpenArm unit files, the committed deploy
scenes that select them, and a sim robot (Franka Panda) with no units at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from openral_core import (
    ROBOT_UNIT_ENV,
    DeployScene,
    RobotDescription,
    SensorOverlay,
    apply_sensor_overlays,
    load_robot_unit,
    publishing_sensors,
    resolve_sensor_overlays,
)
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]
_SO101 = _ROOT / "robots" / "so101_follower" / "robot.yaml"
_OPENARM = _ROOT / "robots" / "openarm" / "robot.yaml"
_FRANKA = _ROOT / "robots" / "franka_panda" / "robot.yaml"


@pytest.fixture(autouse=True)
def _no_host_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host's own ``$OPENRAL_ROBOT_UNIT`` must not leak into these assertions."""
    monkeypatch.delenv(ROBOT_UNIT_ENV, raising=False)


def _effective(robot_yaml: Path, unit: str | None) -> dict[str, object]:
    sensors = RobotDescription.from_yaml(str(robot_yaml)).sensors
    overlays = resolve_sensor_overlays(robot_yaml, unit, required=True)
    return {s.name: s for s in apply_sensor_overlays(sensors, overlays)}


def test_so101_bench_scene_binds_its_host_cameras_through_its_unit() -> None:
    scene = DeployScene.from_yaml(str(_ROOT / "scenes" / "deploy" / "so101_bench.yaml"))
    assert scene.robot_unit == "bench_laptop"
    sensors = _effective(_SO101, scene.robot_unit)
    top = sensors["top"].deploy_binding  # type: ignore[attr-defined]
    assert top.backend_params["device"] == (
        "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11:1.0-video-index0"
    )
    # The shared manifest carries no host path any more.
    assert all(s.deploy_binding is None for s in RobotDescription.from_yaml(str(_SO101)).sensors)
    bound = publishing_sensors(list(sensors.values()), [], "real")  # type: ignore[arg-type]
    assert [s.name for s in bound] == ["top", "wrist"]


def test_openarm_units_keep_frames_and_pin_the_thor_mount() -> None:
    manifest = {s.name: s for s in RobotDescription.from_yaml(str(_OPENARM)).sensors}
    thor, orin = _effective(_OPENARM, "thor"), _effective(_OPENARM, "orin")
    for unit in (thor, orin):
        zed = unit["head_zed"]
        assert zed.frame_id == manifest["head_zed"].frame_id  # type: ignore[attr-defined]
        assert zed.parent_frame == "openarm_base"  # type: ignore[attr-defined]
        assert zed.deploy_binding.backend_params["topic"] == (  # type: ignore[attr-defined]
            "/zed/zed_node/depth/depth_registered"
        )
    # Each unit pins its own fitted mount (2026-09-25), so re-fitting one cell's camera
    # never moves the other's; both differ from the manifest's nominal mount.
    assert thor["head_zed"].static_transform_xyz_rpy == (  # type: ignore[attr-defined]
        -0.0206,
        -0.0023,
        0.2157,
        -0.0123,
        1.1823,
        0.0062,
    )
    assert orin["head_zed"].static_transform_xyz_rpy == (  # type: ignore[attr-defined]
        -0.0170,
        -0.0098,
        0.2163,
        -0.0123,
        1.1823,
        -0.0076,
    )
    nominal = manifest["head_zed"].static_transform_xyz_rpy
    assert nominal not in (
        thor["head_zed"].static_transform_xyz_rpy,
        orin["head_zed"].static_transform_xyz_rpy,
    )  # type: ignore[attr-defined]
    bench = DeployScene.from_yaml(str(_ROOT / "scenes" / "deploy" / "openarm_bench.yaml"))
    assert bench.robot_unit == "orin"


def test_a_real_deploy_of_a_robot_with_units_refuses_without_one() -> None:
    with pytest.raises(ROSConfigError, match="none is selected"):
        resolve_sensor_overlays(_OPENARM, None, required=True)
    assert resolve_sensor_overlays(_OPENARM, None, required=False) == []


def test_the_env_var_wins_over_the_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROBOT_UNIT_ENV, "thor")
    overlays = resolve_sensor_overlays(_OPENARM, "orin", required=True)
    zed = next(o for o in overlays if o.name == "head_zed")
    assert zed.static_transform_xyz_rpy is not None
    assert zed.static_transform_xyz_rpy[0] == -0.0206  # thor's mount, not orin's


def test_a_sim_robot_without_units_needs_none_and_takes_an_overlay() -> None:
    assert resolve_sensor_overlays(_FRANKA, None, required=True) == []
    sensors = RobotDescription.from_yaml(str(_FRANKA)).sensors
    target = sensors[0]
    assert target.intrinsics is not None
    fix = SensorOverlay(
        name=target.name,
        static_transform_xyz_rpy=(0.1, 0.0, 0.5, 0.0, 0.3, 0.0),
        intrinsics=target.intrinsics.model_copy(update={"fx": 123.0}),
    )
    (out, *rest) = apply_sensor_overlays(sensors, [fix])
    assert out.static_transform_xyz_rpy == (0.1, 0.0, 0.5, 0.0, 0.3, 0.0)
    assert out.intrinsics is not None and out.intrinsics.fx == 123.0
    assert (out.name, out.modality, out.frame_id, out.parent_frame) == (
        target.name,
        target.modality,
        target.frame_id,
        target.parent_frame,
    )
    assert rest == sensors[1:]


@pytest.mark.parametrize(
    "field", [{"parent_frame": "x"}, {"frame_id": "x"}, {"modality": "depth"}, {"rate_hz": 1.0}]
)
def test_an_overlay_cannot_change_identity_or_semantics(field: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SensorOverlay.model_validate({"name": "top", **field})


def test_an_overlay_must_name_a_manifest_sensor_once() -> None:
    sensors = RobotDescription.from_yaml(str(_SO101)).sensors
    with pytest.raises(ROSConfigError, match="name no robot-manifest sensor"):
        apply_sensor_overlays(sensors, [SensorOverlay(name="overhead")])
    with pytest.raises(ROSConfigError, match="twice"):
        apply_sensor_overlays(sensors, [SensorOverlay(name="top"), SensorOverlay(name="top")])


def test_a_unit_file_must_belong_to_its_robot(tmp_path: Path) -> None:
    robot_dir = tmp_path / "so101_follower"
    shutil.copytree(_SO101.parent / "units", robot_dir / "units")
    shutil.copy(_SO101, robot_dir / "robot.yaml")
    unit_path = robot_dir / "units" / "bench_laptop.yaml"
    doc = yaml.safe_load(unit_path.read_text(encoding="utf-8"))
    doc["robot_id"] = "openarm"
    unit_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ROSConfigError, match="expected robot_id='so101_follower'"):
        load_robot_unit(robot_dir / "robot.yaml", "bench_laptop")
    with pytest.raises(ROSConfigError, match="available: bench_laptop"):
        load_robot_unit(robot_dir / "robot.yaml", "second_host")
