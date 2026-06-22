"""Deploy-sim camera-slot realignment in rskill_runner_node (ADR-0034).

Deploy-sim keys ``WorldState.image_frames`` by the manifest SENSOR NAME
(the ``/openral/cameras/<name>/image`` topic basename), but VLA adapters
resolve their ``camera_keys`` and look up ``obs["images"]`` by the VLA
slot (``camera1`` / ``camera2`` / ...) — the LIBERO convention
``openral sim run`` and the rldx adapter already use. Without a realignment
a manifest whose RGB sensors are descriptively named (franka: ``front`` /
``wrist`` per ADR-0069) hands the pi0.5 adapter
``obs["images"]["front"]`` while it looks up ``camera1`` and its
``cam_alias`` maps ``camera1 -> image`` for the checkpoint — so the
policy sees no frames.

These tests pin the two helpers that realign the namespaces plus the
``_build_runtime_skill_from_manifest`` scene-camera override, using real
``robots/franka_panda/robot.yaml`` + real ``SensorFrame`` objects (no
mocks, CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# The module under test bundles the full lifecycle node (rclpy / IDL).
# Skip cleanly when those aren't sourced — the helpers are still defined
# at module top-level, just unreachable.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import RobotDescription
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import FrameEncoding, SensorFrame
from openral_rskill_ros.rskill_runner_node import (
    _build_runtime_skill_from_manifest,
    _decode_image_frames,
    _sensor_name_to_vla_slot,
    _vla_camera_slots,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"))


def _rgb_frame(name: str, *, fill: int, h: int = 2, w: int = 2) -> SensorFrame:
    data = np.full((h, w, 3), fill, dtype=np.uint8).tobytes()
    return SensorFrame(
        sensor_id=name,
        stamp_monotonic_ns=1,
        stamp_wall_ns=2,
        encoding=FrameEncoding.RGB8,
        width=w,
        height=h,
        channels=3,
        data=data,
    )


class TestVlaCameraSlots:
    def test_franka_slots_in_manifest_order(self) -> None:
        # front -> observation.images.camera1, wrist -> ...camera2.
        assert _vla_camera_slots(_franka()) == ("camera1", "camera2")

    def test_franka_name_to_slot_map(self) -> None:
        assert _sensor_name_to_vla_slot(_franka()) == {
            "front": "camera1",
            "wrist": "camera2",
        }

    def test_none_description_is_empty(self) -> None:
        assert _vla_camera_slots(None) == ()
        assert _sensor_name_to_vla_slot(None) == {}


class TestDecodeImageFrames:
    def test_remaps_sensor_names_to_vla_slots(self) -> None:
        slot_map = {"front": "camera1", "wrist": "camera2"}
        frames = {
            "front": _rgb_frame("front", fill=10),
            "wrist": _rgb_frame("wrist", fill=20),
        }
        images = _decode_image_frames(frames, slot_map)
        assert set(images) == {"camera1", "camera2"}
        assert int(images["camera1"][0, 0, 0]) == 10
        assert int(images["camera2"][0, 0, 0]) == 20
        assert images["camera1"].shape == (2, 2, 3)

    def test_unmapped_sensor_passes_through_under_its_name(self) -> None:
        images = _decode_image_frames({"extra_cam": _rgb_frame("extra_cam", fill=7)}, {})
        assert set(images) == {"extra_cam"}
        assert int(images["extra_cam"][0, 0, 0]) == 7

    def test_frames_without_data_are_skipped(self) -> None:
        frame = SensorFrame(
            sensor_id="front",
            stamp_monotonic_ns=1,
            stamp_wall_ns=2,
            encoding=FrameEncoding.RGB8,
            width=2,
            height=2,
            topic="/openral/cameras/front/image",  # data=None
        )
        images = _decode_image_frames({"front": frame}, {"front": "camera1"})
        assert images == {}


class TestBuildRuntimeSkillSceneCameras:
    def test_overrides_sensor_name_scene_cameras_with_vla_slots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sensor-name ``scene_cameras`` (what runtime_node passes) → VLA slots.

        ``runtime_node`` forwards ``camera_names`` (manifest sensor names,
        e.g. ``front`` / ``wrist`` per ADR-0069) as ``scene_cameras``; the
        adapter needs the VLA slots (``camera1`` / ``camera2``) so its
        ``cam_alias`` maps ``camera1 -> image`` for the checkpoint. Capture
        the ``env_cfg.scene.cameras`` the policy factory receives by
        monkey-patching ``make_policy`` at the lerobot/torch process
        boundary (CLAUDE.md §1.11) to raise before the heavy import.
        """
        import openral_sim.factory as _sim_factory

        yaml_path = _REPO_ROOT / "rskills" / "pi05-libero-nf4" / "rskill.yaml"
        if not yaml_path.is_file():
            pytest.skip(f"missing in-tree fixture: {yaml_path}")

        captured: dict[str, object] = {}

        def _capture(env_cfg: object) -> object:
            captured["cameras"] = tuple(env_cfg.scene.cameras)  # type: ignore[attr-defined]
            raise ImportError("stop before torch import")

        monkeypatch.setattr(_sim_factory, "make_policy", _capture)

        with pytest.raises(ROSRuntimeError):  # factory translates the ImportError
            _build_runtime_skill_from_manifest(
                yaml_path=yaml_path,
                prompt="pick up the milk",
                scene_cameras=("front", "wrist"),  # sensor names, as runtime_node passes
                description=_franka(),
            )

        assert captured["cameras"] == ("camera1", "camera2")

    def test_vla_slots_supersede_caller_supplied_slots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A description with RGB sensors always wins, even if the caller passed slots.

        Documents the intentional supersede in ``_build_runtime_skill_from_manifest``:
        the manifest's VLA slots replace any caller-supplied ``scene_cameras``
        when the description declares RGB sensors. Here the supplied value
        already equals the derived slots, so the override is idempotent — but
        the assertion pins that the manifest, not the caller, is the source of
        truth.
        """
        import openral_sim.factory as _sim_factory

        yaml_path = _REPO_ROOT / "rskills" / "pi05-libero-nf4" / "rskill.yaml"
        if not yaml_path.is_file():
            pytest.skip(f"missing in-tree fixture: {yaml_path}")

        captured: dict[str, object] = {}

        def _capture(env_cfg: object) -> object:
            captured["cameras"] = tuple(env_cfg.scene.cameras)  # type: ignore[attr-defined]
            raise ImportError("stop before torch import")

        monkeypatch.setattr(_sim_factory, "make_policy", _capture)

        with pytest.raises(ROSRuntimeError):  # factory translates the ImportError
            _build_runtime_skill_from_manifest(
                yaml_path=yaml_path,
                prompt="pick up the milk",
                scene_cameras=("stale", "values"),  # caller's value is discarded
                description=_franka(),
            )

        assert captured["cameras"] == ("camera1", "camera2")

    def test_falls_back_to_passed_scene_cameras_without_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No description → keep the caller's ``scene_cameras`` untouched."""
        import openral_sim.factory as _sim_factory

        yaml_path = _REPO_ROOT / "rskills" / "pi05-libero-nf4" / "rskill.yaml"
        if not yaml_path.is_file():
            pytest.skip(f"missing in-tree fixture: {yaml_path}")

        captured: dict[str, object] = {}

        def _capture(env_cfg: object) -> object:
            captured["cameras"] = tuple(env_cfg.scene.cameras)  # type: ignore[attr-defined]
            raise ImportError("stop before torch import")

        monkeypatch.setattr(_sim_factory, "make_policy", _capture)

        with pytest.raises(ROSRuntimeError):  # factory translates the ImportError
            _build_runtime_skill_from_manifest(
                yaml_path=yaml_path,
                prompt="pick up the milk",
                scene_cameras=("camera1", "camera2"),
                description=None,
            )

        assert captured["cameras"] == ("camera1", "camera2")
