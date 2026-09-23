"""One camera namespace — the VLA slot — across `sim run`, `deploy sim`, `deploy run`.

``image_preprocessing.aliases`` keys are slots (the ``vla_feature_key`` suffix),
so the same alias names the same camera on every path. Real robot manifests and
real in-tree rSkill manifests only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import (
    RobotDescription,
    RSkillManifest,
    required_vla_camera_slots,
    sensor_name_to_slot,
)
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import resolve_image_preprocessing
from openral_sim.sim_runner import _policy_scene_cameras, _rekey_obs_images

_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted(_ROOT.glob("rskills/*/rskill.yaml"))


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROOT / "robots" / name / "robot.yaml"))


def _skill(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_ROOT / "rskills" / name / "rskill.yaml"))


@pytest.mark.parametrize("path", _MANIFESTS, ids=lambda p: p.parent.name)
def test_every_in_tree_alias_key_is_a_declared_slot(path: Path) -> None:
    manifest = RSkillManifest.from_yaml(str(path))
    if manifest.image_preprocessing is not None:
        assert manifest.image_preprocessing.unknown_alias_slots(manifest.sensors_required) == []


def test_sensor_name_alias_key_fails_loud() -> None:
    skill = _skill("pi05-libero-int8")
    assert skill.image_preprocessing is not None
    bad = skill.model_copy(
        update={
            "image_preprocessing": skill.image_preprocessing.model_copy(
                update={"aliases": {"top": "image"}}
            )
        }
    )
    with pytest.raises(ROSConfigError, match=r"\['top'\]"):
        resolve_image_preprocessing(bad, {})


def test_required_slots_trim_to_the_skill() -> None:
    # Franka exposes camera3 (VLABench); a LIBERO skill wants only camera1/2.
    franka = _robot("franka_panda")
    assert tuple(sensor_name_to_slot(franka).values()) == ("camera1", "camera2", "camera3")
    assert required_vla_camera_slots(_skill("smolvla-libero"), franka) == ("camera1", "camera2")


def test_sim_rekeys_robot_sensor_scene_cameras_to_slots() -> None:
    # RoboTwin scene cameras are aloha_agilex sensor names.
    cams, rekey = _policy_scene_cameras(
        ["top", "wrist_left", "wrist_right"], _robot("aloha_agilex"), _skill("smolvla-robotwin")
    )
    assert cams == ["camera1", "camera2", "camera3"]
    assert rekey == {"top": "camera1", "wrist_left": "camera2", "wrist_right": "camera3"}


def test_sim_empty_scene_cameras_default_to_the_skills_slots() -> None:
    cams, rekey = _policy_scene_cameras([], _robot("franka_panda"), _skill("pi05-libero-int8"))
    assert (cams, rekey) == (["camera1", "camera2"], {})


def test_sim_leaves_non_sensor_scene_cameras_alone() -> None:
    # VLABench names its renders camera1..3 — not franka sensor names.
    cams, rekey = _policy_scene_cameras(
        ["camera1", "camera2", "camera3"], _robot("franka_panda"), _skill("smolvla-vlabench")
    )
    assert (cams, rekey) == (["camera1", "camera2", "camera3"], {})


def test_rekey_obs_images_renamed_frame_wins() -> None:
    top = np.zeros((2, 2, 3), dtype=np.uint8)
    stale = np.ones((2, 2, 3), dtype=np.uint8)
    obs = {"images": {"top": top, "camera1": stale, "other": stale}, "state": [0.0]}
    out = _rekey_obs_images(obs, {"top": "camera1"})
    assert out["images"]["camera1"] is top
    assert set(out["images"]) == {"camera1", "other"}
    assert out["state"] == [0.0]
    assert _rekey_obs_images(obs, {}) is obs
