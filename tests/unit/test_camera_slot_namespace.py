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
    ImagePreprocessing,
    RobotDescription,
    RSkillManifest,
    required_vla_camera_slots,
    sensor_name_to_slot,
)
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import (
    checkpoint_image_keys,
    manifest_camera_slots,
    resolve_image_preprocessing,
)
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


# ─── Per-adapter checkpoint keys: real manifests + real robots, no weights ────


def _slots(skill: str, robot: str) -> tuple[str, ...]:
    return required_vla_camera_slots(_skill(skill), _robot(robot))


def _ckpt_keys(skill: str, robot: str) -> tuple[str, ...]:
    manifest = _skill(skill)
    return checkpoint_image_keys(resolve_image_preprocessing(manifest, {}), _slots(skill, robot))


@pytest.mark.parametrize(
    ("skill", "robot", "expected"),
    [
        ("xvla-libero", "franka_panda", ("observation.images.image", "observation.images.image2")),
        (
            "gr00t-n17-libero",
            "franka_panda",
            ("observation.images.image", "observation.images.wrist_image"),
        ),
        ("diffusion-pusht", "pusht_2d", ("observation.image",)),
    ],
)
def test_in_process_adapter_checkpoint_keys(
    skill: str, robot: str, expected: tuple[str, ...]
) -> None:
    assert _ckpt_keys(skill, robot) == expected


@pytest.mark.parametrize("skill", ["lingbot-vla2-robotwin", "lingbot-vla-4b-robotwin"])
def test_lingbot_slots_map_onto_the_sidecar_views(skill: str) -> None:
    from openral_sim.policies.lingbot_vla2 import _resolve_camera_io

    slots = _slots(skill, "aloha_agilex")
    assert _resolve_camera_io(_skill(skill), {}, list(slots)) == (
        ("camera1", "camera2", "camera3"),
        ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    )


def test_lingbot_without_aliases_fails_loud() -> None:
    from openral_sim.policies.lingbot_vla2 import _resolve_camera_io

    skill = _skill("lingbot-vla2-robotwin")
    bare = skill.model_copy(update={"image_preprocessing": None})
    with pytest.raises(ROSConfigError, match="cam_high"):
        _resolve_camera_io(bare, {}, ["camera1", "camera2", "camera3"])


@pytest.mark.parametrize(
    ("skill", "robot"),
    [
        ("rldx1-ft-libero-nf4", "franka_panda"),
        ("rldx1-ft-gr1-nf4", "gr1"),
        ("rldx1-ft-rc365-nf4", "panda_mobile"),
        ("rldx1-ft-simpler-widowx-nf4", "widowx"),
    ],
)
def test_rldx_manifest_aliases_state_the_layout_wire_keys(skill: str, robot: str) -> None:
    from openral_sim.policies.rldx import (
        _RLDX_LAYOUT_VIDEO_KEYS,
        _resolve_state_layout,
        _resolve_video_keys,
    )

    manifest = _skill(skill)
    layout = _resolve_state_layout(manifest)
    slots = _slots(skill, robot)
    aliased = _resolve_video_keys(layout, resolve_image_preprocessing(manifest, {}), slots)
    # The manifest states the layout default explicitly; dropping it changes nothing.
    default = _resolve_video_keys(layout, ImagePreprocessing(), slots)
    assert aliased == default == _RLDX_LAYOUT_VIDEO_KEYS[layout]


def test_openvla_defaults_to_the_manifest_slot() -> None:
    manifest = _skill("openvla-oft-simpler-widowx-nf4")
    assert manifest_camera_slots(manifest) == ("camera1",)
    assert _slots("openvla-oft-simpler-widowx-nf4", "widowx") == ("camera1",)


def test_molmoact2_orders_slots_by_the_checkpoint_camera_keys() -> None:
    from openral_sim.policies.molmoact2 import _checkpoint_camera_order

    manifest = _skill("molmoact2-libero-nf4")
    slots = _slots("molmoact2-libero-nf4", "franka_panda")
    keys = checkpoint_image_keys(resolve_image_preprocessing(manifest, {}), slots)
    # allenai/MolmoAct2-LIBERO norm_stats.json metadata_by_tag.libero.camera_keys.
    ckpt = ("observation.images.image", "observation.images.wrist_image")
    assert _checkpoint_camera_order(slots, keys, ckpt) == ("camera1", "camera2")
    assert _checkpoint_camera_order(slots, keys, ckpt[::-1]) == ("camera2", "camera1")
    # A checkpoint that records no order (MolmoAct2-SO100_101) keeps slot order.
    assert _checkpoint_camera_order(slots, keys, ()) == slots
    with pytest.raises(ROSConfigError, match="image2"):
        _checkpoint_camera_order(slots, keys, ("observation.images.image2",))


def test_molmoact2_libero_aliases_match_the_cached_norm_stats() -> None:
    import json

    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache("allenai/MolmoAct2-LIBERO", "norm_stats.json")
    if not isinstance(path, str):
        pytest.skip("allenai/MolmoAct2-LIBERO norm_stats.json not in the local HF cache")
    with open(path, encoding="utf-8") as fh:
        ckpt = tuple(json.load(fh)["metadata_by_tag"]["libero"]["camera_keys"])
    assert set(_ckpt_keys("molmoact2-libero-nf4", "franka_panda")) == set(ckpt)


def _frames(*slots: str) -> dict[str, np.ndarray]:
    return {s: np.full((8, 8, 3), i, dtype=np.uint8) for i, s in enumerate(slots)}


def test_xvla_batch_takes_frames_by_slot() -> None:
    torch = pytest.importorskip("torch")
    from openral_core import VLASpec
    from openral_sim.policies.xvla import _XVLAAdapter

    keys = _ckpt_keys("xvla-libero", "franka_panda")
    adapter = _XVLAAdapter(
        spec=VLASpec(id="xvla", weights_uri="rskills/xvla-libero"),
        device="cpu",
        _policy=None,
        _env_pre=None,
        _policy_pre=None,
        _policy_post=None,
        _env_post=None,
        _torch=torch,
        _converters=None,
        _camera_keys=("camera1", "camera2"),
        _image_keys=keys,
    )
    obs = {"images": _frames("camera1", "camera2"), "raw": {}}
    batch = adapter._build_raw_batch({}, obs, "pick")
    assert float(batch["observation.images.image"].max()) == 0.0
    assert float(batch["observation.images.image2"].max()) == pytest.approx(1 / 255)


def test_gr00t_batch_takes_frames_by_slot() -> None:
    torch = pytest.importorskip("torch")
    from openral_core import VLASpec
    from openral_sim.policies.gr00t import _GrootAdapter

    adapter = _GrootAdapter(
        spec=VLASpec(id="gr00t", weights_uri="rskills/gr00t-n17-libero"),
        device="cpu",
        _policy=None,
        _preprocessor=None,
        _postprocessor=None,
        _torch=torch,
        _camera_keys=("camera1", "camera2"),
        _image_input_keys=_ckpt_keys("gr00t-n17-libero", "franka_panda"),
    )
    obs = {"images": _frames("camera1", "camera2"), "state": np.zeros(8, dtype=np.float32)}
    batch = adapter._build_batch(obs, "pick")
    assert float(batch["observation.images.wrist_image"].max()) == pytest.approx(1 / 255)
    assert float(batch["observation.images.image"].max()) == 0.0


def test_diffusion_batch_renames_the_slot_to_the_checkpoint_key() -> None:
    torch = pytest.importorskip("torch")
    from openral_core import VLASpec
    from openral_sim.policies.diffusion import _DiffusionAdapter

    (key,) = _ckpt_keys("diffusion-pusht", "pusht_2d")
    adapter = _DiffusionAdapter(
        spec=VLASpec(id="diffusion", weights_uri="rskills/diffusion-pusht"),
        device="cpu",
        _policy=None,
        _preprocessor=None,
        _postprocessor=None,
        _torch=torch,
        _image_key="camera1",
        _batch_key=key,
    )
    batch = adapter._build_batch({"images": _frames("camera1"), "state": [0.0, 0.0]}, "push")
    assert "observation.image" in batch
