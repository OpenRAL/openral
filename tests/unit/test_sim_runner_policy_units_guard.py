"""``sim run`` refuses a unit-converting rSkill on a scene that does not convert.

``sim run`` applies no ``PolicyIOCodec``; only scenes registered with
``converts_policy_units=True`` own the degrees / gripper-scale / joint-order
conversion. Real manifest + real robot description, no doubles.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_rskill._policy_io import PolicyIOCodec
from openral_rskill.loader import load_rskill_manifest
from openral_sim.registry import SCENES
from openral_sim.sim_runner import _check_policy_units_owned

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = load_rskill_manifest(str(_REPO / "rskills" / "rskill-smolvla-so101-eraser_place-bf16"))
_ROBOT = RobotDescription.from_yaml(str(_REPO / "robots" / "so101_follower" / "robot.yaml"))


def test_degrees_checkpoint_codec_is_not_identity() -> None:
    assert not PolicyIOCodec.from_manifest(_MANIFEST, _ROBOT).is_identity
    assert PolicyIOCodec().is_identity


def test_non_converting_scene_raises() -> None:
    assert not SCENES.meta("libero_spatial").get("converts_policy_units")
    with pytest.raises(ROSConfigError, match=r"joint_units_are_degrees=True.*gripper_scale=100"):
        _check_policy_units_owned(_MANIFEST, _ROBOT, "libero_spatial")


@pytest.mark.parametrize("scene_id", ["so101_box", "tabletop_push", "openarm_tabletop_pnp"])
def test_converting_scene_passes(scene_id: str) -> None:
    assert SCENES.meta(scene_id).get("converts_policy_units") is True
    _check_policy_units_owned(_MANIFEST, _ROBOT, scene_id)
