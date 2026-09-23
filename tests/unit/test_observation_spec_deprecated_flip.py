"""``ObservationSpec.image_flip_180`` is deprecated: it still loads, but warns.

Uses a real robot manifest (``robots/so101_follower/robot.yaml``) with the dead
key re-added, since no in-tree manifest carries it any more.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest
import yaml
from openral_core import RobotDescription

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "robots" / "so101_follower" / "robot.yaml"


def test_image_flip_180_warns_but_loads(tmp_path: pathlib.Path) -> None:
    data = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    data["observation_spec"]["image_flip_180"] = True
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="image_flip_180 is deprecated"):
        desc = RobotDescription.from_yaml(str(path))

    assert desc.observation_spec is not None
    assert desc.observation_spec.image_flip_180 is True


def test_manifest_without_image_flip_180_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        RobotDescription.from_yaml(str(_MANIFEST))
