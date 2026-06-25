"""RobotEnvironment.thumbnail_hz defaults to 5 Hz, allows 0, rejects negatives."""

from __future__ import annotations

import pytest
from openral_core import RobotEnvironment
from openral_core.schemas import HalConfig, TaskSpec
from pydantic import ValidationError


def _env(**kw: object) -> RobotEnvironment:
    return RobotEnvironment(
        robot_id="so100_follower",
        hal=HalConfig(adapter="so100_follower"),
        task=TaskSpec(
            id="pick_cube/red",
            scene_id="pick_cube/red",
            instruction="pick up the red cube",
        ),
        **kw,
    )


def test_default_is_25hz() -> None:
    assert _env().thumbnail_hz == 25.0


def test_zero_disables_allowed() -> None:
    assert _env(thumbnail_hz=0.0).thumbnail_hz == 0.0


def test_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _env(thumbnail_hz=-1.0)


def test_round_trip_preserves_field() -> None:
    env = _env(thumbnail_hz=10.0)
    assert RobotEnvironment.model_validate(env.model_dump()).thumbnail_hz == 10.0
