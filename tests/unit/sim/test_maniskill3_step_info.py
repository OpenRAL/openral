"""ManiSkill3 / SimplerEnv step info exposes the env's ``success`` under ``is_success``."""

from __future__ import annotations

import numpy as np
from openral_sim.backends.maniskill3 import _step_info
from openral_sim.rollout import DEFAULT_SUCCESS_KEY


def test_success_is_exposed_under_the_default_key() -> None:
    # ManiSkill3 batches every info value; SimplerEnv carrot-on-plate reports these keys.
    info = {"success": np.array([True]), "src_on_target": np.array([True])}

    out = _step_info(info)

    assert out[DEFAULT_SUCCESS_KEY] is True
    assert bool(out["success"]) is True


def test_an_env_provided_is_success_wins() -> None:
    out = _step_info({"success": np.array([True]), "is_success": np.array([False])})

    assert bool(out[DEFAULT_SUCCESS_KEY]) is False


def test_no_success_key_adds_nothing() -> None:
    assert DEFAULT_SUCCESS_KEY not in _step_info({"elapsed_steps": np.array([3])})
