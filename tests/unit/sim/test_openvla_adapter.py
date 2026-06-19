"""Unit tests for the OpenVLA / OpenVLA-OFT adapter's pure helpers.

These cover the action de-normalization (BOUNDS_Q99) and the OpenVLA prompt
template — the parts that need no GPU / weights, so they run in the unit tier.
The full ``predict_action`` chunk path is exercised live in
``tests/sim/test_widowx_openvla_simpler.py`` (GPU-gated).
"""

from __future__ import annotations

import numpy as np

from openral_sim.policies.openvla import _decode_prompt, _unnormalize_action

# RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood embedded ``bridge_orig`` action
# stats (config.json): 6 EE deltas (masked → rescaled) + gripper (mask False →
# passthrough).
_BRIDGE = {
    "action": {
        "q01": [-0.02872725, -0.04170350, -0.02609386, -0.08092105, -0.09288700, -0.20718276, 0.0],
        "q99": [0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.07792851, 0.20382574, 1.0],
        "mask": [True, True, True, True, True, True, False],
    }
}


def test_unnormalize_maps_minus_one_to_q01() -> None:
    out = _unnormalize_action(np.array([-1, -1, -1, -1, -1, -1, -1.0], dtype=np.float32), _BRIDGE)
    np.testing.assert_allclose(out[:6], _BRIDGE["action"]["q01"][:6], atol=1e-5)


def test_unnormalize_maps_plus_one_to_q99() -> None:
    out = _unnormalize_action(np.array([1, 1, 1, 1, 1, 1, 1.0], dtype=np.float32), _BRIDGE)
    np.testing.assert_allclose(out[:6], _BRIDGE["action"]["q99"][:6], atol=1e-5)


def test_unnormalize_midpoint_is_mean_of_bounds() -> None:
    out = _unnormalize_action(np.zeros(7, dtype=np.float32), _BRIDGE)
    q01 = np.asarray(_BRIDGE["action"]["q01"][:6])
    q99 = np.asarray(_BRIDGE["action"]["q99"][:6])
    np.testing.assert_allclose(out[:6], 0.5 * (q01 + q99), atol=1e-5)


def test_unnormalize_passes_masked_gripper_through_unchanged() -> None:
    # mask[6] is False → dim 6 is NOT rescaled, passed through as-is.
    out = _unnormalize_action(np.array([0, 0, 0, 0, 0, 0, 0.42], dtype=np.float32), _BRIDGE)
    assert out[6] == np.float32(0.42)


def test_decode_prompt_lowercases_and_wraps() -> None:
    assert _decode_prompt("Put The Carrot On The Plate") == (
        "In: What action should the robot take to put the carrot on the plate?\nOut: "
    )


def test_decode_prompt_strips_surrounding_whitespace() -> None:
    assert _decode_prompt("  pick up the cube  ") == (
        "In: What action should the robot take to pick up the cube?\nOut: "
    )
