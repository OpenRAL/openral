"""Unit tests for the OpenVLA / OpenVLA-OFT adapter's pure helpers.

These cover the action de-normalization (BOUNDS_Q99) and the OpenVLA prompt
template — the parts that need no GPU / weights, so they run in the unit tier.
The full ``predict_action`` chunk path is exercised live in
``tests/sim/test_widowx_openvla_simpler.py`` (GPU-gated).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim.policies.openvla import (
    _as_action_chunk,
    _decode_prompt,
    _extra_bool,
    _install_tokenization_compat,
    _postprocess_action_chunk,
    _unnormalize_action,
)

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
_BRIDGE_Q01 = np.array(
    [-0.02872725, -0.04170350, -0.02609386, -0.08092105, -0.09288700, -0.20718276],
    dtype=np.float32,
)
_BRIDGE_Q99 = np.array(
    [0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.07792851, 0.20382574],
    dtype=np.float32,
)


def test_unnormalize_maps_minus_one_to_q01() -> None:
    out = _unnormalize_action(np.array([-1, -1, -1, -1, -1, -1, -1.0], dtype=np.float32), _BRIDGE)
    np.testing.assert_allclose(out[:6], _BRIDGE_Q01, atol=1e-5)


def test_unnormalize_maps_plus_one_to_q99() -> None:
    out = _unnormalize_action(np.array([1, 1, 1, 1, 1, 1, 1.0], dtype=np.float32), _BRIDGE)
    np.testing.assert_allclose(out[:6], _BRIDGE_Q99, atol=1e-5)


def test_unnormalize_midpoint_is_mean_of_bounds() -> None:
    out = _unnormalize_action(np.zeros(7, dtype=np.float32), _BRIDGE)
    np.testing.assert_allclose(out[:6], 0.5 * (_BRIDGE_Q01 + _BRIDGE_Q99), atol=1e-5)


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


def test_as_action_chunk_reshapes_flat_oft_chunk() -> None:
    chunk = _as_action_chunk(np.arange(56, dtype=np.float32), action_dim=7)
    assert chunk.shape == (8, 7)
    np.testing.assert_array_equal(chunk[1], np.arange(7, 14, dtype=np.float32))


def test_as_action_chunk_rejects_wrong_action_dim() -> None:
    with pytest.raises(ROSConfigError, match="last dimension"):
        _as_action_chunk(np.arange(5, dtype=np.float32), action_dim=7)


def test_postprocess_action_chunk_scales_and_binarizes_gripper() -> None:
    raw = np.array(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.49],
            [0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.51],
        ],
        dtype=np.float32,
    )
    out = _postprocess_action_chunk(
        raw,
        action_scale=2.0,
        binarize_gripper=True,
        gripper_threshold=0.5,
    )
    np.testing.assert_allclose(out[:, :6], raw[:, :6] * 2.0, atol=1e-6)
    np.testing.assert_array_equal(out[:, 6], np.array([-1.0, 1.0], dtype=np.float32))


def test_extra_bool_rejects_ambiguous_strings() -> None:
    with pytest.raises(ROSConfigError, match="openvla_do_sample"):
        _extra_bool({"openvla_do_sample": "maybe"}, "openvla_do_sample", False)


def test_tokenization_compat_installs_missing_remote_code_imports() -> None:
    tokenization_utils = SimpleNamespace()
    tokenization_base = SimpleNamespace(
        PaddingStrategy=object(),
        PreTokenizedInput=object(),
        TextInput=object(),
        TruncationStrategy=object(),
    )
    transformers = SimpleNamespace(
        tokenization_utils=tokenization_utils,
        tokenization_utils_base=tokenization_base,
    )

    _install_tokenization_compat(transformers)

    for name in ("PaddingStrategy", "PreTokenizedInput", "TextInput", "TruncationStrategy"):
        assert getattr(tokenization_utils, name) is getattr(tokenization_base, name)
