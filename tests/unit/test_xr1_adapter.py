"""Pure contract checks for the XR-1 policy adapter."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim.policies.xr1 import (
    _history_sample,
    _preprocess_image,
    _quantization_mode,
    _rc365_state,
    _require_remote_code_ack,
    _vlabench_targets,
)


def test_remote_code_requires_explicit_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_REMOTE_CODE=1"):
        _require_remote_code_ack()


def test_int4_manifest_selects_nf4_sidecar() -> None:
    from pathlib import Path

    from openral_core import RSkillManifest

    root = Path(__file__).parents[2]
    from openral_core import VLASpec

    manifest = RSkillManifest.from_yaml(str(root / "rskills" / "xr1-vlabench" / "rskill.yaml"))
    spec = VLASpec(id="xr1", weights_uri="rskills/xr1-vlabench")
    assert _quantization_mode(spec, manifest) == "prequantized_nf4"
    local = manifest.model_copy(
        update={"policy_extras": {**manifest.policy_extras, "prequantized_nf4": False}}
    )
    assert _quantization_mode(spec, local) == "nf4"


def test_xr1_honours_the_shared_override_and_rejects_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XR-1 used to read only the manifest; the per-run override now reaches it."""
    from pathlib import Path

    from openral_core import RSkillManifest, VLASpec
    from openral_sim._quantization import QUANTIZATION_DTYPE_ENV

    root = Path(__file__).parents[2]
    manifest = RSkillManifest.from_yaml(str(root / "rskills" / "xr1-vlabench" / "rskill.yaml"))
    spec = VLASpec(id="xr1", weights_uri="rskills/xr1-vlabench")
    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "bf16")
    assert _quantization_mode(spec, manifest) == "none"
    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "int8")
    with pytest.raises(ROSConfigError, match="xr1 cannot load dtype 'int8'"):
        _quantization_mode(spec, manifest)


def test_rc365_state_reorders_and_converts_quaternions() -> None:
    source = np.array(
        [1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 1, 0.1, 0.2],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        _rc365_state(source),
        np.array([1, 2, 3, 0, 0, 0, 0.1, 0.2, 4, 5, 6, 0, 0, 0], dtype=np.float32),
    )


def test_history_sample_left_pads_then_uses_interval_two() -> None:
    history = deque(
        [np.array([1], dtype=np.float32), np.array([2], dtype=np.float32)],
        maxlen=7,
    )
    np.testing.assert_array_equal(
        _history_sample(history),
        np.array([[1], [1], [1], [2]], dtype=np.float32),
    )


def test_vlabench_deltas_become_absolute_targets() -> None:
    state = np.zeros(7, dtype=np.float32)
    deltas = np.array(
        [[1, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    targets = _vlabench_targets(deltas, state)
    np.testing.assert_allclose(targets[:, 0], np.array([1, 2], dtype=np.float32))
    np.testing.assert_allclose(targets[:, 6], np.array([0, 1], dtype=np.float32))


def test_image_orientation_is_manifest_controlled() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    np.testing.assert_array_equal(
        _preprocess_image(image, flip_180=False, resize=None),
        image,
    )
    np.testing.assert_array_equal(
        _preprocess_image(image, flip_180=True, resize=None),
        image[::-1, ::-1],
    )


def test_image_resize_is_manifest_controlled() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    resized = _preprocess_image(image, flip_180=False, resize=(3, 2))
    assert resized.shape == (2, 3, 3)
