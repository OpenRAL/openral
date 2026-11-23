"""In-process BEHAVIOR-1K GR00T runtime: the ported B1KPolicyWrapper transforms."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_rskill.loader import load_rskill_manifest
from openral_sim import _behavior_wire
from openral_sim._quantization import QUANTIZATION_DTYPE_ENV
from openral_sim.policies import behavior_groot
from openral_sim.policies import behavior_groot_inprocess as bgi

_RSKILL = Path(__file__).resolve().parents[2] / "rskills" / "gr00t-n17-b1k-turning-on-radio"


def test_r1pro_state_slices_groups_in_modality_order() -> None:
    proprio = np.arange(61, dtype=np.float32)
    state = bgi.r1pro_state(proprio)
    assert state.shape == (bgi.STATE_DIM,) == (25,)
    expected = np.r_[0:3, 53:57, 3:10, 24:26, 28:35, 49:51].astype(np.float32)
    np.testing.assert_array_equal(state, expected)


def test_resize_with_pad_letterboxes_wide_frames() -> None:
    img = np.full((360, 640, 3), 200, dtype=np.uint8)
    out = bgi.resize_with_pad(img, 224, 224)
    assert out.shape == (224, 224, 3)
    # 640x360 -> 224 x int(360 * 0.35) = 125 rows (the fork truncates), centred.
    filled = np.flatnonzero(out.any(axis=(1, 2)))
    assert (filled[0], filled[-1]) == (49, 173)
    assert (out[49:174] == 200).all()


def test_temporal_ensemble_reads_each_chunk_at_its_age() -> None:
    ens = bgi.TemporalEnsemble()
    chunks = [np.full((16, 23), 10.0 * (k + 1), dtype=np.float32) for k in range(7)]
    for k in range(7):
        chunks[k][:, :] += np.arange(16, dtype=np.float32)[:, None]
        action = ens.step(chunks[k])
    # Last step: the 5 newest chunks (k=2..6), read at ages 4..0.
    reads = np.array([chunks[k][6 - k] for k in range(2, 7)])
    w = np.exp(0.005 * np.arange(5))
    expected = (reads * (w / w.sum())[:, None]).sum(axis=0)
    for idx in _behavior_wire.GRIPPER_ACTION_INDICES:
        expected[idx] = chunks[6][0, idx]
    np.testing.assert_allclose(action, expected, rtol=1e-6)
    ens.reset()
    np.testing.assert_allclose(ens.step(chunks[0]), chunks[0][0])


def test_runtime_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(bgi.RUNTIME_ENV, raising=False)
    assert not bgi.wants_inprocess({})
    assert bgi.wants_inprocess({"runtime": "inprocess"})
    monkeypatch.setenv(bgi.RUNTIME_ENV, "sidecar")
    assert not bgi.wants_inprocess({"runtime": "inprocess"})
    monkeypatch.delenv(bgi.RUNTIME_ENV)
    with pytest.raises(ROSConfigError):
        bgi.wants_inprocess({"runtime": "bogus"})


def test_temporal_ensemble_refuses_short_or_flat_chunks() -> None:
    with pytest.raises(ROSRuntimeError, match="T>=5"):
        bgi.TemporalEnsemble().step(np.zeros((4, 23), dtype=np.float32))
    with pytest.raises(ROSRuntimeError, match="T>=5"):
        bgi.TemporalEnsemble().step(np.zeros(23 * 16, dtype=np.float32))


def _build(extra: dict[str, object]) -> object:
    manifest = load_rskill_manifest(str(_RSKILL))
    env_cfg = SimpleNamespace(vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)))
    return behavior_groot.build_behavior_groot_policy(env_cfg, manifest, extra)


def test_inprocess_routing_refuses_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """runtime=inprocess reaches the in-process builder, whose pre-load gates refuse."""
    monkeypatch.delenv(bgi.RUNTIME_ENV, raising=False)
    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(tmp_path / "missing"))
    with pytest.raises(ROSConfigError, match="only temporal_ensemble"):
        _build({"runtime": "inprocess", "control_mode": "receding_horizon"})
    with pytest.raises(ROSConfigError, match="checkpoint not found"):
        _build({"runtime": "inprocess"})

    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(tmp_path))
    # bf16 has no in-process load path (unquantized GrootPolicy runs fp32).
    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "bf16")
    with pytest.raises(ROSConfigError, match="cannot load dtype 'bf16'"):
        _build({"runtime": "inprocess"})
    monkeypatch.setenv("OPENRAL_REQUIRE_SIGNED_SKILLS", "1")
    with pytest.raises(ROSConfigError, match="OPENRAL_REQUIRE_SIGNED_SKILLS"):
        _build({"runtime": "inprocess"})
