"""MolmoAct2's slow path re-applies the checkpoint after ``from_pretrained``.

Under transformers 5.x, lerobot's MolmoAct2 ``_init_weights`` (raw ``.data``
fills) runs after the checkpoint load and silently resets every weight, so the
bf16 / fp16 / on-the-fly NF4 paths ran an untrained model.
``_restore_checkpoint_state`` fills every slot from the shards or refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from openral_core.exceptions import ROSConfigError

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from openral_sim.policies import molmoact2  # noqa: E402


def _net() -> Any:
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]) -> None:
    keys = sorted(state)
    st.save_file(
        {k: state[k] for k in keys[:2]}, str(tmp_path / "model-00001-of-00002.safetensors")
    )
    st.save_file(
        {k: state[k] for k in keys[2:]}, str(tmp_path / "model-00002-of-00002.safetensors")
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))


def test_restores_every_tensor_after_an_init_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = {k: v + 3.0 for k, v in _net().state_dict().items()}
    _snapshot(tmp_path, monkeypatch, trained)
    model = _net()  # what from_pretrained leaves: init values, not the checkpoint's
    molmoact2._restore_checkpoint_state(model, "allenai/MolmoAct2-LIBERO", "abc")
    for k, v in model.state_dict().items():
        assert torch.equal(v, trained[k]), k


def test_refuses_a_checkpoint_that_leaves_a_tensor_unfilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = dict(_net().state_dict())
    partial.pop("1.bias")
    _snapshot(tmp_path, monkeypatch, partial)
    with pytest.raises(ROSConfigError, match="does not fill 1 model tensors"):
        molmoact2._restore_checkpoint_state(_net(), "allenai/MolmoAct2-LIBERO", None)


def test_refuses_a_shard_with_a_wrong_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = dict(_net().state_dict())
    state["0.weight"] = torch.zeros(5, 4)
    _snapshot(tmp_path, monkeypatch, state)
    with pytest.raises(ROSConfigError, match="does not fit the model"):
        molmoact2._restore_checkpoint_state(_net(), "allenai/MolmoAct2-LIBERO", None)
