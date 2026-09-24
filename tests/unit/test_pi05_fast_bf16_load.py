"""The fast meta-init loaders: buffers rebuilt, checkpoints streamed in, misses refused.

The family-neutral helpers live in ``openral_sim._quantization``; π0.5 supplies
two hooks (Gemma buffers, lerobot key normalisation). Real ``torch.nn.Module``s,
real safetensors files, and lerobot's real ``PI05Policy`` key fixer (built on
meta); CLAUDE.md §1.11. The full-size parity against ``from_pretrained``
(parameters, buffers and a forward) is ``tests/sim/test_pi05_meta_init_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from openral_core.exceptions import ROSConfigError, ROSRuntimeError  # noqa: E402
from openral_sim._quantization import (  # noqa: E402
    rebuild_non_persistent_buffers,
    resolve_weights_files,
    stream_state_into_policy,
)
from openral_sim.policies.pi05 import _init_pi05_buffers  # noqa: E402


def _stream(policy: torch.nn.Module, snapshot: Path, **kwargs: object) -> None:
    stream_state_into_policy(policy, str(snapshot), torch=torch, family="test", **kwargs)


class _Rope(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope_theta = 10000.0
        self.register_buffer("inv_freq", torch.full((64,), float("nan")))


class _Vision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("position_ids", torch.full((1, 16), -1, dtype=torch.int64))


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope = _Rope()
        self.vision = _Vision()
        self.proj = torch.nn.Linear(8, 4)
        self.head = torch.nn.Linear(4, 2)


def test_rope_and_position_buffers_are_rebuilt() -> None:
    m = _Tiny()
    with torch.no_grad():
        _init_pi05_buffers(m, torch=torch)
    assert torch.equal(m.vision.position_ids, torch.arange(16).unsqueeze(0))
    expected = 1.0 / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
    assert torch.allclose(m.rope.inv_freq, expected)


def test_real_gemma_modules_rebuild_to_their_constructed_values() -> None:
    """The buffers PaliGemma / gemma_expert actually carry, via meta init + ``to_empty``.

    ``embed_scale`` is the one the restock checkpoint ran with as garbage
    (cosine 0.03 vs ``from_pretrained``, Thor 2026-09-23): every token
    embedding is multiplied by it. Real transformers modules, a non-default
    ``rope_theta`` so the config lookup is what is tested, no lerobot policy.
    """
    accelerate = pytest.importorskip("accelerate")
    gemma = pytest.importorskip("transformers.models.gemma.modeling_gemma")
    from transformers import GemmaConfig

    cfg = GemmaConfig(
        vocab_size=64, hidden_size=32, num_attention_heads=4, head_dim=8, rope_theta=50000.0
    )

    class _Stack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = gemma.GemmaTextScaledWordEmbedding(
                cfg.vocab_size, cfg.hidden_size, 0, embed_scale=cfg.hidden_size**0.5
            )
            self.rotary_emb = gemma.GemmaRotaryEmbedding(cfg)

    reference = _Stack()
    with accelerate.init_empty_weights():
        meta = _Stack()
    meta.to_empty(device="cpu")
    with torch.no_grad():
        _init_pi05_buffers(meta, torch=torch)
    assert torch.equal(meta.embed_tokens.embed_scale, reference.embed_tokens.embed_scale)
    assert torch.allclose(meta.rotary_emb.inv_freq, reference.rotary_emb.inv_freq)
    assert torch.allclose(meta.rotary_emb.original_inv_freq, reference.rotary_emb.original_inv_freq)
    # The rebuild used the config's theta, not the 10000 fallback.
    default = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    assert not torch.allclose(meta.rotary_emb.inv_freq, default)


def test_an_unknown_non_persistent_buffer_is_refused() -> None:
    class _Mystery(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("attention_scale", torch.tensor(float("nan")), persistent=False)
            self.register_buffer("kept", torch.zeros(2))  # persistent: loaded from the file

    with pytest.raises(ROSRuntimeError, match=r"smolvla fast meta-init.*attention_scale"):
        rebuild_non_persistent_buffers(_Mystery(), torch=torch, family="smolvla")
    # A family hook that knows the buffer takes it off the refusal list.
    rebuild_non_persistent_buffers(
        _Mystery(),
        torch=torch,
        family="smolvla",
        rebuild=lambda name, _mod, buf: name == "attention_scale" and bool(buf.fill_(1.0) or 1),
    )


def _write_state(tmp_path: Path, model: torch.nn.Module, **edits: torch.Tensor | None) -> Path:
    """Save ``model``'s state as fp32 safetensors; ``edits`` add (tensor) or drop (None) keys."""
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    for raw_key, value in edits.items():
        key = raw_key.replace("__", ".")
        if value is None:
            state.pop(key)
        else:
            state[key] = value
    safetensors_torch.save_file(state, str(snapshot / "model.safetensors"))
    return snapshot


def test_resolve_weights_files_prefers_a_directory_holding_the_file(tmp_path: Path) -> None:
    snapshot = _write_state(tmp_path, _Tiny())
    assert resolve_weights_files(str(snapshot)) == [str(snapshot / "model.safetensors")]


def _stream_and_check(device: str, tmp_path: Path) -> None:
    source = _Tiny()
    with torch.no_grad():
        for p in source.parameters():
            p.copy_(torch.randn_like(p))
    snapshot = _write_state(tmp_path, source)

    target = _Tiny().to(torch.bfloat16).to(device)
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
    _stream(target, snapshot)

    # fp32 file -> bf16 parameters on the target device, matching to bf16 precision.
    assert target.proj.weight.dtype == torch.bfloat16
    assert str(target.proj.weight.device).startswith(device.split(":", maxsplit=1)[0])
    assert torch.allclose(
        target.proj.weight.float().cpu(), source.proj.weight.to(torch.bfloat16).float(), atol=0
    )
    assert torch.allclose(
        target.head.weight.float().cpu(), source.head.weight.to(torch.bfloat16).float()
    )
    assert torch.allclose(
        target.head.bias.float().cpu(), source.head.bias.to(torch.bfloat16).float()
    )


def test_streamed_weights_land_in_place_with_the_targets_dtype(tmp_path: Path) -> None:
    _stream_and_check("cpu", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_streamed_weights_land_on_cuda_without_a_cpu_copy(tmp_path: Path) -> None:
    _stream_and_check("cuda:0", tmp_path)


def test_a_parameter_the_checkpoint_lacks_is_refused(tmp_path: Path) -> None:
    """It would run on its reset value — the silent-miss class of the embed_scale bug."""
    snapshot = _write_state(tmp_path, _Tiny(), head__bias=None)
    with pytest.raises(ROSConfigError, match=r"1 missing .*head\.bias"):
        _stream(_Tiny(), snapshot)


def test_a_key_that_names_nothing_in_the_policy_is_refused(tmp_path: Path) -> None:
    snapshot = _write_state(tmp_path, _Tiny(), not__in__model=torch.zeros(3))
    with pytest.raises(ROSConfigError, match=r"1 unexpected: \['not\.in\.model'\]"):
        _stream(_Tiny(), snapshot)


def test_a_key_map_renames_and_drops_before_matching(tmp_path: Path) -> None:
    """The family's key normalisation runs first; a key it drops is not unexpected."""
    source = _Tiny()
    state = {f"legacy.{k}": v.detach().clone() for k, v in source.state_dict().items()}
    state["legacy.state_proj.weight"] = torch.zeros(2)
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    safetensors_torch.save_file(state, str(snapshot / "model.safetensors"))

    def key_map(keys: list[str]) -> dict[str, str]:
        return {k.removeprefix("legacy."): k for k in keys if "state_proj" not in k}

    target = _Tiny()
    _stream(target, snapshot, key_map=key_map)
    assert torch.equal(target.proj.weight, source.proj.weight)


def test_a_sharded_checkpoint_is_streamed_from_every_shard(tmp_path: Path) -> None:
    """``model.safetensors.index.json`` + shards, the layout ``save_pretrained`` writes."""
    import json

    source = _Tiny()
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    snapshot = tmp_path / "sharded"
    snapshot.mkdir()
    shards = {"model-00001-of-00002.safetensors": [], "model-00002-of-00002.safetensors": []}
    names = list(shards)
    for i, key in enumerate(sorted(state)):
        shards[names[i % 2]].append(key)
    for shard, keys in shards.items():
        safetensors_torch.save_file({k: state[k] for k in keys}, str(snapshot / shard))
    weight_map = {k: shard for shard, keys in shards.items() for k in keys}
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    assert resolve_weights_files(str(snapshot)) == [str(snapshot / n) for n in names]
    target = _Tiny()
    _stream(target, snapshot)
    for name, value in state.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0, equal_nan=True)


def test_lerobot_key_map_applies_from_pretrained_normalisation() -> None:
    """π0.5's hook is lerobot's own ``_fix_pytorch_state_dict_keys`` + the ``model.`` prefix.

    Real ``PI05Policy`` (built on meta, no weights) and real openpi-layout key
    names — the layout the audit found the fast path loading raw.
    """
    accelerate = pytest.importorskip("accelerate")
    modeling = pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from openral_sim.policies.pi05 import _lerobot_key_map

    cfg = PI05Config(device="cpu")
    cfg.device = "meta"  # as _build_pi05 does: keeps __init__'s `.to(device)` off real memory
    with accelerate.init_empty_weights():
        policy = modeling.PI05Policy(cfg)
    params = dict(policy.named_parameters(remove_duplicate=False))
    raw = [
        "action_time_mlp_in.weight",
        "action_time_mlp_out.bias",
        "state_proj.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
        "model.action_in_proj.weight",
    ]
    mapped = _lerobot_key_map(policy)(raw)
    assert mapped == {
        "model.time_mlp_in.weight": "action_time_mlp_in.weight",
        "model.time_mlp_out.bias": "action_time_mlp_out.bias",
        "model.paligemma_with_expert.paligemma.lm_head.weight": (
            "paligemma_with_expert.paligemma.lm_head.weight"
        ),
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": (
            "paligemma_with_expert.paligemma.lm_head.weight"
        ),
        "model.action_in_proj.weight": "model.action_in_proj.weight",
    }
    assert set(mapped) <= set(params), set(mapped) - set(params)


def test_hub_weights_resolve_at_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare repo id forwards ``revision``, so weights and config share one pin."""
    import openral_rskill._vla_core as vla_core

    seen: dict[str, object] = {}

    def _capture(*_args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "/cache/model.safetensors"

    monkeypatch.setattr(vla_core, "hf_download_cached_first", _capture)
    assert resolve_weights_files("OpenRAL/some-rskill", revision="abc123") == [
        "/cache/model.safetensors"
    ]
    assert seen["repo_id"] == "OpenRAL/some-rskill"
    assert seen["filename"] == "model.safetensors"
    assert seen["revision"] == "abc123"


class _Tied(torch.nn.Module):
    """An embedding whose weight is tied to an output head, PaliGemma-style."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(6, 4)
        self.head = torch.nn.Linear(4, 6, bias=False)
        self.head.weight = self.embed.weight


def test_a_checkpoint_may_name_a_tied_weight_by_either_alias(tmp_path: Path) -> None:
    """``named_parameters()`` de-duplicates tied weights; the loader must not.

    PaliGemma ties ``embed_tokens.weight`` to ``lm_head.weight``. A checkpoint
    that carries only the head's spelling used to be reported ``unexpected``
    and the shared storage stayed at its reset value.
    """
    values = torch.randn(6, 4)
    snapshot = tmp_path / "snapshots" / "tied"
    snapshot.mkdir(parents=True)
    safetensors_torch.save_file(
        {"head.weight": values.clone()}, str(snapshot / "model.safetensors")
    )

    target = _Tied().to(torch.bfloat16)
    with torch.no_grad():
        target.embed.weight.zero_()
    _stream(target, snapshot)
    assert torch.equal(target.embed.weight.float(), values.to(torch.bfloat16).float())
    assert target.head.weight.data_ptr() == target.embed.weight.data_ptr()
