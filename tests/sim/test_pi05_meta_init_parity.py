"""π0.5's fast meta-init loads must equal lerobot's own load semantics.

The bf16 and int8 fast paths build the policy on meta and stream the checkpoint
in themselves, so lerobot's key normalisation (``_fix_pytorch_state_dict_keys``
+ the ``model.`` prefix) and every non-persistent buffer are the loader's job.
Both failure classes are silent: PR #289's garbage ``embed_scale`` matched all
813 state_dict tensors and still gave forward cosine 0.03. So parameters AND
buffers are compared bit-exact (``remove_duplicate=False``, dtype included), and
the bf16 path also a forward with fixed inputs and noise — on a checkpoint other
than the OpenArm restock one it was validated on: the LIBERO / Franka
``lerobot/pi05_libero_finetuned_v044`` behind ``rskills/pi05-libero-int8``.

The reference is what ``PI05Policy.from_pretrained`` produces, derived without
``from_pretrained``'s fp32 construction (~14 GB; it OOMs an 8 GiB laptop):

* parameters — lerobot's real ``_fix_pytorch_state_dict_keys`` run on the real
  checkpoint tensors (in bounded chunks; the fixer is per key) + the ``model.``
  prefix, cast to the dtype lerobot's ``__init__`` gives each slot; every
  parameter must receive a value (``from_pretrained`` loads ``strict=True``);
* buffers — lerobot's ``PI05Pytorch.__init__`` built for real
  (``init_empty_weights(include_buffers=False)``);
* forward (bf16 test) — that reference assembled on the GPU with
  ``accelerate.set_module_tensor_to_device``.

The bf16 fast path is ~7.7 GiB of weights, the untied reference 8.7 GiB: it
needs a >= 10 GiB GPU (Orin / Thor / a desktop card) and skips on an 8 GiB
laptop. The int8 path (this rSkill's own config) runs there. Needs the cached
checkpoint (~7.5 GB); absent GPU or checkpoint is the legitimate skip
(CLAUDE.md §12). Run under the shared GPU lock:
    flock <gpu.lock> ./.venv/bin/pytest tests/sim/test_pi05_meta_init_parity.py -v
"""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
pytest.importorskip("accelerate")

_REPO = Path(__file__).resolve().parents[2]
_RSKILL = _REPO / "rskills" / "pi05-libero-int8"
_CONFIG = _REPO / "scenes" / "sim" / "libero_spatial.yaml"
_HF_REPO = "lerobot/pi05_libero_finetuned_v044"
_BF16_MIN_VRAM_GIB = 10

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a local GPU")

Digest = tuple[str, tuple[int, ...], str]


def _weights_or_skip() -> str:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(_HF_REPO, "model.safetensors")
    if not isinstance(path, str):
        pytest.skip(f"{_HF_REPO} is not in the local HF cache (~7.5 GB download)")
    return path


def _digest(t: Any) -> Digest:
    raw = t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
    return str(t.dtype), tuple(t.shape), hashlib.blake2b(raw, digest_size=16).hexdigest()


def _state(module: Any) -> dict[str, Digest]:
    state = {n: _digest(p) for n, p in module.named_parameters(remove_duplicate=False)}
    state.update({n: _digest(b) for n, b in module.named_buffers(remove_duplicate=False)})
    return state


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _meta_models() -> tuple[Any, Any, Any]:
    """``(cfg, PI05Policy on meta, PI05Pytorch with real buffers)``, as the fast path builds."""
    from accelerate import init_empty_weights
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    cfg = PreTrainedConfig.from_pretrained(_HF_REPO)
    cfg.compile_model = False
    cfg.device = "meta"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights():
            policy = PI05Policy(cfg)
        with init_empty_weights(include_buffers=False):
            model = PI05Pytorch(cfg)
    finally:
        torch.set_default_dtype(prev)
    return cfg, policy, model


def _fixed_tensors(policy: Any, cfg: Any, path: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(model.<name>, tensor)`` exactly as ``from_pretrained`` would load them."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for i in range(0, len(keys), 32):
            chunk = {k: f.get_tensor(k) for k in keys[i : i + 32]}
            for key, value in policy._fix_pytorch_state_dict_keys(chunk, cfg).items():
                yield (key if key.startswith("model.") else f"model.{key}"), value


@pytest.fixture(scope="module")
def reference() -> dict[str, Digest]:
    """Digest of every parameter and buffer ``from_pretrained`` would leave, by name."""
    path = _weights_or_skip()
    cfg, policy, model = _meta_models()
    slots = dict(policy.named_parameters(remove_duplicate=False))
    state: dict[str, Digest] = {}
    for name, value in _fixed_tensors(policy, cfg, path):
        state[name] = _digest(value.to(slots[name].dtype))
    unloaded = sorted(set(slots) - set(state))
    assert not unloaded, unloaded[:8]  # from_pretrained loads with strict=True
    state.update({f"model.{n}": _digest(b) for n, b in model.named_buffers(remove_duplicate=False)})
    return state


def _forward(model: Any) -> Any:
    """One ``PI05Pytorch.sample_actions`` on fixed real-shaped inputs + fixed noise."""
    cfg = model.config
    gen = torch.Generator(device="cpu").manual_seed(0)
    images = [
        (torch.rand(1, 3, *cfg.image_resolution, generator=gen) * 2 - 1).cuda() for _ in range(2)
    ]
    img_masks = [torch.ones(1, dtype=torch.bool, device="cuda") for _ in images]
    n_tok = cfg.tokenizer_max_length
    tokens = torch.randint(2, 20000, (1, n_tok), generator=gen).cuda()
    masks = torch.zeros(1, n_tok, dtype=torch.bool)
    masks[:, :24] = True
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_actions(images, img_masks, tokens, masks.cuda(), noise=noise)
    return out.float().cpu()


def _reference_forward() -> Any:
    """The reference model assembled on the GPU (untied, as ``from_pretrained`` leaves it)."""
    from accelerate.utils import set_module_tensor_to_device

    cfg, policy, model = _meta_models()
    for name, buf in list(model.named_buffers()):
        set_module_tensor_to_device(model, name, "cuda", value=buf)
    slots = dict(model.named_parameters(remove_duplicate=False))
    for key, value in _fixed_tensors(policy, cfg, _weights_or_skip()):
        name = key.removeprefix("model.")
        set_module_tensor_to_device(model, name, "cuda", value=value, dtype=slots[name].dtype)
    out = _forward(model.eval())
    del model, policy
    _free()
    return out


def _fast_policy(dtype: str) -> Any:
    from openral_sim.policies.pi05 import _build_pi05

    from tests.sim.conftest import compose_sim_env

    env = compose_sim_env(_CONFIG, rskill_uri=str(_RSKILL))
    extra = {**env.vla.extra, "dtype": dtype}
    vla = env.vla.model_copy(update={"device": "cuda:0", "extra": extra})
    return _build_pi05(env.model_copy(update={"vla": vla}))


def _vram_gib() -> float:
    return float(torch.cuda.get_device_properties(0).total_memory) / 2**30


def test_bf16_fast_meta_init_equals_from_pretrained(reference: dict[str, Digest]) -> None:
    if _vram_gib() < _BF16_MIN_VRAM_GIB:
        pytest.skip(f"bf16 π0.5 needs a >= {_BF16_MIN_VRAM_GIB} GiB GPU; {_vram_gib():.1f} GiB")
    adapter = _fast_policy("bf16")
    try:
        got = _state(adapter._policy)
        assert got.keys() == reference.keys(), sorted(got.keys() ^ reference.keys())[:8]
        diff = [n for n in reference if got[n] != reference[n]]
        assert not diff, (len(diff), [(n, got[n][:2], reference[n][:2]) for n in diff[:8]])
        out = _forward(adapter._policy.model)
    finally:
        adapter.close()
        _free()
    ref_out = _reference_forward()
    assert torch.equal(out, ref_out), (out - ref_out).abs().max()


def test_int8_fast_meta_init_matches_from_pretrained(reference: dict[str, Digest]) -> None:
    """int8 goes through the same streamed, key-normalised, fail-closed load.

    Large Linears are packed to LLM.int8 (bias in bf16), so only their presence is checked;
    every parameter left unpacked (norms, embeddings, the fp32 vision tower, the
    small ``time_mlp_*`` projections the openpi layout renames) and every buffer
    must be bit-identical to the reference, and the forward must be finite.
    """
    bnb = pytest.importorskip("bitsandbytes")
    adapter = _fast_policy("int8")
    try:
        policy = adapter._policy
        # A Linear8bitLt replacement holds its weight packed and its bias in the
        # bf16 compute dtype (the vision tower's fp32 biases included).
        packed = {
            f"{prefix}.{leaf}"
            for prefix, mod in policy.named_modules()
            if isinstance(mod, bnb.nn.Linear8bitLt)
            for leaf in ("weight", "bias")
        }
        got = _state(policy)
        assert got.keys() == reference.keys(), sorted(got.keys() ^ reference.keys())[:8]
        assert packed, "no Linear8bitLt: the int8 path did not run"
        diff = [n for n in reference if n not in packed and got[n] != reference[n]]
        assert not diff, (len(diff), [(n, got[n][:2], reference[n][:2]) for n in diff[:8]])
        out = _forward(policy.model)
    finally:
        adapter.close()
        _free()
    assert torch.isfinite(out).all()
