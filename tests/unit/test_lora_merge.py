"""LoRA merge (``openral_sim.policies._lora``) on real tensors: the math and every refusal.

The adapter files are written with the same layout PEFT 0.19 saves
(``adapter_config.json`` + ``adapter_model.safetensors`` with
``base_model.model.<module>.lora_{A,B}.weight`` keys, as in
``makermods/pi05_metal_pick_place_lora``) and merged into a small real
``torch.nn.Module``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_sim.policies._lora import (  # noqa: E402
    merge_lora_into_policy,
    read_lora_config,
    resolve_base_weights,
)

_R = 2
_TARGETS = r"(.*\.self_attn\.(q|v)_proj|model\.action_out_proj)"


class _Attn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(6, 4, bias=False)
        self.v_proj = torch.nn.Linear(6, 2, bias=False)
        self.o_proj = torch.nn.Linear(4, 6, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.action_out_proj = torch.nn.Linear(4, 3)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


_MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
    "model.action_out_proj",
)


def _write_adapter(
    tmp: Path, policy: Any, *, config: dict[str, Any] | None = None, drop: str | None = None
) -> dict[str, Any]:
    torch.manual_seed(0)
    modules = dict(policy.named_modules())
    tensors = {}
    for name in _MODULES:
        out_f, in_f = modules[name].weight.shape
        tensors[f"base_model.model.{name}.lora_A.weight"] = torch.randn(_R, in_f)
        tensors[f"base_model.model.{name}.lora_B.weight"] = torch.randn(out_f, _R)
    if drop is not None:
        del tensors[drop]
    cfg = {
        "peft_type": "LORA",
        "r": _R,
        "lora_alpha": 4,
        "target_modules": _TARGETS,
        "base_model_name_or_path": "lerobot/pi05_base",
        "bias": "none",
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": [],
        "fan_in_fan_out": False,
        "layers_to_transform": None,
        "lora_bias": False,
    }
    cfg.update(config or {})
    (tmp / "adapter_config.json").write_text(json.dumps(cfg))
    safetensors_torch.save_file(tensors, str(tmp / "adapter_model.safetensors"))
    return tensors


def test_merge_matches_dense_update(tmp_path: Path) -> None:
    policy = _Policy()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    tensors = _write_adapter(tmp_path, policy)
    stats = merge_lora_into_policy(policy, tmp_path, torch=torch, base_repo_id="lerobot/pi05_base")
    assert stats["modules"] == len(_MODULES)
    assert stats["scale"] == 2.0
    assert stats["params"] == sum(t.numel() for t in tensors.values())
    assert stats["max_rel_delta"] > 0.0
    params = dict(policy.named_parameters())
    for name in _MODULES:
        a = tensors[f"base_model.model.{name}.lora_A.weight"]
        b = tensors[f"base_model.model.{name}.lora_B.weight"]
        expect = before[f"{name}.weight"] + 2.0 * b @ a
        torch.testing.assert_close(params[f"{name}.weight"], expect)
    # Untargeted weights and every bias are untouched.
    for name in ("model.layers.0.self_attn.o_proj.weight", "model.action_out_proj.bias"):
        assert torch.equal(params[name], before[name])


def test_merge_bf16_is_fp32_then_one_cast(tmp_path: Path) -> None:
    policy = _Policy().to(torch.bfloat16)
    before = policy.model.action_out_proj.weight.detach().clone()
    tensors = _write_adapter(tmp_path, policy)
    merge_lora_into_policy(policy, tmp_path, torch=torch)
    a = tensors["base_model.model.model.action_out_proj.lora_A.weight"]
    b = tensors["base_model.model.model.action_out_proj.lora_B.weight"]
    expect = (before.float() + 2.0 * b @ a).to(torch.bfloat16)
    assert policy.model.action_out_proj.weight.dtype == torch.bfloat16
    assert torch.equal(policy.model.action_out_proj.weight, expect)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"use_dora": True}, "use_dora"),
        ({"use_rslora": True}, "use_rslora"),
        ({"rank_pattern": {"q_proj": 8}}, "rank_pattern"),
        ({"alpha_pattern": {"q_proj": 8}}, "alpha_pattern"),
        ({"bias": "all"}, "bias"),
        ({"lora_bias": True}, "lora_bias"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out"),
        ({"layers_to_transform": [0]}, "layers_to_transform"),
        ({"modules_to_save": ["model.action_out_proj"]}, "modules_to_save"),
        ({"peft_type": "LOHA"}, "not LORA"),
        ({"r": 0}, "rank/alpha"),
        ({"target_modules": []}, "target_modules"),
    ],
)
def test_refuses_non_plain_lora(tmp_path: Path, override: dict[str, Any], match: str) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy, config=override)
    with pytest.raises(ROSConfigError, match=match):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_base_mismatch(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy)
    with pytest.raises(ROSConfigError, match="trained on"):
        merge_lora_into_policy(policy, tmp_path, torch=torch, base_repo_id="lerobot/pi0_base")


def test_refuses_unpaired(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy, drop="base_model.model.model.action_out_proj.lora_B.weight")
    with pytest.raises(ROSConfigError, match="without its pair"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_missing_target(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy)
    # A regex wider than the shipped pairs: o_proj now targeted but absent.
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    cfg["target_modules"] = r"(.*\.self_attn\.(q|v|o)_proj|model\.action_out_proj)"
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ROSConfigError, match="no adapter weights"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_out_of_target(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy, config={"target_modules": r".*\.self_attn\.(q|v)_proj"})
    with pytest.raises(ROSConfigError, match="outside target_modules"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def _rewrite(tmp: Path, mutate: Any) -> None:
    path = tmp / "adapter_model.safetensors"
    tensors = dict(safetensors_torch.load_file(str(path)))
    mutate(tensors)
    safetensors_torch.save_file(tensors, str(path))


def test_refuses_unexpected_key(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy)
    _rewrite(tmp_path, lambda t: t.update({"base_model.model.model.extra.weight": torch.ones(1)}))
    with pytest.raises(ROSConfigError, match="unexpected adapter key"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_unknown_module(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy)
    _rewrite(
        tmp_path,
        lambda t: t.update(
            {
                "base_model.model.model.ghost.lora_A.weight": torch.ones(_R, 2),
                "base_model.model.model.ghost.lora_B.weight": torch.ones(2, _R),
            }
        ),
    )
    with pytest.raises(ROSConfigError, match="names no Linear"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_wrong_shape(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(tmp_path, policy)
    key = "base_model.model.model.action_out_proj.lora_A.weight"
    _rewrite(tmp_path, lambda t: t.update({key: torch.ones(_R, 5)}))
    with pytest.raises(ROSConfigError, match="do not fit"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)


def test_refuses_tied_weight_and_changes_nothing(tmp_path: Path) -> None:
    policy = _Policy()
    # Tie layer 1's q_proj to layer 0's: merging one would move both.
    mods = dict(policy.named_modules())
    mods["model.layers.1.self_attn.q_proj"].weight = mods["model.layers.0.self_attn.q_proj"].weight
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    _write_adapter(tmp_path, policy)
    with pytest.raises(ROSConfigError, match="tied"):
        merge_lora_into_policy(policy, tmp_path, torch=torch)
    for name, p in policy.named_parameters():
        assert torch.equal(p, before[name]), name


def test_read_config_missing(tmp_path: Path) -> None:
    with pytest.raises(ROSConfigError, match=r"no adapter_config\.json"):
        read_lora_config(tmp_path)


def test_list_target_modules_use_suffix_match(tmp_path: Path) -> None:
    policy = _Policy()
    _write_adapter(
        tmp_path, policy, config={"target_modules": ["q_proj", "v_proj", "action_out_proj"]}
    )
    assert merge_lora_into_policy(policy, tmp_path, torch=torch)["modules"] == len(_MODULES)


def test_resolve_base_weights(tmp_path: Path) -> None:
    assert resolve_base_weights("hf://lerobot/pi05_base@22c30e5") == (
        "lerobot/pi05_base",
        "22c30e5",
    )
    assert resolve_base_weights("hf://lerobot/pi05_base") == ("lerobot/pi05_base", None)
    assert resolve_base_weights(f"local://{tmp_path}") == (str(tmp_path.resolve()), None)
    with pytest.raises(ROSConfigError, match="missing"):
        resolve_base_weights(f"local://{tmp_path}/nope")
