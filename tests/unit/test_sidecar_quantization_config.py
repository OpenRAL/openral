"""Pro's sidecar settings reach real bitsandbytes layers (ADR-0123 step 3).

The LingBot and BEHAVIOR GR00T sidecars rewrite their Linears with OpenRAL
Pro's sidecar file (``--pro-quant-module``); the layer tests need Pro installed
and skip without it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _pro_module_path() -> str:
    from openral_rskill.quantization_gate import pro_quantization, pro_quantization_installed

    if not pro_quantization_installed():
        pytest.skip("OpenRAL Pro (openral.pro_modules: quantization) is not installed")
    return str(pro_quantization(what="the sidecar tests").sidecar_module_path())


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["_lingbot_vla2_server", "behavior_groot_sidecar"])
@pytest.mark.parametrize("raw", ["invalid", "[]", '{"load_in_8bit":true}'])
def test_nf4_config_rejects_invalid_input(name: str, raw: str) -> None:
    with pytest.raises(SystemExit):
        _load(name)._check_pro_config("nf4", raw, launcher="test")


@pytest.mark.parametrize("name", ["_lingbot_vla2_server", "behavior_groot_sidecar"])
@pytest.mark.parametrize("skip_name", ["1", "model.qwenvl_with_expert.qwenvl.1.0"])
def test_nf4_layers_use_pro_settings(name: str, skip_name: str) -> None:
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes packing needs CUDA")
    config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float32",
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        llm_int8_skip_modules=[
            skip_name
            if name == "_lingbot_vla2_server"
            else skip_name.removeprefix("model.qwenvl_with_expert.qwenvl.")
        ],
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Sequential(torch.nn.Linear(64, 64))
    )
    module = _load(name)
    pro = module._load_pro_quant(_pro_module_path(), launcher="test")
    if name == "_lingbot_vla2_server":
        module._nf4_backbone_in_place(pro, model, torch=torch, config=config, min_params=1)
        model.cuda()
    else:
        module._quantize(
            pro.nf4_linears_in_place, model, device="cuda", config=config, min_params=1
        )
    layer = model[0]
    assert isinstance(layer, bnb.nn.Linear4bit)
    assert layer.compute_dtype == torch.float32
    assert layer.weight.quant_type == "fp4"
    assert layer.weight.compress_statistics is False
    assert layer.weight.quant_state.nested is False
    assert not isinstance(model[1][0], bnb.nn.Linear4bit)
    assert torch.isfinite(layer(torch.ones(1, 64, device="cuda"))).all()


def test_int8_layers_use_pro_settings() -> None:
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes packing needs CUDA")
    config = transformers.BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=3.0,
        llm_int8_has_fp16_weight=True,
        llm_int8_skip_modules=["1.0"],
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Sequential(torch.nn.Linear(64, 64))
    )
    module = _load("behavior_groot_sidecar")
    pro = module._load_pro_quant(_pro_module_path(), launcher="test")
    module._quantize(pro.int8_linears_in_place, model, device="cuda", config=config, min_params=1)
    layer = model[0]
    assert isinstance(layer, bnb.nn.Linear8bitLt)
    assert not isinstance(model[1][0], bnb.nn.Linear8bitLt)
    assert layer.state.threshold == 3.0
    assert layer.weight.has_fp16_weights is True
    assert torch.isfinite(layer(torch.ones(1, 64, device="cuda", dtype=torch.float16))).all()


def test_pack_rejects_invalid_config_before_loading_weights() -> None:
    with pytest.raises(SystemExit, match="not valid JSON"):
        _load("quantize_lingbot_vla2").main(
            [
                "--ckpt",
                str(_ROOT / "rskills" / "lingbot-vla2-robotwin"),
                "--out",
                "/unused-pack-output",
                "--quantization-config",
                "invalid",
                "--pro-quant-module",
                "/unused-pro-module.py",
            ]
        )


@pytest.mark.parametrize("recorded", [None, [], ["backbone.0"]])
@pytest.mark.parametrize("launch", [None, [], ["backbone.0"]])
def test_pack_skip_modules_must_match(
    tmp_path: Path, recorded: list[str] | None, launch: list[str] | None
) -> None:
    # Legacy packs have no config and quantized every qualifying module.
    quantization: dict[str, object] = {"scheme": "nf4"}
    if recorded is not None:
        quantization["config"] = {"llm_int8_skip_modules": recorded}
    (tmp_path / "quantization_metadata.json").write_text(json.dumps({"quantization": quantization}))
    config = {
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_storage": "uint8",
        "llm_int8_skip_modules": launch,
    }
    module = _load("_lingbot_vla2_server")
    if set(recorded or []) != set(launch or []):
        with pytest.raises(SystemExit, match="differs from the NF4 pack"):
            module._check_pack_config(str(tmp_path), config)
    else:
        module._check_pack_config(str(tmp_path), config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bnb_4bit_quant_type", "fp4"),
        ("bnb_4bit_compute_dtype", "float32"),
        ("bnb_4bit_use_double_quant", False),
        ("bnb_4bit_quant_storage", "float16"),
    ],
)
def test_pack_precision_must_match(tmp_path: Path, key: str, value: object) -> None:
    config = {
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_storage": "uint8",
    }
    (tmp_path / "quantization_metadata.json").write_text(
        json.dumps({"quantization": {"scheme": "nf4", "config": {**config, key: value}}})
    )
    with pytest.raises(SystemExit, match=key):
        _load("_lingbot_vla2_server")._check_pack_config(str(tmp_path), config)


def test_lingbot_skips_a_model_root_subtree() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    config = transformers.BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["model"])
    model = torch.nn.Sequential(torch.nn.Linear(64, 64))
    module = _load("_lingbot_vla2_server")
    pro = module._load_pro_quant(_pro_module_path(), launcher="test")
    module._nf4_backbone_in_place(pro, model, torch=torch, config=config, min_params=1)
    assert type(model[0]) is torch.nn.Linear


@pytest.mark.parametrize("name", ["_lingbot_vla2_server", "behavior_groot_sidecar"])
@pytest.mark.parametrize("path", [None, "", "/no/such/pro_sidecar.py"])
def test_sub_16_bit_needs_pro_rewrites_file(name: str, path: str | None) -> None:
    with pytest.raises(SystemExit, match="--pro-quant-module"):
        _load(name)._load_pro_quant(path, launcher="test")


def test_gate_passes_pro_rewrites_file_only_when_asked() -> None:
    from openral_rskill.quantization_gate import sidecar_quantization_args

    path = _pro_module_path()
    assert "--pro-quant-module" not in sidecar_quantization_args("nf4", what="test")
    for dtype in ("nf4", "int8", "prequantized_nf4"):
        args = sidecar_quantization_args(dtype, what="test", rewrites=True)
        assert args[2:] == ["--pro-quant-module", path]
        assert Path(path).is_file()
    assert sidecar_quantization_args("none", what="test", rewrites=True) == []
