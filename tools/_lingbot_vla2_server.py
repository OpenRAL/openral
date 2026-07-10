"""LingBot-VLA 2.0 inference server — runs inside the sidecar venv (no openral import).

This is the **server side** of the LingBot-VLA 2.0 sidecar: it loads the upstream
``LingbotVLAv2Server`` (NF4 Qwen3-VL backbone + bf16 sparse-MoE action expert) from
Robbyant's ``lingbotvla`` package (https://github.com/robbyant/lingbot-vla-v2,
Apache-2.0 code + weights) and answers ``ping`` / ``reset`` / ``get_action`` /
``close`` over ZMQ REQ/REP framed by msgpack — the same ndarray wire the
:class:`openral_sim.sidecar.SidecarClient` speaks. It is ``os.execvpe``-d by the
boot helper :mod:`tools.lingbot_vla2_sidecar` *after* the repo checkout + torch-2.8
venv are provisioned, so it only ever runs under the sidecar interpreter and never
imports ``openral_*`` (that stack pins torch>=2.9 / transformers>=5, incompatible
with the upstream torch==2.8.0 / transformers==4.57.3 pins — CLAUDE.md §3).

Wire protocol (msgpack ``{"endpoint","data"}`` in, dict out; ndarrays via the
``__ndarray__`` / ``np.save`` sentinel that mirrors
``openral_sim.sidecar.encode_ndarray``)::

    ping        -> {"ok": True, "model": <id>, "robo_name": <robo>}
    reset       -> {"ok": True}                       data: {"robo_name": <robo>}
    get_action  -> {"action": (chunk, 14) float32,    data: {"observation": {
                    "action_keys": [...]}                "images": {cam_high,cam_left_wrist,cam_right_wrist},
                                                          "state": (14,), "task": <str>}}
    close       -> {"ok": True}  (stops the server)

The 6.38 B model does not fit an 8 GB card at bf16 (12.8 GB), so the Qwen3-VL
backbone is NF4-quantized in place (``--quantization nf4``) while the MoE action
expert stays bf16; ``none`` loads bf16 for ≥16 GB GPUs.

Attention backend: flash-attn is intentionally NOT installed in the sidecar venv
(it is not in the upstream ``requirements.txt`` and needs a source build). The
upstream ``QwenvlWithExpertV2Model.__init__`` hardcodes ``_attn_implementation =
"flash_attention_2"`` on the VLM / text / expert configs, so we coerce those
configs to a flash-free backend (``--attn sdpa`` by default; ``eager`` as the
universally-correct fallback) before and after the model is built (see
:func:`_install_attn_fallback`).

CLAUDE.md compliance: real upstream code in a real subprocess (no mocks, §1.11);
py-version/dep isolation is the only safe bridge (§3); the model's Apache-2.0
weights carry no license guard.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

_NDARRAY_SENTINEL = "__ndarray__"


def _encode_ndarray(obj: Any) -> Any:
    """Mirror ``openral_sim.sidecar.encode_ndarray`` (msgpack-only ndarray wire)."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_SENTINEL: True, "npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    if _NDARRAY_SENTINEL in obj and "npy" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _resolve_repo() -> Path:
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_REPO")
    default = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar" / "source"
    repo = Path(override).expanduser() if override else default
    if not (repo / "lingbotvla").is_dir():
        raise SystemExit(
            f"lingbotvla repo checkout not found at {repo}. The boot helper "
            "(tools/lingbot_vla2_sidecar.py) clones it; set "
            "OPENRAL_LINGBOT_VLA2_REPO to reuse an existing checkout."
        )
    return repo


def _resolve_qwen() -> str:
    for env in ("OPENRAL_QWEN3VL_PATH", "QWEN3VL_PATH"):
        val = os.environ.get(env)
        if val:
            return val
    # Fall back to the HF id; the server will fetch it if not cached.
    return "Qwen/Qwen3-VL-4B-Instruct"


def _resolve_checkpoint(model: str) -> str:
    """Return a local checkpoint dir (dir of ``*.safetensors``) for ``model``.

    A local path is used verbatim; an HF id is snapshot-downloaded.
    """
    p = Path(model).expanduser()
    if p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])


def _write_cli_yaml(repo: Path, ckpt_dir: str, qwen: str) -> Path:
    """Reconstruct the training config the upstream loader reads.

    ``LingbotVLAv2Server.load_vla`` reads ``lingbotvla_cli.yaml`` at
    ``ckpt.parent.parent.parent`` — the HF release does not ship it, but the
    repo's ``configs/vla/robotwin/robotwin.yaml`` carries every architecture dim.
    We copy it with the checkpoint / backbone / norm-stats paths patched and
    place it at the location the loader expects.
    """
    import yaml

    tmpl = yaml.safe_load((repo / "configs/vla/robotwin/robotwin.yaml").read_text())
    tmpl["model"]["model_path"] = ckpt_dir
    tmpl["model"]["tokenizer_path"] = qwen
    tmpl["data"]["norm_stats_file"] = "assets/norm_stats/robotwin.json"
    tmpl["data"].setdefault("img_size", 256)
    # FeatureInfo.update_info / get_normalizer ast.literal_eval each joints +
    # norm_type entry, so they must be string-reprs of the {name: val} dicts,
    # not YAML-parsed dicts (verified live: FeatureTransform crashes otherwise).
    for _key in ("joints", "norm_type"):
        if _key in tmpl["data"]:
            tmpl["data"][_key] = [
                str(entry) if isinstance(entry, dict) else entry
                for entry in tmpl["data"][_key]
            ]
    # loader path: Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"
    cli = Path(ckpt_dir).parent.parent.parent / "lingbotvla_cli.yaml"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(yaml.safe_dump(tmpl))
    return cli


def _install_lerobot_stub() -> None:
    """Stub any ``lerobot.*`` import so the model class imports.

    The upstream model import chain pulls in ``lingbotvla.data`` (training-only
    ``lerobot`` imports scattered across submodules) — not needed for inference,
    and ``lerobot`` cannot be installed here anyway (it wants transformers 5.x,
    conflicting with the pinned 4.57.3). A meta-path finder returns a permissive
    stub module for every ``lerobot`` submodule. Verified live: without this the
    ``from deploy.lingbot_vla_v2_policy import ...`` chain ModuleNotFound's.
    """
    import importlib.abc
    import importlib.machinery
    import types

    class _Dummy:
        def __init__(self, *a: Any, **k: Any) -> None: ...

        def __call__(self, *a: Any, **k: Any) -> None:
            return None

    class _AnyStub(types.ModuleType):
        __path__: ClassVar[list[str]] = []

        def __getattr__(self, name: str) -> Any:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Dummy

    class _LerobotFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
            if fullname == "lerobot" or fullname.startswith("lerobot."):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec: Any) -> Any:
            return _AnyStub(spec.name)

        def exec_module(self, module: Any) -> None: ...

    if not any(type(f).__name__ == "_LerobotFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _LerobotFinder())


def _coerce_attn_config(config: Any, target: str) -> None:
    """Recursively rewrite ``_attn_implementation`` on ``config`` + sub-configs.

    The upstream ``QwenvlWithExpertV2Model.__init__`` hardcodes
    ``flash_attention_2`` on the VLM / ``text_config`` / ``qwen_expert_config``
    (and ``vision_config`` inherits ``vit_attn_implementation`` which also
    defaults to flash). We walk every attribute that is itself a config
    (``*_config``) so the whole tree lands on a flash-free backend.
    """
    if config is None:
        return
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = target
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = target
    for name in dir(config):
        if not name.endswith("_config"):
            continue
        child = getattr(config, name, None)
        if child is not None and hasattr(child, "_attn_implementation"):
            _coerce_attn_config(child, target)


def _install_attn_fallback(*, target: str) -> None:
    """Patch the two Qwen ``_from_config`` classmethods to drop flash attention.

    ``QwenvlWithExpertV2Model.__init__`` calls
    ``Qwen3VLForConditionalGeneration._from_config(vlm_config)`` and
    ``Qwen2ForCausalLM._from_config(qwen_expert_config)`` with configs it has
    just pinned to ``flash_attention_2``. flash-attn is not installed in the
    sidecar venv, so we intercept both classmethods and coerce the incoming
    config (and its sub-configs) to ``target`` before delegating — the attention
    modules bind their backend at construction from ``config._attn_implementation``.
    Idempotent; a no-op for anyone who already passes an explicit backend.
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2ForCausalLM
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration

    for cls in (Qwen3VLForConditionalGeneration, Qwen2ForCausalLM):
        orig = cls._from_config.__func__  # type: ignore[attr-defined]  # reason: classmethod unwrap

        def _patched(inner_cls: Any, config: Any, *args: Any, _orig: Any = orig, **kwargs: Any) -> Any:
            _coerce_attn_config(config, target)
            kwargs.setdefault("attn_implementation", target)
            try:
                return _orig(inner_cls, config, *args, **kwargs)
            except TypeError:
                # Older signatures do not accept the attn_implementation kwarg.
                kwargs.pop("attn_implementation", None)
                return _orig(inner_cls, config, *args, **kwargs)

        cls._from_config = classmethod(_patched)  # type: ignore[assignment]  # reason: monkeypatch


def _nf4_backbone_in_place(backbone: Any, *, torch: Any, min_params: int = 2_000_000) -> None:
    """Self-contained NF4 rewrite of ``nn.Linear`` -> ``Linear4bit`` (bnb packs on ``.cuda()``)."""
    import bitsandbytes as bnb

    def _replace(module: Any) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= min_params:
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16, quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4")
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(torch.bfloat16), requires_grad=False)
                setattr(module, name, new)
            else:
                _replace(child)

    _replace(backbone)


class _LingBotPolicy:
    """Loads ``LingbotVLAv2Server`` (NF4 backbone / bf16 expert) and serves actions."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        repo = _resolve_repo()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # The loader resolves configs/ + assets/ relative to CWD.
        os.chdir(repo)
        qwen = _resolve_qwen()
        os.environ["QWEN3VL_PATH"] = qwen
        ckpt_dir = _resolve_checkpoint(args.model)
        _write_cli_yaml(repo, ckpt_dir, qwen)

        # The model import chain pulls training-only lerobot imports; stub them
        # before importing the deploy module (see _install_lerobot_stub).
        _install_lerobot_stub()

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

        self._torch = torch
        self._robo_name = args.robo_name
        quant = args.quantization

        # flash-attn is not installed in the sidecar venv; coerce the upstream
        # hardcoded flash_attention_2 configs to a flash-free backend BEFORE the
        # model is built (the attention modules bind at construction).
        _install_attn_fallback(target=args.attn)

        # Replicate LingbotVLAv2Server.__init__ but inject quantization before
        # the device move (the stock server only offers fp32 / bf16).
        srv = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
        srv.adaptive_ensemble_alpha = 0.1
        srv.action_ensemble_horizon = 8
        srv.use_length = 1
        srv.chunk_ret = True
        srv.robot_norm_path = None
        srv.task_description = None
        srv.use_compile = False
        apply_lingbot_qwen3_vl_patch()
        srv.vla = srv.load_vla(ckpt_dir)  # CPU build + strict load_state_dict
        # Belt-and-suspenders: coerce any attn config the build left on flash.
        _coerce_attn_config(getattr(srv.vla, "config", None), args.attn)
        srv.vla.model = srv.vla.model.to(torch.bfloat16)
        if quant == "nf4":
            _nf4_backbone_in_place(srv.vla.model.qwenvl_with_expert.qwenvl, torch=torch)
        srv.vla = srv.vla.to("cuda").eval()
        srv.global_step = 0
        srv.last_action_chunk = None
        srv.last_normalized_action_chunk = None
        srv.use_bf16 = True
        srv.use_fp32 = False
        srv.action_key = "action"
        srv.reset(self._robo_name)
        self._srv = srv
        self._action_keys = list(srv.vla.feature_transform.org_features["actions"])
        if torch.cuda.is_available():
            print(
                f"[lingbot_vla2_server] loaded ({quant}, attn={args.attn}); "
                f"VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                flush=True,
            )

    def reset(self, robo_name: str | None = None) -> None:
        self._srv.reset(robo_name or self._robo_name)

    def get_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        images = obs["images"]
        raw = {
            "observation.images.cam_high": np.asarray(images["cam_high"], dtype=np.uint8),
            "observation.images.cam_left_wrist": np.asarray(images["cam_left_wrist"], dtype=np.uint8),
            "observation.images.cam_right_wrist": np.asarray(images["cam_right_wrist"], dtype=np.uint8),
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
            "task": str(obs.get("task", "")),
        }
        out = self._srv.infer(raw)  # chunk_ret -> {action_key: (chunk, dim)}
        chunk = np.concatenate(
            [np.asarray(out[k], dtype=np.float32) for k in self._action_keys], axis=-1
        )
        return chunk, self._action_keys


def _serve(policy: _LingBotPolicy, *, host: str, port: int, model: str, robo_name: str) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[lingbot_vla2_server] serving on tcp://{host}:{port}", flush=True)
    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {"ok": True, "model": model, "robo_name": robo_name}
            elif endpoint == "reset":
                policy.reset(data.get("robo_name"))
                reply = {"ok": True}
            elif endpoint == "get_action":
                action, keys = policy.get_action(data["observation"])
                reply = {"action": action, "action_keys": keys}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:  # surface any sidecar-side fault to the client
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))
    sock.close(linger=0)
    ctx.term()
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenRAL LingBot-VLA 2.0 policy server (sidecar venv)")
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--robo-name", default="robotwin", help="upstream robot config stem")
    p.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    p.add_argument(
        "--attn",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="flash-free attention backend to replace the upstream flash_attention_2 hardcode.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args(argv)
    policy = _LingBotPolicy(args)
    return _serve(
        policy, host=args.host, port=args.port, model=args.model, robo_name=args.robo_name
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
