"""Boot the LingBot-VLA 2.0 inference server in an isolated py3.12 sidecar venv.

Robbyant's ``lingbotvla`` package (https://github.com/robbyant/lingbot-vla-v2)
pins ``torch==2.8.0`` / ``transformers==4.57.3`` / ``triton==3.4.0`` and carries
custom Triton MoE kernels + a ``sys.path``-based layout, so it cannot coexist in
the openral py3.12 (torch>=2.9 / transformers>=5) workspace (CLAUDE.md §3). We
run the upstream ``LingbotVLAv2Server`` out-of-process and drive it from
:mod:`openral_sim.policies.lingbot_vla2` over ZMQ REQ/REP framed by msgpack —
the same transport the rldx / rlbench-3dda sidecars use.

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

Provisioning (repo checkout, Qwen3-VL backbone, this script's venv) is resolved
from env / args below. This is the Phase-1 boot helper: it assumes the venv +
repo checkout already exist (set ``OPENRAL_LINGBOT_VLA2_REPO`` /
``OPENRAL_QWEN3VL_PATH``); Phase 2 wires the auto git-clone + venv build via
``openral_sim._sidecar_common`` once the 25 GB checkpoint fetch is unblocked.

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
from typing import Any

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
            f"lingbotvla repo checkout not found at {repo}. Clone "
            "https://github.com/robbyant/lingbot-vla-v2 there or set "
            "OPENRAL_LINGBOT_VLA2_REPO."
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
    # loader path: Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"
    cli = Path(ckpt_dir).parent.parent.parent / "lingbotvla_cli.yaml"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(yaml.safe_dump(tmpl))
    return cli


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

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

        self._torch = torch
        self._robo_name = args.robo_name
        quant = args.quantization

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
                f"[lingbot_vla2_sidecar] loaded ({quant}); "
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
    print(f"[lingbot_vla2_sidecar] serving on tcp://{host}:{port}", flush=True)
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
    p = argparse.ArgumentParser(description="OpenRAL LingBot-VLA 2.0 policy sidecar")
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--robo-name", default="robotwin", help="upstream robot config stem")
    p.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
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
