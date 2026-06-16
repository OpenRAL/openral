"""Stateless Robometer reward-scoring server (ADR-0057), run in the sidecar venv.

ZMQ REQ/REP + msgpack. Loads the NF4 RBM reward model once, then scores clips on
demand. Stateless — the rolling frame buffer / windowing lives node-side
(:class:`openral_runner.backends.reward.frame_source.RollingFrameBuffer`).

Two load paths (``--mode``, default ``auto``):
  * **prequantized** — the published ``OpenRAL/rskill-robometer-4b-nf4`` checkpoint
    is loaded DIRECTLY as 4-bit on the meta device (no 8 GB bf16 materialisation,
    no requantize): build the RBM skeleton on ``meta`` -> install empty Linear4bit
    shells -> ``Params4bit.from_prequantized`` the packed weights -> assign the
    folded rotary buffers. ~1.7 s, 3.32 GB VRAM.
  * **bf16** — the upstream Apache-2.0 ``robometer/Robometer-4B`` is loaded bf16 via
    the pinned robometer loader, then NF4-quantized in place (the path used to
    BUILD the pre-quantized checkpoint; ~90 s + 19 GB transient CPU RSS).

``auto`` picks prequantized for a local pre-quantized dir or an HF repo id
containing ``nf4``, else bf16. The meta path is byte-identical to the bf16 path
(verified) and, with the determinism pins applied here, reproducible across
launches.

Protocol (msgpack dict, key ``op``):
  ping     {}                                          -> {ok, model}
  score    {frames: bytes(n*h*w*3 BGR), n, width,      -> {ok, progress:[float],
            height, task: str, num_bins: int}                success:[float]}
  shutdown {}                                          -> {ok}

Booted by ``tools/robometer_sidecar.py``. Not imported by the main package.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import fields

import _robometer_quant as q

q.set_cublas_workspace_env()  # MUST precede CUDA init (torch import below)

import msgpack
import numpy as np
import torch
import zmq

q.apply_determinism()


def _split_repo_rev(weights: str) -> tuple[str, str | None]:
    """``repo@rev`` -> (repo, rev); a bare repo/path -> (repo, None)."""
    if "@" in weights and not os.path.isdir(weights):
        repo, rev = weights.rsplit("@", 1)
        return repo, rev
    return weights, None


def _resolve_local_dir(weights: str) -> str:
    """Return a local directory for ``weights`` (snapshot_download if it's an HF id)."""
    weights = weights.removeprefix("local://")
    if os.path.isdir(weights):
        return weights
    from huggingface_hub import snapshot_download

    repo, rev = _split_repo_rev(weights)
    return snapshot_download(repo, revision=rev)


class _Scorer:
    """Loads the NF4 RBM once and scores clips (discrete mode -> [0,1])."""

    def __init__(self, weights: str, device: str = "cuda", mode: str = "auto") -> None:
        from robometer.utils.setup_utils import setup_batch_collator

        self.device = device
        resolved = mode
        if mode == "auto":
            local_first = weights if os.path.isdir(weights) else None
            if local_first and q.is_prequantized_checkpoint(local_first):
                resolved = "prequantized"
            elif "nf4" in os.path.basename(_split_repo_rev(weights)[0]).lower():
                resolved = "prequantized"
            else:
                resolved = "bf16"

        if resolved == "prequantized":
            self.exp_config, self.tokenizer, self.processor, self.model = (
                self._load_prequantized(weights))
        else:
            self.exp_config, self.tokenizer, self.processor, self.model = (
                self._load_bf16_and_quantize(weights))

        self.model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"[robometer-server] ready ({resolved}): {vram:.2f} GB on {device}",
                  flush=True)
        self.collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True)

    def _load_prequantized(self, weights: str):
        """Direct 4-bit load of the published checkpoint (meta device, no bf16)."""
        import yaml
        from robometer.configs.experiment_configs import ExperimentConfig
        from robometer.models.rbm import RBM
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        local = _resolve_local_dir(weights)
        print(f"[robometer-server] prequantized meta-load from {local} ...", flush=True)
        raw = yaml.safe_load(open(os.path.join(local, "config.yaml")))
        valid = {f.name for f in fields(ExperimentConfig)}
        exp = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})
        base_id = getattr(exp.model, "base_model_id", "Qwen/Qwen3-VL-4B-Instruct")

        config = AutoConfig.from_pretrained(local)
        processor = AutoProcessor.from_pretrained(local)
        tokenizer = AutoTokenizer.from_pretrained(local)
        # Direct construction defaults to "eager"; production uses "sdpa".
        for c in (config, getattr(config, "text_config", None),
                  getattr(config, "vision_config", None)):
            if c is not None:
                c._attn_implementation = "sdpa"

        with torch.device("meta"):
            model = RBM(config, processor, tokenizer, base_model=None,
                        base_model_id=base_id, model_config=exp.model)
        q.install_linear4bit_shells(model, torch.bfloat16)
        state = load_file(os.path.join(local, "model.safetensors"), device=self.device)
        consumed = q.install_prequantized(model, state, self.device)
        leftover = {k: v for k, v in state.items() if k not in consumed}
        model.load_state_dict(leftover, strict=False, assign=True)
        n_buf = q.assign_meta_buffers(model, state, self.device)
        still_meta = [n for n, p in model.named_parameters() if p.is_meta]
        if still_meta:
            raise RuntimeError(f"meta params left after prequant load: {still_meta[:5]}")
        print(f"[robometer-server] installed NF4 + {n_buf} rotary buffers", flush=True)
        return exp, tokenizer, processor, model

    def _load_bf16_and_quantize(self, weights: str):
        """Upstream bf16 load + in-place NF4 (used to build the prequant checkpoint)."""
        from robometer.utils.save import load_model_from_hf

        print(f"[robometer-server] loading {weights} bf16 on CPU ...", flush=True)
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=weights, device="cpu")
        n = q.quantize_nf4_in_place(model, compute_dtype=torch.bfloat16)
        model.to(self.device)
        print(f"[robometer-server] quantized {n} NF4 modules", flush=True)
        return exp_config, tokenizer, processor, model

    @torch.no_grad()
    def score(self, frames_rgb: np.ndarray, task: str, num_bins: int) -> tuple[list, list]:
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        t = int(frames_rgb.shape[0])
        traj = Trajectory(frames=frames_rgb, frames_shape=tuple(frames_rgb.shape), task=task,
                          id="0", metadata={"subsequence_length": t}, video_embeddings=None)
        batch = self.collator([ProgressSample(trajectory=traj, sample_type="progress")])
        inp = batch["progress_inputs"]
        for k, v in inp.items():
            if hasattr(v, "to"):
                inp[k] = v.to(self.device)
        res = compute_batch_outputs(self.model, self.tokenizer, inp,
                                    sample_type="progress", is_discrete_mode=True,
                                    num_bins=num_bins)
        prog = np.asarray(res["progress_pred"][0] if isinstance(res["progress_pred"], list)
                          else res["progress_pred"], dtype=np.float32)
        succ_raw = res.get("outputs_success", {}).get("success_probs")
        succ = (np.asarray(succ_raw[0] if isinstance(succ_raw, list) and succ_raw else succ_raw,
                           dtype=np.float32) if succ_raw is not None else np.zeros_like(prog))
        return prog.tolist(), succ.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="OpenRAL/rskill-robometer-4b-nf4")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5769)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="auto", choices=("auto", "prequantized", "bf16"))
    args = ap.parse_args()

    scorer = _Scorer(args.weights, device=args.device, mode=args.mode)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[robometer-server] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        if op == "ping":
            sock.send(msgpack.packb({"ok": True, "model": args.weights}))
        elif op == "score":
            try:
                n, w, h = int(req["n"]), int(req["width"]), int(req["height"])
                bgr = np.frombuffer(req["frames"], dtype=np.uint8).reshape(n, h, w, 3)
                rgb = bgr[:, :, :, ::-1]  # BGR -> RGB (model expects RGB)
                progress, success = scorer.score(
                    np.ascontiguousarray(rgb), str(req["task"]), int(req.get("num_bins", 100)))
                sock.send(msgpack.packb({"ok": True, "progress": progress, "success": success}))
            except Exception as exc:  # noqa: BLE001 — report to client, keep serving
                sock.send(msgpack.packb({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        elif op == "shutdown":
            sock.send(msgpack.packb({"ok": True}))
            break
        else:
            sock.send(msgpack.packb({"ok": False, "error": f"unknown op {op!r}"}))

    sock.close()
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
