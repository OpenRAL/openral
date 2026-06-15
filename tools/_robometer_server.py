"""Stateless Robometer reward-scoring server (ADR-0057), run in the sidecar venv.

ZMQ REQ/REP + msgpack. Loads the NF4-quantized RBM reward model once via the
pinned ``robometer`` package, then scores clips on demand. Stateless — the
rolling frame buffer / windowing lives node-side
(:class:`openral_runner.backends.reward.frame_source.RollingFrameBuffer`).

Protocol (msgpack dict, key ``op``):
  ping     {}                                          -> {ok, model}
  score    {frames: bytes(n*h*w*3 BGR), n, width,      -> {ok, progress:[float],
            height, task: str, num_bins: int}                success:[float]}
  shutdown {}                                          -> {ok}

Booted by ``tools/robometer_sidecar.py``. Not imported by the main package.
"""

from __future__ import annotations

import argparse

import msgpack
import numpy as np
import torch
import zmq

_MIN_PARAMS = 4_000_000  # openral_sim._quantization.DEFAULT_MIN_PARAMS_TO_QUANTIZE


def _quantize_nf4_in_place(root: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    """Rewrite large ``nn.Linear`` -> ``bnb.nn.Linear4bit`` (the repo's NF4 rule).

    Inlined from openral_sim._quantization.quantize_nf4_in_place because that
    module is not importable in the isolated sidecar venv.
    """
    import bitsandbytes as bnb

    n = 0

    def _replace(module: torch.nn.Module) -> None:
        nonlocal n
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= _MIN_PARAMS:
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype, quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4")
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype), requires_grad=False)
                setattr(module, name, new)
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


class _Scorer:
    """Loads the NF4 RBM once and scores clips (discrete mode -> [0,1])."""

    def __init__(self, weights: str, device: str = "cuda") -> None:
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.device = device
        print(f"[robometer-server] loading {weights} bf16 on CPU ...", flush=True)
        self.exp_config, self.tokenizer, self.processor, self.model = load_model_from_hf(
            model_path=weights, device="cpu")
        n = _quantize_nf4_in_place(self.model, compute_dtype=torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"[robometer-server] ready: {n} NF4 modules, {vram:.2f} GB on {device}",
                  flush=True)
        self.collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True)

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
    ap.add_argument("--weights", default="robometer/Robometer-4B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5769)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scorer = _Scorer(args.weights, device=args.device)

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
