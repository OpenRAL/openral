"""Phase 3 PROOF-OF-CONCEPT reward sidecar for Robometer-4B (ADR-0057).

A long-lived out-of-process ZMQ REP server that loads the NF4-quantized RBM once,
maintains a rolling time-indexed frame buffer, and answers windowed progress/success
queries — the runtime shape the Reasoner's QueryTaskProgressTool will call.

This is a spike that runs in the isolated /tmp/robometer-env (robometer + transformers
4.57.1). The production version lives at python/runner/.../backends/reward/ with a
ROS FrameSource instead of ZMQ-pushed frames. Protocol: msgpack req/rep.

Commands (msgpack dict, key "cmd"):
  set_task   {task: str}
  ingest     {frames: bytes, shape: [n,h,w,3], stamp_ns: int, fps: float}
  query      {window_s: float}  -> per-frame progress/success over the window
  shutdown   {}

Run: /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/reward_sidecar.py --port 5599
"""

from __future__ import annotations

import argparse
import collections

import msgpack
import numpy as np
import torch
import zmq

MIN_PARAMS = 4_000_000


def quantize_nf4_in_place(root: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    import bitsandbytes as bnb

    n = 0

    def _replace(module: torch.nn.Module) -> None:
        nonlocal n
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype), requires_grad=False
                    )
                setattr(module, name, new)
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


class RewardMonitor:
    """Loads NF4 RBM once; scores a clip of frames -> per-frame progress/success."""

    def __init__(self, device: str = "cuda", num_bins: int = 100) -> None:
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.device = device
        self.num_bins = num_bins
        print("[sidecar] loading bf16 on CPU ...", flush=True)
        self.exp_config, self.tokenizer, self.processor, self.model = load_model_from_hf(
            model_path="robometer/Robometer-4B", device="cpu"
        )
        print("[sidecar] NF4 rewrite ...", flush=True)
        n = quantize_nf4_in_place(self.model, compute_dtype=torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[sidecar] ready: {n} NF4 modules, {vram:.2f} GB resident on {device}", flush=True)
        self.collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True
        )

    @torch.no_grad()
    def score(self, frames: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        T = int(frames.shape[0])
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        batch = self.collator([ProgressSample(trajectory=traj, sample_type="progress")])
        inp = batch["progress_inputs"]
        for k, v in inp.items():
            if hasattr(v, "to"):
                inp[k] = v.to(self.device)
        res = compute_batch_outputs(
            self.model,
            self.tokenizer,
            inp,
            sample_type="progress",
            is_discrete_mode=True,
            num_bins=self.num_bins,
        )
        prog = np.asarray(
            res["progress_pred"][0]
            if isinstance(res["progress_pred"], list)
            else res["progress_pred"],
            dtype=np.float32,
        )
        succ_raw = res.get("outputs_success", {}).get("success_probs")
        succ = (
            np.asarray(
                succ_raw[0] if isinstance(succ_raw, list) and succ_raw else succ_raw,
                dtype=np.float32,
            )
            if succ_raw is not None
            else np.zeros_like(prog)
        )
        return prog, succ


def _trend(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float32)
    return float(np.polyfit(x, arr, 1)[0])  # slope per frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--stale-deadline-s", type=float, default=3.0)
    args = ap.parse_args()

    monitor = RewardMonitor()
    task = ""
    buf: collections.deque[tuple[int, np.ndarray]] = collections.deque()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[sidecar] listening on tcp://127.0.0.1:{args.port}", flush=True)

    while True:
        msg = msgpack.unpackb(sock.recv(), raw=False)
        cmd = msg.get("cmd")
        if cmd == "set_task":
            task = msg["task"]
            buf.clear()
            sock.send(msgpack.packb({"ok": True}))
        elif cmd == "ingest":
            frames = np.frombuffer(msg["frames"], dtype=np.uint8).reshape(msg["shape"])
            stamp = int(msg["stamp_ns"])
            for i, fr in enumerate(frames):
                buf.append((stamp + i, fr))
            # evict frames older than window relative to newest
            newest = buf[-1][0]
            horizon = newest - int(args.window_s * 1e9)
            while buf and buf[0][0] < horizon:
                buf.popleft()
            sock.send(msgpack.packb({"ok": True, "buffered": len(buf)}))
        elif cmd == "query":
            window_s = float(msg.get("window_s", args.window_s))
            if not buf:
                sock.send(msgpack.packb({"stale": True, "frames_seen": 0}))
                continue
            newest = buf[-1][0]
            horizon = newest - int(window_s * 1e9)
            frames = np.stack([fr for ts, fr in buf if ts >= horizon])
            prog, succ = monitor.score(frames, task)
            stale = False  # ZMQ push path: freshness is the caller's responsibility here
            sock.send(
                msgpack.packb(
                    {
                        "progress_now": float(prog[-1]),
                        "success_now": float(succ[-1]),
                        "progress_trend": _trend(prog),
                        "success_trend": _trend(succ),
                        "progress_series": [round(float(x), 4) for x in prog.tolist()],
                        "success_series": [round(float(x), 4) for x in succ.tolist()],
                        "stalled": bool(abs(_trend(prog)) < 0.002),
                        "frames_seen": int(frames.shape[0]),
                        "stale": stale,
                    }
                )
            )
        elif cmd == "shutdown":
            sock.send(msgpack.packb({"ok": True}))
            break
        else:
            sock.send(msgpack.packb({"error": f"unknown cmd {cmd}"}))

    sock.close()
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
