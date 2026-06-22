"""ADR-0064 — DA3 monocular metric-depth inference server (sidecar process).

Runs Depth Anything 3 (`depth-anything/DA3-SMALL` by default) in its own
isolated venv and serves metric depth over a ZMQ REP socket using msgpack
frames — the same wire pattern as :mod:`tools._locateanything_server`. The
openral-side client lives in the `openral_perception_ros` depth-provider node,
which republishes the result as a `32FC1` depth Image + CameraInfo for nvblox.

DA3-SMALL is the default because it measured **0.27 GB / ~27 Hz** on an 8 GB
Ada (vs Depth Pro's ~4 GB / ~1 Hz) and its depth range matched Depth Pro's on
the same frame (metric corroborated; absolute scale unverified vs ground truth).
It ships as the `depth-anything-3` package, not transformers-native, hence the
isolated venv.

Wire protocol (msgpack dict in, msgpack dict out, ZMQ REQ/REP):
  in : {"op": "ping"}                          -> {"ok": True, "model": str}
  in : {"op": "shutdown"}                       -> {"ok": True}
  in : {"op": "depth", "image": <png bytes>,
        "process_res": 504}
       -> {"ok": True, "depth": <float32 bytes, metres>, "h": int, "w": int,
           "fx": float, "fy": float, "cx": float, "cy": float,
           "latency_ms": float}
  on error                                      -> {"ok": False, "error": str}
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from typing import Any

import msgpack
import numpy as np
import zmq
from PIL import Image


def _load(model_id: str) -> tuple[Any, Any]:
    import torch
    from depth_anything_3.api import (  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: runs in the isolated DA3 sidecar venv (da3venv), absent from the main lint env
        DepthAnything3,
    )

    model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
    return torch, model


def _infer(torch: Any, model: Any, image: Image.Image, process_res: int) -> tuple[Any, Any, float]:
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.inference([image], process_res=process_res)
    torch.cuda.synchronize()
    depth = np.ascontiguousarray(np.asarray(pred.depth, dtype=np.float32)[0])
    k = np.asarray(pred.intrinsics, dtype=np.float32)[0]
    return depth, k, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="depth-anything/DA3-SMALL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5771)
    ap.add_argument("--process-res", type=int, default=504)
    args = ap.parse_args()

    print(f"[da3-server] loading {args.model}...", flush=True)
    t0 = time.perf_counter()
    torch, model = _load(args.model)
    print(
        f"[da3-server] loaded in {time.perf_counter() - t0:.1f}s; "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[da3-server] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        try:
            if op == "ping":
                sock.send(msgpack.packb({"ok": True, "model": args.model}, use_bin_type=True))
            elif op == "shutdown":
                sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
                break
            elif op == "depth":
                image = Image.open(io.BytesIO(req["image"])).convert("RGB")
                depth, k, latency = _infer(
                    torch, model, image, int(req.get("process_res", args.process_res))
                )
                h, w = depth.shape
                sock.send(
                    msgpack.packb(
                        {
                            "ok": True,
                            "depth": depth.tobytes(),
                            "h": int(h),
                            "w": int(w),
                            "fx": float(k[0, 0]),
                            "fy": float(k[1, 1]),
                            "cx": float(k[0, 2]),
                            "cy": float(k[1, 2]),
                            "latency_ms": latency,
                        },
                        use_bin_type=True,
                    )
                )
            else:
                sock.send(
                    msgpack.packb({"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True)
                )
        except Exception as exc:  # server must reply, not die, on a bad frame
            sock.send(
                msgpack.packb(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, use_bin_type=True
                )
            )

    print("[da3-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
