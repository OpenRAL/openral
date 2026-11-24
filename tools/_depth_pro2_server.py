"""Depth Pro 2 monocular metric-depth inference server (sidecar process).

Runs Apple Depth Pro 2 (``apple/DepthPro2-*``) in its own venv and serves depth over
ZMQ REQ/REP + msgpack with the same wire protocol as ``tools/_da3_depth_server.py``, so
the ``openral_perception_ros`` depth provider talks to either. The weights are under the
Apple Machine Learning Research Model license (research only); the launcher
(``tools/depth_estimator_sidecar.py``) refuses them without ``OPENRAL_ALLOW_NONCOMMERCIAL=1``.

Each frame is inferred on its own (``DepthPro2.infer``), never as a clip: a clip's frames
share one focal length and attend to each other, which a live provider cannot wait for.
The checkpoint has no field-of-view head (``config.json``: ``fov: false``), so every
request must carry ``f_px``, the camera's focal length in pixels of the image sent.

Wire protocol (msgpack dict in, msgpack dict out):
  in : {"op": "ping"}                          -> {"ok": True, "model": str}
  in : {"op": "shutdown"}                       -> {"ok": True}
  in : {"op": "depth", "image": <png bytes>, "f_px": float}
       -> {"ok": True, "depth": <float32 bytes>, "h": int, "w": int,
           "fx": nan, "fy": nan, "cx": nan, "cy": nan,
           "latency_ms": float, "model": str}

  ``depth`` is metres along the optical axis at the input's own size; sky / no return
  is non-finite (the provider zeroes it). The model estimates no camera, so K is NaN.
  on error                                      -> {"ok": False, "error": str}
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from typing import Any

import msgpack
import numpy as np
import zmq
from PIL import Image


def _load(model_id: str, revision: str | None) -> tuple[Any, Any, Any]:
    import torch
    from depth_pro_2.model import (  # type: ignore[import-not-found,unused-ignore]  # reason: runs in the isolated Depth Pro 2 sidecar venv, absent from the main lint env
        DepthPro2,
        to_normalized_tensor,
    )

    device = torch.device("cuda")
    # bfloat16 autocast on CUDA, as the model was trained and evaluated (upstream default).
    model = DepthPro2.from_pretrained(model_id, revision=revision, precision=torch.bfloat16)
    model = model.to(device).eval()
    return torch, model, lambda rgb: to_normalized_tensor(rgb, device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo, e.g. apple/DepthPro2-S")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5772)
    args = ap.parse_args()

    print(f"[depth-pro2-server] loading {args.model}@{args.revision}...", flush=True)
    t0 = time.perf_counter()
    torch, model, transform = _load(args.model, args.revision)
    print(
        f"[depth-pro2-server] loaded in {time.perf_counter() - t0:.1f}s; "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[depth-pro2-server] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        try:
            if op == "ping":
                reply: dict[str, Any] = {"ok": True, "model": args.model}
            elif op == "shutdown":
                sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
                break
            elif op == "depth":
                f_px = float(req.get("f_px", math.nan))
                if not (math.isfinite(f_px) and f_px > 0.0):
                    raise ValueError(f"f_px must be a positive focal in pixels, got {f_px}")
                rgb = np.asarray(Image.open(io.BytesIO(req["image"])).convert("RGB"))
                t1 = time.perf_counter()
                with torch.no_grad():
                    depth_t = model.infer(transform(rgb), f_px=f_px)["depth"]
                torch.cuda.synchronize()
                latency = (time.perf_counter() - t1) * 1000.0
                depth = np.ascontiguousarray(depth_t.float().cpu().numpy(), dtype=np.float32)
                h, w = depth.shape
                nan = float("nan")
                reply = {
                    "ok": True,
                    "depth": depth.tobytes(),
                    "h": int(h),
                    "w": int(w),
                    "fx": nan,
                    "fy": nan,
                    "cx": nan,
                    "cy": nan,
                    "latency_ms": latency,
                    "model": args.model,
                }
            else:
                reply = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:  # server must reply, not die, on a bad frame
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, use_bin_type=True))

    print("[depth-pro2-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
