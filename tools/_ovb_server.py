"""Open Vision Banana inference server — runs INSIDE the isolated sidecar venv.

Exec'd by ``tools.ovb_sidecar``. Loads the base ``Flux2KleinPipeline`` with the
OVB fine-tuned transformer swapped in (upstream ``common/flux2_runtime.py``)
once, in bf16, and answers image-to-image requests over ZMQ REP + msgpack.

Deliberately thin: it returns the painted image as PNG. The prompt, the colour
decode and the region split live in the main-env backend
(``openral_runner.backends.gstreamer.ovb_detector``), unit-testable without a GPU.

Wire protocol (msgpack dict in, msgpack dict out, ZMQ REQ/REP):

    {"op": "ping"}                                   -> {"ok": True, "model": <id>}
    {"op": "generate", "image": <png bytes>, "prompt": <str>,
     "steps": 28, "guidance": 3.5, "seed": 0}         -> {"ok": True, "image": <png>,
                                                         "elapsed_s": <float>}
    {"op": "shutdown"}                               -> {"ok": True}

On any exception the reply is ``{"ok": False, "error": <str>}``. The output is
generated at the input's own size (``width``/``height`` set from the image).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from typing import Any

import msgpack
import torch
import zmq
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image

_SHA = re.compile(r"[0-9a-f]{40}")


def _split_ref(ref: str) -> tuple[str, str]:
    """Split ``repo@<sha>``; only a full commit sha pins the weights immutably."""
    repo_id, _, revision = ref.partition("@")
    if _SHA.fullmatch(revision) is None:
        raise ValueError(f"expected an immutable 'repo@<40-hex sha>' reference, got {ref!r}")
    return repo_id, revision


def _load(model: str, subfolder: str, base: str, offload: str) -> Flux2KleinPipeline:
    ovb_repo, ovb_rev = _split_ref(model)
    base_repo, base_rev = _split_ref(base)
    transformer = Flux2Transformer2DModel.from_pretrained(
        ovb_repo, subfolder=subfolder, revision=ovb_rev, torch_dtype=torch.bfloat16
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        base_repo, revision=base_rev, transformer=transformer, torch_dtype=torch.bfloat16
    )
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def _generate(pipe: Flux2KleinPipeline, req: dict[str, Any]) -> bytes:
    image = Image.open(io.BytesIO(req["image"])).convert("RGB")
    out = pipe(
        prompt=str(req["prompt"]),
        image=image,
        width=image.width,
        height=image.height,
        num_inference_steps=int(req.get("steps", 28)),
        guidance_scale=float(req.get("guidance", 3.5)),
        generator=torch.Generator(device="cpu").manual_seed(int(req.get("seed", 0))),
    ).images[0]
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5759)
    ap.add_argument("--offload", default="model", choices=["none", "model", "sequential"])
    args = ap.parse_args()

    print(f"[ovb-server] loading {args.model} ({args.subfolder}) on {args.base}...", flush=True)
    t0 = time.perf_counter()
    pipe = _load(args.model, args.subfolder, args.base, args.offload)
    print(
        f"[ovb-server] loaded in {time.perf_counter() - t0:.1f}s (offload={args.offload}); "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[ovb-server] listening on tcp://{args.host}:{args.port}", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        try:
            if op == "ping":
                reply: dict[str, object] = {"ok": True, "model": args.model}
            elif op == "shutdown":
                sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
                break
            elif op == "generate":
                t = time.perf_counter()
                png = _generate(pipe, req)
                reply = {"ok": True, "image": png, "elapsed_s": time.perf_counter() - t}
            else:
                reply = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, use_bin_type=True))
    print("[ovb-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
