"""LocateAnything-3B inference server — runs INSIDE the isolated sidecar venv.

This module is exec'd by :mod:`tools.locateanything_sidecar` inside a
``transformers==4.57.1`` virtualenv (see that file for *why* the model cannot
share the openral runtime's ``transformers>=5`` environment). It loads the NF4
bitsandbytes-quantized model once and answers detection requests over a ZMQ
REP socket using msgpack frames.

It is deliberately **thin**: it returns the model's *raw* generated text. All
``<ref>``/``<box>`` parsing, degenerate-box filtering, and ``ObjectsMetadata``
construction live in the main-env backend
(:mod:`openral_runner.backends.gstreamer.locateanything_detector`) so that
logic is unit-testable without a GPU or this venv.

Wire protocol (msgpack dict in, msgpack dict out, ZMQ REQ/REP):

    {"op": "ping"}                              -> {"ok": True, "model": <id>}
    {"op": "detect", "image": <png/jpeg bytes>,
     "query": "person", "max_side": 1024,
     "mode": "hybrid", "max_new_tokens": 1024}  -> {"ok": True, "answer": <str>,
                                                    "norm": 1000}
    {"op": "shutdown"}                          -> {"ok": True}

On any exception the reply is ``{"ok": False, "error": <str>}``.

This is real upstream model code (no mocks, CLAUDE.md §1.11). ``trust_remote_code``
is True because the model ships custom modeling files; the operator opts into
that risk by launching the sidecar for this specific model.
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import msgpack
import torch
import zmq
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


def _load(model_id: str) -> tuple[object, object, object]:
    """Load tokenizer, processor, and the NF4-quantized model on the GPU."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    ).eval()
    return tokenizer, processor, model


@torch.no_grad()
def _detect(
    tokenizer: object,
    processor: object,
    model: object,
    *,
    image: Image.Image,
    query: str,
    max_side: int,
    mode: str,
    max_new_tokens: int,
) -> str:
    """Run one open-vocabulary detection and return the raw generated text."""
    w, h = image.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    prompt = f"Locate all the instances that matches the following description: {query}."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")

    response = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        generation_mode=mode,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        verbose=False,
    )
    return response[0] if isinstance(response, tuple) else response


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5757)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    args = ap.parse_args()

    print(f"[la-server] loading {args.model} (NF4)...", flush=True)
    t0 = time.perf_counter()
    tokenizer, processor, model = _load(args.model)
    print(
        f"[la-server] loaded in {time.perf_counter() - t0:.1f}s; "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[la-server] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        try:
            if op == "ping":
                sock.send(msgpack.packb({"ok": True, "model": args.model}, use_bin_type=True))
            elif op == "shutdown":
                sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
                break
            elif op == "detect":
                image = Image.open(io.BytesIO(req["image"])).convert("RGB")
                answer = _detect(
                    tokenizer,
                    processor,
                    model,
                    image=image,
                    query=req["query"],
                    max_side=int(req.get("max_side", args.max_side)),
                    mode=req.get("mode", args.mode),
                    max_new_tokens=int(req.get("max_new_tokens", args.max_new_tokens)),
                )
                sock.send(
                    msgpack.packb({"ok": True, "answer": answer, "norm": 1000}, use_bin_type=True)
                )
            else:
                sock.send(
                    msgpack.packb({"ok": False, "error": f"unknown op {op!r}"}, use_bin_type=True)
                )
        except Exception as exc:
            sock.send(
                msgpack.packb(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, use_bin_type=True
                )
            )

    print("[la-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
