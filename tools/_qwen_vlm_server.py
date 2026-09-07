"""Qwen3.5-4B scene-VLM inference server — runs INSIDE the isolated sidecar venv.

Exec'd by :mod:`tools.qwen_vlm_sidecar` (see that file for why out-of-process).
Loads the NF4 bitsandbytes-quantized model once and answers scene questions
over ZMQ REP + msgpack. Deliberately thin: returns the model's answer text
verbatim; the main-env backend
(:mod:`openral_runner.backends.gstreamer.qwen_scene_vlm`) only strips
whitespace, keeping the protocol unit-testable without a GPU.

Wire protocol (msgpack dict in/out, ZMQ REQ/REP):

    {"op": "ping"}                                  -> {"ok": True, "model": <id>}
    {"op": "query", "image": <png/jpeg bytes>,
     "question": "Has the robot grasped the mug?",
     "max_side": 1024, "max_new_tokens": 256}       -> {"ok": True, "answer": <str>}
    {"op": "shutdown"}                              -> {"ok": True}

On any exception: ``{"ok": False, "error": <str>}``.

Real upstream model code, no mocks (§1.11); validated by the GPU-gated
``tests/sim/test_qwen_scene_vlm_e2e.py``, not asserted blind here (§1.2).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import os
import sys
import time

# Reduce CUDA allocator fragmentation for the NF4 load on tight (8 GB) GPUs —
# transformers 5.x's parallel tensor loader spikes VRAM before bitsandbytes
# quantizes. Must be set before torch initializes its CUDA allocator, so the
# version is read from installed metadata rather than `import torch`. torch
# renamed this var in 2.9 (PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF) and
# warns on the old spelling. `make_isolated_env` (the boot helper) usually
# sets this already; duplicated here so running this file directly is still
# correct — this module has no `openral_*` on the path to share the logic.
if (_v := importlib.metadata.version("torch").split(".")[:2]) and tuple(map(int, _v)) >= (2, 9):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import msgpack
import torch
import zmq
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


def _load(model_id: str) -> tuple[object, object]:
    """Load the processor and the NF4 model on the GPU.

    ``Qwen/Qwen3.5-4B`` registers as ``Qwen3_5ForConditionalGeneration``
    (model_type ``qwen3_5``) — ``AutoModelForImageTextToText`` in
    transformers 5.x.

    Two paths, auto-selected by whether ``model_id`` is already quantized:
    pre-quantized (e.g. ``OpenRAL/rskill-qwen35_4b-any-general-nf4``) loads
    4-bit weights directly (~3.3 GB), no bf16 spike; raw upstream (e.g.
    ``Qwen/Qwen3.5-4B``) quantizes NF4 at load, forcing serial materialization
    since transformers 5.x's concurrent bf16 loader OOMs a tight 8 GB card.
    """
    cfg = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    if getattr(cfg, "quantization_config", None) is not None:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map={"": 0},
        ).eval()
        return processor, model

    import transformers.core_model_loading as _cml

    _cml.GLOBAL_WORKERS = 1
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    return processor, model


@torch.no_grad()
def _query(
    processor: object,
    model: object,
    *,
    image: Image.Image,
    question: str,
    max_side: int,
    max_new_tokens: int,
) -> str:
    """Run one scene query and return the model's generated answer text."""
    w, h = image.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    # Thinking left ON deliberately: `enable_thinking=False` is ~3x faster
    # (GB10: ~132 tok/8.4s vs 801 tok/28.9s, same question) but less accurate —
    # non-thinking mode described an occupied gripper as empty, twice. Cost:
    # the trace must finish inside `max_new_tokens` or the </think> strip
    # below no-ops and a truncated scratchpad reads like an answer — hence the
    # 1024 default + the truncation guard below (256/512 both truncated on a
    # question that took ~801 tokens).
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Trim the prompt tokens so only the answer is decoded.
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated, strict=True)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # Qwen3.5 is a "thinking" model: it emits a <think>…</think> reasoning trace
    # before the answer. The reasoner wants the conclusion, not the scratchpad, so
    # return only the text after the final </think>.
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1]
    # Guard on truncation, not a tag. Hitting the ceiling means the trace
    # never closed, so the strip above no-opped and `answer` is the raw
    # scratchpad — returning that as {"ok": True} lets a reasoner act on
    # garbage (e.g. "The tray is on the right side [603, 126,"). A tag check
    # can't substitute: the opening <think> is in the (trimmed) prompt, so a
    # completion only ever holds the closing tag — `elif "<think>" in answer`
    # is unreachable.
    if len(trimmed[0]) >= max_new_tokens:
        raise RuntimeError(
            f"Qwen hit the {max_new_tokens}-token ceiling, so its <think> trace never "
            "closed and the reply is a truncated reasoning scratchpad, not an answer. "
            "Raise --max-new-tokens."
        )
    return answer.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5759)
    ap.add_argument("--max-side", type=int, default=1024)
    # 1024, not 256: thinking is on (see `_query`), so the budget must cover
    # the <think> trace plus the answer — 256/512 both truncated mid-trace on
    # a question measured at ~801 tokens.
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    print(f"[qwen-server] loading {args.model} (NF4)...", flush=True)
    t0 = time.perf_counter()
    processor, model = _load(args.model)
    print(
        f"[qwen-server] loaded in {time.perf_counter() - t0:.1f}s; "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[qwen-server] listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        op = req.get("op")
        try:
            if op == "ping":
                sock.send(msgpack.packb({"ok": True, "model": args.model}, use_bin_type=True))
            elif op == "shutdown":
                sock.send(msgpack.packb({"ok": True}, use_bin_type=True))
                break
            elif op == "query":
                image = Image.open(io.BytesIO(req["image"])).convert("RGB")
                answer = _query(
                    processor,
                    model,
                    image=image,
                    question=req["question"],
                    max_side=int(req.get("max_side", args.max_side)),
                    max_new_tokens=int(req.get("max_new_tokens", args.max_new_tokens)),
                )
                sock.send(msgpack.packb({"ok": True, "answer": answer}, use_bin_type=True))
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

    print("[qwen-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
