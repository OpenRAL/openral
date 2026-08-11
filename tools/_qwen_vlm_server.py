"""Qwen3.5-4B scene-VLM inference server — runs INSIDE the isolated sidecar venv.

Exec'd by :mod:`tools.qwen_vlm_sidecar` inside the sidecar virtualenv (see that
file for *why* the model runs out-of-process). It loads the NF4
bitsandbytes-quantized model once and answers open-ended scene questions over a
ZMQ REP socket using msgpack frames.

It is deliberately **thin**: it returns the model's generated answer text
verbatim. The main-env backend
(:mod:`openral_runner.backends.gstreamer.qwen_scene_vlm`) only strips
whitespace, so the protocol stays trivially unit-testable without a GPU or this
venv.

Wire protocol (msgpack dict in, msgpack dict out, ZMQ REQ/REP):

    {"op": "ping"}                                  -> {"ok": True, "model": <id>}
    {"op": "query", "image": <png/jpeg bytes>,
     "question": "Has the robot grasped the mug?",
     "max_side": 1024, "max_new_tokens": 256}       -> {"ok": True, "answer": <str>}
    {"op": "shutdown"}                              -> {"ok": True}

On any exception the reply is ``{"ok": False, "error": <str>}``.

Real upstream model code (no mocks, CLAUDE.md §1.11). The exact processor /
generate entrypoints below follow the canonical Qwen-VL transformers recipe;
they are validated for real by the GPU-gated end-to-end test
(``tests/sim/test_qwen_scene_vlm_e2e.py``), which the operator runs once against
a provisioned sidecar venv — not asserted blind here (CLAUDE.md §1.2).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

# Reduce CUDA allocator fragmentation during the NF4 load on tight (8 GB) GPUs —
# transformers 5.x's parallel tensor loader spikes VRAM before bitsandbytes
# quantizes, and the recommended mitigation is expandable segments. Must be set
# before torch initializes its CUDA caching allocator.
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
    (model_type ``qwen3_5``), the image-text-to-text auto class in
    transformers 5.x — hence ``AutoModelForImageTextToText``.

    Two load paths, auto-selected by whether ``model_id`` is already quantized:

    * **Pre-quantized** (e.g. ``OpenRAL/rskill-qwen35_4b-any-general-nf4``, whose config
      embeds a bitsandbytes ``quantization_config``): the 4-bit weights load
      directly (~3.3 GB) with no bf16 spike — the clean path for an 8 GB GPU.
    * **Raw upstream** (e.g. ``Qwen/Qwen3.5-4B``): quantize NF4 at load.
      transformers 5.x materializes weights in bf16 on-GPU before bitsandbytes
      quantizes; the 4-way-concurrent transient OOMs a tight card, so force
      serial materialization.
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
    # enable_thinking=False pre-closes the <think> block in the *prompt*
    # (`…assistant\n<think>\n\n</think>\n\n`) so the model answers directly
    # instead of opening a reasoning trace it has to finish within the token
    # budget. Without it the trace routinely outruns `max_new_tokens` and the
    # </think> strip below silently no-ops, so the sidecar returns a **truncated
    # scratchpad as the answer** — a reasoner parsing "The tray is on the right
    # side [603, 126," for a completion signal acts on garbage. Raising the
    # budget is not a fix: 256 and 512 both truncate on a question that 1024
    # answers, and the threshold is question-dependent.
    #
    # Measured on a GB10, same image + question: a complete, correct answer in
    # ~132 tokens / 8.4 s without thinking, versus 801 tokens / 28.9 s with it.
    # Absolute numbers are host-specific; the point is that the non-thinking
    # path is several times faster on a call the reasoner makes synchronously,
    # and its answers fit comfortably inside the token budget. Tradeoff worth
    # knowing: non-thinking mode is measurably weaker at fine spatial reasoning
    # (it has been observed calling an occupied gripper empty).
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
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
    # return only the text after the final </think> when present. With
    # enable_thinking=False above this normally no-ops — kept for a checkpoint
    # whose template ignores the toggle.
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1]
    # Guard on *truncation*, not on a tag. Hitting the token ceiling means the
    # reply stops mid-sentence — either a half-finished answer or, if the
    # enable_thinking toggle ever stops working, a reasoning scratchpad that
    # reads exactly like one. Returning either as {"ok": True} is the dangerous
    # outcome: a reasoner parsing "The tray is on the right side [603, 126," for
    # a completion signal acts on garbage.
    #
    # A tag-based check cannot do this job, which was verified the hard way: the
    # chat template puts the *opening* <think> in the prompt, and the prompt
    # tokens are trimmed above, so a completion can only ever contain the
    # closing </think>. `elif "<think>" in answer` is therefore unreachable, and
    # replaying the original defect showed it staying silent while the sidecar
    # handed back the scratchpad.
    if len(trimmed[0]) >= max_new_tokens:
        raise RuntimeError(
            f"Qwen hit the {max_new_tokens}-token ceiling, so the reply is cut off "
            "mid-generation rather than a complete answer. Raise --max-new-tokens, "
            "or check that the checkpoint still honours enable_thinking=False."
        )
    return answer.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5759)
    ap.add_argument("--max-side", type=int, default=1024)
    # 1024, not 256. With enable_thinking=False a scene answer runs ~130-250
    # tokens, so 256 looks sufficient — but it is not: a legitimate "describe
    # what the wrist camera sees" question ran past it on real weights, and with
    # the truncation guard in `_query` that now fails the query outright instead
    # of quietly returning a clipped answer. 1024 leaves headroom for the long
    # tail while keeping the guard meaningful for genuinely runaway generations.
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
