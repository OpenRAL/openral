"""Standalone NF4 4-bit inference spike for nvidia/LocateAnything-3B.

Proves the model loads and runs an open-vocabulary detection query on an
8 GB Ada GPU using bitsandbytes NF4 quantization at load time (shard-wise,
so peak VRAM never holds the full bf16 model). Draws parsed boxes onto the
image and writes an overlay PNG.

This is a feasibility spike / reference recipe for the runtime detector
adapter — NOT part of CI. Run with the project venv:

    OPENRAL_ALLOW_REMOTE_CODE=1 \
      .venv/bin/python tools/locateanything_infer_demo.py \
      --image <path/to/image.png> --query person
"""

from __future__ import annotations

import argparse
import re
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

MODEL = "nvidia/LocateAnything-3B"
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer: str, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Parse ``<box><x1><y1><x2><y2></box>`` (coords normalized to [0,1000]).

    Drops degenerate boxes (zero-area slivers, near-full-image boxes) and
    exact duplicates — the model can loop on a repeated box token when it
    fails to emit an end token, producing a tail of identical garbage boxes.
    """
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for m in _BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        bw, bh = (x2 - x1) / 1000, (y2 - y1) / 1000
        if bw <= 0.02 or bh <= 0.02:  # zero-area / sliver
            continue
        if bw * bh >= 0.85:  # near-whole-image degenerate box
            continue
        out.append((x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", default="person")
    ap.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Resize so the longest image side <= this before inference (caps MoonViT "
        "attention memory). Boxes are mapped back to the original resolution for overlay.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or args.image.rsplit(".", 1)[0] + f"_la_{args.query.replace(' ', '_')}.png"

    print(f"[demo] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    ).eval()
    print(f"[demo] load+quantize: {time.perf_counter() - t0:.1f}s", flush=True)
    print(
        f"[demo] VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
        f"(reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB)",
        flush=True,
    )

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    # Downscale a copy for inference so MoonViT's dense attention stays within
    # 8 GB. Normalized [0,1000] box coords map back to the original resolution.
    infer_img = img
    longest = max(w, h)
    if longest > args.max_side:
        scale = args.max_side / longest
        infer_img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        print(f"[demo] resized {w}x{h} -> {infer_img.size} for inference", flush=True)
    prompt = f"Locate all the instances that matches the following description: {args.query}."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": infer_img}, {"type": "text", "text": prompt}],
        }
    ]

    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")

    t0 = time.perf_counter()
    with torch.no_grad():
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            generation_mode=args.mode,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
    answer = response[0] if isinstance(response, tuple) else response
    print(f"[demo] generate: {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[demo] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[demo] raw answer:\n{answer}\n", flush=True)

    boxes = parse_boxes(answer, w, h)
    print(f"[demo] parsed {len(boxes)} boxes", flush=True)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.text((x1 + 4, y1 + 4), f"{args.query} {i}", fill=(255, 255, 0), font=font)
    img.save(out_path)
    print(f"[demo] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
