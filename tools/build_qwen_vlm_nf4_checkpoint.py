"""Build a pre-quantized NF4 checkpoint of Qwen3.5-4B for the scene-VLM rSkill.

Why pre-quantize (vs quantizing at load): loading the raw bf16 model and letting
bitsandbytes quantize on-GPU spikes VRAM to ~7.4 GB before it shrinks to the
~3.3 GB NF4 resident — which OOMs an 8 GB card unless the transformers loader is
forced serial (ADR-0047). Saving the *already-4bit* weights once and loading
that checkpoint skips the bf16 spike entirely: the 4-bit tensors load directly,
so deployment on an 8 GB GPU "just works" with no loader workaround.

Why not ``tools/quantize_rskill.py``? That tool also packs NF4 with
bitsandbytes, but it serialises a raw ``Params4bit`` state dict that is loaded
back by ``openral_sim.policies.install_prequantized_linears`` — the *in-process*
lerobot/π0.5 policy runtime. This scene VLM instead runs in an **isolated
sidecar venv** (``tools/_qwen_vlm_server.py``) that cannot import
``openral_sim`` and loads the model with plain ``transformers.from_pretrained``.
That path needs the **transformers-native** layout this script writes —
``save_pretrained`` with an embedded ``quantization_config`` in ``config.json``,
which ``from_pretrained`` auto-detects. Same quantizer (bitsandbytes nf4),
different serialization for a different loader; see ADR-0047.

This is the reproducible recipe behind the published
``OpenRAL/rskill-qwen35-4b-nf4`` weights. Run it INSIDE the sidecar venv (it
needs the same transformers / bitsandbytes / qwen-vl-utils stack as
``tools/_qwen_vlm_server.py``)::

    OPENRAL_QWEN_VLM_SIDECAR_VENV/bin/python tools/build_qwen_vlm_nf4_checkpoint.py \
        --source Qwen/Qwen3.5-4B \
        --out ~/.cache/openral/qwen35-4b-nf4-ckpt

It writes the NF4 ``model.safetensors`` + config (with the embedded
``quantization_config``) + processor files to ``--out``, then verifies the
checkpoint reloads directly as 4-bit and answers a smoke query.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Same allocator + serial-loader settings as the sidecar server: the *build*
# still loads the raw bf16 model once to quantize it, so it needs the 8 GB
# headroom workaround. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="Qwen/Qwen3.5-4B", help="Upstream HF model id to quantize.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache" / "openral" / "qwen35-4b-nf4-ckpt",
        help="Output directory for the NF4 checkpoint.",
    )
    args = ap.parse_args()
    out = args.out.expanduser()

    import torch
    import transformers.core_model_loading as cml
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    cml.GLOBAL_WORKERS = 1  # serial materialization so the bf16 load fits 8 GB

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[build] loading + quantizing {args.source} (NF4)...", flush=True)
    processor = AutoProcessor.from_pretrained(args.source)
    model = AutoModelForImageTextToText.from_pretrained(
        args.source,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    print(f"[build] resident VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[build] saving NF4 checkpoint to {out}...", flush=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))

    # Free the build model before the verify reload so both don't co-reside.
    del model
    torch.cuda.empty_cache()

    print("[build] verifying the saved checkpoint reloads directly as 4-bit...", flush=True)
    reloaded = AutoModelForImageTextToText.from_pretrained(str(out), device_map={"": 0}).eval()
    print(
        f"[build] reloaded resident VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True
    )

    # Smoke query on a synthetic image so the recipe is self-checking.
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    img = Image.new("RGB", (448, 448), (40, 40, 40))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What color is this image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        gen = reloaded.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen, strict=True)]
    ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    if "</think>" in ans:
        ans = ans.rsplit("</think>", 1)[-1]
    print(f"[build] smoke answer: {ans.strip()[:200]!r}", flush=True)
    print(f"[build] DONE — NF4 checkpoint at {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
