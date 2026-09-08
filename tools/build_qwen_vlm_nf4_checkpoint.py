"""Build a pre-quantized NF4 checkpoint of Qwen3.5-4B for the scene-VLM rSkill.

Pre-quantizing avoids the on-GPU quantize spike: loading raw bf16 then
letting bitsandbytes quantize spikes VRAM to ~7.4 GB before shrinking to the
~3.3 GB NF4 resident, which OOMs an 8 GB card unless the loader is forced
serial. Saving the already-4bit weights once skips that spike entirely.

Not ``tools/quantize_rskill.py``: that tool serializes a raw ``Params4bit``
state dict for the in-process lerobot/pi0.5 runtime
(``install_prequantized_linears``). This VLM runs in an isolated sidecar venv
(``tools/_qwen_vlm_server.py``) via plain ``transformers.from_pretrained``,
which needs the transformers-native layout instead: ``save_pretrained`` with
an embedded ``quantization_config`` in ``config.json`` that
``from_pretrained`` auto-detects. Same quantizer (bitsandbytes nf4),
different serialization.

Recipe behind the published ``OpenRAL/rskill-qwen35_4b-any-general-nf4``
weights. Run INSIDE the sidecar venv (same transformers / bitsandbytes /
qwen-vl-utils stack as ``tools/_qwen_vlm_server.py``)::

    OPENRAL_QWEN_VLM_SIDECAR_VENV/bin/python tools/build_qwen_vlm_nf4_checkpoint.py \
        --source Qwen/Qwen3.5-4B \
        --out ~/.cache/openral/qwen35-4b-nf4-ckpt

Writes NF4 ``model.safetensors`` + config + processor files to ``--out``,
then verifies the checkpoint reloads as 4-bit and answers a smoke query.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

# Same allocator + serial-loader workaround as the sidecar server (the build
# also loads raw bf16 once); must be set before torch initializes CUDA. torch
# renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF in 2.9 (warns on the
# old spelling), so resolve the name from metadata instead of hardcoding it.
try:
    _torch_mm = tuple(int(p) for p in importlib.metadata.version("torch").split(".")[:2])
    _alloc_var = "PYTORCH_ALLOC_CONF" if _torch_mm >= (2, 9) else "PYTORCH_CUDA_ALLOC_CONF"
except (importlib.metadata.PackageNotFoundError, ValueError):
    _alloc_var = "PYTORCH_CUDA_ALLOC_CONF"
os.environ.setdefault(_alloc_var, "expandable_segments:True")


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
