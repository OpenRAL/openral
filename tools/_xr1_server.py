#!/usr/bin/env python3
"""XR-1 inference server for the isolated sidecar venv."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_NDARRAY_SENTINEL = "__ndarray__"
_STATE_DIM = 60
_ACTION_DIMS = {"robocasa_mg": 7, "robocasa365": 12, "vlabench_choice": 7}


def _encode_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_SENTINEL: True, "npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    if _NDARRAY_SENTINEL in obj and "npy" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _split_model_ref(model: str) -> tuple[str, str | None]:
    path = Path(model).expanduser()
    if path.exists():
        return str(path), None
    ref = model.removeprefix("hf://")
    repo_id, separator, revision = ref.partition("@")
    return repo_id, revision if separator else None


def _messages(profile: str, images: list[Any], instruction: str) -> list[dict[str, Any]]:
    if profile == "robocasa_mg":
        content = [
            {
                "type": "text",
                "text": "The following observations are captured from multiple views.\n# Base View\n",
            },
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": images[2]},
            {
                "type": "text",
                "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
            },
        ]
    elif profile == "robocasa365":
        content = [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": images[0]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": images[1]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": images[2]},
            {
                "type": "text",
                "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
            },
        ]
    else:
        content = [
            {
                "type": "text",
                "text": "The following observations are captured from multiple views.\n# Ego View\n",
            },
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "\n# Base View\n"},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": images[2]},
            {
                "type": "text",
                "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
            },
        ]
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def _pad_state(state: NDArray[np.float32]) -> NDArray[np.float32]:
    source = np.asarray(state, dtype=np.float32)
    if source.ndim == 1:
        source = source[None, :]
    out = np.zeros((1, source.shape[0], _STATE_DIM), dtype=np.float32)
    out[0, :, : source.shape[1]] = source
    return out


def _center_crop(image: Any, ratio: float) -> Any:
    """Center-crop to ``ratio`` then resize back to the original resolution."""
    if ratio >= 1.0:
        return image
    width, height = image.size
    crop_width = max(1, round(width * ratio))
    crop_height = max(1, round(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    from PIL import Image

    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )


def _is_metadata_asset(path: Path) -> bool:
    """True for custom-code/processor metadata; never for weight tensors."""
    return path.is_file() and path.suffix != ".safetensors"


class _XR1Policy:
    def __init__(self, model: str, profile: str, quantization: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

        repo_id, revision = _split_model_ref(model)
        self._torch = torch
        self._source_model = model
        self._profile = profile
        self._quantization = quantization
        # reason: transformers ships AutoProcessor / BitsAndBytesConfig untyped,
        # same as the safetensors `safe_open` sites in quantize_lingbot_vla2.py.
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call, unused-ignore]
            repo_id,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if quantization == "nf4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call, unused-ignore]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": 0}
        elif quantization == "prequantized_nf4":
            model_kwargs["device_map"] = {"": 0}
        self._model = AutoModel.from_pretrained(repo_id, **model_kwargs)
        if quantization == "none":
            self._model = self._model.cuda()
        self._model.eval()

    def reset(self) -> None:
        return

    def export_pretrained(self, output_dir: Path) -> None:
        """Persist the loaded NF4 model and all custom-code processor assets."""
        if self._quantization != "nf4":
            raise ValueError("XR-1 export requires --quantization nf4.")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"XR-1 export directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        self._processor.save_pretrained(output_dir)

        repo_id, revision = _split_model_ref(self._source_model)
        if revision is not None or not Path(repo_id).exists():
            from huggingface_hub import snapshot_download

            metadata_dir = Path(
                snapshot_download(
                    repo_id,
                    revision=revision,
                    allow_patterns=[
                        "*.py",
                        "*.json",
                        "*.jinja",
                        "tokenizer*",
                        "vocab*",
                        "merges.txt",
                        "added_tokens.json",
                        "special_tokens_map.json",
                    ],
                )
            )
            for source in metadata_dir.iterdir():
                # The snapshot may already contain cached BF16 weight shards
                # outside this call's allow_patterns. Never copy any
                # safetensors file into the packed export.
                if _is_metadata_asset(source) and not (output_dir / source.name).exists():
                    shutil.copy2(source, output_dir / source.name)

        metadata = {
            "source_repo": repo_id,
            "source_revision": revision,
            "quantization": {
                "scheme": "nf4",
                "backend": "bitsandbytes",
                "compute_dtype": "bfloat16",
                "double_quant": True,
                "prequantized": True,
            },
        }
        (output_dir / "quantization_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_action(self, payload: dict[str, Any]) -> NDArray[np.float32]:
        from PIL import Image

        profile = str(payload["profile"])
        if profile != self._profile:
            raise ValueError(f"sidecar profile is {self._profile!r}, got {profile!r}")
        raw_images = payload["images"]
        image_values = list(raw_images.values())
        crop_ratio = float(payload.get("crop_ratio", 1.0))
        if not 0.0 < crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
        images: list[Any]
        if profile == "robocasa365":
            images = [
                [
                    _center_crop(
                        Image.fromarray(np.asarray(frame, dtype=np.uint8)),
                        crop_ratio,
                    )
                    for frame in history
                ]
                for history in image_values
            ]
        else:
            images = [
                _center_crop(
                    Image.fromarray(np.asarray(image, dtype=np.uint8)),
                    crop_ratio,
                )
                for image in image_values
            ]
        state = self._torch.from_numpy(_pad_state(np.asarray(payload["state"], dtype=np.float32)))
        processor_kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "state": state,
            "robot_type": profile,
        }
        if profile == "robocasa365":
            processor_kwargs["do_resize"] = False
        inputs = self._processor.apply_chat_template(
            _messages(profile, images, str(payload["task"])),
            **processor_kwargs,
        )
        model_inputs = {
            key: value.cuda() if hasattr(value, "cuda") else value
            for key, value in dict(inputs).items()
        }
        model_inputs["task_id"] = profile
        model_inputs["seed"] = int(payload.get("seed", 42))
        with self._torch.no_grad():
            outputs = self._model(**model_inputs)
        decoded = self._processor.decode_action(
            outputs.actions.detach().cpu(),
            robot_type=profile,
        )
        return np.asarray(
            decoded[0, :, : _ACTION_DIMS[profile]].detach().cpu().to(self._torch.float32).numpy(),
            dtype=np.float32,
        )


def _serve(
    policy: _XR1Policy,
    *,
    host: str,
    port: int,
    model: str,
    profile: str,
    quantization: str,
) -> int:
    import msgpack
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    running = True
    while running:
        request = msgpack.unpackb(
            socket.recv(),
            object_hook=_decode_ndarray,
            raw=False,
        )
        endpoint = request.get("endpoint")
        data = request.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {
                    "ok": True,
                    "model": model,
                    "profile": profile,
                    "quantization": quantization,
                }
            elif endpoint == "reset":
                policy.reset()
                reply = {"ok": True}
            elif endpoint == "get_action":
                reply = {"action": policy.get_action(data)}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        socket.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))
    socket.close(linger=0)
    context.term()
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("robocasa_mg", "robocasa365", "vlabench_choice"),
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "nf4", "prequantized_nf4"),
        default="nf4",
    )
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    policy = _XR1Policy(args.model, args.profile, args.quantization)
    if args.export_dir is not None:
        policy.export_pretrained(args.export_dir)
        return 0
    return _serve(
        policy,
        host=args.host,
        port=args.port,
        model=args.model,
        profile=args.profile,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
