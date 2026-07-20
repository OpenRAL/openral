"""Unit tests for the Cosmos 3 reasoner sidecar boot helper.

Covers the pure, GPU-free logic: the ``vllm serve`` argv builder, the Edge
diffusers-layout detector, and the flattened-reasoner-view materializer. These
are the pieces that got vLLM to actually load ``nvidia/Cosmos3-Edge`` on an
8 GB card (the diffusers subfolder layout is invisible to vLLM's weight loader
otherwise); see ``docs/reference/cosmos3-edge-reasoner.md``.

No mocks: the layout/view tests build a real on-disk fixture with real
safetensors shard files (empty is fine — only paths + the index matter) and
assert against the real filesystem result.

Run with:
    uv run pytest tests/unit/test_cosmos3_sidecar.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SIDECAR = Path(__file__).resolve().parents[2] / "tools" / "cosmos3_reasoner_sidecar.py"
_spec = importlib.util.spec_from_file_location("cosmos3_reasoner_sidecar", _SIDECAR)
assert _spec is not None and _spec.loader is not None
sidecar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sidecar)


# ── build_serve_argv ───────────────────────────────────────────────────────────


def test_build_serve_argv_core_flags() -> None:
    argv = sidecar.build_serve_argv(
        vllm_bin=Path("/venv/bin/vllm"),
        model="/local/view",
        host="127.0.0.1",
        port=8901,
        tool_call_parser="hermes",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    assert argv[:3] == ["/venv/bin/vllm", "serve", "/local/view"]
    # Tool-calling is on (the reasoner sends tool_choice="required").
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    # enforce_eager=True → the flag is present (saves VRAM + startup on 8 GB).
    assert "--enforce-eager" in argv


def test_build_serve_argv_no_enforce_eager_omits_flag() -> None:
    argv = sidecar.build_serve_argv(
        vllm_bin=Path("/venv/bin/vllm"),
        model="nvidia/Cosmos3-Nano",
        host="0.0.0.0",
        port=8000,
        tool_call_parser="hermes",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    assert "--enforce-eager" not in argv
    # No local view → no served-model-name override.
    assert "--served-model-name" not in argv


def test_build_serve_argv_served_model_name() -> None:
    argv = sidecar.build_serve_argv(
        vllm_bin=Path("/venv/bin/vllm"),
        model="/cache/Cosmos3-Edge-reasoner",
        host="127.0.0.1",
        port=8901,
        tool_call_parser="hermes",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        served_model_name="nvidia/Cosmos3-Edge",
    )
    assert argv[argv.index("--served-model-name") + 1] == "nvidia/Cosmos3-Edge"


# ── is_diffusers_reasoner_layout ───────────────────────────────────────────────


def _write_index(dir_: Path, weight_map: dict[str, str]) -> None:
    (dir_ / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )


def test_edge_layout_detected(tmp_path: Path) -> None:
    """Subfolder weight_map paths → Edge diffusers layout (needs a view)."""
    _write_index(
        tmp_path,
        {
            "embed_tokens.weight": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "lm_head.weight": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        },
    )
    assert sidecar.is_diffusers_reasoner_layout(tmp_path) is True


def test_standard_layout_not_flagged(tmp_path: Path) -> None:
    """Bare top-level shard names (Nano/Super) → served as-is, no view."""
    _write_index(
        tmp_path,
        {"embed_tokens.weight": "model-00001-of-00002.safetensors"},
    )
    assert sidecar.is_diffusers_reasoner_layout(tmp_path) is False


def test_no_index_not_flagged(tmp_path: Path) -> None:
    assert sidecar.is_diffusers_reasoner_layout(tmp_path) is False


# ── materialize_reasoner_view ──────────────────────────────────────────────────


def test_materialize_reasoner_view(tmp_path: Path) -> None:
    """The view flattens subfolder shards to bare symlinks + a rewritten index."""
    snap = tmp_path / "snapshot"
    (snap / "transformer").mkdir(parents=True)
    (snap / "vision_encoder").mkdir(parents=True)
    # Real (empty) shard files at the subfolder paths the index references.
    (snap / "transformer/diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"")
    (snap / "transformer/diffusion_pytorch_model-00002-of-00002.safetensors").write_bytes(b"")
    (snap / "vision_encoder/model.safetensors").write_bytes(b"")
    _write_index(
        snap,
        {
            "embed_tokens.weight": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "layers.0.mlp.up_proj.weight": (
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
            ),
            "visual.patch_embed.proj.weight": "vision_encoder/model.safetensors",
        },
    )
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer_config.json").write_text("{}")

    view = sidecar.materialize_reasoner_view(snap, tmp_path / "view")

    # Index weight_map is rewritten to bare basenames.
    idx = json.loads((view / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"].values()) == {
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
        "model.safetensors",
    }
    # Each bare shard name exists as a symlink resolving to the real file.
    for base in set(idx["weight_map"].values()):
        link = view / base
        assert link.is_symlink()
        assert link.resolve().is_file()
    # Config/tokenizer carried over.
    assert (view / "config.json").exists()
    assert (view / "tokenizer_config.json").exists()


def test_materialize_reasoner_view_idempotent(tmp_path: Path) -> None:
    """Re-running refreshes links without error (sentinel-free re-provision)."""
    snap = tmp_path / "snapshot"
    (snap / "transformer").mkdir(parents=True)
    (snap / "transformer/diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"")
    _write_index(
        snap,
        {"embed_tokens.weight": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"},
    )
    dest = tmp_path / "view"
    sidecar.materialize_reasoner_view(snap, dest)
    # Second call must not raise on the already-present symlink.
    view = sidecar.materialize_reasoner_view(snap, dest)
    link = view / "diffusion_pytorch_model-00001-of-00002.safetensors"
    assert link.is_symlink()
