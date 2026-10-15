"""Static message/state packing checks for ``tools/_xr1_server.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_PATH = Path(__file__).parents[2] / "tools" / "_xr1_server.py"
_SPEC = importlib.util.spec_from_file_location("_xr1_server_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_robocasa365_message_uses_video_items() -> None:
    messages = _MODULE._messages("robocasa365", [["left"], ["right"], ["wrist"]], "close lid")
    content = messages[0]["content"]
    assert [item["type"] for item in content].count("video") == 3
    assert content[-1]["text"].endswith("close lid /no_cot")


def test_state_padding_preserves_history_and_uses_60_dims() -> None:
    state = np.ones((4, 14), dtype=np.float32)
    padded = _MODULE._pad_state(state)
    assert padded.shape == (1, 4, 60)
    np.testing.assert_array_equal(padded[0, :, :14], state)
    assert np.count_nonzero(padded[0, :, 14:]) == 0


def test_center_crop_preserves_size_and_removes_border() -> None:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[2:8, 2:8] = 255
    cropped = np.asarray(_MODULE._center_crop(Image.fromarray(pixels), 0.6))
    assert cropped.shape == pixels.shape
    assert int(cropped[0, 0, 0]) > 0


def test_export_metadata_filter_never_copies_weight_shards(tmp_path: Path) -> None:
    metadata = tmp_path / "processing_mibot.py"
    weights = tmp_path / "model-00001-of-00003.safetensors"
    metadata.write_text("# custom code\n")
    weights.write_bytes(b"weights")
    assert _MODULE._is_metadata_asset(metadata) is True
    assert _MODULE._is_metadata_asset(weights) is False


def test_non_mibot_processor_fails_at_load() -> None:
    with pytest.raises(TypeError, match="not the checkpoint's MiBotProcessor"):
        _MODULE._require_mibot_processor(object(), "hf://OpenRAL/x@abc")


def test_pinned_hf_ref_resolves_to_its_snapshot_offline_without_refs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pack fetched by sha has no refs/main: transformers' AutoProcessor probes
    # ignore the revision and miss it, so the server must hand over the snapshot dir.
    from huggingface_hub import constants

    sha = "5e6c1eaac1e40f68cf592bd348ba07cbfc926ce6"
    repo_dir = tmp_path / "models--OpenRAL--rskill-xr1-panda_mobile-robocasa-nf4"
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "processor_config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    ref = f"hf://OpenRAL/rskill-xr1-panda_mobile-robocasa-nf4@{sha}"
    assert Path(_MODULE._local_snapshot(ref)) == snapshot
    assert _MODULE._local_snapshot(str(snapshot)) == str(snapshot)
