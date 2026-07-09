"""Whole-model ONNX export parity for the ACT so101-pen policy.

Loads the **real** ``gabrycina/so101-passing-pen-policy`` checkpoint (skips when
the snapshot is not in the local HF cache and no network), exports it via
:func:`tools.export_act_onnx.export`, and asserts ONNXRuntime output parity
against the untouched torch ``ACTPolicy.predict_action_chunk`` on the same
normalized inputs.

ACT is a plain CNN+transformer, so unlike SmolVLA (see
``test_smolvla_onnx_split.py``) the whole model exports to one graph. Everything
runs on CPU in float32 — no GPU. Parity is exact to ~1e-6 because normalization
is external (the MEAN_STD processor sidecars run in Python), so the graph is a
pure function of the normalized observation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("lerobot")
pytest.importorskip("cv2")

import onnxruntime as ort  # noqa: E402

_CHECKPOINT = "gabrycina/so101-passing-pen-policy"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot_cached(repo_id: str) -> bool:
    """True when a local snapshot of ``repo_id`` exists in the HF cache."""
    d = _HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return d.is_dir() and any(d.iterdir())


def _load_export_tool() -> Any:
    """Import ``tools/export_act_onnx.py`` (not an installed package)."""
    path = _REPO_ROOT / "tools" / "export_act_onnx.py"
    spec = importlib.util.spec_from_file_location("export_act_onnx", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_act_onnx"] = mod
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not _snapshot_cached(_CHECKPOINT),
    reason=f"{_CHECKPOINT} not in the local HF cache; skip offline",
)


def test_act_whole_model_onnx_matches_torch(tmp_path: Path) -> None:
    """ONNX graph == torch predict_action_chunk on the same normalized inputs."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(_CHECKPOINT).eval().to("cpu")
    keys = list(policy.config.image_features)  # ordered: [wrist, front]
    _, h, w = next(iter(policy.config.image_features.values())).shape
    state_dim = policy.config.robot_state_feature.shape[0]

    onnx_path = tmp_path / "act_model.onnx"
    _load_export_tool().export(onnx_path, _CHECKPOINT, device="cpu")

    torch.manual_seed(0)
    imgs = {k: torch.randn(1, 3, h, w) for k in keys}
    state = torch.randn(1, state_dim)

    with torch.no_grad():
        ref = policy.predict_action_chunk({"observation.state": state, **imgs}).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {"state": state.numpy()}
    for k in keys:
        feed[k.replace("observation.images.", "img_")] = imgs[k].numpy()
    out = sess.run(["action_chunk"], feed)[0]

    assert out.shape == ref.shape == (1, policy.config.chunk_size, state_dim)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)
