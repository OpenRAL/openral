"""Export a LeRobot ACT policy to a single whole-model ONNX graph.

Unlike SmolVLA (whose VLM + flow-matching graph needs a *split* export — see
``openral_rskill.smolvla_export``), ACT is a plain CNN + transformer: the core
network maps normalized observations straight to the normalized action chunk,
so it exports whole-model in one shot.

Export target: ``ACTPolicy.model`` (the ``ACT`` nn.Module), reached exactly as
``predict_action_chunk`` does — it builds ``batch[OBS_IMAGES]`` as an ordered
list over ``config.image_features`` and returns ``model(batch)[0]`` of shape
``(B, chunk_size, action_dim)``. Normalization is **external** (the checkpoint's
``policy_preprocessor.json`` / ``policy_postprocessor.json`` MEAN_STD sidecars,
applied in Python by the ACT adapter), so the ONNX graph takes already-normalized
images + state and emits the normalized action chunk. The parity test
(``tests/integration/test_act_onnx.py``) compares this graph against the torch
``predict_action_chunk`` on the same normalized inputs.

Inputs are ordered by ``config.image_features`` (for so101-passing-pen that is
``[wrist, front]``) plus the 6-D state. The gabrycina/so101-passing-pen-policy
checkpoint trains at 480x640.

Usage (uses the project venv, which already carries torch + lerobot + onnx):

    uv run python tools/export_act_onnx.py \
        --repo-id gabrycina/so101-passing-pen-policy \
        --out rskills/act-so101-pen/model.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch


class _ACTExportWrapper(torch.nn.Module):
    """Flat-tensor facade over ``ACTPolicy.model`` for ONNX tracing.

    Rebuilds the dict/list batch structure ``predict_action_chunk`` feeds the
    core network, from positional image + state tensors the exporter can trace.
    """

    def __init__(
        self, policy: Any, image_feature_keys: list[str], state_key: str
    ) -> None:  # reason: lerobot ACTPolicy is untyped
        super().__init__()
        from lerobot.utils.constants import OBS_IMAGES

        self.model = policy.model
        self._image_feature_keys = image_feature_keys
        self._state_key = state_key
        self._obs_images = OBS_IMAGES

    def forward(self, *images_and_state: torch.Tensor) -> torch.Tensor:
        *images, state = images_and_state
        batch: dict[str, Any] = {self._state_key: state}
        for key, img in zip(self._image_feature_keys, images, strict=True):
            batch[key] = img
        # predict_action_chunk: batch[OBS_IMAGES] = [batch[k] for k in image_features]
        batch[self._obs_images] = list(images)
        actions: torch.Tensor = self.model(batch)[0]  # (B, chunk_size, action_dim)
        return actions


class _ACTDeviceWrapper(torch.nn.Module):
    """Device-preprocess ACT facade: takes ``/255`` RGB images, bakes image normalize.

    The zero-copy NVMM path (``openral_rskill.act_nvmm``) hands the engine
    ``[0,1]`` RGB NCHW tensors straight off the nvrtc RGBA→NCHW/255 kernel — so
    the per-channel MEAN_STD image normalize the lerobot processor would apply on
    the host is folded into the graph here instead. State stays host-normalized
    (6-D, negligible); the graph consumes the already-normalized ``state``.
    """

    def __init__(
        self,
        policy: Any,  # reason: lerobot ACTPolicy is untyped
        image_feature_keys: list[str],
        state_key: str,
        stats: _ActStats,
    ) -> None:
        super().__init__()
        from lerobot.utils.constants import OBS_IMAGES

        self.model = policy.model
        self._image_feature_keys = image_feature_keys
        self._state_key = state_key
        self._obs_images = OBS_IMAGES
        self.register_buffer("_img_mean", stats.img_mean.view(1, 3, 1, 1))
        self.register_buffer("_img_std", stats.img_std.view(1, 3, 1, 1))
        self.register_buffer("_state_mean", stats.state_mean.view(1, -1))
        self.register_buffer("_state_std", stats.state_std.view(1, -1))
        self.register_buffer("_action_mean", stats.action_mean.view(1, 1, -1))
        self.register_buffer("_action_std", stats.action_std.view(1, 1, -1))

    def forward(self, *images_and_state: torch.Tensor) -> torch.Tensor:
        *images_rgb01, state_raw = images_and_state
        img_mean = cast(torch.Tensor, self._img_mean)
        img_std = cast(torch.Tensor, self._img_std)
        state_mean = cast(torch.Tensor, self._state_mean)
        state_std = cast(torch.Tensor, self._state_std)
        action_mean = cast(torch.Tensor, self._action_mean)
        action_std = cast(torch.Tensor, self._action_std)
        normed = [(img - img_mean) / img_std for img in images_rgb01]
        state = (state_raw - state_mean) / state_std
        batch: dict[str, Any] = {self._state_key: state}
        for key, img in zip(self._image_feature_keys, normed, strict=True):
            batch[key] = img
        batch[self._obs_images] = list(normed)
        actions_norm: torch.Tensor = self.model(batch)[0]
        return actions_norm * action_std + action_mean  # unnormalized


class _ActStats(NamedTuple):
    img_mean: torch.Tensor
    img_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor


def _norm_stats(repo_id: str) -> _ActStats:
    """Image / state / action MEAN_STD stats from the checkpoint's preprocessor normalizer.

    The ``device`` engine bakes all three (image + state normalize, action
    unnormalize) so its executor stays a pure device-pointer runner — feeding
    ``/255`` RGB + raw state and reading back raw action.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    s = load_file(
        hf_hub_download(repo_id, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    )
    img_key = next(
        k for k in s if k.startswith("observation.images.") and k.endswith(".mean")
    ).rsplit(".", 1)[0]
    return _ActStats(
        img_mean=s[f"{img_key}.mean"].reshape(3).float(),
        img_std=s[f"{img_key}.std"].reshape(3).float(),
        state_mean=s["observation.state.mean"].reshape(-1).float(),
        state_std=s["observation.state.std"].reshape(-1).float(),
        action_mean=s["action.mean"].reshape(-1).float(),
        action_std=s["action.std"].reshape(-1).float(),
    )


def export(out_path: Path, repo_id: str, *, device: str = "cpu", preprocess: str = "host") -> str:
    """Export ``repo_id``'s ACT policy to ``out_path`` as ONNX; return sha256.

    ``preprocess="host"`` (default): image inputs are already MEAN_STD-normalized
    (the lerobot processor runs on the host). ``preprocess="device"``: image
    inputs are ``[0,1]`` RGB and the image normalize is baked into the graph —
    for the zero-copy NVMM path where an nvrtc kernel produces ``/255`` frames on
    the GPU (:mod:`openral_rskill.act_nvmm`).
    """
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_STATE

    if preprocess not in ("host", "device"):
        raise ValueError(f"preprocess must be 'host' or 'device', got {preprocess!r}")

    policy: Any = ACTPolicy.from_pretrained(repo_id)
    policy.eval().to(device)

    cfg = policy.config
    image_feature_keys = list(cfg.image_features)
    _, h, w = next(iter(cfg.image_features.values())).shape
    state_dim = cfg.robot_state_feature.shape[0]

    if preprocess == "device":
        wrapped: Any = (
            _ACTDeviceWrapper(policy, image_feature_keys, OBS_STATE, _norm_stats(repo_id))
            .eval()
            .to(device)
        )
    else:
        wrapped = _ACTExportWrapper(policy, image_feature_keys, OBS_STATE).eval().to(device)
    dummy_images = tuple(torch.randn(1, 3, h, w, device=device) for _ in image_feature_keys)
    dummy_state = torch.randn(1, state_dim, device=device)
    example = (*dummy_images, dummy_state)

    image_input_names = [k.replace("observation.images.", "img_") for k in image_feature_keys]
    input_names = [*image_input_names, "state"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        example,
        str(out_path),
        input_names=input_names,
        output_names=["action_chunk"],
        opset_version=17,
        dynamo=False,
    )
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(  # reason: CLI export tool
        f"exported {out_path} sha256={sha}\n"
        f"  inputs: {input_names} (images 1x3x{h}x{w}, state 1x{state_dim})\n"
        f"  output: action_chunk 1x{cfg.chunk_size}x{cfg.action_feature.shape[0]}"
    )
    return sha


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="gabrycina/so101-passing-pen-policy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--preprocess",
        choices=("host", "device"),
        default="host",
        help="'device' bakes image normalize (for the zero-copy NVMM path).",
    )
    ns = ap.parse_args()
    export(ns.out, ns.repo_id, device=ns.device, preprocess=ns.preprocess)
