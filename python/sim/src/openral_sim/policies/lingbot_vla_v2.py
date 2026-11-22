"""LingBot-VLA 2.0 in-process adapter — wraps lerobot's ``LingbotVLAV2Policy``.

lerobot 0.7 vendors LingBot-VLA 2.0 (Qwen3-VL-4B backbone + sparse-MoE
flow-matching action expert) without upstream ``lingbotvla``'s
``torch==2.8`` / ``transformers==4.57.3`` / Triton-MoE pins, so it loads in the
workspace interpreter. This family (``lingbot_vla_v2``, lerobot's own type name)
runs LeRobot-format checkpoints — ``config.json`` ``type: lingbot_vla_v2`` plus
the ``policy_{pre,post}processor.json`` pipeline — in-process. Upstream-format
checkpoints (``robbyant/lingbot-vla-v2-6b``) stay on the ``lingbot_vla2``
sidecar family (``openral_sim.policies.lingbot_vla2``).

IO contract (RoboTwin / AgileX, same as the sidecar): three RGB slots renamed by
the manifest ``image_preprocessing`` to ``observation.images.cam_high`` /
``cam_left_wrist`` / ``cam_right_wrist``, a 14-D ``observation.state``
(``[arm_l6 grip_l1 arm_r6 grip_r1]``), and the task string. The checkpoint's own
pipeline does the rest: quantile normalisation, the 55-D canonical slot mapping,
Qwen3-VL image patching, the chat template and tokenisation on the way in; the
inverse slot mapping and unnormalisation back to 14-D on the way out.

The processor steps carry artefact directories (image processor, tokenizers), so
the pipelines load from the checkpoint snapshot itself, not the manifest's two-file
``processors`` fetch, through lerobot's own
``make_lingbot_vla_v2_pre_post_processors_from_pretrained`` (it links the
relative/absolute action steps and re-pins slot spans and tokenizer length from the
config). Relative-action checkpoints and RTC are refused: neither is verified here.

Imports torch / lerobot lazily so installing ``openral-sim`` never pulls them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_rskill._vla_core import (
    apply_chunk_replay,
    assert_all_parameters_finite,
    build_chunk_executor,
    checkpoint_image_keys,
    image_to_chw_tensor,
    local_snapshot_dir,
    release_torch_modules,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_revision,
    rtc_enabled_in_extra,
    run_inference,
    to_numpy_action,
    validate_processor_steps,
)

from openral_sim._quantization import require_supported_dtype, resolve_quant_plan
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.policies._video_capture import to_input_frame
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

_STATE_DIM = 14


@dataclass
class _LingbotVLAV2Adapter:
    """Per-step adapter over lerobot's ``LingbotVLAV2Policy`` and its processors."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _camera_keys: tuple[str, ...]
    _image_keys: tuple[str, ...]
    # bf16 inference runs under autocast: lerobot's vendored Qwen3-VL vision tower
    # interpolates its position embeddings with fp32 weights, promoting the patch
    # stream to fp32 ahead of the bf16 blocks. None = fp32, no autocast.
    _autocast_dtype: Any = None
    _last_input_frame: NDArray[np.uint8] | None = None
    _chunk_executor: Any = field(default=None)

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if self._chunk_executor is not None:
            self._chunk_executor.reset()
        else:
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if self._chunk_executor is not None:
            action: NDArray[np.float32] = self._chunk_executor.select_action(
                lambda: self._prepared_batch(observation, instruction)
            )
            return action
        batch = self._prepared_batch(observation, instruction)
        with self._autocast():
            return self._finished_action(run_inference(self._policy, batch))

    def _autocast(self) -> Any:
        """Mixed-precision context for a bf16 model (see ``_autocast_dtype``)."""
        device_type = self.device.split(":", 1)[0]
        if self._autocast_dtype is not None and device_type in {"cuda", "cpu"}:
            return self._torch.amp.autocast(device_type=device_type, dtype=self._autocast_dtype)
        import contextlib

        return contextlib.nullcontext()

    def _chunk_forward(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Chunk producer for the executor — predict under this adapter's autocast."""
        with self._torch.no_grad(), self._autocast():
            return self._policy.predict_action_chunk(batch, **kwargs)

    def _prepared_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        images = observation.get("images", {})
        batch: dict[str, Any] = {"task": instruction or str(observation.get("task", ""))}
        for slot, key in zip(self._camera_keys, self._image_keys, strict=True):
            frame = images.get(slot)
            if frame is None:
                raise ROSCapabilityMismatch(
                    f"lingbot_vla_v2 needs camera slot {slot!r}; scene provides {sorted(images)}"
                )
            batch[key] = image_to_chw_tensor(frame, torch=torch, device=self.device).unsqueeze(0)
            if slot == self._camera_keys[0]:  # the head camera is the preview
                self._last_input_frame = to_input_frame(frame)
        state = observation.get("state")
        if state is None:
            raise ROSConfigError("lingbot_vla_v2 requires a 14-D proprio `state` observation.")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != _STATE_DIM:
            raise ROSConfigError(
                f"lingbot_vla_v2 expects a {_STATE_DIM}-D state "
                f"(arm_l6+grip1+arm_r6+grip1), got {state_arr.shape[0]}-D."
            )
        batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
        return cast("dict[str, Any]", self._preprocessor(batch))

    def _finished_action(self, action_tensor: Any) -> NDArray[np.float32]:
        """Canonical 55-D normalised step -> the checkpoint's 14-D action."""
        return to_numpy_action(self._postprocessor(action_tensor))

    def close(self) -> None:
        if self._chunk_executor is not None:
            self._chunk_executor.stop()
            self._chunk_executor = None
        release_torch_modules(
            self,
            "_policy",
            "_preprocessor",
            "_postprocessor",
            device=self.device,
            torch=self._torch,
        )


@POLICIES.register(
    "lingbot_vla_v2",
    install_groups=("vla",),
    required_imports=("transformers", "lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2"),
)
def _build_lingbot_vla_v2(env_cfg: Any) -> _LingbotVLAV2Adapter:
    """Load a LeRobot-format LingBot-VLA 2.0 checkpoint and its processor pipeline."""
    spec = env_cfg.vla
    device = resolve_device(spec)
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
        from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
            make_lingbot_vla_v2_pre_post_processors_from_pretrained,
        )
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "lingbot_vla_v2 needs torch + lerobot>=0.7; install with: "
            "just sync --all-packages --group vla"
        ) from exc

    extra = dict(spec.extra or {})
    if rtc_enabled_in_extra(extra, adapter_name="lingbot_vla_v2"):
        raise ROSConfigError(
            "lingbot_vla_v2: real-time chunking is not verified for this family; "
            "drop policy_extras.rtc."
        )
    manifest = load_manifest_for_spec(spec)
    # lerobot casts the whole model to one dtype; bf16 halves the 25.5 GB fp32
    # checkpoint. NF4 is the sidecar family's path; "none" is not a dtype here.
    plan = resolve_quant_plan(spec, manifest)
    require_supported_dtype(plan, frozenset({"bf16", "fp32"}), "lingbot_vla_v2")
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if plan.dtype not in dtypes:
        raise ROSConfigError(
            f"lingbot_vla_v2 needs an explicit bf16 or fp32 dtype, got {plan.dtype!r} "
            "(declare quantization.dtype in the manifest)."
        )
    dtype = dtypes[plan.dtype]
    _log.info("lingbot_vla_v2.dtype", dtype=plan.dtype, source=plan.source)

    repo_id, revision = resolve_rskill_repo_revision(
        spec.weights_uri, adapter_name="lingbot_vla_v2"
    )
    # One snapshot holds weights + the processor JSONs + their artefact dirs.
    ckpt = local_snapshot_dir(repo_id, revision=revision)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        validate_processor_steps(str(Path(ckpt) / name))

    config = PreTrainedConfig.from_pretrained(ckpt)
    if getattr(config, "use_relative_actions", False):
        raise ROSConfigError(
            f"lingbot_vla_v2: {repo_id} predicts relative actions; their anchoring is not "
            "verified for this adapter."
        )
    config.dtype = dtype
    # Build + load on the CPU: lerobot loads the safetensors straight onto
    # `config.device`, so a CUDA load first lands the full fp32 checkpoint
    # (25.5 GB) on the GPU before the copy into the bf16 parameters.
    config.device = "cpu"
    policy = LingbotVLAV2Policy.from_pretrained(ckpt, config=config)
    # __init__ casts the model to config.dtype, but from_pretrained then loads the
    # checkpoint's fp32 tensors dtype-preserving, leaving a mixed model whose
    # `_extract_model_inputs` casts the inputs to the FIRST parameter's dtype.
    # Re-apply the one-dtype cast upstream intends, then move.
    policy.model.to(dtype)
    policy = policy.to(device).eval()
    policy.config.device = device
    assert_all_parameters_finite(policy, repo_id=repo_id)
    apply_chunk_replay(policy, extra, manifest=manifest)
    policy.reset()

    preprocessor, postprocessor = make_lingbot_vla_v2_pre_post_processors_from_pretrained(
        policy.config,
        ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(
        manifest,
        extra,
        scene_cameras=scene_cameras,
        default=("camera1", "camera2", "camera3"),
    )
    image_keys = checkpoint_image_keys(resolve_image_preprocessing(manifest, extra), cam_keys)

    adapter = _LingbotVLAV2Adapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _camera_keys=cam_keys,
        _image_keys=image_keys,
        _autocast_dtype=None if dtype == torch.float32 else dtype,
    )
    adapter._chunk_executor = build_chunk_executor(
        extra,
        policy=policy,
        chunk_fn=adapter._chunk_forward,
        adapter_name="lingbot_vla_v2",
        postprocess_action=adapter._finished_action,
    )
    return adapter
