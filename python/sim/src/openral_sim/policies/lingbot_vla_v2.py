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
the processor directory is the checkpoint snapshot itself, not the manifest's
two-file ``processors`` fetch.

Imports torch / lerobot lazily so installing ``openral-sim`` never pulls them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
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
    run_inference,
    to_numpy_action,
    validate_processor_steps,
)

from openral_sim._quantization import require_supported_dtype, resolve_quant_plan
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation

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
        return self._finished_action(
            run_inference(self._policy, self._prepared_batch(observation, instruction))
        )

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
            if self._last_input_frame is None or slot == self._camera_keys[0]:
                self._last_input_frame = np.asarray(frame, dtype=np.uint8)
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
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "lingbot_vla_v2 needs torch + lerobot>=0.7; install with: "
            "just sync --all-packages --group vla"
        ) from exc

    manifest = load_manifest_for_spec(spec)
    # The model loads in one dtype (lerobot casts every module to config.dtype);
    # bf16 halves the 25.5 GB fp32 checkpoint. NF4 is the sidecar family's path.
    plan = resolve_quant_plan(spec, manifest)
    require_supported_dtype(plan, frozenset({"bf16", "fp32"}), "lingbot_vla_v2")
    dtype = torch.float32 if plan.dtype == "fp32" else torch.bfloat16

    repo_id, revision = resolve_rskill_repo_revision(
        spec.weights_uri, adapter_name="lingbot_vla_v2"
    )
    # One snapshot holds weights + the processor JSONs + their artefact dirs.
    ckpt = local_snapshot_dir(repo_id, revision=revision)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        validate_processor_steps(str(Path(ckpt) / name))

    config = PreTrainedConfig.from_pretrained(ckpt)
    config.dtype = dtype
    # Build + load on the CPU: lerobot loads the safetensors straight onto
    # `config.device`, so a CUDA load first lands the full fp32 checkpoint
    # (25.5 GB) on the GPU before the copy into the bf16 parameters. Loading on
    # the host keeps the GPU at the bf16 footprint (~12.8 GB).
    config.device = "cpu"
    policy = LingbotVLAV2Policy.from_pretrained(ckpt, config=config)
    assert_all_parameters_finite(policy, repo_id=repo_id)
    # The policy's __init__ casts the model to config.dtype, but from_pretrained
    # then loads the checkpoint's fp32 tensors dtype-preserving, leaving a mixed
    # model; `_extract_model_inputs` casts the images to the FIRST parameter's
    # dtype, so the bf16 vision blocks get fp32 activations ("expected scalar
    # type Float but found BFloat16"). Re-apply the one-dtype cast upstream intends.
    policy.model.to(dtype)
    policy = policy.to(device).eval()
    if dtype != torch.float32:
        # lerobot's vendored Qwen3-VL vision tower interpolates its position
        # embeddings with fp32 weights (`pos_embed(...) * interpolation_weights`),
        # promoting the patch stream to fp32 ahead of the bf16 blocks — the
        # checkpoint config and lerobot's own runs are fp32. Autocast resolves the
        # mixed-dtype ops (matmuls in bf16, norms in fp32) without patching lerobot.
        _predict = policy.predict_action_chunk
        autocast_device = torch.device(device).type

        def _predict_autocast(*args: Any, **kwargs: Any) -> Any:
            with torch.autocast(device_type=autocast_device, dtype=dtype):
                return _predict(*args, **kwargs)

        policy.predict_action_chunk = _predict_autocast
    policy.config.device = device
    apply_chunk_replay(policy, dict(spec.extra or {}), manifest=manifest)
    policy.reset()

    overrides = {"device_processor": {"device": device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(
        manifest,
        dict(spec.extra or {}),
        scene_cameras=scene_cameras,
        default=("camera1", "camera2", "camera3"),
    )
    image_keys = checkpoint_image_keys(
        resolve_image_preprocessing(manifest, dict(spec.extra or {})), cam_keys
    )

    adapter = _LingbotVLAV2Adapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _camera_keys=cam_keys,
        _image_keys=image_keys,
    )
    adapter._chunk_executor = build_chunk_executor(
        spec.extra,
        policy=policy,
        adapter_name="lingbot_vla_v2",
        postprocess_action=adapter._finished_action,
    )
    return adapter
