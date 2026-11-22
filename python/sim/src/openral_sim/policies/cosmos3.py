"""NVIDIA Cosmos 3 action-policy adapter — wraps lerobot's ``Cosmos3Policy`` (LIBERO).

lerobot 0.7 ships Cosmos 3 (``lerobot.policies.cosmos3``): the Mixture-of-Transformers
world action model that denoises future video latents and a robot action chunk together
from the camera views and a language instruction. This family (``cosmos3``, covering
lerobot's ``cosmos3_edge`` and ``cosmos3_nano`` policy types) runs it in-process as an
action policy: the predicted video is not decoded or kept (``save_predicted_video``
stays off). It is not the ``cosmos3-edge`` reasoner (``openral_reasoner.cosmos3``), which
serves the text tower of the base model for S2 planning.

IO contract (the rot6d LIBERO recipe, ``lerobot/LIBERO-Cosmos3``): two RGB slots renamed
by the manifest ``image_preprocessing`` to ``observation.images.image`` (third person)
and ``image2`` (wrist), rotated 180 degrees as the dataset holds them (``flip_180``), the
8-D canonical LIBERO state (the checkpoint declares it; ``use_state`` is off, so the model
ignores it), and the task string. The checkpoint's own pipeline builds the two-view video
canvas, the prompt and the tokens; its postprocessor unnormalises the 10-D
``[dpos(3), rot6d(6), gripper]`` action, which lerobot's
``Cosmos3LiberoActionProcessorStep`` turns into the 7-D LIBERO OSC delta command
(axis-angle, gripper -1 open / +1 closed) — the step ``lerobot-eval`` composes for a
Cosmos 3 policy on LIBERO.

Imports torch / lerobot lazily so installing ``openral-sim`` never pulls them.
"""

from __future__ import annotations

from dataclasses import dataclass
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

_LIBERO_STATE_DIM = 8


@dataclass
class _Cosmos3Adapter:
    """Per-step adapter over lerobot's ``Cosmos3Policy``, its processors and the LIBERO decode."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _libero_action: Any  # Cosmos3LiberoActionProcessorStep: 10-D rot6d -> 7-D LIBERO
    _torch: Any
    _camera_keys: tuple[str, ...]
    _image_keys: tuple[str, ...]
    _flip_180: bool = False
    _last_input_frame: NDArray[np.uint8] | None = None
    _chunk_executor: Any = None

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
        return self._finished_action(run_inference(self._policy, batch))

    def _prepared_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        images = observation.get("images", {})
        batch: dict[str, Any] = {"task": instruction or str(observation.get("task", ""))}
        for slot, key in zip(self._camera_keys, self._image_keys, strict=True):
            frame = images.get(slot)
            if frame is None:
                raise ROSCapabilityMismatch(
                    f"cosmos3 needs camera slot {slot!r}; scene provides {sorted(images)}"
                )
            tensor = image_to_chw_tensor(frame, torch=torch, device=self.device)
            batch[key] = (torch.flip(tensor, dims=[1, 2]) if self._flip_180 else tensor).unsqueeze(
                0
            )
            if slot == self._camera_keys[0]:  # the third-person view is the preview
                self._last_input_frame = to_input_frame(frame, flip_180=self._flip_180)
        state = observation.get("state")
        if state is None:
            raise ROSCapabilityMismatch(
                "cosmos3 (LIBERO) needs the 8-D canonical LIBERO state as observation['state']."
            )
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != _LIBERO_STATE_DIM:
            raise ROSCapabilityMismatch(
                f"cosmos3 (LIBERO) expects the {_LIBERO_STATE_DIM}-D canonical LIBERO state "
                f"eef_pos | axis-angle | gripper_qpos, got {state_arr.shape[0]}-D."
            )
        batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
        return cast("dict[str, Any]", self._preprocessor(batch))

    def _finished_action(self, action_tensor: Any) -> NDArray[np.float32]:
        """Normalised 10-D rot6d step -> unnormalised -> 7-D LIBERO command."""
        return to_numpy_action(self._libero_action.action(self._postprocessor(action_tensor)))

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
    "cosmos3",
    install_groups=("vla",),
    required_imports=("transformers", "diffusers", "lerobot.policies.cosmos3.modeling_cosmos3"),
)
def _build_cosmos3(env_cfg: Any) -> _Cosmos3Adapter:
    """Load a LeRobot-format Cosmos 3 LIBERO checkpoint and its processor pipeline."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    # The checkpoint's action decode targets LIBERO's OSC command, and the 8-D state it
    # declares is LIBERO's task-space state: refuse any other scene before loading.
    from openral_sim.backends.libero import _LIBERO_SUITES

    if env_cfg.scene.id not in _LIBERO_SUITES:
        raise ROSCapabilityMismatch(
            f"cosmos3 runs the rot6d LIBERO recipe only: scene {env_cfg.scene.id!r} is not one "
            f"of {_LIBERO_SUITES}."
        )
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3Policy
        from lerobot.policies.cosmos3.processor_cosmos3 import (
            Cosmos3LiberoActionProcessorStep,
            make_cosmos3_pre_post_processors_from_pretrained,
        )
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "cosmos3 needs torch + diffusers + lerobot>=0.7; install with: "
            "just sync --all-packages --group libero"
        ) from exc

    extra = dict(spec.extra or {})
    if rtc_enabled_in_extra(extra, adapter_name="cosmos3"):
        raise ROSConfigError(
            "cosmos3: real-time chunking is not verified for this family; drop policy_extras.rtc."
        )
    manifest = load_manifest_for_spec(spec)
    plan = resolve_quant_plan(spec, manifest)
    require_supported_dtype(plan, frozenset({"bf16", "fp32"}), "cosmos3")
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if plan.dtype not in dtypes:
        raise ROSConfigError(
            f"cosmos3 needs an explicit bf16 or fp32 dtype, got {plan.dtype!r} "
            "(declare quantization.dtype in the manifest)."
        )

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="cosmos3")
    # One snapshot holds weights + the processor JSONs + the tokenizer directory.
    ckpt = local_snapshot_dir(repo_id, revision=revision)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        validate_processor_steps(str(Path(ckpt) / name))

    config = PreTrainedConfig.from_pretrained(ckpt)
    if getattr(config, "use_state", False):
        raise ROSConfigError(
            f"cosmos3: {repo_id} conditions on the state action (use_state); only the "
            "state-free LIBERO recipe is verified for this adapter."
        )
    # The checkpoint config says float32 so fine-tuning keeps fp32 master weights; the
    # model card evaluates in bf16 (the stored tensors). The model keeps its own fp32
    # exceptions (timestep embedding) whatever config.dtype is.
    config.dtype = dtypes[plan.dtype]
    config.device = device
    config.gradient_checkpointing = False  # a training-time memory trade; inference only here
    _log.info(
        "cosmos3.load",
        repo=repo_id,
        revision=revision,
        policy_type=config.type,
        dtype=plan.dtype,
        source=plan.source,
    )
    policy = Cosmos3Policy.from_pretrained(ckpt, config=config).to(device).eval()
    assert_all_parameters_finite(policy, repo_id=repo_id)
    apply_chunk_replay(policy, extra, manifest=manifest)
    policy.reset()

    preprocessor, postprocessor = make_cosmos3_pre_post_processors_from_pretrained(
        policy.config,
        ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, extra, scene_cameras=scene_cameras)
    image_prep = resolve_image_preprocessing(manifest, extra)
    image_keys = checkpoint_image_keys(image_prep, cam_keys)

    adapter = _Cosmos3Adapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _libero_action=Cosmos3LiberoActionProcessorStep(),
        _torch=torch,
        _camera_keys=cam_keys,
        _image_keys=image_keys,
        _flip_180=image_prep.flip_180,
    )
    adapter._chunk_executor = build_chunk_executor(
        extra,
        policy=policy,
        adapter_name="cosmos3",
        postprocess_action=adapter._finished_action,
    )
    return adapter
