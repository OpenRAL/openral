"""NaT VLA adapter — LFM2-VL-450M writing NAT action tokens (lerobot ``nat_vla``).

NaT VLA (``ahad-j/nat-vla-450m``) is a 450M vision-language-action model built on
LiquidAI LFM2-VL-450M that emits actions as tokens of the NAT neural action
tokenizer and decodes them while they are generated. lerobot merged it after its
0.7.0 release, so this adapter uses ``lerobot.policies.nat_vla`` when the installed
lerobot ships it and otherwise the copy vendored at the merge commit
(``openral_sim.policies._nat_vla``, Apache-2.0); the source is logged.

The published checkpoint is the multi-robot *pretrained* model: its config carries
no input/output features and its processors no normalisation stats. The rSkill
manifest supplies them:

- ``state_contract.dim`` / ``action_contract.dim`` — the state and action widths;
- ``image_preprocessing`` — the camera slots and their checkpoint keys;
- ``dataset_uri`` — a LeRobot dataset whose ``meta/stats.json`` holds ``q01``/``q99``
  for the state (quantile normalisation) and ``min``/``max`` for the action;
- ``policy_extras.setup`` / ``policy_extras.control_mode`` — the prompt's robot setup
  and control mode (pretraining used ``single so100/so101 robotic arm in molmoact2``
  / ``absolute joint pose`` for the SO-ARM).

Actions stream: ``select_action`` returns each step as soon as its tokens are
generated, so this adapter runs the policy's own queue rather than the shared chunk
executor. ``n_action_steps`` must be one of the tokenizer's whole-latent prefixes
(8, 16, 24 or 32).

Imports torch / lerobot lazily so installing ``openral-sim`` never pulls them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_rskill._vla_core import (
    checkpoint_image_keys,
    image_to_chw_tensor,
    release_torch_modules,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_n_action_steps,
    resolve_rskill_repo_revision,
    run_inference,
    to_numpy_action,
)

from openral_sim._quantization import require_supported_dtype, resolve_quant_plan
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.policies._video_capture import to_input_frame
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)


def _nat_vla_module() -> tuple[Any, Any, Any, str]:
    """``(NATVLAConfig, NATVLAPolicy, make_nat_vla_pre_post_processors, source)``.

    lerobot's own package when the installed version ships it (>0.7.0), else the
    copy vendored at the lerobot merge commit. Both register ``nat_vla`` with
    lerobot's config and processor registries, so only one may be imported.
    """
    import importlib.util

    if importlib.util.find_spec("lerobot.policies.nat_vla") is not None:
        from lerobot.policies.nat_vla import (
            NATVLAConfig,
            NATVLAPolicy,
            make_nat_vla_pre_post_processors,
        )

        return NATVLAConfig, NATVLAPolicy, make_nat_vla_pre_post_processors, "lerobot"
    from openral_sim.policies._nat_vla import (
        NATVLAConfig,
        NATVLAPolicy,
        make_nat_vla_pre_post_processors,
    )

    return NATVLAConfig, NATVLAPolicy, make_nat_vla_pre_post_processors, "vendored"


@dataclass
class _NATVLAAdapter:
    """Per-step adapter over ``NATVLAPolicy`` and the pipelines built for its rSkill."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _camera_keys: tuple[str, ...]
    _image_keys: tuple[str, ...]
    _state_dim: int
    _last_input_frame: NDArray[np.uint8] | None = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        torch = self._torch
        images = observation.get("images", {})
        batch: dict[str, Any] = {"task": instruction or str(observation.get("task", ""))}
        for slot, key in zip(self._camera_keys, self._image_keys, strict=True):
            frame = images.get(slot)
            if frame is None:
                raise ROSCapabilityMismatch(
                    f"nat_vla needs camera slot {slot!r}; scene provides {sorted(images)}"
                )
            batch[key] = image_to_chw_tensor(frame, torch=torch, device=self.device)
            if slot == self._camera_keys[0]:
                self._last_input_frame = to_input_frame(frame)
        state = observation.get("state")
        if state is None:
            raise ROSConfigError("nat_vla requires a proprio `state` observation.")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != self._state_dim:
            raise ROSConfigError(
                f"nat_vla expects a {self._state_dim}-D state, got {state_arr.shape[0]}-D."
            )
        batch["observation.state"] = torch.from_numpy(state_arr).to(self.device)
        prepared = cast("dict[str, Any]", self._preprocessor(batch))
        return to_numpy_action(self._postprocessor(run_inference(self._policy, prepared)))

    def close(self) -> None:
        release_torch_modules(
            self,
            "_policy",
            "_preprocessor",
            "_postprocessor",
            device=self.device,
            torch=self._torch,
        )


@POLICIES.register(
    "nat_vla",
    install_groups=("vla",),
    required_imports=("transformers", "lerobot.policies.pretrained"),
)
def _build_nat_vla(env_cfg: Any) -> _NATVLAAdapter:
    """Build the pretrained NaT VLA for an rSkill's robot (features + stats from the manifest)."""
    spec = env_cfg.vla
    device = resolve_device(spec)
    try:
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "nat_vla needs torch + lerobot>=0.7; install with: just sync --all-packages --group vla"
        ) from exc
    config_cls, policy_cls, make_processors, source = _nat_vla_module()

    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError(
            "nat_vla needs a bare rSkill reference: the manifest supplies its features and stats."
        )
    extra = {**manifest.policy_extras, **dict(spec.extra or {})}
    plan = resolve_quant_plan(spec, manifest)
    require_supported_dtype(plan, frozenset({"bf16", "fp32"}), "nat_vla")
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if plan.dtype not in dtypes:
        raise ROSConfigError(f"nat_vla needs an explicit bf16 or fp32 dtype, got {plan.dtype!r}.")
    state_dim = manifest.state_contract.dim if manifest.state_contract else None
    action_dim = manifest.action_contract.dim if manifest.action_contract else None
    if state_dim is None or action_dim is None:
        raise ROSConfigError("nat_vla needs state_contract.dim and action_contract.dim.")
    if manifest.dataset_uri is None:
        raise ROSConfigError(
            "nat_vla's pretrained checkpoint carries no normalisation stats: declare dataset_uri."
        )
    setup, control_mode = extra.get("setup"), extra.get("control_mode")
    if not isinstance(setup, str) or not isinstance(control_mode, str):
        raise ROSConfigError(
            "nat_vla needs policy_extras.setup and policy_extras.control_mode (prompt strings)."
        )

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, extra, scene_cameras=scene_cameras)
    image_keys = checkpoint_image_keys(resolve_image_preprocessing(manifest, extra), cam_keys)
    height = int(getattr(env_cfg.scene, "observation_height", 480))
    width = int(getattr(env_cfg.scene, "observation_width", 640))

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="nat_vla")
    n_action_steps = resolve_n_action_steps(manifest, extra, default=32)
    config = config_cls(
        vlm=repo_id,
        vlm_revision=revision,
        setup=setup,
        control_mode=control_mode,
        dtype=dtypes[plan.dtype],
        n_action_steps=n_action_steps,
        device=device,
        input_features={
            **{key: PolicyFeature(FeatureType.VISUAL, (3, height, width)) for key in image_keys},
            "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    _log.info(
        "nat_vla.load",
        source=source,
        repo=repo_id,
        revision=revision,
        dtype=plan.dtype,
        n_action_steps=n_action_steps,
        dataset=manifest.dataset_uri,
    )
    # Constructing a fresh policy whose `vlm` is the pretrained repo loads its weights
    # (strict: anything but the frozen tokenizer copy missing or extra raises).
    policy = policy_cls(config).to(device).eval()

    from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

    raw_stats = _load_lerobot_dataset_stats(manifest.dataset_uri)
    stats = {
        key: {name: torch.as_tensor(np.asarray(value)) for name, value in raw_stats[key].items()}
        for key in ("observation.state", "action")
    }
    preprocessor, postprocessor = make_processors(config, stats)
    return _NATVLAAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _camera_keys=cam_keys,
        _image_keys=image_keys,
        _state_dim=state_dim,
    )
