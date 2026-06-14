"""xVLA policy adapter — wraps :class:`lerobot.policies.xvla.XVLAPolicy`.

xVLA's IO contract is structurally distinct from SmolVLA, so it gets its own
adapter rather than reusing the SmolVLA one:

- Three RGB camera slots: ``camera1``, ``camera2`` plus an in-process
  zero-tensor ``empty_camera_0`` (224x224) injected here. The xVLA
  checkpoint's processor pipeline expects all three.
- 20-D ``observation.state`` derived from the env's full ``robot_state``
  (eef pose + 6D rotation + joints + gripper) by ``LiberoProcessorStep``.
  The eval-layer ``Observation`` exposes the raw nested LIBERO obs under
  ``observation["raw"]`` so the env preprocessor can run.
- BART tokenizer for the task language.
- 20-D action output that ``XVLARotation6DToAxisAngle`` collapses to the
  7-D LIBERO control vector (eef pos + axis-angle + gripper).

The full pipeline (env_preprocessor → policy_preprocessor →
policy.select_action → policy_postprocessor → env_postprocessor) is the
same one ``lerobot-eval`` would compose; only the *raw obs / action plumbing*
is openral-specific.

This module imports torch / lerobot lazily so installing ``openral-sim``
never pulls them transitively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_rskill._vla_core import (
    resolve_camera_keys,
    resolve_device,
    resolve_rskill_repo_revision,
    run_inference,
    to_numpy_action,
)

from openral_sim.policies._processors import resolve_processor_dir
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


@dataclass
class _XVLAAdapter:
    """xVLA policy adapter — applies the four-stage processor pipeline per step."""

    spec: VLASpec
    device: str
    _policy: Any
    _env_pre: Any  # env_preprocessor: raw env obs → policy input batch
    _policy_pre: Any  # policy_preprocessor (saved with checkpoint)
    _policy_post: Any  # policy_postprocessor (saved with checkpoint)
    _env_post: Any  # env_postprocessor: 20-D action → 7-D LIBERO action
    _torch: Any
    _converters: Any  # lerobot.processor.converters (for transition wrapping)
    _empty_camera_size: tuple[int, int] = (224, 224)
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    _last_input_frame: NDArray[np.uint8] | None = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        raw = observation.get("raw")
        if not isinstance(raw, dict):
            raise ROSCapabilityMismatch(
                "xVLA is a LIBERO-engine-only checkpoint: its env preprocessor "
                "(LiberoProcessorStep) consumes the nested LiberoEnv obs that the "
                "scene must expose as observation['raw']. This scene does not "
                f"populate it (got observation['raw'] of type {type(raw).__name__}), "
                "so xvla cannot run here — use a LIBERO scene (e.g. libero_spatial / "
                "franka_libero_pnp) or a non-LIBERO-bound checkpoint."
            )
        batch = self._build_raw_batch(raw, observation, instruction)
        # Stage 1: env preprocessor (LiberoProcessorStep + image normalize + domain id).
        batch = self._env_pre(batch)
        # Stage 2: policy preprocessor (rename, tokenize, device move, normalize).
        batch = self._policy_pre(batch)
        # Stage 3: forward pass.
        action_tensor = run_inference(self._policy, batch)
        # Stage 4: policy postprocessor (unnormalize, CPU move).
        action_tensor = self._policy_post(action_tensor)
        # Stage 5: env postprocessor (rot6d → axis-angle, 20-D → 7-D).
        transition = self._converters.policy_action_to_transition(action_tensor)
        transition = self._env_post(transition)
        action_tensor = self._converters.transition_to_policy_action(transition)
        return to_numpy_action(action_tensor)

    def close(self) -> None:
        if self.device.startswith("cuda"):
            import contextlib

            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _build_raw_batch(
        self,
        raw_obs: dict[str, Any],
        observation: Observation,
        instruction: str,
    ) -> dict[str, Any]:
        """Convert the LIBERO raw obs into the env-preprocessor's input dict.

        The xVLA env preprocessor (``LiberoProcessorStep``) consumes
        ``observation.images.image{,2}``, ``observation.images.empty_camera_0``,
        and a nested ``observation.robot_state`` dict with tensor fields. We
        build that here from the raw LiberoEnv observation.
        """
        torch = self._torch
        device = self.device

        def _hwc_uint8_to_bchw_float(img: NDArray[np.uint8]) -> Any:
            t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to(device)

        def _to_tensor(arr: Any, shape: tuple[int, ...]) -> Any:
            if arr is None:
                return torch.zeros((1, *shape), dtype=torch.float32, device=device)
            np_arr = np.asarray(arr, dtype=np.float32)
            return torch.from_numpy(np_arr).reshape(1, *shape).to(device)

        pixels = raw_obs.get("pixels", {})
        h, w = self._empty_camera_size
        batch: dict[str, Any] = {"task": instruction or observation.get("task", "")}
        from openral_sim.policies._video_capture import to_input_frame

        if "image" in pixels:
            batch["observation.images.image"] = _hwc_uint8_to_bchw_float(pixels["image"])
        if "image2" in pixels:
            batch["observation.images.image2"] = _hwc_uint8_to_bchw_float(pixels["image2"])
        # Capture the wrist / mounted camera for the top-left "VLA input"
        # panel — image2 is the eye-in-hand view in LIBERO; fall back to
        # the agent view (image) when only one stream is available.
        wrist = pixels.get("image2", pixels.get("image"))
        if wrist is not None:
            self._last_input_frame = to_input_frame(wrist)
        batch["observation.images.empty_camera_0"] = torch.zeros(
            1, 3, h, w, dtype=torch.float32, device=device
        )

        robot_state = raw_obs.get("robot_state", {}) or {}
        eef = robot_state.get("eef", {}) or {}
        gripper = robot_state.get("gripper", {}) or {}
        joints = robot_state.get("joints", {}) or {}
        batch["observation.robot_state"] = {
            "eef": {
                "pos": _to_tensor(eef.get("pos"), (3,)),
                "mat": _to_tensor(eef.get("mat"), (3, 3)),
                "quat": _to_tensor(eef.get("quat"), (4,)),
            },
            "gripper": {
                "qpos": _to_tensor(gripper.get("qpos"), (2,)),
                "qvel": _to_tensor(gripper.get("qvel"), (2,)),
            },
            "joints": {
                "pos": _to_tensor(joints.get("pos"), (7,)),
                "vel": _to_tensor(joints.get("vel"), (7,)),
            },
        }
        return batch


@POLICIES.register("xvla")
def _build_xvla(env_cfg: Any) -> _XVLAAdapter:
    """Load an xVLA-LIBERO checkpoint + its four-stage processor pipeline."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    try:
        import torch
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import (
            make_xvla_libero_pre_post_processors,
        )
        from lerobot.processor import (
            PolicyProcessorPipeline,
        )
        from lerobot.processor import converters as _converters
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from openral_rskill import _lerobot_compat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "xVLA adapter requires torch + lerobot[libero]; install with: "
            "just sync --all-packages --group libero"
        ) from exc

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="xVLA")
    policy = XVLAPolicy.from_pretrained(repo_id, revision=revision).to(device)
    policy.eval()
    if getattr(policy.config, "chunk_size", None):
        policy.config.n_action_steps = policy.config.chunk_size
        policy.reset()

    # lerobot 0.5.1 split the four-stage pipeline:
    #   * env-side  : make_xvla_libero_pre_post_processors() returns
    #     (env_pre = LiberoProcessorStep + ImageNet + DomainId;
    #      env_post = Rot6DToAxisAngle, 20-D → 7-D action).
    #   * policy-side: serialised to ``policy_preprocessor.json`` /
    #     ``policy_postprocessor.json`` inside the HF checkpoint and
    #     restored via PolicyProcessorPipeline.from_pretrained — this is
    #     the rename/tokenize/device/normalize chain that wraps
    #     ``policy.select_action``. Building it from
    #     ``make_xvla_pre_post_processors`` would double-apply
    #     ImageNetNormalize and trip ``XVLAImageToFloatProcessorStep``'s
    #     [0, 255] guard.
    env_pre, env_post = make_xvla_libero_pre_post_processors()
    # Manifest-first: when spec.weights_uri is a bare rSkill reference and
    # the manifest declares a `processors` block, fetch exactly those two
    # files per-file (ADR-0013). The snapshot fallback covers legacy
    # hf:// URIs that predate the per-file contract.
    pretrained_path = resolve_processor_dir(spec, repo_id)
    policy_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    policy_post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Manifest-first camera-key resolution. xVLA's env preprocessor
    # (LiberoProcessorStep) builds its own observation.images.* batch
    # from the LIBERO raw obs; this list only drives the wrist-frame
    # surface for the eval video helper.
    manifest = None
    weights_uri = str(getattr(spec, "weights_uri", "") or "")
    if not weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        from openral_rskill.loader import load_rskill_manifest

        manifest = load_rskill_manifest(weights_uri)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    return _XVLAAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _env_pre=env_pre,
        _policy_pre=policy_pre,
        _policy_post=policy_post,
        _env_post=env_post,
        _torch=torch,
        _converters=_converters,
        _camera_keys=cam_keys,
    )
