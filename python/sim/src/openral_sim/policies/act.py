"""ACT (Action Chunking Transformer) policy adapter.

Wraps :class:`lerobot.policies.act.modeling_act.ACTPolicy` (Zhao et al.,
2023). ACT's IO contract differs from SmolVLA / π0.5:

- One observation key per camera (``observation.images.<name>``);
  by default a single ``observation.images.top`` 480x640 stream.
- ``observation.state`` is the raw joint position vector (e.g. 14-D for
  bimanual ALOHA).
- Action chunks of size 100; each chunk pop is allocation-free.
- ``observation.state`` and image tensors must be CHW / float32 in
  ``[0, 1]`` on the policy device — same plumbing as the SmolVLA adapter
  but with the ACT-native key naming (no ``rename_observations_processor``
  step).

This module imports torch / lerobot lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import (
    apply_chunk_replay,
    call_make_processors_cached_first,
    materialize_processor_dir,
    maybe_compile_chunk_forward,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_revision,
    resolve_state_dim,
    run_inference,
    to_numpy_action,
)

from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


def _load_manifest_for_spec(spec: Any) -> Any:
    """Load the rSkill manifest from ``spec.weights_uri`` (bare rSkill reference).

    Mirrors :func:`openral_sim.policies.smolvla._load_manifest_for_spec`.
    Returns ``None`` for explicit-scheme URIs (``hf://``, ``local://``,
    etc.) so legacy tests that construct such a ``weights_uri`` still work.
    """
    weights_uri = str(getattr(spec, "weights_uri", "") or "")
    if weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        return None
    from openral_rskill.loader import load_rskill_manifest

    return load_rskill_manifest(weights_uri)


@dataclass
class _ACTAdapter:
    """ACT policy adapter — straight passthrough, no separate preprocessor pipeline.

    Many ACT checkpoints (including ``lerobot/act_aloha_sim_transfer_cube_human``)
    predate lerobot's ``PolicyProcessorPipeline`` migration and ship without
    a saved preprocessor. The adapter therefore feeds raw float32 [0,1]
    images and raw state directly to ``policy.select_action``; if the
    upstream checkpoint *does* expose normaliser stats, set
    ``vla.extra.use_lerobot_processors=True`` and we will compose them.
    """

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any | None
    _postprocessor: Any | None
    _torch: Any
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("top",))
    # Manifest-driven scene-key → checkpoint-input-feature renames.
    _cam_alias: dict[str, str] = field(default_factory=dict)
    _image_input_template: str = "observation.images.{cam}"
    _flip_images_180: bool = False
    _state_dim: int | None = None
    _last_input_frame: NDArray[np.uint8] | None = None
    # Normalization stats extracted from the checkpoint when the
    # underlying lerobot ACTPolicy class no longer carries
    # normalize_inputs / unnormalize_outputs modules. ACT uses
    # mean/std (Gaussian) normalization — without these the 14-D state
    # gets fed raw (~80× the training scale) and the action is emitted
    # in zero-mean unit-std space, so the gripper flails near the home
    # pose instead of executing the demonstrated cube-transfer
    # trajectory. Set on construction; applied per step.
    _state_mean: Any = None  # tensor (1, D)
    _state_std: Any = None
    _image_mean: dict[str, Any] = field(default_factory=dict)  # cam_key → tensor (1, 3, 1, 1)
    _image_std: dict[str, Any] = field(default_factory=dict)
    _action_mean: Any = None  # tensor (1, A)
    _action_std: Any = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        batch = self._build_batch(observation, instruction)
        if self._preprocessor is not None:
            batch = self._preprocessor(batch)
        elif self._state_mean is not None or self._image_mean:
            self._normalize_inplace(batch)
        action_tensor = run_inference(self._policy, batch)
        if self._postprocessor is not None:
            action_tensor = self._postprocessor(action_tensor)
        elif self._action_mean is not None:
            # Unnormalize from zero-mean unit-std training space →
            # raw joint-position command units.
            action_tensor = action_tensor * self._action_std + self._action_mean
        return to_numpy_action(action_tensor)

    def _normalize_inplace(self, batch: dict[str, Any]) -> None:
        for cam_key in self._camera_keys:
            key = f"observation.images.{cam_key}"
            if key in batch and cam_key in self._image_mean:
                batch[key] = (batch[key] - self._image_mean[cam_key]) / self._image_std[cam_key]
        if "observation.state" in batch and self._state_mean is not None:
            batch["observation.state"] = (
                batch["observation.state"] - self._state_mean
            ) / self._state_std

    def close(self) -> None:
        if self.device.startswith("cuda"):
            import contextlib

            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {"task": [instruction or str(observation.get("task", ""))]}

        images = observation.get("images", {})
        from openral_sim.policies._video_capture import tile_input_frames, to_input_frame

        # Record every camera the policy consumed so the debug video does not
        # hide multi-camera setups behind a single wrist-only preview.
        preview_frames: list[NDArray[np.uint8]] = []
        for cam_key in self._camera_keys:
            img = images.get(cam_key)
            if img is None:
                continue
            preview = to_input_frame(img, flip_180=self._flip_images_180)
            if preview is not None:
                preview_frames.append(preview)
            t = torch.from_numpy(np.asarray(img)).float().div(255.0).permute(2, 0, 1)
            if self._flip_images_180:
                t = torch.flip(t, dims=[1, 2])
            t = t.unsqueeze(0).to(self.device)
            batch[self._image_input_template.format(cam=self._cam_alias.get(cam_key, cam_key))] = t
        self._last_input_frame = tile_input_frames(preview_frames)

        state = observation.get("state")
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            if self._state_dim is not None and state_np.shape[0] != self._state_dim:
                if state_np.shape[0] < self._state_dim:
                    pad = np.zeros(self._state_dim - state_np.shape[0], dtype=np.float32)
                    state_np = np.concatenate([state_np, pad])
                else:
                    state_np = state_np[: self._state_dim]
            batch["observation.state"] = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        return batch


@POLICIES.register("act")
def _build_act(env_cfg: Any) -> _ACTAdapter:
    """Load an ACTPolicy checkpoint."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from openral_rskill import _lerobot_compat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "ACT adapter requires torch + lerobot; install with: "
            "just sync --all-packages --group sim"
        ) from exc

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="ACT")
    manifest = _load_manifest_for_spec(spec)
    # Snapshot first so we can (a) probe for modern processor sidecars
    # below and (b) sanitize a ``config.json`` that may carry training-only
    # fields the installed lerobot ACTConfig doesn't recognize (e.g.
    # ``n_state_dim`` on ``JunnDooChoi/act_libero_spatial_finetuned_kaf_64``).
    from huggingface_hub import snapshot_download

    pretrained_path = snapshot_download(
        repo_id=repo_id, revision=revision, ignore_patterns=["*.md"]
    )
    _sanitize_act_config_json(pretrained_path)
    policy = ACTPolicy.from_pretrained(pretrained_path).to(device)
    policy.eval()

    # ACT predicts ``chunk_size=100`` actions per heavy forward. The paper
    # protocol (Zhao et al. §V-B) is *temporal ensembling*: re-infer every
    # env step and weighted-average overlapping chunks. Default replay
    # therefore is ``n_action_steps=1`` (per-step re-inference); pair with
    # ``temporal_ensemble_coeff`` below for the paper-faithful eval mode.
    # ``vla.extra.n_action_steps`` overrides for users who want simpler
    # chunked execution.
    apply_chunk_replay(policy, spec.extra, manifest=manifest, default_n_action_steps=1)
    _apply_temporal_ensemble(policy, spec.extra)
    maybe_compile_chunk_forward(policy, spec.extra, device, torch)

    # Two ACT shapes coexist in tree:
    #
    # - **Modern** (`manifest.processors is not None`): the upstream
    #   checkpoint ships ``policy_preprocessor.json`` /
    #   ``policy_postprocessor.json`` sidecars (e.g.
    #   ``Deepkar/libero-test-act`` wrapped by ``rskills/act-libero``).
    #   We materialize them via per-file ``hf_hub_download`` (driven by
    #   ``manifest.processors``) and let the lerobot factory compose the
    #   pipeline. This is the "rSkill self-containment audit Gap 1+3"
    #   path — no implicit snapshot_download.
    #
    # - **Legacy** (`manifest.processors is None`): the
    #   ``lerobot/act_aloha_sim_transfer_cube_human`` / `_insertion_human`
    #   checkpoints pre-date the PolicyProcessorPipeline migration and
    #   carry their norm stats inside ``model.safetensors``. The schema
    #   permits these to omit the processors block; the existing
    #   ``_try_load_act_norm_stats`` path below reads the safetensors
    #   directly. ``rskills/act-aloha`` / ``act-aloha-insertion`` keep
    #   working unchanged.
    preprocessor: Any | None = None
    postprocessor: Any | None = None

    if manifest is not None and manifest.processors is not None:
        from lerobot.policies.factory import make_pre_post_processors

        # Per-file download from manifest.processors (no snapshot_download).
        processors_dir = materialize_processor_dir(manifest)

        # Override the saved ``device_processor`` step's device with the
        # resolved runtime device. The stored preprocessor json bakes in
        # whatever device the trainer used (e.g. ``mps`` on Apple Silicon
        # for ``Deepkar/libero-test-act``); without this override the
        # pipeline crashes at instantiation on hosts without that backend.
        preprocessor, postprocessor = call_make_processors_cached_first(
            make_pre_post_processors,
            policy.config,
            pretrained_path=processors_dir,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )

    ip = resolve_image_preprocessing(manifest, spec.extra)
    state_dim = resolve_state_dim(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    # When the manifest declares image aliases but the scene doesn't list
    # cameras, derive the source key tuple from the alias map. Falls
    # through to the ALOHA-shaped ``("top",)`` default for single-cam.
    default_keys: tuple[str, ...] = tuple(ip.aliases.keys()) if ip.aliases else ("top",)
    cam_keys = resolve_camera_keys(
        manifest, spec.extra, scene_cameras=scene_cameras, default=default_keys
    )

    # Recover normalization stats from the checkpoint's safetensors when
    # the loaded ACTPolicy class doesn't expose normalize_inputs /
    # unnormalize_outputs modules. The published
    # ``lerobot/act_aloha_sim_transfer_cube_human`` checkpoint still
    # carries the buffers; without applying them the policy emits
    # unnormalized-action-space outputs and the gripper never picks up
    # the cube.
    stats: dict[str, Any] = {}
    if not preprocessor and not postprocessor:
        stats = _try_load_act_norm_stats(repo_id, device, torch, cam_keys)

    return _ACTAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _camera_keys=cam_keys,
        _cam_alias=dict(ip.aliases),
        _image_input_template=ip.input_template,
        _flip_images_180=ip.flip_180,
        _state_dim=state_dim,
        _state_mean=stats.get("state_mean"),
        _state_std=stats.get("state_std"),
        _image_mean=stats.get("image_mean", {}),
        _image_std=stats.get("image_std", {}),
        _action_mean=stats.get("action_mean"),
        _action_std=stats.get("action_std"),
    )


def _sanitize_act_config_json(snapshot_dir: str) -> None:
    """Drop ACTConfig fields the installed lerobot version doesn't accept.

    Training-side forks of ACT (e.g. ``kafarobotics/lerobot``) sometimes
    add knobs like ``n_state_dim`` to ``ACTConfig`` and bake them into
    the published ``config.json``. The vanilla ``lerobot==0.5.x``
    ``ACTConfig`` rejects unknown fields under draccus's strict
    dataclass decoding, breaking ``ACTPolicy.from_pretrained``. Strip
    unknown keys in-place before the from-pretrained call so this is
    transparent to the rest of the adapter.

    Mutates ``config.json`` only when (a) the file exists, (b) it parses
    as JSON, and (c) at least one key is unknown — otherwise it's a
    no-op so re-runs are idempotent.
    """
    import dataclasses
    import json
    import os

    from lerobot.policies.act.configuration_act import ACTConfig

    cfg_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path) as f:
            raw = json.load(f)
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    valid = {f.name for f in dataclasses.fields(ACTConfig)} | {"type"}
    unknown = [k for k in raw if k not in valid]
    if not unknown:
        return
    for k in unknown:
        raw.pop(k, None)
    with open(cfg_path, "w") as f:
        json.dump(raw, f, indent=4)


def _apply_temporal_ensemble(policy: Any, spec_extra: dict[str, Any]) -> float | None:
    """Engage ACT's temporal-ensembling buffer (Zhao et al. §V-B).

    ACT's paper eval mode predicts a fresh ``chunk_size``-step chunk every
    env step and takes the action for timestep ``t`` as an exponentially
    weighted average of every chunk that includes ``t``. Lerobot exposes
    this via ``policy.config.temporal_ensemble_coeff``: a float > 0
    enables ensembling; ``None`` falls back to plain chunked execution.

    Paper default is ``0.01``; the published
    ``lerobot/act_aloha_sim_transfer_cube_human`` checkpoint ships with
    the field set to ``None`` so plain chunked execution wins by default
    — that's the difference between the harness's previous 0.46 and the
    paper's 0.95 on aloha_transfer_cube. This helper restores the paper
    value unless ``vla.extra.temporal_ensemble_coeff`` overrides; pass
    ``null`` (YAML) / ``None`` (Python) to disable.

    Args:
        policy: A lerobot ACTPolicy with a ``config.temporal_ensemble_coeff``
            attribute. Setting before the first ``select_action`` is what
            lerobot's ``ACTPolicy.reset`` reads when sizing its buffer.
        spec_extra: ``VLASpec.extra``. ``temporal_ensemble_coeff`` is
            looked up here; missing → paper default 0.01.

    Returns:
        The applied coefficient (``None`` if disabled).
    """
    if not hasattr(policy.config, "temporal_ensemble_coeff"):
        return None  # older lerobot ACT config without the field — nothing to do.
    sentinel = object()
    raw = spec_extra.get("temporal_ensemble_coeff", sentinel)
    if raw is sentinel:
        coeff: float | None = 0.01
    elif raw is None:
        coeff = None
    else:
        coeff = float(raw)
    policy.config.temporal_ensemble_coeff = coeff
    # ACTPolicy constructs `self.temporal_ensembler` ONLY when the config
    # has a non-None coeff at __init__ time (lerobot/policies/act/modeling_act.py).
    # rSkill manifests typically load the policy with coeff=None and let this
    # helper enable ensembling post-construction; we therefore have to build
    # the ensembler ourselves or `policy.reset()` will explode on the missing
    # attribute.
    if coeff is not None and not hasattr(policy, "temporal_ensembler"):
        try:
            from lerobot.policies.act.modeling_act import (
                ACTTemporalEnsembler,  # reason: lazy — heavy lerobot import
            )

            policy.temporal_ensembler = ACTTemporalEnsembler(coeff, policy.config.chunk_size)
        except ImportError:  # pragma: no cover - lerobot always present in sim group
            policy.config.temporal_ensemble_coeff = None
            return None
    # Re-initialise the ensembling buffer (if any) against the new coeff.
    if hasattr(policy, "reset"):
        policy.reset()
    return coeff


def _try_load_act_norm_stats(
    repo_id: str,
    device: str,
    torch: Any,
    cam_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Pull ``normalize_*`` / ``unnormalize_*`` mean/std tensors from the checkpoint.

    Returns an empty dict if the checkpoint doesn't carry them — in
    which case the adapter falls back to feeding raw inputs (the lerobot
    behaviour and what we did before this fix).
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError:
        return {}
    try:
        local = snapshot_download(repo_id=repo_id, ignore_patterns=["*.md"])
    except Exception:
        return {}
    import os

    weights = os.path.join(local, "model.safetensors")
    if not os.path.exists(weights):
        return {}

    out: dict[str, Any] = {"image_mean": {}, "image_std": {}}
    try:
        with safe_open(weights, framework="pt") as f:  # type: ignore[no-untyped-call]
            available = set(f.keys())

            def _maybe(key: str) -> Any | None:
                if key in available:
                    return f.get_tensor(key).to(device=device, dtype=torch.float32)
                return None

            sm = _maybe("normalize_inputs.buffer_observation_state.mean")
            ss = _maybe("normalize_inputs.buffer_observation_state.std")
            if sm is not None and ss is not None:
                out["state_mean"] = sm.unsqueeze(0)
                out["state_std"] = ss.unsqueeze(0)

            am = _maybe("unnormalize_outputs.buffer_action.mean")
            astd = _maybe("unnormalize_outputs.buffer_action.std")
            if am is not None and astd is not None:
                out["action_mean"] = am.unsqueeze(0)
                out["action_std"] = astd.unsqueeze(0)

            for cam in cam_keys:
                im = _maybe(f"normalize_inputs.buffer_observation_images_{cam}.mean")
                istd = _maybe(f"normalize_inputs.buffer_observation_images_{cam}.std")
                if im is not None and istd is not None:
                    # Already (3, 1, 1) in the checkpoint; add batch dim.
                    out["image_mean"][cam] = im.unsqueeze(0)
                    out["image_std"][cam] = istd.unsqueeze(0)
    except Exception:
        return {}
    return out
