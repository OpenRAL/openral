"""SmolVLA / generic lerobot-policy adapter.

Wraps any ``lerobot.policies.*`` checkpoint that follows the SmolVLA
``select_action(batch) → Tensor`` interface — this includes ``smolvla_libero``,
``smolvla_metaworld``, ``pi05_libero``, and any compatible finetune.

The adapter only accepts bare rSkill references in
:attr:`VLASpec.weights_uri`. The rSkill manifest is the contract between
robot/sensors/preprocessing and the policy weights — the eval layer never
loads weights without one. The manifest is resolved to a bare HF Hub repo id
via :func:`openral_rskill.loader.resolve_rskill_to_hf`.

Like the other adapters, this module imports torch / lerobot / transformers
lazily so installing ``openral-sim`` never pulls them transitively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_rskill._diagnostics import phase_timer
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

from openral_sim.policies._policy_loading import (
    lazy_import_lerobot,
    load_manifest_for_spec,
)
from openral_sim.registry import POLICIES

_log = structlog.get_logger(__name__)

try:
    from huggingface_hub.errors import EntryNotFoundError as _HFEntryNotFoundError
    from huggingface_hub.errors import RemoteEntryNotFoundError as _HFRemoteEntryNotFoundError

    _PROCESSOR_MISSING_EXC: tuple[type[BaseException], ...] = (
        _HFRemoteEntryNotFoundError,
        _HFEntryNotFoundError,
    )
except ImportError:  # pragma: no cover — huggingface_hub is a core dep
    _PROCESSOR_MISSING_EXC = ()


def _is_processor_missing(exc: BaseException) -> bool:
    """True when an exception chain wraps a 404 from a per-file processor download.

    ``materialize_processor_dir`` re-raises HF Hub's ``RemoteEntryNotFoundError`` /
    ``EntryNotFoundError`` as a typed ``ROSConfigError``. Walk the cause chain by
    type-name so we stay decoupled from HF Hub's exception module path.
    """
    cur: BaseException | None = exc
    while cur is not None:
        if type(cur).__name__ in ("RemoteEntryNotFoundError", "EntryNotFoundError"):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _load_lerobot_dataset_stats(dataset_uri: str) -> dict[str, dict[str, Any]]:
    """Aggregate per-feature stats from a LeRobotDataset on HF Hub.

    Handles both layouts:

    * v3 — single ``meta/stats.json`` at the top level.
    * v2.1 — ``meta/episodes_stats.jsonl`` with one record per episode,
      aggregated via ``lerobot.datasets.compute_stats.aggregate_stats``.

    Args:
        dataset_uri: ``hf://owner/repo[@rev]`` reference (no file tail).

    Returns:
        ``{feature_key: {mean|std|min|max|count: np.ndarray}}`` suitable for
        ``make_pre_post_processors(..., dataset_stats=...)``.

    Raises:
        ROSConfigError: ``dataset_uri`` is malformed or the dataset repo ships
            neither layout.
    """
    if not dataset_uri.startswith("hf://"):
        raise ROSConfigError(
            f"_load_lerobot_dataset_stats: only hf:// URIs are accepted, got {dataset_uri!r}"
        )
    repo_id = dataset_uri[len("hf://") :]
    revision: str | None = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
    except ImportError as exc:  # pragma: no cover — huggingface_hub is a core dep
        raise ROSConfigError("_load_lerobot_dataset_stats requires huggingface_hub") from exc

    import json

    _not_found = (RemoteEntryNotFoundError, EntryNotFoundError)

    try:
        stats_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/stats.json",
            revision=revision,
            repo_type="dataset",
        )
        with open(stats_path) as f:
            raw = json.load(f)
        return {feat: {k: np.asarray(v) for k, v in st.items()} for feat, st in raw.items()}
    except _not_found:
        pass

    try:
        episodes_stats_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/episodes_stats.jsonl",
            revision=revision,
            repo_type="dataset",
        )
    except _not_found as exc:
        raise ROSConfigError(
            f"Dataset {dataset_uri!r} has neither meta/stats.json (v3) nor "
            "meta/episodes_stats.jsonl (v2.1) — cannot rebuild SmolVLA "
            "processors without normalization stats."
        ) from exc

    from lerobot.datasets.compute_stats import aggregate_stats

    stats_list: list[dict[str, dict[str, NDArray[np.float64]]]] = []
    with open(episodes_stats_path) as f:
        for line in f:
            record = json.loads(line)
            stats_list.append(
                {
                    feat: {k: np.asarray(v) for k, v in st.items()}
                    for feat, st in record["stats"].items()
                }
            )
    aggregated: dict[str, dict[str, Any]] = aggregate_stats(stats_list)
    return aggregated


def _smolvla_phase(name: str, **fields: Any) -> Any:
    """Shortcut for ``phase_timer(name, prefix="smolvla", log=_log)``.

    Mirrors the pi05 adapter's ``_pi05_phase`` helper so a SmolVLA load
    emits the same ``smolvla_<name>_{start,heartbeat,done}`` event shape
    that the operator already knows from pi05. ``gpu_mb`` is opt-in per
    phase because SmolVLA is small (~400 M params); GPU memory only
    moves during ``.to(device)`` and the warm-up forward.
    """
    return phase_timer(name, prefix="smolvla", log=_log, **fields)


if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


@dataclass
class _SmolVLAAdapter:
    """Lerobot-style policy adapter that returns per-step actions."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _flip_images_180: bool = False
    _state_dim: int | None = None
    # Dtype of the VLM backbone (the model's majority dtype — bf16 for the
    # smolvla-libero checkpoint lerobot now loads dtype-preserving). The image
    # tensors we build are float32 and feed the bf16 vision tower, so step()
    # casts *only the image inputs* to this. ``observation.state`` is left
    # float32 to match the model's float32 action expert (and the sampler's
    # internal float32 noise/time). ``None`` (a fully-float32 load) → no cast.
    _image_dtype: Any = None
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    # Camera renames + image-batch-key template come from the rSkill
    # manifest's ``image_preprocessing`` block (or ``vla.extra``). The
    # historical default of ``{camera1: image, camera2: image2}`` is now
    # encoded on the LIBERO skill manifests instead of being hard-wired
    # in this dataclass — see commit "feat(core): RSkillManifest
    # image_preprocessing + state_contract + n_action_steps".
    _cam_alias: dict[str, str] = field(default_factory=dict)
    _image_input_template: str = "observation.images.{cam}"
    # Last image fed to the policy, post-flip — populated by step() for the
    # eval-layer video helper. Not part of the public API.
    _last_input_frame: NDArray[np.uint8] | None = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        batch = self._build_batch(observation, instruction)
        batch = self._preprocessor(batch)
        # Belt-and-suspenders device move (preprocessor sometimes returns CPU tensors).
        device_kind = self.device.split(":", 1)[0]
        for k, v in list(batch.items()):
            if not hasattr(v, "device") or getattr(v, "device", None) is None:
                continue
            if str(v.device) != self.device and not str(v.device).startswith(device_kind):
                v = v.to(self.device)
            # Cast image inputs to the bf16 backbone dtype (float32 image × bf16
            # vision weight raises "mat1 and mat2 must have the same dtype").
            # state / everything else stays float32 to match the float32 action
            # expert and the sampler's internal float32 tensors.
            if (
                self._image_dtype is not None
                and "image" in k
                and hasattr(v, "is_floating_point")
                and v.is_floating_point()
                and v.dtype != self._image_dtype
            ):
                v = v.to(self._image_dtype)
            batch[k] = v

        action_tensor = run_inference(self._policy, batch)
        action_tensor = self._postprocessor(action_tensor)
        return to_numpy_action(action_tensor)

    def close(self) -> None:
        # lerobot policies do not expose an explicit close — rely on GC.
        # Free GPU memory on CUDA to avoid leaks across configs in long-running drivers.
        if self.device.startswith("cuda"):
            import contextlib

            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Convert the eval-layer Observation into a lerobot batch dict.

        Args:
            observation: Eval-layer observation (``images`` dict + ``state``).
            instruction: Task instruction string.

        Returns:
            Dict with ``observation.images.*`` tensors, ``observation.state``,
            and ``task``. Image keys mirror the underlying lerobot convention
            so the stored ``rename_observations_processor`` step finds them.
        """
        torch = self._torch
        batch: dict[str, Any] = {"task": instruction or observation.get("task", "")}

        images = observation.get("images", {})
        # Preserve the original LIBERO key naming so the stored preprocessor
        # rename step (image → camera1, image2 → camera2) lines up.
        from openral_sim.policies._video_capture import tile_input_frames, to_input_frame

        # Record every camera the policy consumed so the debug video does not
        # collapse multi-camera input to only the last stream in the loop.
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


@POLICIES.register("smolvla")
def _build_smolvla(env_cfg: Any) -> _SmolVLAAdapter:
    """Load a SmolVLA-compatible lerobot policy from HF Hub."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    # Heavy first-import cost (torch + transformers + lerobot pulls in
    # safetensors / huggingface_hub / accelerate) is paid once per
    # process but invisible to the operator otherwise. Wrap it so the
    # 10–30 s first-call cost shows up in the timeline. Shared
    # torch+factory import lives in ``_policy_loading``; the SmolVLA
    # ``Policy`` class is pulled here so the choice doesn't leak into
    # adapters that don't need it.
    with _smolvla_phase("imports"):
        torch, make_pre_post_processors = lazy_import_lerobot("SmolVLA")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="SmolVLA")
    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError(
            "SmolVLA adapter requires a bare rSkill reference as weights_uri so the "
            "loader can fetch the manifest's `processors` block (per-file URIs for "
            "the lerobot PolicyProcessorPipeline). Explicit-scheme URIs (hf://, "
            "local://, etc.) are not accepted by the sim layer."
        )

    # ``SmolVLAPolicy.from_pretrained`` allocates the full graph on CPU,
    # downloads + mmaps the safetensors, and (on a cold HF connection)
    # HEAD-validates every cached file. Split the device transfer into
    # its own phase so a slow ``.to(device)`` is distinguishable from a
    # slow ``from_pretrained``.
    with _smolvla_phase("from_pretrained", repo=repo_id):
        policy = SmolVLAPolicy.from_pretrained(repo_id, revision=revision)
    with _smolvla_phase("to_device", device=device):
        policy = policy.to(device)
    policy.eval()
    # lerobot's from_pretrained now loads SmolVLA *mixed* — a bf16 VLM backbone
    # but a float32 action expert (the flow-matching sampler allocates float32
    # noise/time internally and needs the expert in float32). Keep that native
    # layout: don't unify (fp32-unify ~doubles memory and OOMs the reward
    # sidecar on 8 GB; bf16-unify breaks the sampler). step() instead casts only
    # the image inputs to the backbone dtype. The backbone is the model's
    # majority dtype by param count (the VLM dwarfs the small action expert).
    from collections import Counter

    dtypes = Counter(p.dtype for p in policy.parameters())
    image_dtype = dtypes.most_common(1)[0][0] if dtypes else None
    # No cast needed on a fully-float32 load (the float32 images already match).
    if image_dtype == torch.float32:
        image_dtype = None

    # `n_action_steps` precedence (apply_chunk_replay):
    # vla.extra > manifest.n_action_steps > chunk_size. SmolVLA on LIBERO
    # ships chunk_size=50 and the manifest pins n_action_steps=25
    # (closed-loop replan every half-chunk -- paper-faithful for the
    # validated 3/3 success run on libero_10/4).
    apply_chunk_replay(policy, spec.extra, manifest=manifest)
    maybe_compile_chunk_forward(policy, spec.extra, device, torch)

    # Per-file download from manifest.processors (no snapshot_download).
    # `materialize_processor_dir` symlinks the two URI targets under the
    # filenames `make_pre_post_processors` reads from a pretrained_path.
    #
    # Community finetunes routinely upload only `config.json` +
    # `model.safetensors` (no `policy_*processor.json`). On a 404 from
    # `materialize_processor_dir`, fall back to building processors from
    # the training dataset's normalization stats — manifest.dataset_uri
    # points at the LeRobotDataset (v3 `meta/stats.json` or v2.1
    # `meta/episodes_stats.jsonl`). The resulting processors are
    # functionally identical to what the trainer would have saved.
    pretrained_path: str | None = None
    dataset_stats: dict[str, dict[str, Any]] | None = None
    _missing_reason: BaseException | None = None
    with _smolvla_phase("processor_dir", repo=repo_id):
        try:
            pretrained_path = materialize_processor_dir(manifest)
        except _PROCESSOR_MISSING_EXC as exc:
            _missing_reason = exc
        except ROSConfigError as exc:
            if not _is_processor_missing(exc):
                raise
            _missing_reason = exc

        if _missing_reason is not None:
            _log.warning(
                "smolvla_processor_files_missing_falling_back_to_dataset_stats",
                repo_id=repo_id,
                dataset_uri=manifest.dataset_uri,
                exc=str(_missing_reason),
            )
            if not manifest.dataset_uri:
                raise ROSConfigError(
                    f"SmolVLA adapter: {repo_id} ships no policy_preprocessor.json "
                    "/ policy_postprocessor.json and the manifest has no "
                    "`dataset_uri` to recompute the normalization stats from. "
                    "Either upload the processor pair to the model repo or set "
                    "`dataset_uri: hf://<owner>/<dataset>` on the manifest so "
                    "the adapter can rebuild them locally."
                ) from _missing_reason
            dataset_stats = _load_lerobot_dataset_stats(manifest.dataset_uri)
    with _smolvla_phase("make_processors"):
        if pretrained_path is not None:
            # Pretrained-path branch: `call_make_processors_cached_first`
            # peeks at the preprocessor JSON for a TokenizerProcessorStep
            # and flips HF_HUB_OFFLINE for the duration of the inner call
            # so lerobot's unconditional `AutoTokenizer.from_pretrained`
            # doesn't fire 5 HEAD round-trips against a warm tokenizer
            # cache on every reload.
            preprocessor, postprocessor = call_make_processors_cached_first(
                make_pre_post_processors,
                policy.config,
                pretrained_path=pretrained_path,
            )
        else:
            # Stats-fallback branch: no preprocessor JSON on disk → no
            # tokenizer step to warm-cache. Drop straight into the lerobot
            # factory so it builds the pipeline from scratch.
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                dataset_stats=dataset_stats,
            )

    # Manifest-first resolution (no auto-derive). LIBERO checkpoints whose
    # manifests don't declare `aliases: {camera1: image, camera2: image2}`
    # will now feed `observation.images.camera1` to a policy that expects
    # `observation.images.image` -- by design, that's a loud failure that
    # tells the user to update their manifest.
    ip = resolve_image_preprocessing(manifest, spec.extra)
    state_dim = resolve_state_dim(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    return _SmolVLAAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _flip_images_180=ip.flip_180,
        _state_dim=state_dim,
        _image_dtype=image_dtype,
        _camera_keys=cam_keys,
        _cam_alias=dict(ip.aliases),
        _image_input_template=ip.input_template,
    )
