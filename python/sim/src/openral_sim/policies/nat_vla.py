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
- the normalisation stats, from one of:

  - ``policy_extras.norm_stats`` + ``policy_extras.norm_tag`` — a MolmoAct2
    ``norm_stats.json`` (``hf://<repo>@<rev>/norm_stats.json``) and the tag of the
    embodiment's pooled stats. NaT VLA was pretrained with MolmoAct2's data pipeline,
    which normalises state and action alike by each tag's ``q01``/``q99``, so these
    are the stats the pretrained checkpoint saw (``so100_so101_molmoact2`` for the
    SO-ARM, whose ``setup_type`` is the prompt's setup string);
  - ``dataset_uri`` — a LeRobot dataset whose ``meta/stats.json`` holds ``q01``/``q99``
    for the state (quantile normalisation) and ``min``/``max`` for the action: what a
    fine-tune on that dataset normalised with;

- ``policy_extras.joint_affine`` (optional) — one ``[scale, offset]`` per state/action
  column mapping this robot's joint convention to the one the stats describe
  (``model = scale * robot + offset``; actions map back by the inverse). The SO-ARM
  pretraining data is mostly in lerobot's old calibration convention (rest pose
  shoulder_lift ≈ 190°, elbow ≈ 180°), the opposite direction on shoulder_lift from
  the current range-centred one (rest ≈ -103° / 95°);
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

#: ``owner``, ``repo[@rev]`` and at least one path segment of an ``hf://`` file URI.
_HF_FILE_MIN_PARTS = 3


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
    _flip_180: bool = False
    _last_input_frame: NDArray[np.uint8] | None = None
    # ``joint_affine`` columns: model = scale * robot + offset (identity when undeclared).
    _affine_scale: NDArray[np.float32] | None = None
    _affine_offset: NDArray[np.float32] | None = None

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
            tensor = image_to_chw_tensor(frame, torch=torch, device=self.device)
            batch[key] = torch.flip(tensor, dims=[1, 2]) if self._flip_180 else tensor
            if slot == self._camera_keys[0]:
                self._last_input_frame = to_input_frame(frame, flip_180=self._flip_180)
        state = observation.get("state")
        if state is None:
            raise ROSConfigError("nat_vla requires a proprio `state` observation.")
        state_arr: NDArray[np.float32] = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != self._state_dim:
            raise ROSConfigError(
                f"nat_vla expects a {self._state_dim}-D state, got {state_arr.shape[0]}-D."
            )
        if self._affine_scale is not None and self._affine_offset is not None:
            state_arr = np.asarray(
                self._affine_scale * state_arr + self._affine_offset, dtype=np.float32
            )
        batch["observation.state"] = torch.from_numpy(state_arr).to(self.device)
        prepared = cast("dict[str, Any]", self._preprocessor(batch))
        action = to_numpy_action(self._postprocessor(run_inference(self._policy, prepared)))
        if self._affine_scale is not None and self._affine_offset is not None:
            return np.asarray((action - self._affine_offset) / self._affine_scale, dtype=np.float32)
        return action

    def close(self) -> None:
        # The streamed chunk keeps generating in a background thread after
        # select_action returns; reset() joins it so the release frees the model.
        if self._policy is not None:
            self._policy.reset()
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
    norm_stats, norm_tag = extra.get("norm_stats"), extra.get("norm_tag")
    if (norm_stats is None) != (norm_tag is None) or not all(
        isinstance(v, str) for v in (norm_stats, norm_tag) if v is not None
    ):
        raise ROSConfigError(
            "nat_vla: policy_extras.norm_stats (hf://<repo>@<rev>/norm_stats.json) and "
            "norm_tag go together, both strings."
        )
    if norm_stats is None and manifest.dataset_uri is None:
        raise ROSConfigError(
            "nat_vla's pretrained checkpoint carries no normalisation stats: declare "
            "policy_extras.norm_stats + norm_tag, or dataset_uri."
        )
    affine = joint_affine(extra.get("joint_affine"), state_dim, action_dim)
    setup, control_mode = extra.get("setup"), extra.get("control_mode")
    if not isinstance(setup, str) or not isinstance(control_mode, str):
        raise ROSConfigError(
            "nat_vla needs policy_extras.setup and policy_extras.control_mode (prompt strings)."
        )

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, extra, scene_cameras=scene_cameras)
    image_prep = resolve_image_preprocessing(manifest, extra)
    image_keys = checkpoint_image_keys(image_prep, cam_keys)
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
        stats=f"{norm_stats}#{norm_tag}" if norm_stats is not None else manifest.dataset_uri,
        joint_affine=None if affine is None else np.stack(affine, axis=1).tolist(),
    )
    # Constructing a fresh policy whose `vlm` is the pretrained repo loads its weights
    # (strict: anything but the frozen tokenizer copy missing or extra raises).
    policy = policy_cls(config).to(device).eval()

    raw_stats: dict[str, dict[str, Any]]
    if norm_stats is not None:
        raw_stats = dict(molmoact2_tag_stats(_hf_file(str(norm_stats)), str(norm_tag)))
    else:
        from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

        raw_stats = _load_lerobot_dataset_stats(cast("str", manifest.dataset_uri))
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
        _flip_180=image_prep.flip_180,
        _affine_scale=None if affine is None else affine[0],
        _affine_offset=None if affine is None else affine[1],
    )


def joint_affine(
    value: object, state_dim: int, action_dim: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """``(scale, offset)`` arrays from ``policy_extras.joint_affine``, or ``None`` when unset.

    One ``[scale, offset]`` pair per column maps the robot's joint convention to the one
    the normalisation stats describe (``model = scale * robot + offset``); actions map
    back by the inverse, so the state and action columns must be the same joints.

    Args:
        value: The manifest value: ``None`` or a list of ``[scale, offset]`` pairs.
        state_dim: ``state_contract.dim``.
        action_dim: ``action_contract.dim``.

    Returns:
        ``(scale, offset)`` float32 arrays of length ``state_dim``, or ``None``.

    Raises:
        ROSConfigError: The value is not one non-zero-scale pair per column, or the state
            and action widths differ.

    Example:
        >>> scale, offset = joint_affine([[1, 0], [-1, 87.0]], 2, 2)
        >>> (scale * np.array([0.0, -103.0]) + offset).tolist()
        [0.0, 190.0]
    """
    if value is None:
        return None
    if state_dim != action_dim:
        raise ROSConfigError(
            f"nat_vla joint_affine maps state and action alike: their widths differ "
            f"({state_dim} vs {action_dim})."
        )
    try:
        pairs = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(f"nat_vla joint_affine must be [scale, offset] pairs: {exc}") from exc
    if pairs.shape != (state_dim, 2):
        raise ROSConfigError(
            f"nat_vla joint_affine needs {state_dim} [scale, offset] pairs, got shape "
            f"{pairs.shape}."
        )
    if not np.isfinite(pairs).all() or (pairs[:, 0] == 0).any():
        raise ROSConfigError("nat_vla joint_affine scales must be finite and non-zero.")
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def molmoact2_tag_stats(path: str, norm_tag: str) -> dict[str, dict[str, list[float]]]:
    """The NaT processors' stats from one tag of a MolmoAct2 ``norm_stats.json``.

    MolmoAct2's data pipeline (``norm_mode: q01_q99``) maps state and action alike from
    each tag's 1st/99th percentiles to [-1, 1]. NaT's processors normalise the state by
    quantiles (``q01``/``q99``) and the action by ``min``/``max``, so the action's
    ``q01``/``q99`` are passed as its ``min``/``max``: the pooled ``min``/``max`` span
    every calibration convention in the mixture (the SO-ARM tag's reach ±270°) and would
    squash the actions the checkpoint learned.

    Args:
        path: Local path of the ``norm_stats.json``.
        norm_tag: The tag under ``metadata_by_tag``.

    Returns:
        ``{"observation.state": {...}, "action": {...}}`` with ``q01``/``q99`` (state)
        and ``min``/``max`` (action) lists.

    Raises:
        ROSConfigError: The file is not MolmoAct2's ``q01_q99`` format or lacks the tag.

    Example:
        >>> import json, os, tempfile
        >>> doc = {
        ...     "norm_mode": "q01_q99",
        ...     "metadata_by_tag": {
        ...         "t": {
        ...             "state_stats": {"q01": [-1.0], "q99": [1.0]},
        ...             "action_stats": {"q01": [-2.0], "q99": [2.0]},
        ...         }
        ...     },
        ... }
        >>> with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        ...     json.dump(doc, fh)
        >>> molmoact2_tag_stats(fh.name, "t")["action"]
        {'min': [-2.0], 'max': [2.0]}
        >>> os.unlink(fh.name)
    """
    import json

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("norm_mode") != "q01_q99":
        raise ROSConfigError(
            f"nat_vla norm_stats {path}: expected MolmoAct2 norm_mode 'q01_q99', got "
            f"{doc.get('norm_mode')!r}."
        )
    meta = doc.get("metadata_by_tag", {}).get(norm_tag)
    if meta is None:
        known = sorted(doc.get("metadata_by_tag", {}))
        raise ROSConfigError(f"nat_vla norm_stats {path}: no tag {norm_tag!r} (has {known}).")
    try:
        state, action = meta["state_stats"], meta["action_stats"]
        return {
            "observation.state": {
                "q01": [float(v) for v in state["q01"]],
                "q99": [float(v) for v in state["q99"]],
            },
            "action": {
                "min": [float(v) for v in action["q01"]],
                "max": [float(v) for v in action["q99"]],
            },
        }
    except KeyError as exc:
        raise ROSConfigError(f"nat_vla norm_stats {path} tag {norm_tag!r} lacks {exc}.") from exc


def _hf_file(uri: str) -> str:
    """Local path of ``hf://<owner>/<repo>[@<rev>]/<file>`` (model repo), cache first.

    ``<rev>`` must not contain ``/`` (a commit SHA or a plain branch/tag name).
    """
    if not uri.startswith("hf://"):
        raise ROSConfigError(f"nat_vla norm_stats must be an hf:// file URI, got {uri!r}.")
    parts = uri[len("hf://") :].split("/")
    if len(parts) < _HF_FILE_MIN_PARTS:
        raise ROSConfigError(f"nat_vla norm_stats {uri!r}: expected hf://<owner>/<repo>/<file>.")
    repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
    repo, _, revision = repo.partition("@")
    from huggingface_hub import hf_hub_download

    from openral_sim.policies.molmoact2 import _hf_cached_first

    return _hf_cached_first(
        lambda: str(hf_hub_download(repo, filename, revision=revision or None)),
        repo_id=repo,
    )
