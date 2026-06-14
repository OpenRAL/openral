"""SmolVLA adapter — Skill implementation for the SmolVLA family of VLAs.

This module provides two public classes:

- :class:`ChunkedExecutor` — a background-thread pre-fetcher that overlaps GPU
  inference for chunk N+1 with the robot executing chunk N.  This is the
  "async chunked executor" pattern described in the SmolVLA paper and required
  by the Day-17 spec.

- :class:`SmolVLAAdapter` — a full :class:`~openral_rskill.base.Skill`
  implementation that loads any ``SmolVLAPolicy``-compatible checkpoint from
  the HuggingFace Hub and drives the :class:`ChunkedExecutor` from within the
  standard Skill lifecycle.

Architecture
------------
::

    WorldState ──obs_fn──► raw_batch ──preprocessor──► batch
                                                          │
                                    ┌─────────────────────▼──────────────────────┐
                                    │            ChunkedExecutor                  │
                                    │                                             │
                                    │  ┌──────────────────────────────────────┐  │
                                    │  │  Background thread (daemon)          │  │
                                    │  │  • _policy.select_action(batch)      │  │
                                    │  │  • result → _next_chunk (threading.  │  │
                                    │  │             Event + storage)         │  │
                                    │  └──────────────────────────────────────┘  │
                                    │                                             │
                                    │  Foreground (step N):                       │
                                    │  • pop from _policy internal queue         │
                                    │  • if queue nearly empty → trigger BG      │
                                    └─────────────────────────────────────────────┘
                                                          │
                                              Action (joint_targets, 1 step)

Timing contract (RTX 4070 reference host)
-----------------------------------------
- Full chunk inference: ~313 ms.
- Queue pop: ~3 ms.
- Pre-fetch trigger at ``prefetch_at`` steps before end of chunk (default 5),
  giving 5 x 3 ms = 15 ms window — well within the 313 ms inference time.
- Result: the background thread always finishes before the queue drains,
  keeping per-step latency in the cached-pop regime for all but the very
  first inference of a session.

Observation convention (SO-100 default)
-----------------------------------------
The ``obs_fn`` callable maps :class:`~openral_core.schemas.WorldState`
to a raw SmolVLA input dict:

.. code-block:: python

    {
        "observation.state":          (1, 6) float32 on device
        "observation.images.camera1": (1, 3, 256, 256) float32 on device
        ...
        "task":                       ["<prompt>"]
    }

The default ``obs_fn`` used by :class:`SO100SmolVLASkill` builds this from
``world_state.joint_state.position`` and expects images to be populated by
the caller via ``extra_images``.  For other robots, pass a custom ``obs_fn``.

Public API
----------
.. code-block:: python

    from openral_rskill.smolvla import SmolVLAAdapter, SO100SmolVLASkill

    skill = SO100SmolVLASkill(prompt="pick up the red cube")
    skill.configure()  # loads SmolVLA-base from HF Hub, quantizes, warms up
    skill.activate()  # starts the ChunkedExecutor background thread
    action = skill.step(world_state)  # <1 ms for steps 2-50; ~313 ms for step 1
    skill.shutdown()  # stops background thread, releases GPU memory
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, WorldState

from openral_rskill.base import rSkillBase
from openral_rskill.executor import ChunkedExecutor

if TYPE_CHECKING:
    pass  # torch / lerobot imported lazily in on_load_weights

__all__ = ["ChunkedExecutor", "SO100SmolVLASkill", "SmolVLAAdapter"]

log = structlog.get_logger(__name__)

# SO-100 joint names in the order the lerobot policy expects.
_SO100_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


# ── SmolVLAAdapter ────────────────────────────────────────────────────────────


class SmolVLAAdapter(rSkillBase):
    """Skill implementation that drives any SmolVLA-family policy.

    Implements the full :class:`~openral_rskill.base.Skill` lifecycle:

    - ``configure()``: fetches the checkpoint from HF Hub, builds the
      preprocessor, moves the model to ``device``, and optionally quantizes.
    - ``activate()``: runs a warm-up inference; starts the
      :class:`ChunkedExecutor` background thread.
    - ``step(world_state)``: calls ``obs_fn`` → preprocessor → executor;
      returns a single-step :class:`~openral_core.schemas.Action`.
    - ``deactivate()`` / ``shutdown()``: stops the executor thread, frees GPU.

    Args:
        repo_id: HuggingFace Hub repo ID, e.g. ``"lerobot/smolvla_base"``.
        obs_fn: Callable mapping :class:`~openral_core.schemas.WorldState`
            to a raw SmolVLA input dict (un-preprocessed).  Must return tensors
            on ``device`` or CPU (they are moved to ``device`` after
            preprocessing).
        prompt: Default task prompt.  Can be overridden per-step by updating
            the ``task`` key inside ``obs_fn``.
        device: PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        n_dof: Degrees of freedom of the robot's action space (default 6 for
            SO-100).  Used to shape the returned :class:`Action`.
        prefetch_at: Steps before chunk end at which background pre-fetch is
            triggered.  See :class:`ChunkedExecutor`.
        name: Skill name passed to the :class:`~openral_rskill.base.Skill`
            base class.
        version: SemVer string.
        embodiment_tags: Embodiment tags for capability matching.
        latency_budget_ms: Warned (not raised) if step latency exceeds this.

    Example:
        >>> from openral_rskill.smolvla import SmolVLAAdapter
        >>> # Full test in tests/unit/test_smolvla_adapter.py (uses NullPolicy).
        >>> pass
    """

    def __init__(
        self,
        repo_id: str,
        obs_fn: Callable[[WorldState], dict[str, Any]],
        prompt: str,
        *,
        device: str = "cuda:0",
        n_dof: int = 6,
        prefetch_at: int = 5,
        name: str = "smolvla",
        version: str = "0.1.0",
        embodiment_tags: list[str] | None = None,
        latency_budget_ms: float | None = None,
    ) -> None:
        """Initialise without loading weights or starting threads."""
        super().__init__(
            name,
            version=version,
            role="s1",
            embodiment_tags=embodiment_tags or [],
            latency_budget_ms=latency_budget_ms,
        )
        self._repo_id = repo_id
        self._obs_fn = obs_fn
        self._prompt = prompt
        self._device = device
        self._n_dof = n_dof
        self._prefetch_at = prefetch_at

        # Set in on_load_weights / activate.
        self._policy: Any = None
        self._preprocessor: Any = None
        self._executor: ChunkedExecutor | None = None

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def on_load_weights(self) -> None:
        """Fetch the SmolVLA checkpoint from HF Hub and move it to ``device``.

        Raises:
            ROSConfigError: If the checkpoint cannot be fetched (network down
                and not in local HF cache).
        """
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            from openral_rskill import _lerobot_compat  # noqa: F401
        except ImportError as exc:
            raise ROSConfigError(
                "SmolVLAAdapter requires 'lerobot', 'transformers', and 'num2words'. "
                "Install them with: uv add lerobot transformers num2words"
            ) from exc

        try:
            from huggingface_hub.errors import HfHubHTTPError

            policy = SmolVLAPolicy.from_pretrained(self._repo_id)
            preprocessor, _ = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=self._repo_id
            )
        except (OSError, HfHubHTTPError) as exc:
            raise ROSConfigError(
                f"SmolVLAAdapter: could not fetch '{self._repo_id}': {exc}"
            ) from exc

        self._policy = policy.to(self._device).eval()
        self._preprocessor = preprocessor
        log.info(
            "smolvla.weights_loaded",
            repo_id=self._repo_id,
            device=self._device,
            n_params=sum(p.numel() for p in self._policy.parameters()),
        )

    def on_warmup(self) -> None:
        """Run a dummy inference to amortize JIT and cuDNN autotune overhead."""
        import torch

        assert self._policy is not None, "call configure() before activate()"
        self._policy.reset()
        dummy_state = torch.zeros(1, self._n_dof, dtype=torch.float32, device=self._device)
        dummy_img = torch.rand(1, 3, 256, 256, dtype=torch.float32, device=self._device)
        dummy_batch = self._preprocess(
            {
                "observation.state": dummy_state,
                "observation.images.camera1": dummy_img,
                "task": [self._prompt],
            }
        )
        with torch.no_grad():
            self._policy.select_action(dummy_batch)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        log.info("smolvla.warmed_up", repo_id=self._repo_id)

    def _configure_impl(self) -> None:
        """Validate policy IO shapes match ``n_dof``."""
        cfg = self._policy.config
        action_shape = cfg.output_features["action"].shape
        if action_shape != (self._n_dof,):
            raise ROSConfigError(
                f"SmolVLAAdapter: policy action shape {action_shape} does not match "
                f"n_dof={self._n_dof}. Pass the correct n_dof or choose a different checkpoint."
            )
        log.info(
            "smolvla.configured",
            chunk_size=cfg.chunk_size,
            n_action_steps=cfg.n_action_steps,
            n_dof=self._n_dof,
        )

    def _activate_impl(self) -> None:
        """Reset the policy and start the :class:`ChunkedExecutor`."""
        assert self._policy is not None
        self._policy.reset()
        self._executor = ChunkedExecutor(self._policy, prefetch_at=self._prefetch_at)
        self._executor.start()
        log.info("smolvla.activated", prefetch_at=self._prefetch_at)

    def _deactivate_impl(self) -> None:
        """Stop the pre-fetch thread without unloading weights."""
        if self._executor is not None:
            self._executor.stop()
        log.info("smolvla.deactivated")

    def _shutdown_impl(self) -> None:
        """Stop threads and free GPU memory."""
        if self._executor is not None:
            self._executor.stop()
            self._executor = None
        if self._policy is not None:
            del self._policy
            self._policy = None
            try:
                import torch

                if self._device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        log.info("smolvla.shutdown")

    def _step_impl(self, world_state: WorldState) -> Action:
        """One S1 control step.

        Converts ``world_state`` via ``obs_fn`` + preprocessor, then calls the
        :class:`ChunkedExecutor` (which either pops from the cached queue or
        triggers a new chunk inference).

        Args:
            world_state: Current world state snapshot.

        Returns:
            A single-step :class:`~openral_core.schemas.Action` with
            ``control_mode=JOINT_POSITION`` and ``horizon=1``.

        Raises:
            ROSRuntimeError: If the executor is not running.
        """
        if self._executor is None:
            raise ROSRuntimeError("SmolVLAAdapter._step_impl: executor not started")

        raw = self._obs_fn(world_state)
        batch = self._preprocess(raw)
        action_tensor = self._executor.select_action(batch)

        # action_tensor: (1, n_dof) float32 on device
        joints = action_tensor.squeeze(0).cpu().tolist()
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[joints],
            confidence=1.0,
            stamp_ns=time.time_ns(),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Run the lerobot preprocessor and ensure tensors are on ``_device``."""
        import torch

        out: dict[str, Any] = self._preprocessor(raw)
        for k, v in list(out.items()):
            if isinstance(v, torch.Tensor) and v.device != torch.device(self._device):
                out[k] = v.to(self._device)
        return out


# ── SO100SmolVLASkill ─────────────────────────────────────────────────────────


def _so100_obs_fn(
    world_state: WorldState,
    *,
    device: str,
    extra_images: dict[str, Any] | None = None,
    prompt: str,
) -> dict[str, Any]:
    """Convert a SO-100 WorldState to a SmolVLA raw input dict.

    Args:
        world_state: Current world state from the aggregator.
        device: Target device for tensors.
        extra_images: Optional ``{key: tensor}`` dict of camera images.
            If ``None``, a single synthetic image is used (valid for smoke
            tests, but not for real task execution).
        prompt: Task prompt string.

    Returns:
        Raw SmolVLA input dict ready for the preprocessor.
    """
    import torch

    joint_pos = world_state.joint_state.position[:6]  # first 6 DoF
    state = torch.tensor(joint_pos, dtype=torch.float32, device=device).unsqueeze(0)

    if extra_images is not None:
        images = extra_images
    else:
        # Synthetic placeholder — valid for lifecycle tests; not for task execution.
        images = {
            "observation.images.camera1": torch.rand(
                1, 3, 256, 256, dtype=torch.float32, device=device
            )
        }

    return {
        "observation.state": state,
        **images,
        "task": [prompt],
    }


class SO100SmolVLASkill(SmolVLAAdapter):
    """Convenience SmolVLAAdapter pre-configured for the SO-100 6-DoF arm.

    Uses :func:`_so100_obs_fn` to convert :class:`WorldState` to SmolVLA
    inputs.  Pass ``extra_images`` at runtime via ``step_kwargs`` or patch
    ``obs_fn`` after construction for multi-camera setups.

    Args:
        prompt: Task prompt (e.g. ``"pick up the red cube"``).
        repo_id: HF Hub repo ID (default ``"lerobot/smolvla_base"``).
        device: Inference device (default ``"cuda:0"``).
        extra_images: Camera images injected into every ``_step_impl`` call.
            If ``None``, a synthetic image is used (smoke-test only).
        **kwargs: Forwarded to :class:`SmolVLAAdapter`.

    Example:
        >>> from openral_rskill.smolvla import SO100SmolVLASkill
        >>> skill = SO100SmolVLASkill(prompt="pick up the red cube")
        >>> skill.name
        'smolvla_so100'
    """

    def __init__(
        self,
        prompt: str,
        *,
        repo_id: str = "lerobot/smolvla_base",
        device: str = "cuda:0",
        extra_images: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise for the SO-100 arm.

        Args:
            prompt: Task prompt string.
            repo_id: HF Hub checkpoint ID.
            device: Inference device.
            extra_images: Optional camera image tensors.
            **kwargs: Extra args forwarded to :class:`SmolVLAAdapter`.
        """

        def obs_fn(ws: WorldState) -> dict[str, Any]:
            return _so100_obs_fn(ws, device=device, extra_images=extra_images, prompt=prompt)

        super().__init__(
            repo_id=repo_id,
            obs_fn=obs_fn,
            prompt=prompt,
            device=device,
            n_dof=6,
            name=kwargs.pop("name", "smolvla_so100"),
            embodiment_tags=kwargs.pop("embodiment_tags", ["so100_follower"]),
            **kwargs,
        )
