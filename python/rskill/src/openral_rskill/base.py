"""Skill — abstract base class with lifecycle state machine.

A Skill is the fundamental unit of robot behaviour in openral.  It maps
directly onto a ROS 2 Managed Node (lifecycle node) but contains no ROS
imports so it can be unit-tested without a live ROS 2 installation.

State machine
-------------
::

    unconfigured ──configure()──► inactive ──activate()──► active
                                      ▲                        │
                                  deactivate()            shutdown()
                                      │                        │
                                   active ◄──────────────── (loop)
                                                               │
                                   any ──shutdown()──► finalized
                                   any ──on_error()──► error

Sub-states (visible in :class:`~openral_core.schemas.RSkillInfo`)
-----------------------------------------------------------------------
While in ``inactive`` or ``active``, three boolean flags refine the
sub-state reported on ``/skill/<name>/info``:

- ``weights_loaded`` — set by :meth:`on_load_weights`
- ``quantized``      — set by :meth:`on_quantize`
- ``warmed_up``      — set by :meth:`on_warmup`

Concrete subclasses override the hook methods; the base class enforces the
allowed transitions and updates :class:`RSkillInfo` atomically.

Hot path
--------
:meth:`step` is the only hot-path method.  It is called by the ROS 2 action
server at the skill's control frequency.  Override :meth:`_step_impl` in
subclasses; do not override :meth:`step` directly.

Concrete subclasses implement five hooks (`_configure_impl`,
`_activate_impl`, `_deactivate_impl`, `_shutdown_impl`, `_step_impl`)
and may override the optional `on_load_weights` / `on_quantize` /
`on_warmup` hooks. See `openral_rskill.gpu_passthrough` for a
minimal real example and `openral_rskill.smolvla.SmolVLAAdapter`
for a production VLA wiring.
"""

from __future__ import annotations

import abc
import time
from typing import final

import structlog
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import Action, RSkillInfo, RSkillState, WorldState
from openral_observability import rskill_span

__all__ = ["rSkillBase"]

log = structlog.get_logger(__name__)

# Valid primary-state transitions.
# Key: current state.  Value: set of states reachable from it.
_TRANSITIONS: dict[RSkillState, frozenset[RSkillState]] = {
    RSkillState.UNCONFIGURED: frozenset({RSkillState.INACTIVE}),
    RSkillState.INACTIVE: frozenset({RSkillState.ACTIVE, RSkillState.FINALIZED}),
    RSkillState.ACTIVE: frozenset({RSkillState.INACTIVE, RSkillState.FINALIZED}),
    RSkillState.FINALIZED: frozenset(),
    RSkillState.ERROR: frozenset({RSkillState.UNCONFIGURED}),
}


class rSkillBase(abc.ABC):  # noqa: N801  # reason: rSkill is the official package-format name (CLAUDE.md §6.4); rSkillBase is its ABC
    """Abstract base class for all OpenRAL skills.

    Subclasses must implement:

    - :meth:`_configure_impl` — load config, parse manifest, etc.
    - :meth:`_activate_impl`  — final pre-execution setup (e.g. warm-up).
    - :meth:`_deactivate_impl` — pause execution without unloading weights.
    - :meth:`_shutdown_impl`  — release all resources.
    - :meth:`_step_impl`      — one inference step; returns an ``Action``.

    Subclasses may optionally override:

    - :meth:`on_load_weights` — called during ``configure`` to load model
      weights.  Sets ``info.weights_loaded = True`` on return.
    - :meth:`on_quantize`     — called after weight loading to quantize.
      Sets ``info.quantized = True`` on return.
    - :meth:`on_warmup`       — called during ``activate`` to run a dummy
      inference.  Sets ``info.warmed_up = True`` on return.

    Args:
        name: Skill name, used as the ROS 2 node name and in
            ``/skill/<name>/info``.
        version: SemVer string, forwarded to :class:`RSkillInfo`.
        role: Skill slot — ``"s0"``, ``"s1"``, or ``"s2"``.
        embodiment_tags: Embodiment tags for capability matching.
        latency_budget_ms: Maximum allowed inference latency.  Exceeded
            latency is logged as a warning but does not raise.
    """

    def __init__(
        self,
        name: str,
        *,
        version: str = "0.1.0",
        role: str = "s1",
        embodiment_tags: list[str] | None = None,
        latency_budget_ms: float | None = None,
    ) -> None:
        """Initialise the skill; does not configure or load weights."""
        self._info = RSkillInfo(
            name=name,
            version=version,
            state=RSkillState.UNCONFIGURED,
            role=role,
            embodiment_tags=embodiment_tags or [],
            latency_budget_ms=latency_budget_ms,
            stamp_ns=time.time_ns(),
        )
        log.info("skill.created", name=name, role=role, version=version)

    # ── Public read-only info ─────────────────────────────────────────────────

    @property
    def info(self) -> RSkillInfo:
        """Current :class:`~openral_core.schemas.RSkillInfo` snapshot.

        Returns a copy — mutating the returned object has no effect.
        """
        return self._info.model_copy()

    @property
    def name(self) -> str:
        """Skill name."""
        return self._info.name

    @property
    def state(self) -> RSkillState:
        """Current primary lifecycle state."""
        return self._info.state

    # ── Lifecycle transitions (final — do not override) ───────────────────────

    @final
    def configure(self) -> None:
        """Transition ``unconfigured → inactive``.

        Calls :meth:`on_load_weights`, then :meth:`on_quantize`, then
        :meth:`_configure_impl` in order.

        Raises:
            ROSRuntimeError: If the current state does not allow this
                transition.
        """
        self._require_transition(RSkillState.INACTIVE)
        log.info("rskill.configure", name=self.name)
        try:
            with rskill_span("rskill.configure", rskill_id=self.name, role=self._info.role):
                self.on_load_weights()
                self._update(weights_loaded=True)
                self.on_quantize()
                self._update(quantized=True)
                self._configure_impl()
                self._transition(RSkillState.INACTIVE)
        except Exception as exc:
            self._enter_error(str(exc))
            raise

    @final
    def activate(self) -> None:
        """Transition ``inactive → active``.

        Calls :meth:`on_warmup`, then :meth:`_activate_impl`.

        Raises:
            ROSRuntimeError: If the current state does not allow this
                transition.
        """
        self._require_transition(RSkillState.ACTIVE)
        log.info("rskill.activate", name=self.name)
        try:
            with rskill_span("rskill.activate", rskill_id=self.name, role=self._info.role):
                self.on_warmup()
                self._update(warmed_up=True)
                self._activate_impl()
                self._transition(RSkillState.ACTIVE)
        except Exception as exc:
            self._enter_error(str(exc))
            raise

    @final
    def deactivate(self) -> None:
        """Transition ``active → inactive``.

        Raises:
            ROSRuntimeError: If the current state is not ``active``.
        """
        if self._info.state is not RSkillState.ACTIVE:
            raise ROSRuntimeError(
                f"Skill '{self.name}': deactivate() requires 'active' state, "
                f"current state is '{self._info.state.value}'."
            )
        log.info("rskill.deactivate", name=self.name)
        try:
            self._deactivate_impl()
            self._transition(RSkillState.INACTIVE)
        except Exception as exc:
            self._enter_error(str(exc))
            raise

    @final
    def shutdown(self) -> None:
        """Transition any state → ``finalized``.

        Idempotent if already ``finalized``.
        """
        if self._info.state is RSkillState.FINALIZED:
            return
        log.info("rskill.shutdown", name=self.name)
        try:
            # ADR-0050 — release GPU weights on the way down so the skill
            # runner's single-resident eviction frees VRAM before the next
            # skill loads. Guarded by ``weights_loaded`` so a shutdown before
            # configure() never invokes a subclass hook that assumes loaded
            # state.
            if self._info.weights_loaded:
                self.on_unload_weights()
                self._update(weights_loaded=False)
            self._shutdown_impl()
        except Exception as exc:  # reason: shutdown must always reach finalized; log and swallow
            log.error("skill.shutdown.error", name=self.name, exc=str(exc))
        finally:
            self._transition(RSkillState.FINALIZED)

    @final
    def step(self, world_state: WorldState) -> Action | list[Action]:
        """Execute one inference step and return one or more ``Action`` chunks.

        Records wall-clock latency and logs a warning if it exceeds
        ``latency_budget_ms``.

        Args:
            world_state: Current :class:`~openral_core.schemas.WorldState`
                snapshot from the aggregator.

        Returns:
            Either a single :class:`~openral_core.schemas.Action` chunk
            (the legacy single-control-surface path used by every skill
            shipped before ADR-0028b) OR a list of :class:`Action`
            chunks (ADR-0028b multi-surface dispatch — used by skills
            whose manifest declares an ``action_contract.slots`` block;
            each slot in the manifest becomes one :class:`Action` in
            the returned list, all routed by the HAL according to their
            individual ``control_mode``).

        Raises:
            ROSRuntimeError: If the skill is not in the ``active`` state.
        """
        if self._info.state is not RSkillState.ACTIVE:
            raise ROSRuntimeError(
                f"Skill '{self.name}' is in state '{self._info.state.value}'; "
                "must be 'active' to call step()."
            )
        t0 = time.perf_counter()
        with rskill_span("rskill.execute", rskill_id=self.name, role=self._info.role):
            action = self._step_impl(world_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._update(last_inference_ms=elapsed_ms)
        budget = self._info.latency_budget_ms
        if budget is not None and elapsed_ms > budget:
            log.warning(
                "skill.step.latency_exceeded",
                name=self.name,
                elapsed_ms=round(elapsed_ms, 2),
                budget_ms=budget,
            )
        log.debug("skill.step", name=self.name, elapsed_ms=round(elapsed_ms, 2))
        return action

    # ── Hooks (override in subclasses) ────────────────────────────────────────

    def on_load_weights(self) -> None:
        """Load model weights into memory.

        Called during :meth:`configure` before :meth:`on_quantize`.
        Default is a no-op; override to load ``safetensors`` / ONNX files.
        """

    def on_unload_weights(self) -> None:
        """Release model weights from memory (ADR-0050).

        Symmetric with :meth:`on_load_weights`. Called by :meth:`shutdown`
        when ``weights_loaded`` is set (before :meth:`_shutdown_impl`), so the
        skill runner's single-resident eviction frees GPU VRAM before the next
        skill loads. Default is a no-op; override to drop model references and
        call ``torch.cuda.empty_cache()`` (or terminate an inference sidecar).
        """

    def on_quantize(self) -> None:
        """Apply quantization to loaded weights.

        Called during :meth:`configure` after :meth:`on_load_weights`.
        Default is a no-op; override to apply INT8 / INT4 / NVFP4.
        """

    def on_warmup(self) -> None:
        """Run a dummy inference to warm up the model.

        Called during :meth:`activate` before :meth:`_activate_impl`.
        Default is a no-op; override to call the model once with dummy input.
        """

    # ── Abstract implementation hooks ─────────────────────────────────────────

    @abc.abstractmethod
    def _configure_impl(self) -> None:
        """Skill-specific configuration logic.

        Called at the end of :meth:`configure`, after weights are loaded and
        quantized.  Raise any exception to abort the transition and enter
        ``error`` state.
        """

    @abc.abstractmethod
    def _activate_impl(self) -> None:
        """Skill-specific activation logic.

        Called at the end of :meth:`activate`, after warm-up.
        """

    @abc.abstractmethod
    def _deactivate_impl(self) -> None:
        """Skill-specific deactivation logic.

        Called by :meth:`deactivate`.  Weights remain loaded.
        """

    @abc.abstractmethod
    def _shutdown_impl(self) -> None:
        """Release all resources held by the skill.

        Called by :meth:`shutdown`.  Must not raise — exceptions are caught
        and logged, then the state is set to ``finalized`` regardless.
        """

    @abc.abstractmethod
    def _step_impl(self, world_state: WorldState) -> Action | list[Action]:
        """Core inference logic.

        Args:
            world_state: Current world state snapshot.

        Returns:
            An :class:`~openral_core.schemas.Action` chunk.
        """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transition(self, target: RSkillState) -> None:
        """Unconditionally move to *target* and stamp the info."""
        self._info = self._info.model_copy(update={"state": target, "stamp_ns": time.time_ns()})
        log.debug("skill.state", name=self.name, state=target.value)

    def _update(self, **kwargs: object) -> None:
        """Patch one or more RSkillInfo fields without changing state."""
        self._info = self._info.model_copy(update={**kwargs, "stamp_ns": time.time_ns()})

    def _require_transition(self, target: RSkillState) -> None:
        """Raise :class:`ROSRuntimeError` if *target* is not reachable."""
        allowed = _TRANSITIONS.get(self._info.state, frozenset())
        if target not in allowed:
            raise ROSRuntimeError(
                f"Skill '{self.name}': cannot transition from "
                f"'{self._info.state.value}' to '{target.value}'. "
                f"Allowed targets: {[s.value for s in sorted(allowed, key=lambda s: s.value)]}"
            )

    def _enter_error(self, msg: str) -> None:
        """Latch error state with a human-readable message."""
        self._info = self._info.model_copy(
            update={"state": RSkillState.ERROR, "error_msg": msg, "stamp_ns": time.time_ns()}
        )
        log.error("skill.error", name=self.name, msg=msg)
