"""Sim rollout protocol — the typed contract every scene adapter must satisfy.

A :class:`SimRollout` is what comes out of :func:`openral_sim.make_env`.
It is *gym-flavoured* (``reset``, ``step``, ``close``) but intentionally
narrower so adapters can wrap any underlying engine (gymnasium, dm_env,
custom) without leaking framework specifics into the eval layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from openral_core import SceneSpec, TaskSpec


def sim_time_ns_from_mujoco_handles(handles: tuple[Any, Any] | None) -> int | None:
    """Derive sim-time nanoseconds from a MuJoCo ``(model, data)`` handle pair.

    Shared helper for the MuJoCo-backed :class:`SimRollout` adapters
    (``robocasa`` / ``libero`` / ``metaworld`` / ``aloha`` / the native
    backends) so the ``data.time``-to-nanoseconds conversion lives in exactly
    one place (ADR-0048 Phase 1). ``mujoco.MjData.time`` is the authoritative
    elapsed simulation time in seconds, advanced by ``model.opt.timestep`` on
    every physics step; this rounds ``data.time * 1e9`` to the nearest integer
    nanosecond.

    Args:
        handles: The adapter's ``mujoco_handles()`` return — a
            ``(MjModel, MjData)`` tuple, or ``None`` when the backend exposes
            no live MuJoCo handle. ``model`` is unused (kept so callers pass the
            tuple verbatim).

    Returns:
        ``round(data.time * 1e9)`` when ``handles`` is a tuple, else ``None``
        (no MuJoCo clock available).

    Example:
        >>> sim_time_ns_from_mujoco_handles(None) is None
        True
    """
    if handles is None:
        return None
    _model, data = handles
    return round(float(data.time) * 1e9)


Observation = dict[str, Any]
"""Free-form observation dict — keys are adapter-specific.

Adapters SHOULD include at least:
    * ``"images"`` → ``dict[str, np.ndarray]`` of HWC uint8 RGB frames.
    * ``"state"``  → 1-D float32 NumPy array of proprioception.
    * ``"task"``   → str natural-language instruction.

The :class:`~openral_sim.policy.PolicyAdapter` is responsible for mapping
this dict to its own input format. This keeps the eval layer agnostic to VLA
input conventions.
"""


@dataclass
class StepResult:
    """One environment transition.

    Attributes:
        observation: New observation dict after applying the action.
        reward: Scalar reward (gym semantics).
        terminated: Whether the episode ended naturally (success / failure).
        truncated: Whether the episode hit its step budget.
        info: Adapter-specific dict; runners read ``info[task.success_key]``.
    """

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@runtime_checkable
class SimRollout(Protocol):
    """Minimal gym-style env contract used by the eval runner.

    Every scene factory in :data:`openral_sim.SCENES` must return an
    object satisfying this protocol.

    Attributes:
        scene: Scene spec the rollout was built from.
        task:  Task spec it is currently solving.

    Optional duck-typed extensions (any of, mutually exclusive per adapter):
        ``mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData] | None``
            — MuJoCo-backed adapters expose this so ``openral sim run --view``
            can open a passive ``mujoco.viewer`` window against the
            adapter's own ``MjModel`` / ``MjData``.

        ``sim_time_ns(self) -> int | None``
            — ADR-0048 Phase 1. The backend's authoritative elapsed
            simulation time in nanoseconds — the seam a sim ``/clock``
            publisher reads so the deploy-sim ROS graph runs on simulation
            time rather than wall time. Contract:

            * MuJoCo-backed adapters return ``round(MjData.time * 1e9)`` (the
              physics clock advanced by ``model.opt.timestep`` per step) via
              :func:`sim_time_ns_from_mujoco_handles`.
            * SAPIEN / ManiSkill-backed adapters derive elapsed control time
              from the live env's elapsed step counter and control timestep.
            * The value is **monotonic non-decreasing within a single
              episode**. It is NOT guaranteed monotonic across ``reset`` —
              backends that rewind ``MjData.time`` to ``0`` on reset (e.g.
              robocasa) restart their clock, so a consumer that needs
              cross-reset monotonicity maintains its own offset (see
              :meth:`openral_hal.sim_attached.SimAttachedHAL.sim_time_ns`).
            * Returns ``None`` when the backend has **no sim clock** —
              clock-less backends (PushT) and sidecars whose wire protocol does
              not carry elapsed time.

            Like the other extensions it is intentionally NOT part of the
            Protocol, so a clock-less adapter need not stub it. Callers MUST
            use ``getattr(env, "sim_time_ns", None)`` and treat both a missing
            attribute and a ``None`` return as "no sim clock" (fall back to
            wall time).

        ``viewer_render(self) -> None``
            — Adapters whose underlying engine owns the live viewer
            and needs per-step pumping (e.g. SAPIEN / ManiSkill3 envs
            constructed with ``render_mode='human'``). :class:`SimRunner`
            calls this once per applied step to pump the viewer; the
            adapter is responsible for promoting / opening the window
            on the first call and tearing it down in :meth:`close`.

        ``enable_intrinsic_viewer(self) -> None``
            — Adapters whose engine draws its own self-managed window
            and does not need per-step pumping (e.g. ``gym_pusht`` with
            ``render_mode="human"``; the engine updates the window inside
            its own ``step`` / ``reset``). :class:`SimRunner` calls this
            ONCE before the first ``reset()`` when ``--view`` is set, then
            skips both the MuJoCo viewer path AND the per-step
            ``viewer_render`` pump.

        All three extensions are intentionally *not* part of the Protocol
        so adapters do not need to stub a method they cannot honour;
        callers MUST use ``getattr(env, "<name>", None)`` and treat both
        a missing attribute and a ``None`` return as "viewer unsupported".
    """

    scene: SceneSpec
    task: TaskSpec

    def reset(self, seed: int | None = ...) -> Observation:
        """Reset the simulator and return the initial observation."""

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Apply ``action`` for one timestep and return the transition."""

    def render(self) -> NDArray[np.uint8] | None:
        """Return an HWC uint8 RGB frame, or ``None`` when rendering is unavailable."""

    def close(self) -> None:
        """Release any underlying engine resources."""


@dataclass
class EpisodeResult:
    """Outcome of one episode rollout.

    Attributes:
        success: Whether the task was completed (driven by ``task.success_key``).
        steps: Number of ``env.step()`` calls executed.
        total_reward: Sum of per-step rewards.
        max_step_reward: Maximum per-step reward observed during the episode.
            Used as the paper-faithful metric on scenes where the per-step
            reward is a "best-so-far" continuous signal (PushT reports
            coverage IoU between the T-block and its goal pose; the
            Diffusion Policy paper averages the per-rollout max across
            seeds).
        mean_step_latency_ms: Mean policy ``step()`` latency in milliseconds.
        max_step_latency_ms: Max policy ``step()`` latency in milliseconds.
        latency_budget_ms: Latency budget from the rSkill manifest, if known.
        budget_violations: Number of steps exceeding the latency budget.
        frames: Captured RGB frames (HWC uint8) when ``record_video`` is True;
            empty otherwise.
        metadata: Free-form per-run metadata propagated from
            :class:`openral_core.SimEnvironment.metadata`.
    """

    success: bool = False
    steps: int = 0
    total_reward: float = 0.0
    max_step_reward: float = 0.0
    mean_step_latency_ms: float = 0.0
    max_step_latency_ms: float = 0.0
    latency_budget_ms: float | None = None
    budget_violations: int = 0
    frames: list[NDArray[np.uint8]] = field(default_factory=list)
    # Per-step input frame the VLA actually saw (after the adapter's
    # preprocessing + crop / resize / flip). Populated only when
    # ``record_video`` is True and the adapter implements
    # ``PolicyAdapter.last_input_frame``. May be empty for scripted /
    # mock policies that don't consume images.
    vla_input_frames: list[NDArray[np.uint8]] = field(default_factory=list)
    # Per-step proprioception (1-D float32 state vector) recorded BEFORE
    # the policy step. Useful for joint-position plots in episode videos.
    joint_positions: list[NDArray[np.float32]] = field(default_factory=list)
    # Per-step action vector returned by the policy. Same length as
    # ``steps``.
    actions: list[NDArray[np.float32]] = field(default_factory=list)
    # Number of distinct camera streams the policy consumed. The runner
    # prefers the adapter's resolved camera keys and falls back to
    # ``env_cfg.vla.extra.camera_keys`` when the adapter doesn't expose them.
    # The video helper uses this to decide whether the top of the debug video
    # shows separate policy/world panels or collapses to one full-width tile.
    num_input_cameras: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable single-line summary.

        Returns:
            A compact summary string suitable for CLI output.
        """
        return (
            f"success={self.success} steps={self.steps} reward={self.total_reward:.3f} "
            f"mean_lat={self.mean_step_latency_ms:.1f}ms"
            f" budget_viol={self.budget_violations}"
        )
