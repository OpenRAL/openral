"""Tier-C critic progress-stall watchdog for ``/openral/failure/critic`` (R3).

The OpenRAL failure bus reserves ``/openral/failure/critic`` for **Tier-C**
triggers (ADR-0018 §F3 + the 2026-05-25 amendment failure-tier taxonomy:
``safety→A``, ``hal/sensor/rskill/wam→B``, ``critic→C``). Until now that topic
had no default producer, so a robot whose task progress silently *stalls* —
the policy keeps emitting action chunks but the scene stops moving toward the
goal — emitted no structured signal and the S2 reasoner could not react.

This module ships the **decision core**: :class:`CriticWatchdog`, a pure,
import-safe state machine (no ``rclpy``) that consumes a stream of per-frame
progress/critic scores and decides *when* a stall has occurred. It emits the
real :class:`openral_core.CriticEvidence` (it does **not** invent a schema) so
a thin ROS node can publish it on the bus unchanged.

Stall semantics (deterministic, fully covered by ``tests/test_critic_watchdog.py``):

- The watchdog keeps a running **best** score (the highest seen since the last
  reset or recovery) and a consecutive-**stall** counter.
- An observation counts as **progress** iff ``score > best + min_delta`` — it
  strictly beats the running best by more than ``min_delta``. Progress updates
  ``best``, zeroes the stall counter, and clears the fired latch.
- An observation with ``score >= threshold`` is a **recovery**: it zeroes the
  counter and clears the latch (and updates ``best`` when it is also a new
  best). The task is making/holding acceptable progress, so no stall is owed.
- Otherwise (``score < threshold`` and not progress) the observation is a
  **stall** and increments the counter.
- When the counter reaches ``stall_patience`` consecutive stalls **and** the
  watchdog is not already latched, :meth:`observe` returns one
  :class:`CriticEvidence` and **latches** — subsequent stalled observations
  return ``None`` to avoid spamming the bus. The latch clears only on progress
  or recovery (above threshold), or on :meth:`reset`.

Intended wiring (the explicit next increment — **not** in this commit):
a ROS node subscribes to the Robometer reward monitor's per-frame progress
signal (ADR-0057 — the Robometer-4B reward rSkill emits progress/success
∈ [0, 1]), calls :meth:`observe` each frame, and on a non-``None`` return
publishes via
``FailureBusPublisher(node, FailureSource.CRITIC).publish(kind=KIND_CRITIC,
severity=SEVERITY_FAIL, evidence=<that CriticEvidence>)``
(see :mod:`openral_observability.failure_bus`). The ``reasoner_node`` then maps
the resulting ``/openral/failure/critic`` (FAIL) event onto a forced Tier-C
tick — ``ReasonerCore.tick(..., force=True, tier="C")`` — which is already
supported (the tick stamps ``reasoner.tier`` on its OTel span). The node calls
:meth:`reset` whenever the reasoner context shifts (new operator prompt / new
task), mirroring ``ReasonerCore.reset_kind_streak``. Building that node and its
reward subscription is the documented follow-up.

Example:
    >>> from openral_reasoner import CriticWatchdog
    >>> wd = CriticWatchdog(
    ...     critic_id="OpenRAL/rskill-robometer-4b",
    ...     threshold=0.8,
    ...     stall_patience=3,
    ... )
    >>> wd.observe(0.4) is None  # stall 1 (below threshold, sets best)
    True
    >>> wd.observe(0.4) is None  # stall 2
    True
    >>> evidence = wd.observe(0.4)  # stall 3 → fire
    >>> evidence.kind, evidence.critic_id, evidence.score, evidence.threshold
    ('critic', 'OpenRAL/rskill-robometer-4b', 0.4, 0.8)
    >>> wd.observe(0.4) is None  # latched — no repeat fire
    True
"""

from __future__ import annotations

from openral_core import CriticEvidence


class CriticWatchdog:
    """Progress-stall decision core for the Tier-C ``critic`` failure source.

    Pure logic and import-safe (no ``rclpy``): feed one score per frame via
    :meth:`observe`; it returns a :class:`~openral_core.CriticEvidence` exactly
    once when a stall trips, then latches until progress recovers. See the
    module docstring for the precise, deterministic stall semantics and the
    intended ROS wiring.

    Attributes:
        critic_id: Identifier of the upstream critic (e.g. the Robometer reward
            rSkill id) stamped onto every emitted :class:`CriticEvidence`.
        threshold: Pass threshold; observations at or above it are recoveries.
        stall_patience: Consecutive stalled observations required to fire.
        min_delta: Minimum strict improvement over the running best for an
            observation to count as progress.
    """

    __slots__ = (
        "_best",
        "_critic_id",
        "_latched",
        "_min_delta",
        "_stall_count",
        "_stall_patience",
        "_threshold",
    )

    def __init__(
        self,
        critic_id: str,
        threshold: float,
        stall_patience: int,
        *,
        min_delta: float = 0.0,
    ) -> None:
        """Configure the watchdog.

        Args:
            critic_id: Identifier of the upstream critic; copied onto every
                emitted :class:`CriticEvidence`.
            threshold: Pass threshold in the critic's native range. An
                observation ``>= threshold`` is a recovery (no stall owed).
            stall_patience: Number of consecutive stalled observations before
                firing. Must be ``>= 1``.
            min_delta: An observation counts as progress only when it beats the
                running best by **more than** this. Must be ``>= 0.0``.

        Raises:
            ValueError: If ``stall_patience < 1`` or ``min_delta < 0.0``.
        """
        if stall_patience < 1:
            raise ValueError(f"stall_patience must be >= 1, got {stall_patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0.0, got {min_delta}")
        self._critic_id = critic_id
        self._threshold = threshold
        self._stall_patience = stall_patience
        self._min_delta = min_delta
        self._best: float | None = None
        self._stall_count: int = 0
        self._latched: bool = False

    @property
    def critic_id(self) -> str:
        """Identifier of the upstream critic stamped onto emitted evidence."""
        return self._critic_id

    @property
    def threshold(self) -> float:
        """Pass threshold; observations at or above it are recoveries."""
        return self._threshold

    @property
    def stall_patience(self) -> int:
        """Consecutive stalled observations required to fire."""
        return self._stall_patience

    @property
    def min_delta(self) -> float:
        """Minimum strict improvement over the running best to count as progress."""
        return self._min_delta

    def observe(self, score: float) -> CriticEvidence | None:
        """Feed one progress/critic score and decide whether to fire.

        Args:
            score: One frame's progress/critic score (e.g. a Robometer
                progress estimate ∈ [0, 1]) in the critic's native range.

        Returns:
            A :class:`~openral_core.CriticEvidence` carrying this
            :attr:`critic_id`, ``score`` and :attr:`threshold` exactly once,
            when ``stall_patience`` consecutive stalled observations accumulate
            while not already latched; otherwise ``None``. See the module
            docstring for the full stall semantics.
        """
        # Progress requires a prior best to beat; the very first observation
        # establishes the baseline and is never itself "progress".
        is_progress = self._best is not None and score > self._best + self._min_delta
        is_recovery = score >= self._threshold

        if is_progress or is_recovery:
            if self._best is None or score > self._best:
                self._best = score
            self._stall_count = 0
            self._latched = False
            return None

        # Stalled: below threshold and not improving beyond min_delta.
        if self._best is None or score > self._best:
            self._best = score
        self._stall_count += 1
        if self._latched or self._stall_count < self._stall_patience:
            return None
        self._latched = True
        return CriticEvidence(
            critic_id=self._critic_id,
            score=score,
            threshold=self._threshold,
        )

    def reset(self) -> None:
        """Clear all state when the reasoner context shifts / a new task starts.

        Mirrors :meth:`ReasonerCore.reset_kind_streak`'s rationale. Forgets the
        running best, zeroes the stall counter, and clears the fired latch, so
        the next stall starts a fresh patience countdown.
        """
        self._best = None
        self._stall_count = 0
        self._latched = False
