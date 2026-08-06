"""Action-chunk executor — overlap inference for chunk N+1 with execution of chunk N.

This module provides :class:`ChunkedExecutor`, a background-thread pre-fetcher
for lerobot action-chunk policies exposing ``predict_action_chunk`` and
``config.n_action_steps``.

Architecture
------------
::

    obs → preprocessor → batch
                                                          │
                                ┌─────────────────────────▼──────────────────────┐
                                │            ChunkedExecutor                      │
                                │                                                 │
                                │  ┌──────────────────────────────────────────┐  │
                                │  │  Background thread (daemon)              │  │
                                │  │  • _policy.predict_action_chunk(batch)   │  │
                                │  │  • result → _bg_result (threading.Event) │  │
                                │  └──────────────────────────────────────────┘  │
                                │                                                 │
                                │  Foreground (step N):                           │
                                │  • pop from executor-owned action deque         │
                                │  • if queue nearly empty → trigger BG           │
                                └─────────────────────────────────────────────────┘
                                                          │
                                              Action (joint_targets, 1 step)

Timing contract (RTX 4070 reference host, SmolVLA-base)
-------------------------------------------------------
- Full chunk inference: ~313 ms.
- Queue pop: ~3 ms.
- Pre-fetch trigger at ``prefetch_at`` steps before end of chunk (default 20),
  giving ~667 ms at a 30 Hz controller — enough to cover the measured
  313–600 ms chunk inference without a boundary pause.
- Result: the background thread always finishes before the queue drains,
  keeping per-step latency in the cached-pop regime for all but the very first
  inference of a session.

The executor owns its action deque. It calls the lerobot
``predict_action_chunk`` surface directly; it never resets or consumes the
policy's internal ``select_action`` queue from two threads.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_rskill._vla_core import InferenceKind, run_inference

__all__ = ["ChunkedExecutor"]

log = structlog.get_logger(__name__)


class ChunkedExecutor:
    """Overlaps GPU chunk inference with robot execution via a background thread.

    The executor calls ``predict_action_chunk`` and owns the resulting action
    deque. The foreground only pops tensors from that deque while the background
    computes the next independent chunk. This is deliberately separate from
    lerobot's mutable ``select_action`` queue: resetting that shared queue in a
    prefetch thread reordered live robot commands.

    This means chunk N+1 is computed while the robot is executing the last
    ``prefetch_at`` steps of chunk N, keeping the observable per-step latency
    in the cached-pop regime (< 5 ms on the reference host) rather than pausing
    for a full ~313 ms re-inference.

    Args:
        policy: A lerobot-style action-chunk policy exposing
            ``predict_action_chunk(batch)`` and ``config.n_action_steps``.
        prefetch_at: Number of remaining steps at which background prefetch
            starts. Default 20, providing ~667 ms at a 30 Hz control rate.

    Example:
        >>> # (doctest requires torch + lerobot — skipped in fast unit tests)
        >>> pass
    """

    def __init__(self, policy: Any, *, prefetch_at: int = 20) -> None:
        """Initialise without starting any threads.

        Args:
            policy: lerobot-style policy with ``predict_action_chunk`` and
                ``config.n_action_steps``.
            prefetch_at: Pre-fetch trigger threshold (steps before queue empty).
        """
        self._policy = policy
        self._chunk_size: int = policy.config.n_action_steps
        if prefetch_at < 0:
            raise ROSConfigError(f"prefetch_at must be >= 0; got {prefetch_at}.")
        self._prefetch_at = min(prefetch_at, max(0, self._chunk_size - 1))
        if self._prefetch_at != prefetch_at:
            log.warning(
                "chunked_executor.prefetch_clamped",
                requested=prefetch_at,
                applied=self._prefetch_at,
                chunk_size=self._chunk_size,
            )
        self._actions: deque[Any] = deque()

        # Background pre-fetch state.
        self._bg_thread: threading.Thread | None = None
        self._bg_result: Any = None  # the pre-fetched action tensor
        self._bg_event = threading.Event()  # set when result is ready
        self._bg_lock = threading.Lock()
        self._bg_error: Exception | None = None

        # Monotonic index of the chunk currently being replayed.
        self._chunk_index: int = 0

        self._running = False

    def start(self) -> None:
        """Mark the executor as running. Call after the policy is on-device."""
        self._running = True

    def stop(self) -> None:
        """Signal the background thread to stop and join it."""
        self._running = False
        # Unblock any waiting join.
        self._bg_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)

    def reset(self) -> None:
        """Reset the executor state (e.g. between episodes)."""
        self.stop()
        self._bg_thread = None
        self._bg_result = None
        self._bg_event.clear()
        self._bg_error = None
        self._actions.clear()
        self._chunk_index = 0
        self._running = True
        self._policy.reset()

    def select_action(self, batch: dict[str, Any]) -> Any:
        """Return the next action, pre-fetching the following chunk if needed.

        The first call computes a full chunk. Subsequent calls pop the
        executor-owned deque; when ``prefetch_at`` actions remain, a background
        thread computes the next chunk from that tick's observation. At the
        boundary, the completed chunk replaces the empty deque.

        Args:
            batch: Pre-processed observation dict on the inference device.

        Returns:
            Action tensor from ``policy.select_action``.

        Raises:
            ROSRuntimeError: If the background pre-fetch thread raised.
        """
        if not self._actions:
            self._load_next_chunk(batch)

        action = self._actions.popleft()
        if self._prefetch_at > 0 and len(self._actions) == self._prefetch_at and self._running:
            self._launch_prefetch(batch)
        return action

    # ── Internal ─────────────────────────────────────────────────────────────

    def _infer_chunk(
        self,
        batch: dict[str, Any],
        *,
        index: int,
        kind: InferenceKind,
    ) -> Any:
        """Run one full chunk inference without touching the policy action queue."""
        return run_inference(
            self._policy,
            batch,
            chunk_index=index,
            kind=kind,
            chunk_size=self._chunk_size,
            method_name="predict_action_chunk",
            synchronize=True,
        )

    def _load_next_chunk(self, batch: dict[str, Any]) -> None:
        """Fill the foreground deque from prefetch, or infer synchronously."""
        next_index = self._chunk_index + 1
        if self._bg_thread is None:
            chunk = self._infer_chunk(batch, index=next_index, kind="foreground")
        else:
            self._bg_event.wait()
            with self._bg_lock:
                error = self._bg_error
                chunk = self._bg_result
            self._bg_thread = None
            self._bg_event.clear()
            self._bg_result = None
            self._bg_error = None
            if error is not None:
                raise ROSRuntimeError(f"VLA pre-fetch thread raised: {error}") from error
            if chunk is None:
                raise ROSRuntimeError("VLA pre-fetch stopped without producing an action chunk.")

        try:
            actions = chunk.transpose(0, 1)[: self._chunk_size]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ROSRuntimeError(
                "VLA predict_action_chunk must return shape (batch, steps, action_dim)."
            ) from exc
        if len(actions) == 0:
            raise ROSRuntimeError("VLA predict_action_chunk returned an empty action chunk.")
        self._actions.extend(actions)
        self._chunk_index = next_index

    def _launch_prefetch(self, batch: dict[str, Any]) -> None:
        """Start or restart the background pre-fetch thread."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # already running
        self._bg_event.clear()
        self._bg_error = None

        prefetch_index = self._chunk_index + 1

        def _run() -> None:
            try:
                result = self._infer_chunk(batch, index=prefetch_index, kind="prefetch")
                with self._bg_lock:
                    self._bg_result = result
            except Exception as exc:  # reason: propagate to foreground via event
                with self._bg_lock:
                    self._bg_error = exc
            finally:
                self._bg_event.set()

        self._bg_thread = threading.Thread(target=_run, daemon=True)
        self._bg_thread.start()
        log.debug("chunked_executor.prefetch_launched", prefetch_at=self._prefetch_at)
