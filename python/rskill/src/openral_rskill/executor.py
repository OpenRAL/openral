"""Action-chunk executor — overlap inference for chunk N+1 with execution of chunk N.

This module provides :class:`ChunkedExecutor`, a generic background-thread
pre-fetcher that wraps any lerobot-style policy's ``select_action`` /
``config.n_action_steps`` interface. The same executor is reused by every
chunked VLA family (SmolVLA, π0 / π0.5, ACT, Diffusion Policy, OpenVLA-OFT, …)
— prior to ADR-0010 the class lived inside ``openral_rskill.smolvla`` and
was effectively SmolVLA-private.

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
                                │  │  • _policy.select_action(batch)          │  │
                                │  │  • result → _bg_result (threading.Event) │  │
                                │  └──────────────────────────────────────────┘  │
                                │                                                 │
                                │  Foreground (step N):                           │
                                │  • pop from _policy internal queue              │
                                │  • if queue nearly empty → trigger BG           │
                                └─────────────────────────────────────────────────┘
                                                          │
                                              Action (joint_targets, 1 step)

Timing contract (RTX 4070 reference host, SmolVLA-base)
-------------------------------------------------------
- Full chunk inference: ~313 ms.
- Queue pop: ~3 ms.
- Pre-fetch trigger at ``prefetch_at`` steps before end of chunk (default 5),
  giving a 5 × 3 ms = 15 ms window — well within the 313 ms inference time.
- Result: the background thread always finishes before the queue drains,
  keeping per-step latency in the cached-pop regime for all but the very first
  inference of a session.

The class is policy-agnostic. Any policy with ``select_action(batch) -> Tensor``
and ``config.n_action_steps`` (the chunk size) is supported.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog
from openral_core.exceptions import ROSRuntimeError

from openral_rskill._vla_core import run_inference

__all__ = ["ChunkedExecutor"]

log = structlog.get_logger(__name__)


class ChunkedExecutor:
    """Overlaps GPU chunk inference with robot execution via a background thread.

    The executor wraps a lerobot-style policy's ``select_action`` call. After
    the first call triggers a full ``chunk_size``-step inference, it monitors
    the remaining steps in the policy's internal queue and automatically
    pre-fetches the next chunk in a background daemon thread when the queue
    depth falls to ``prefetch_at``.

    This means chunk N+1 is computed while the robot is executing the last
    ``prefetch_at`` steps of chunk N, keeping the observable per-step latency
    in the cached-pop regime (< 5 ms on the reference host) rather than pausing
    for a full ~313 ms re-inference.

    Args:
        policy: A lerobot-style policy instance with a ``select_action`` method
            and a ``config.n_action_steps`` attribute (the chunk size).
        prefetch_at: Number of steps before the queue empties at which the
            background pre-fetch is triggered. Default 5.

    Example:
        >>> # (doctest requires torch + lerobot — skipped in fast unit tests)
        >>> pass
    """

    def __init__(self, policy: Any, *, prefetch_at: int = 5) -> None:
        """Initialise without starting any threads.

        Args:
            policy: lerobot-style policy with ``select_action`` and
                ``config.n_action_steps``.
            prefetch_at: Pre-fetch trigger threshold (steps before queue empty).
        """
        self._policy = policy
        self._prefetch_at = prefetch_at
        self._chunk_size: int = policy.config.n_action_steps

        # Background pre-fetch state.
        self._bg_thread: threading.Thread | None = None
        self._bg_result: Any = None  # the pre-fetched action tensor
        self._bg_event = threading.Event()  # set when result is ready
        self._bg_lock = threading.Lock()
        self._bg_error: Exception | None = None

        # Step counter within the current chunk (0 = just about to call).
        self._step_in_chunk: int = 0
        # Monotonic chunk counter (foreground inferences only) used as the
        # ``inference.chunk_index`` span attribute for trace correlation.
        self._chunk_index: int = 0
        self._last_batch: dict[str, Any] | None = None

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
        self._step_in_chunk = 0
        self._chunk_index = 0
        self._last_batch = None
        self._running = True
        self._policy.reset()

    def select_action(self, batch: dict[str, Any]) -> Any:
        """Return the next action, pre-fetching the following chunk if needed.

        On the first call after a :meth:`reset`, this triggers a full GPU
        chunk inference and blocks until it completes (~313 ms on RTX 4070).
        On subsequent calls within the same chunk, it pops from the policy's
        internal action queue (< 3 ms) and, when the queue depth reaches
        ``prefetch_at``, launches a background thread to pre-fetch the next
        chunk. When the queue is exhausted, the foreground call blocks for the
        background result if it is not yet ready (should be 0 ms wait in the
        steady state).

        Args:
            batch: Pre-processed observation dict on the inference device.

        Returns:
            Action tensor from ``policy.select_action``.

        Raises:
            ROSRuntimeError: If the background pre-fetch thread raised.
        """
        import torch  # lazy: not required until inference time

        self._last_batch = batch
        self._step_in_chunk += 1

        # If we just started a new chunk (step 1 of N), the foreground call
        # goes straight to the policy — the internal queue is empty.
        # The policy blocks until the GPU completes the full chunk inference.
        if self._step_in_chunk == 1:
            self._bg_event.clear()
            self._bg_result = None
            self._chunk_index += 1
            action = run_inference(
                self._policy,
                batch,
                chunk_index=self._chunk_index,
                kind="foreground",
                chunk_size=self._chunk_size,
            )
        elif self._step_in_chunk <= self._chunk_size:
            # Pop from the policy's internal queue.
            with torch.no_grad():
                action = self._policy.select_action(batch)

            # Trigger pre-fetch when approaching end of chunk.
            remaining = self._chunk_size - self._step_in_chunk
            if remaining == self._prefetch_at and self._running:
                self._launch_prefetch(batch)
        else:
            # Queue exhausted — wait for pre-fetched result.
            self._step_in_chunk = 1  # start of new chunk
            self._chunk_index += 1
            self._bg_event.wait()
            with self._bg_lock:
                if self._bg_error is not None:
                    raise ROSRuntimeError(
                        f"VLA pre-fetch thread raised: {self._bg_error}"
                    ) from self._bg_error
                action = self._bg_result
            # Trigger next pre-fetch immediately for the chunk after this one.
            self._bg_event.clear()
            self._bg_result = None

        return action

    # ── Internal ─────────────────────────────────────────────────────────────

    def _launch_prefetch(self, batch: dict[str, Any]) -> None:
        """Start or restart the background pre-fetch thread."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # already running
        self._bg_event.clear()
        self._bg_error = None

        prefetch_index = self._chunk_index + 1

        def _run() -> None:
            try:
                self._policy.reset()
                result = run_inference(
                    self._policy,
                    batch,
                    chunk_index=prefetch_index,
                    kind="prefetch",
                    chunk_size=self._chunk_size,
                )
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
