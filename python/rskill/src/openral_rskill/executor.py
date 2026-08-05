"""Action-chunk executor — overlap inference for chunk N+1 with execution of chunk N.

This module provides :class:`ChunkedExecutor`, a generic background-thread
pre-fetcher that wraps any lerobot-style policy's ``select_action`` /
``config.n_action_steps`` interface. The same executor is reused by every
chunked VLA family (SmolVLA, π0 / π0.5, ACT, Diffusion Policy, OpenVLA-OFT, …)
— the class previously lived inside ``openral_rskill.smolvla`` and
was effectively SmolVLA-private.

Architecture
------------
::

    obs → (lazy) preprocessor → batch
                                                          │
                                ┌─────────────────────────▼──────────────────────┐
                                │            ChunkedExecutor                      │
                                │                                                 │
                                │  ┌──────────────────────────────────────────┐  │
                                │  │  Background thread (daemon)              │  │
                                │  │  • _policy.predict_action_chunk(batch)   │  │
                                │  │    (pure — never touches policy queues)  │  │
                                │  │  • result → _bg_result (threading.Event) │  │
                                │  └──────────────────────────────────────────┘  │
                                │                                                 │
                                │  Foreground (step N):                           │
                                │  • pop from the EXECUTOR-owned buffer           │
                                │  • if buffer nearly empty → trigger BG          │
                                └─────────────────────────────────────────────────┘
                                                          │
                                              Action (joint_targets, 1 step)

Timing contract (RTX 4070 reference host, SmolVLA-base)
-------------------------------------------------------
- Full chunk inference: ~313 ms.
- Buffer pop: ~0 ms (a deque popleft; with a lazy ``batch`` callable, no
  observation preprocessing runs on pop steps at all).
- Pre-fetch trigger at ``prefetch_at`` remaining actions. At a 30 Hz tick the
  lead is ``prefetch_at × 33 ms`` — size it to cover the chunk inference time
  (e.g. 15 → ~500 ms of cover); a too-small value stalls the foreground at the
  chunk boundary but is never incorrect.

The class is policy-agnostic. Any policy with lerobot's
``predict_action_chunk(batch) -> Tensor`` (all ``PreTrainedPolicy``
subclasses in lerobot ≥ 0.6) and ``config.n_action_steps`` is supported.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import structlog
from openral_core.exceptions import ROSRuntimeError

from openral_rskill._vla_core import run_inference

__all__ = ["ChunkedExecutor"]

log = structlog.get_logger(__name__)


class ChunkedExecutor:
    """Overlaps GPU chunk inference with robot execution via a background thread.

    The executor calls the policy's ``predict_action_chunk`` and owns the
    per-step action buffer itself. After the first call triggers a full
    ``chunk_size``-step inference, it monitors the remaining actions in its
    buffer and automatically pre-fetches the next chunk in a background daemon
    thread when the depth falls to ``prefetch_at``.

    This means chunk N+1 is computed while the robot is executing the last
    ``prefetch_at`` steps of chunk N, keeping the observable per-step latency
    in the buffer-pop regime (~0 ms) rather than pausing for a full ~313 ms
    re-inference.

    Args:
        policy: A lerobot-style policy instance with a ``predict_action_chunk``
            method and a ``config.n_action_steps`` attribute (the chunk size).
        prefetch_at: Number of remaining buffered actions at which the
            background pre-fetch is triggered. Default 15 (≈500 ms of cover at
            a 30 Hz tick — enough for a ~313 ms SmolVLA chunk inference plus
            co-resident-thread contention headroom).

    Example:
        >>> # (doctest requires torch + lerobot — skipped in fast unit tests)
        >>> pass
    """

    def __init__(
        self,
        policy: Any = None,
        *,
        chunk_fn: Callable[[Any], Any] | None = None,
        chunk_size: int | None = None,
        prefetch_at: int = 15,
    ) -> None:
        """Initialise without starting any threads.

        Args:
            policy: lerobot-style policy with ``predict_action_chunk`` and
                ``config.n_action_steps``. Optional when ``chunk_fn`` +
                ``chunk_size`` are given; still useful alongside ``chunk_fn``
                for ``reset()`` delegation and the span's device label.
            chunk_fn: Custom chunk producer ``payload -> chunk`` for adapters
                whose forward is not a bare ``policy.predict_action_chunk``
                (extra autocast contexts, chunk-level decode/postprocessing,
                non-lerobot APIs). The payload is whatever
                :meth:`select_action` was given — the executor treats it
                opaquely, so it need not be a lerobot batch dict. The chunk
                may be a ``(batch, chunk, dof)`` tensor OR any sequence of
                per-step actions. Instrumented by the same
                ``run_inference`` seam as the default path.
            chunk_size: Actions consumed per inference. Defaults to
                ``policy.config.n_action_steps``; required with a bare
                ``chunk_fn``.
            prefetch_at: Pre-fetch trigger threshold (remaining buffered
                actions). ``0`` disables the background thread entirely —
                the executor is then a plain synchronous chunk buffer
                (identical action semantics to lerobot's internal queue,
                minus the wasted per-tick batch builds).

        Raises:
            ValueError: Neither a policy nor ``chunk_fn`` + ``chunk_size``
                was provided.
        """
        if policy is None and (chunk_fn is None or chunk_size is None):
            raise ValueError("ChunkedExecutor needs a policy, or chunk_fn together with chunk_size")
        self._policy = policy
        self._chunk_fn = chunk_fn
        self._prefetch_at = prefetch_at
        self._chunk_size: int = (
            int(chunk_size) if chunk_size is not None else int(policy.config.n_action_steps)
        )

        # Executor-owned per-step action buffer. The policy's internal queue
        # is deliberately NOT used: sharing it with a background pre-fetch is
        # what caused the double-inference race this class now avoids (see
        # ``select_action``).
        self._buffer: deque[Any] = deque()

        # Background pre-fetch state.
        self._bg_thread: threading.Thread | None = None
        self._bg_pending: bool = False  # a pre-fetch is in flight/unconsumed
        self._bg_result: Any = None  # the pre-fetched chunk tensor
        self._bg_event = threading.Event()  # set when result is ready
        self._bg_lock = threading.Lock()
        self._bg_error: Exception | None = None

        # Monotonic chunk counter used as the ``inference.chunk_index`` span
        # attribute for trace correlation.
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
        self._bg_pending = False
        self._bg_result = None
        self._bg_event.clear()
        self._bg_error = None
        self._buffer.clear()
        self._chunk_index = 0
        self._running = True
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def select_action(self, batch: dict[str, Any] | Callable[[], dict[str, Any]]) -> Any:
        """Return the next action, pre-fetching the following chunk if needed.

        On the first call after a :meth:`reset`, this triggers a full GPU
        chunk inference (``policy.predict_action_chunk``) and blocks until it
        completes (~313 ms on RTX 4070); the resulting chunk fills the
        executor-owned buffer. Subsequent calls pop from that buffer (~0 ms)
        and, when the buffer depth falls to ``prefetch_at``, launch a
        background thread computing the next chunk. When the buffer is
        exhausted, the foreground call blocks for the background result if it
        is not yet ready.

        The background thread calls ``predict_action_chunk`` — a pure chunk
        producer — never ``select_action``/``reset``, so it cannot mutate the
        policy's internal action queue. (An earlier revision called
        ``policy.reset()`` from the background thread, which cleared the live
        queue mid-chunk: the foreground then found it empty and ran a SECOND
        full inference racing the pre-fetch on the same GPU — every chunk
        boundary paid two contending forwards.)

        Args:
            batch: Pre-processed observation dict on the inference device, or
                a zero-arg callable producing one. Pass a callable to skip
                observation preprocessing entirely on buffer-pop calls — it is
                only invoked when an inference actually launches, with the
                freshest observation at that moment.

        Returns:
            Action tensor of shape ``(batch, action_dim)``.

        Raises:
            ROSRuntimeError: If the background pre-fetch thread raised, or the
                executor was stopped while waiting on a pre-fetch.
        """
        if self._buffer:
            return self._pop_and_maybe_prefetch(batch)

        if self._bg_pending:
            # Buffer drained before the pre-fetch finished — block for it.
            self._bg_event.wait()
            with self._bg_lock:
                self._bg_pending = False
                if self._bg_error is not None:
                    raise ROSRuntimeError(
                        f"VLA pre-fetch thread raised: {self._bg_error}"
                    ) from self._bg_error
                result = self._bg_result
                self._bg_result = None
            self._bg_event.clear()
            if result is None:
                raise ROSRuntimeError("ChunkedExecutor stopped while waiting on pre-fetch")
            self._extend_buffer(result)
            return self._pop_and_maybe_prefetch(batch)

        # Cold start (first call after reset) — synchronous foreground chunk.
        self._chunk_index += 1
        chunk = self._produce(self._materialize(batch), self._chunk_index, "foreground")
        self._extend_buffer(chunk)
        return self._pop_and_maybe_prefetch(batch)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _pop_and_maybe_prefetch(self, batch: Any) -> Any:
        """Pop one action; launch the pre-fetch when the buffer is low.

        Every pop routes through here — cold start, buffered, and
        bg-consume alike — so the trigger cannot be skipped by which branch
        served the action (a chunk shorter than ``prefetch_at`` triggers
        right after its first pop, the maximum lead it can give).
        """
        action = self._buffer.popleft()
        trigger = min(self._prefetch_at, self._chunk_size - 1)
        if 0 < trigger >= len(self._buffer) and self._running and not self._bg_pending:
            self._launch_prefetch(self._materialize(batch))
        return action

    def _produce(self, payload: Any, chunk_index: int, kind: str) -> Any:
        """Run one chunk inference through the instrumented ``run_inference`` seam."""
        return run_inference(
            self._policy,
            payload,
            chunk_index=chunk_index,
            kind=kind,  # type: ignore[arg-type]  # reason: executor only passes valid InferenceKind values
            chunk_size=self._chunk_size,
            method="predict_action_chunk",
            call=self._chunk_fn,
        )

    @staticmethod
    def _materialize(batch: dict[str, Any] | Callable[[], Any]) -> Any:
        """Resolve a lazy payload factory to the concrete payload."""
        return batch() if callable(batch) else batch

    def _extend_buffer(self, chunk: Any) -> None:
        """Slice one produced chunk into per-step actions.

        A torch ``(batch, chunk, dof)`` tensor follows lerobot's own queue
        layout — ``chunk.transpose(0, 1)`` yields ``chunk_size`` tensors of
        shape ``(batch, dof)``, truncated to ``chunk_size`` (lerobot's
        ``n_action_steps`` semantics). Anything else is treated as a sequence
        of ready per-step actions from a custom ``chunk_fn`` producer that
        decodes / sizes the chunk itself — NOT truncated, so ``chunk_size``
        can be nominal (trigger + telemetry) for models whose chunk length is
        only known at inference time.
        """
        if hasattr(chunk, "transpose") and hasattr(chunk, "dim"):  # torch tensor
            self._buffer.extend(chunk.transpose(0, 1)[: self._chunk_size])
        else:
            self._buffer.extend(list(chunk))

    def _launch_prefetch(self, batch: Any) -> None:
        """Start the background pre-fetch thread (no live policy state touched)."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # already running
        self._bg_event.clear()
        self._bg_error = None
        self._bg_pending = True

        self._chunk_index += 1
        prefetch_index = self._chunk_index

        def _run() -> None:
            try:
                result = self._produce(batch, prefetch_index, "prefetch")
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
