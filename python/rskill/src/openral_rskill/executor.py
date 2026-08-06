"""Buffered VLA chunk execution with optional background prefetch."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import structlog
from openral_core.exceptions import ROSRuntimeError

from openral_rskill._vla_core import InferenceKind, run_inference

__all__ = ["ChunkedExecutor"]

log = structlog.get_logger(__name__)

_CHUNK_TENSOR_RANK = 3


class ChunkedExecutor:
    """Serve per-step actions from whole-chunk inference results."""

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
        self._chunk_fn = chunk_fn or getattr(policy, "predict_action_chunk", None)
        if not callable(self._chunk_fn):
            raise ValueError("ChunkedExecutor policy must expose predict_action_chunk")
        self._prefetch_at = prefetch_at
        self._chunk_size: int = (
            int(chunk_size) if chunk_size is not None else int(policy.config.n_action_steps)
        )

        self._buffer: deque[Any] = deque()

        # Background pre-fetch state.
        self._bg_thread: threading.Thread | None = None
        self._bg_pending: bool = False  # a pre-fetch is in flight/unconsumed
        self._bg_result: Any = None  # the pre-fetched chunk tensor
        self._bg_event = threading.Event()  # set when result is ready
        self._bg_lock = threading.Lock()
        self._bg_error: Exception | None = None

        self._chunk_index: int = 0

        self._running = False

    def start(self) -> None:
        """Mark the executor as running. Call after the policy is on-device."""
        self._running = True

    def stop(self) -> None:
        """Signal the background thread to stop and join it."""
        self._running = False
        self._bg_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join()

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

        A callable payload is evaluated only when an inference starts, so
        buffered actions avoid observation preprocessing.

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

        Every pop routes through this method so all branches share one trigger.
        """
        action = self._buffer.popleft()
        trigger = min(self._prefetch_at, self._chunk_size - 1)
        if 0 < trigger >= len(self._buffer) and self._running and not self._bg_pending:
            self._launch_prefetch(self._materialize(batch))
        return action

    def _produce(self, payload: Any, chunk_index: int, kind: InferenceKind) -> Any:
        """Run one chunk inference through the instrumented ``run_inference`` seam."""
        return run_inference(
            self._policy,
            payload,
            chunk_index=chunk_index,
            kind=kind,
            chunk_size=self._chunk_size,
            call=self._chunk_fn,
        )

    @staticmethod
    def _materialize(batch: dict[str, Any] | Callable[[], Any]) -> Any:
        """Resolve a lazy payload factory to the concrete payload."""
        return batch() if callable(batch) else batch

    def _extend_buffer(self, chunk: Any) -> None:
        """Slice one produced chunk into per-step actions.

        Tensor producers may return more than ``chunk_size`` actions; lerobot
        consumes the configured prefix. Custom sequence producers must return
        exactly ``chunk_size`` actions so scheduling and telemetry stay honest.
        """
        if hasattr(chunk, "transpose") and hasattr(chunk, "dim"):  # torch tensor
            if chunk.dim() != _CHUNK_TENSOR_RANK or chunk.shape[1] < self._chunk_size:
                raise ROSRuntimeError(
                    "ChunkedExecutor expected a (batch, chunk, action) tensor "
                    f"with at least {self._chunk_size} actions, got shape "
                    f"{tuple(chunk.shape)!r}"
                )
            actions = list(chunk.transpose(0, 1)[: self._chunk_size])
        else:
            actions = list(chunk)
            if len(actions) != self._chunk_size:
                raise ROSRuntimeError(
                    f"ChunkedExecutor expected {self._chunk_size} actions, "
                    f"producer returned {len(actions)}"
                )
        self._buffer.extend(actions)

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
