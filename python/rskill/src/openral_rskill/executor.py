"""Buffered VLA chunk execution with optional background prefetch.

With an enabled lerobot ``RTCConfig`` the buffer becomes a Real-Time-Chunking
``ActionQueue``: a pre-fetched chunk replaces the queue tail as soon as it
lands instead of being appended behind it.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

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
        chunk_fn: Callable[..., Any] | None = None,
        chunk_size: int | None = None,
        prefetch_at: int = 15,
        rtc_config: Any = None,
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
                minus the wasted per-tick batch builds). RTC mode has no
                such synchronous form and requires ``>= 1``.
            rtc_config: lerobot ``RTCConfig``. When ``enabled``, the deque is
                replaced by lerobot's ``ActionQueue``: each prefetched chunk
                *replaces* the queue tail the moment it lands (dropping the
                actions consumed during inference), and the producer is called
                with ``inference_delay=`` / ``prev_chunk_left_over=`` so the
                policy's flow-matching guidance can blend chunks. The tail is
                truncated to ``execution_horizon`` rows; a ``prefetch_at``
                below the horizon cannot fill it, which shortens the blend and
                logs a warning at construction. Requires ``prefetch_at >= 1``
                and a producer returning a ``(1, chunk, action)`` tensor.

        Raises:
            ValueError: Neither a policy nor ``chunk_fn`` + ``chunk_size``
                was provided.
            ROSConfigError: RTC is enabled without a background pre-fetch.
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

        # ── RTC (lerobot Real-Time Chunking) state ──────────────────────────
        self._rtc_cfg = rtc_config
        self._rtc_enabled = bool(getattr(rtc_config, "enabled", False))
        self._rtc_queue: Any = None
        self._rtc_last_delay = 0
        self._rtc_horizon = 0
        if self._rtc_enabled:
            if prefetch_at < 1:
                raise ROSConfigError(
                    "ChunkedExecutor: RTC needs chunk_prefetch (prefetch_at >= 1) — "
                    "without an overlapping prefetch there is no previous-chunk tail to blend"
                )
            self._rtc_horizon = int(getattr(rtc_config, "execution_horizon", 0) or 0)
            if prefetch_at < self._rtc_horizon:
                # The tail handed to the policy is at most `prefetch_at` long, so a
                # lead shorter than the horizon silently shortens the guidance ramp.
                # A degradation, not an error — the blend still runs, over fewer steps.
                log.warning(
                    "chunked_executor.rtc_prefetch_below_horizon",
                    prefetch_at=prefetch_at,
                    execution_horizon=self._rtc_horizon,
                    effect="previous-chunk tail is shorter than the RTC guidance ramp; "
                    "chunks blend over prefetch_at steps instead of execution_horizon",
                )
            # lerobot is a heavy optional dep — deferred to RTC-enabled construction.
            from lerobot.policies.rtc import ActionQueue

            self._rtc_queue = ActionQueue(rtc_config)

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
        if self._rtc_queue is not None:
            self._rtc_queue.clear()
        self._rtc_last_delay = 0
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
        if self._rtc_enabled:
            return self._select_action_rtc(batch)

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

    # ── RTC mode ─────────────────────────────────────────────────────────────

    def _select_action_rtc(self, batch: dict[str, Any] | Callable[[], dict[str, Any]]) -> Any:
        """RTC pop: serve from the ActionQueue; the bg thread merges directly."""
        self._raise_bg_error_if_any()
        action = self._rtc_queue.get()
        if action is None:
            if self._bg_pending:
                # Drained before the pre-fetch finished — block for the merge.
                self._bg_event.wait()
                self._raise_bg_error_if_any()
                if not self._running:
                    raise ROSRuntimeError("ChunkedExecutor stopped while waiting on pre-fetch")
            # Look again whether or not we waited: the bg thread can merge between
            # the first get() and the _bg_pending read above, and the cold start
            # below *replaces* the queue — it would discard the chunk that just
            # landed and re-run inference synchronously.
            action = self._rtc_queue.get()
            if action is None:
                # Cold start (first call after reset) — no previous tail to blend.
                self._chunk_index += 1
                chunk = self._produce(
                    self._materialize(batch),
                    self._chunk_index,
                    "foreground",
                    rtc_kwargs={"inference_delay": 0, "prev_chunk_left_over": None},
                )
                self._rtc_merge(chunk, idx_before=self._rtc_queue.get_action_index())
                action = self._rtc_queue.get()
            if action is None:
                raise ROSRuntimeError("ChunkedExecutor: RTC queue empty after inference")
        trigger = min(self._prefetch_at, self._chunk_size - 1)
        if 0 < trigger >= self._rtc_queue.qsize() and self._running and not self._bg_pending:
            self._launch_prefetch(self._materialize(batch))
        return action.unsqueeze(0)

    def _rtc_merge(self, chunk: Any, *, idx_before: int) -> None:
        """Replace the queue tail with a fresh chunk; delay = actions consumed meanwhile."""
        if not (hasattr(chunk, "dim") and chunk.dim() == _CHUNK_TENSOR_RANK):
            raise ROSRuntimeError(
                "ChunkedExecutor: RTC mode needs a (batch, chunk, action) tensor producer, "
                f"got {type(chunk).__name__}"
            )
        if chunk.shape[0] != 1:
            # squeeze(0) is a no-op on a real batch, which would silently feed the
            # queue a (batch, chunk, action) tensor and consume batch rows as time.
            raise ROSRuntimeError(
                "ChunkedExecutor: RTC serves a single action stream, so the producer's "
                f"batch dimension must be 1, got {int(chunk.shape[0])}"
            )
        actions = chunk.squeeze(0).detach()
        # Ground-truth delay: how many actions the consumer popped while this
        # inference ran. Valid in wall-clock deploy AND fast-forward sim, unlike
        # a latency/control-period estimate.
        #
        # get_action_index() and merge() take the ActionQueue lock separately, so a
        # consumer pop landing between them makes lerobot's own cross-check log
        # "Indexes diff is not equal to real delay" and fall back to real_delay —
        # one action of staleness in the drop count, not a bug. Don't triage it as one.
        real_delay = max(0, self._rtc_queue.get_action_index() - idx_before)
        self._rtc_last_delay = real_delay
        self._rtc_queue.merge(actions, actions, real_delay, idx_before)

    def _raise_bg_error_if_any(self) -> None:
        """Re-raise a latched background error on the foreground thread."""
        with self._bg_lock:
            error, self._bg_error = self._bg_error, None
            if error is not None:
                self._bg_pending = False
        if error is not None:
            raise ROSRuntimeError(f"VLA pre-fetch thread raised: {error}") from error

    def _produce(
        self,
        payload: Any,
        chunk_index: int,
        kind: InferenceKind,
        rtc_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run one chunk inference through the instrumented ``run_inference`` seam."""
        return run_inference(
            self._policy,
            payload,
            chunk_index=chunk_index,
            kind=kind,
            chunk_size=self._chunk_size,
            call=self._chunk_fn,
            call_kwargs=rtc_kwargs,
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

        rtc_kwargs: dict[str, Any] | None = None
        idx_before = 0
        if self._rtc_enabled:
            idx_before = self._rtc_queue.get_action_index()
            left_over = self._rtc_queue.get_left_over()
            if left_over is not None and 0 < self._rtc_horizon < len(left_over):
                # Guidance weights are zero past the execution horizon, so the extra
                # rows only cost a host->policy copy. Normalizing here also keeps the
                # tail length independent of how early prefetch_at fires.
                left_over = left_over[: self._rtc_horizon]
            # The delay fed here is the *previous* merge's measured one, so the
            # first guided prefetch after a cold start or reset bootstraps with
            # 0 (the reset value): no merge has happened yet, so nothing has
            # been measured. It under-states the real delay for exactly one
            # chunk — the guidance simply starts freeing a few steps late — and
            # self-corrects as soon as that merge records a real delay.
            rtc_kwargs = {
                "inference_delay": self._rtc_last_delay,
                "prev_chunk_left_over": left_over,
            }

        def _run() -> None:
            try:
                result = self._produce(batch, prefetch_index, "prefetch", rtc_kwargs=rtc_kwargs)
                if self._rtc_enabled:
                    # Merge NOW — RTC's point is that the new chunk takes over
                    # immediately, not when the old buffer drains.
                    self._rtc_merge(result, idx_before=idx_before)
                    with self._bg_lock:
                        self._bg_pending = False
                else:
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
