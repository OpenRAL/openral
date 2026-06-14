"""High-precision cadence helpers for the inference runner.

The :class:`~openral_runner.InferenceRunner` foreground loop ticks at
``RobotEnvironment.rate_hz`` (default 30 Hz to match
:class:`~openral_world_state.WorldStateAggregator`). To hit that cadence
on hosts where ``time.sleep`` has ~1-2 ms jitter, the runner uses
:func:`precise_sleep` — a hybrid that delegates the bulk of the wait to
``time.sleep`` and then busy-waits the final millisecond on
``time.perf_counter()``. This is the same shape lerobot uses in
``src/lerobot/scripts/lerobot_record.py:record_loop``.

The helper is intentionally tiny; the runner's PR C deadline-overrun policy
(``warn`` / ``drop`` / ``raise``) is applied separately around it.
"""

from __future__ import annotations

import time

__all__ = ["precise_sleep", "sleep_until"]

# Coarse-grain hand-off threshold: anything shorter than this is busy-waited
# end-to-end (avoids the ``time.sleep`` minimum quantum on Linux which can
# overshoot by ~1 ms on a non-RT kernel). 1 ms is the same value lerobot
# uses (private ``BUSY_LOOP_THRESHOLD`` in their record loop).
_BUSY_LOOP_THRESHOLD_S: float = 1e-3


def precise_sleep(duration_s: float) -> None:
    """Sleep for ``duration_s`` seconds with sub-millisecond accuracy.

    Uses ``time.sleep`` for the bulk of the wait (CPU-friendly) and busy-waits
    on ``time.perf_counter()`` for the final ~1 ms (cadence-accurate).
    Mirrors lerobot's ``precise_sleep`` shape — see
    ``huggingface/lerobot:src/lerobot/scripts/lerobot_record.py``.

    Args:
        duration_s: Wall-clock duration to wait, in seconds. Non-positive
            values return immediately.

    Example:
        >>> import time
        >>> t0 = time.perf_counter()
        >>> precise_sleep(0.01)
        >>> elapsed = time.perf_counter() - t0
        >>> 0.009 < elapsed < 0.02
        True
    """
    if duration_s <= 0:
        return
    deadline = time.perf_counter() + duration_s
    coarse = duration_s - _BUSY_LOOP_THRESHOLD_S
    if coarse > 0:
        time.sleep(coarse)
    # Busy-wait the final ~1 ms.
    while time.perf_counter() < deadline:
        pass


def sleep_until(deadline_perf_counter_s: float) -> None:
    """Sleep until ``time.perf_counter() >= deadline_perf_counter_s``.

    Convenience wrapper around :func:`precise_sleep` that takes an absolute
    monotonic deadline instead of a relative duration. Useful for the
    rate-limited loop pattern::

        deadline = time.perf_counter()
        while running:
            tick()
            deadline += 1.0 / rate_hz
            sleep_until(deadline)

    Args:
        deadline_perf_counter_s: Target value of ``time.perf_counter()`` to
            wait for.

    Example:
        >>> import time
        >>> target = time.perf_counter() + 0.005
        >>> sleep_until(target)
        >>> time.perf_counter() >= target
        True
    """
    remaining = deadline_perf_counter_s - time.perf_counter()
    if remaining > 0:
        precise_sleep(remaining)
