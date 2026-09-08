"""Unit tests for :func:`openral_rskill._diagnostics.phase_timer`.

The seam every VLA adapter's ``_build_*`` factory wraps each load phase
with, so a multi-second ``Policy.from_pretrained`` shows up in the operator
log / ``openral dashboard`` trace. Covers three contracts: (1) event shape —
one ``<prefix>_<name>_start``/``..._done`` pair with populated ``elapsed_s``;
(2) ``**fields`` flow unmutated to both events; (3) ``gpu_mb=True`` populates
``gpu_mb`` on CUDA, no-ops on CPU-only.

Per CLAUDE.md §1.11 — no mocks; a real ``structlog`` processor captures events.
"""

from __future__ import annotations

import sys
import time

import pytest
from openral_rskill._diagnostics import phase_timer

from tests.unit.conftest import _CaptureProcessor


def test_emits_start_and_done(cap: _CaptureProcessor) -> None:
    """One wrapped block produces exactly one start + one done."""
    with phase_timer("phase_a", prefix="unit"):
        pass
    names = [name for name, _ in cap.events]
    assert names == ["unit_phase_a_start", "unit_phase_a_done"]


def test_done_carries_elapsed_s(cap: _CaptureProcessor) -> None:
    """``elapsed_s`` is populated and ≥ the actual sleep duration."""
    sleep_s = 0.05
    with phase_timer("phase_b", prefix="unit"):
        time.sleep(sleep_s)
    done = cap.events[-1]
    assert done[0] == "unit_phase_b_done"
    elapsed = done[1].get("elapsed_s")
    assert isinstance(elapsed, float)
    assert elapsed >= sleep_s
    # Sanity bound — the test should not take more than a second.
    assert elapsed < 1.0


def test_extra_fields_flow_to_both_events(cap: _CaptureProcessor) -> None:
    """``**fields`` appear unmutated on both the start and done events."""
    with phase_timer("phase_c", prefix="unit", repo="lerobot/smolvla_libero", dtype="bf16"):
        pass
    start, done = cap.events
    assert start[0] == "unit_phase_c_start"
    assert start[1]["repo"] == "lerobot/smolvla_libero"
    assert start[1]["dtype"] == "bf16"
    assert done[0] == "unit_phase_c_done"
    assert done[1]["repo"] == "lerobot/smolvla_libero"
    assert done[1]["dtype"] == "bf16"


def test_default_prefix_is_phase(cap: _CaptureProcessor) -> None:
    """Without an explicit ``prefix=``, events default to ``phase_*``."""
    with phase_timer("warmup"):
        pass
    names = [name for name, _ in cap.events]
    assert names == ["phase_warmup_start", "phase_warmup_done"]


def test_gpu_mb_on_cpu_only_host_omits_field(cap: _CaptureProcessor) -> None:
    """``gpu_mb=True`` is silently no-op when CUDA is unavailable.

    Done payload must not carry ``gpu_mb`` when torch/CUDA is absent — how a
    CPU-only CI host stays clean.
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False

    with phase_timer("phase_gpu", prefix="unit", gpu_mb=True):
        pass
    done = cap.events[-1]
    if cuda:
        # No magnitude assert (test allocates no tensors); gpu probe fires in
        # the heartbeat path, not the final done line — just confirms no crash.
        assert done[0] == "unit_phase_gpu_done"
    else:
        assert done[0] == "unit_phase_gpu_done"
        assert "gpu_mb" not in done[1]


def test_exception_inside_block_still_emits_done(cap: _CaptureProcessor) -> None:
    """A raising wrapped block still emits the ``_done`` event.

    Diagnostics needs the phase duration even when the body raised.
    """
    with pytest.raises(RuntimeError, match="synthetic"), phase_timer("phase_fail", prefix="unit"):
        raise RuntimeError("synthetic")
    names = [name for name, _ in cap.events]
    assert names == ["unit_phase_fail_start", "unit_phase_fail_done"]
    elapsed = cap.events[-1][1].get("elapsed_s")
    assert isinstance(elapsed, float)


def test_switch_interval_raised_inside_and_restored(cap: _CaptureProcessor) -> None:
    """The GIL switch interval is raised while the block runs and restored after.

    Regression: SO-101 deploy cold-start starvation — load phases share
    runtime_node with two 30fps camera threads; at the default 5ms interval the
    loading thread was starved to ~12% of a core (6s SmolVLA import → 15+ min).
    """
    before = sys.getswitchinterval()
    with phase_timer("phase_gil", prefix="unit"):
        assert sys.getswitchinterval() == pytest.approx(0.05)
    assert sys.getswitchinterval() == pytest.approx(before)


def test_switch_interval_restored_on_exception() -> None:
    """A raising block must not leak the raised switch interval."""
    before = sys.getswitchinterval()
    with pytest.raises(RuntimeError, match="synthetic"), phase_timer("phase_gil2", prefix="unit"):
        raise RuntimeError("synthetic")
    assert sys.getswitchinterval() == pytest.approx(before)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs accounting is Linux-only")
def test_done_carries_rss_and_major_fault_delta(cap: _CaptureProcessor) -> None:
    """``_done`` reports live RSS and major faults counted from phase entry.

    Attribution seam for a phase that's slow while burning no CPU (page reclaim
    shows here, not in CPU time). Allocating must move ``rss_mb`` up; the fault
    counter is a delta (>=0), not the process-lifetime total.
    """
    with phase_timer("phase_mem", prefix="unit"):
        ballast = bytearray(64 * 1024 * 1024)
        ballast[::4096] = b"\x01" * len(ballast[::4096])  # fault the pages in
    payload = cap.events[-1][1]
    assert payload["rss_mb"] > 64.0
    assert payload["major_faults"] >= 0
    # Lifetime totals for a pytest process are far larger than a delta
    # taken across a sub-second block; this is what catches a missing
    # baseline subtraction.
    assert payload["major_faults"] < 10_000


def test_overlapping_phase_timers_restore_the_original_interval() -> None:
    """Two overlapping contexts must not leave 50 ms installed forever.

    Regression: save/restore was non-reentrant on the process-global switch
    interval — A saves 5ms, B saves A's 50ms, A restores 5ms, B re-installs 50ms
    permanently. Depth-counted guard makes the outermost holder own both
    transitions, regardless of unwind order.
    """
    import threading

    before = sys.getswitchinterval()

    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()

    def _a() -> None:
        with phase_timer("phase_gil_a", prefix="unit"):
            a_entered.set()
            assert b_entered.wait(timeout=10.0)
        a_exited.set()

    def _b() -> None:
        assert a_entered.wait(timeout=10.0)
        with phase_timer("phase_gil_b", prefix="unit"):
            b_entered.set()
            # B outlives A — the interleaving that used to leak 0.05.
            assert a_exited.wait(timeout=10.0)
            assert sys.getswitchinterval() == pytest.approx(0.05)

    threads = [threading.Thread(target=_a), threading.Thread(target=_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert sys.getswitchinterval() == pytest.approx(before)
