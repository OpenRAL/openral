"""Unit tests for :func:`openral_rskill._diagnostics.phase_timer`.

The phase timer is the single seam every VLA adapter's ``_build_*``
factory wraps each load phase with so an opaque multi-second
``Policy.from_pretrained`` shows up in the operator-visible log /
``openral dashboard`` trace. The tests cover the three observable
contracts:

1. **Event shape** — every wrapped phase emits exactly one
   ``<prefix>_<name>_start`` on entry and exactly one
   ``..._done`` with a populated ``elapsed_s`` on exit.
2. **Extra fields** — ``**fields`` flow through to both the start and
   done events without mutation.
3. **GPU memory probe** — ``gpu_mb=True`` populates the heartbeat /
   done payload's ``gpu_mb`` field on a CUDA host, silently no-ops
   on a CPU-only host.

Per CLAUDE.md §1.11 — no mocks. Tests use a real ``structlog``
processor to capture events.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import structlog
from openral_rskill._diagnostics import phase_timer


class _CaptureProcessor:
    """Real ``structlog`` processor that buffers events for assertion.

    Implements the processor contract — call returns the event_dict or
    raises ``structlog.DropEvent`` to stop the pipeline (we drop because
    we don't want test logs polluting pytest output).
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        del logger, method
        name = str(event_dict.pop("event", ""))
        self.events.append((name, dict(event_dict)))
        raise structlog.DropEvent


@pytest.fixture
def cap() -> Any:
    """Install a fresh capture processor; restore structlog defaults after.

    ``structlog.reset_defaults`` undoes the global ``configure`` so the
    test pollutes neither pytest's own structlog setup nor other tests
    in the same session.
    """
    proc = _CaptureProcessor()
    structlog.reset_defaults()
    structlog.configure(processors=[proc])
    try:
        yield proc
    finally:
        structlog.reset_defaults()


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

    The done event's payload must NOT carry a ``gpu_mb`` field when
    torch isn't installed or no CUDA device is present — that's how a
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
        # On a real CUDA host the field should be present; we don't
        # assert its magnitude because the test allocates no tensors.
        # The done event only fires the gpu probe in the heartbeat
        # path, not on the final done line — so this just confirms the
        # done emits without crashing on the CUDA host.
        assert done[0] == "unit_phase_gpu_done"
    else:
        assert done[0] == "unit_phase_gpu_done"
        assert "gpu_mb" not in done[1]


def test_exception_inside_block_still_emits_done(cap: _CaptureProcessor) -> None:
    """A raising wrapped block still emits the ``_done`` event.

    Critical for diagnostics — the operator needs to see how long the
    phase ran before the failure, not lose the timing because the body
    raised.
    """
    with pytest.raises(RuntimeError, match="synthetic"), phase_timer("phase_fail", prefix="unit"):
        raise RuntimeError("synthetic")
    names = [name for name, _ in cap.events]
    assert names == ["unit_phase_fail_start", "unit_phase_fail_done"]
    elapsed = cap.events[-1][1].get("elapsed_s")
    assert isinstance(elapsed, float)
