"""The dispatch helper waits on the graph clock, not the wall (#418 r4po_on3 / r4po_on4).

Isaac ran at ~0.17x: a 60 s goal deadline (graph clock) needs ~6 min of wall time, but the
helper gave up after deadline + 180 s of WALL time and printed ``status: -1`` with no
reason while the goal was still running. It now overruns only past the graph-clock
deadline plus a margin (wall time only when no /clock is published, or past a wall cap).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_validation_matrix_dispatch.py"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("vm_dispatch_under_test", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_slow_sim_is_not_an_overrun_while_its_graph_clock_is_inside_the_deadline() -> None:
    overran = _tool()._overran
    # r4po_on3: 240 s of wall, 40.7 s of graph clock into a 60 s deadline.
    assert not overran(sim_elapsed_s=40.7, wall_elapsed_s=240.0, deadline_s=60.0)
    assert overran(sim_elapsed_s=60.0 + 31.0, wall_elapsed_s=500.0, deadline_s=60.0)


def test_without_a_graph_clock_the_wall_decides_and_a_wall_cap_always_holds() -> None:
    overran = _tool()._overran
    assert not overran(sim_elapsed_s=None, wall_elapsed_s=200.0, deadline_s=60.0)
    assert overran(sim_elapsed_s=None, wall_elapsed_s=241.0, deadline_s=60.0)
    assert overran(sim_elapsed_s=10.0, wall_elapsed_s=3601.0, deadline_s=60.0)


def test_the_graph_clock_baseline_latches_on_the_first_sample_seen_while_waiting() -> None:
    """No /clock yet when the wait began used to pin the baseline to None for the whole
    wait, so a slow sim fell back to the wall limit (#421 review)."""
    sim_elapsed = _tool()._sim_elapsed
    sim0: list[float] = []
    assert sim_elapsed([], sim0) is None
    assert sim_elapsed([500.0], sim0) == 0.0
    assert sim_elapsed([540.7], sim0) == pytest.approx(40.7)
