"""runtime_node VLA pre-warm ordering contract — structural regression guard.

``_prewarm_vla_imports()`` imports ``torch`` + ``lerobot.policies.factory``
while the process is still single-threaded. The *ordering* is the entire
fix, not the import itself: ``transformers``' import runs
``importlib.metadata.packages_distributions()``, which does a ``stat()``
per file of every installed distribution, and every ``stat`` releases the
GIL for a 30 fps camera reader thread to snatch. Measured live on the
SO-101 bench, the identical import chain takes

* 23 s in a quiet process, and
* 8+ minutes (never finished) once two readers are up.

The convoy is on GIL *re-acquisition after a syscall*, so neither
``UV_COMPILE_BYTECODE=1`` nor ``phase_timer``'s raised switch interval
rescues it — running before any reader thread exists is the only fix that
works. A refactor that moves the pre-warm below
``open_deploy_sensor_readers`` silently reinstates the 8-minute stall, and
it would do so only on real hardware, where it is most expensive to
discover.

Same static-shape approach as ``test_runtime_node_sigint_shape.py``: parse
the script and assert the contract, so the guard needs no ROS graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_NODE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "scripts" / "runtime_node"

_PREWARM = "_prewarm_vla_imports"
_OPEN_READERS = "open_deploy_sensor_readers"


def _parse() -> ast.Module:
    """Parse the runtime_node script as Python. Fail loudly if missing."""
    assert _RUNTIME_NODE.is_file(), f"runtime_node not found at {_RUNTIME_NODE}"
    return ast.parse(_RUNTIME_NODE.read_text(), filename=str(_RUNTIME_NODE))


def _call_lines(tree: ast.Module, func_name: str) -> list[int]:
    """Line numbers of every call to ``func_name`` in the module."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    ]


def test_prewarm_runs_before_the_first_camera_reader_is_opened() -> None:
    """The pre-warm call must precede every ``open_deploy_sensor_readers`` call."""
    tree = _parse()
    prewarm_lines = _call_lines(tree, _PREWARM)
    reader_lines = _call_lines(tree, _OPEN_READERS)

    assert prewarm_lines, f"{_PREWARM}() is never called in runtime_node"
    assert reader_lines, (
        f"{_OPEN_READERS}() is never called in runtime_node — if the camera leg "
        "moved, this guard needs to move with it."
    )
    assert max(prewarm_lines) < min(reader_lines), (
        f"{_PREWARM}() must run BEFORE {_OPEN_READERS}() — pre-warm at lines "
        f"{prewarm_lines}, readers opened at {reader_lines}. Importing the VLA "
        "stack once a 30 fps reader thread exists stretches it from 23 s to 8+ "
        "minutes via GIL convoy on syscall re-acquisition."
    )


def test_prewarm_is_not_gated_on_deploy_config() -> None:
    """The pre-warm must run on the sim path too, not only on real deploys.

    Sim has no in-process reader threads and so no convoy, but the ~7 s
    import otherwise lands in the first ``ExecuteRskill`` goal's latency
    instead of overlapping the HAL's concurrent ``on_configure``.
    """
    tree = _parse()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        guarded = {
            call.func.id
            for stmt in node.body
            for call in ast.walk(stmt)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert _PREWARM not in guarded, (
            f"{_PREWARM}() is inside an `if` at line {node.lineno}; it must run "
            "unconditionally so `openral deploy sim` pre-warms too."
        )
