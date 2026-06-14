"""Regression test for the one-shot autostart in ``sim_e2e.launch.py`` (ADR-0050).

``_autostart_lifecycle`` drives a node UNCONFIGURED → INACTIVE → ACTIVE at boot.
The activate handler must be scoped to the **configure** transition
(``start_state="configuring"``), NOT a bare ``goal_state="inactive"`` — otherwise
it also re-fires on a *runtime* deactivate (``active → deactivating → inactive``)
and immediately re-activates the node. That fights ADR-0050 VRAM eviction: the
reasoner deactivates the object detector to free its VRAM before a VLA, and an
auto-reactivate reloads the detector model and OOMs an 8 GB card (observed live
2026-06-12). This test pins the scoping so the bug can't silently return.

Hermetic — only ``launch`` / ``launch_ros`` are needed; the module's heavy
imports are deferred inside ``compose_runtime_graph``. Skips cleanly without a
sourced ROS 2 overlay (CLAUDE.md §1.11 legitimate skip path).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "sim_e2e.launch.py"


def _import_launch_module() -> ModuleType:
    """Import ``sim_e2e.launch.py`` by absolute path (skip without a ROS overlay)."""
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")
    spec = importlib.util.spec_from_file_location(
        "openral_sim_e2e_autostart_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _activate_handler_start_state(mod: ModuleType) -> str | None:
    """Return the ``start_state`` the autostart ACTIVATE handler is scoped to.

    ``_autostart_lifecycle`` returns ``[OnProcessStart(configure),
    OnStateTransition(activate)]``. The ``OnStateTransition`` matcher closes over
    ``start_state`` / ``goal_state`` — read it back from the closure.
    """
    from launch_ros.actions import LifecycleNode
    from launch_ros.event_handlers import OnStateTransition

    node = LifecycleNode(package="x", executable="y", name="openral_test_node", namespace="")
    handlers = mod._autostart_lifecycle(node, "openral_test_node")
    for register in handlers:
        inner = register.event_handler
        if isinstance(inner, OnStateTransition):
            matcher = vars(inner)["_OnStateTransition__custom_matcher"]
            freevars = matcher.__code__.co_freevars
            cells = dict(zip(freevars, (matcher.__closure__ or ()), strict=False))
            cell = cells.get("start_state")
            return None if cell is None else cell.cell_contents
    pytest.fail("no OnStateTransition (activate) handler found in _autostart_lifecycle output")


def test_autostart_activate_is_scoped_to_configure_not_runtime_deactivate() -> None:
    """The activate handler fires on configure→inactive only, never on deactivate→inactive.

    ``start_state="configuring"`` is what makes the autostart one-shot: a runtime
    deactivate produces ``start_state="deactivating"``, which must NOT match (else
    the reasoner can never free the detector's VRAM — the 8 GB OOM, ADR-0050).
    """
    mod = _import_launch_module()
    start_state = _activate_handler_start_state(mod)
    assert start_state == "configuring", (
        "autostart ACTIVATE handler must be scoped to the configure transition "
        f"(start_state='configuring'), got {start_state!r}. A bare goal_state='inactive' "
        "matcher re-activates a node on a runtime deactivate and breaks ADR-0050 VRAM eviction."
    )
