"""HAL lifecycle node SIGINT teardown contract — structural regression guard.

Mirrors ``packages/openral_reasoner_ros/test/test_reasoner_node_sigint_shape.py``
(landed in abd594f for the reasoner_node) and the original runtime_node guard
(caae96f). ROS 2 Jazzy installs a SIGINT signal handler in :func:`rclpy.init`
that:

1. Shuts down the rclpy context.
2. Raises ``KeyboardInterrupt`` out of :func:`rclpy.spin`.

Before this guard, both ``main()`` factories in ``openral_hal.lifecycle``
(:func:`make_lifecycle_main` and :func:`make_lifecycle_main_from_manifest`)
wrapped ``rclpy.spin(node)`` in a bare ``try/finally`` and called plain
``rclpy.shutdown()`` in the finally block. On every operator Ctrl-C during
``openral deploy sim`` the finally then crashed with::

    rclpy._rclpy_pybind11.RCLError: failed to shutdown:
    rcl_shutdown already called on the given context

which (a) replaced the ``KeyboardInterrupt`` with a confusing traceback in
stderr and (b) stalled the launch shutdown supervisor's wait-for-children past
the 30 s ``shutdown_grace`` window, forcing a SIGKILL of the deploy graph
(``ros2 launch`` exit 250).

This module has *two* spin-wrapping ``main()`` factories, so the spin-wrap
assertion below requires **every** spin Try to catch both exceptions (not just
one of them).

This is the structural counterpart to the behavioural deploy probe: it parses
``lifecycle.py`` as Python and asserts the *shape* of the SIGINT-handling
contract, so a future refactor can't silently revert to the broken pattern.
The HAL lifecycle node is the robot bring-up node (not the safety kernel); it
only subscribes to ``/openral/estop`` defensively — this guard touches only the
``main()`` spin/shutdown wrapper, never the estop latch.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE = _REPO_ROOT / "python" / "hal" / "src" / "openral_hal" / "lifecycle.py"


def _parse() -> ast.Module:
    """Parse the HAL lifecycle module as Python. Fail loudly if missing."""
    assert _NODE.is_file(), f"lifecycle.py not found at {_NODE}"
    return ast.parse(_NODE.read_text(), filename=str(_NODE))


def _walk_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``ast.Call`` node anywhere in the tree."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _is_rclpy_attr(node: ast.expr, attr: str) -> bool:
    """``rclpy.<attr>`` reference (Attribute on Name(id='rclpy'))."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "rclpy"
    )


def test_imports_external_shutdown_exception() -> None:
    """rclpy.executors.ExternalShutdownException must be imported."""
    tree = _parse()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "rclpy.executors":
            imported.update(alias.name for alias in node.names)
    assert "ExternalShutdownException" in imported, (
        f"HAL lifecycle must import ExternalShutdownException from "
        f"rclpy.executors. Found imports from rclpy.executors: {sorted(imported)}"
    )


def test_no_bare_rclpy_shutdown_call() -> None:
    """``rclpy.shutdown()`` may not be called anywhere in the HAL lifecycle.

    All shutdown sites must use :func:`rclpy.try_shutdown`, which is
    idempotent and a no-op when the context is already shut down.
    """
    bare_calls: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "shutdown"):
            bare_calls.append(call.lineno)
    assert not bare_calls, (
        f"HAL lifecycle must use rclpy.try_shutdown() (idempotent); "
        f"found bare rclpy.shutdown() at lines: {bare_calls}. "
        f"See docstring for why this breaks SIGINT teardown."
    )


def test_uses_try_shutdown() -> None:
    """At least one ``rclpy.try_shutdown()`` call must exist.

    Both ``main()`` factories converted their bare ``rclpy.shutdown()``, so
    two are expected; one is the minimum to catch a wholesale drop.
    """
    try_shutdown_lines: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "try_shutdown"):
            try_shutdown_lines.append(call.lineno)
    assert try_shutdown_lines, "HAL lifecycle must call rclpy.try_shutdown() on the teardown path."


def test_spin_wrapped_in_sigint_except() -> None:
    """Every ``rclpy.spin(node)`` must be in ``try / except (KI, ESE) / finally``.

    Unlike the single-``main()`` nodes, this module has two spin-wrapping
    factories, so *every* spin Try must catch both exceptions.
    """
    tree = _parse()
    spin_trys: list[ast.Try] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "spin"
                ):
                    spin_trys.append(node)
                    break
            else:
                continue
            break
    assert len(spin_trys) >= 2, (
        "expected both HAL lifecycle main() factories to wrap a `.spin()` "
        f"call in a `try:`; found {len(spin_trys)} spin-wrapping Try block(s)."
    )

    needs = {"KeyboardInterrupt", "ExternalShutdownException"}
    for try_node in spin_trys:
        names_in_excepts: set[str] = set()
        for handler in try_node.handlers:
            exc = handler.type
            for sub in ast.walk(exc) if exc is not None else ():
                if isinstance(sub, ast.Name):
                    names_in_excepts.add(sub.id)
        assert needs.issubset(names_in_excepts), (
            "every try-block wrapping rclpy.spin() must `except "
            "(KeyboardInterrupt, ExternalShutdownException)`; a spin Try at "
            f"line {try_node.lineno} catches only {sorted(names_in_excepts)}."
        )
        assert try_node.finalbody, (
            "a `finally:` clause must run cleanup (destroy_node / "
            f"try_shutdown) on the spin Try at line {try_node.lineno}."
        )
