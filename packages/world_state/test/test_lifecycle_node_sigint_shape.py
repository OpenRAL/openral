"""world_state lifecycle_node SIGINT teardown contract — structural guard.

Mirrors ``packages/openral_reasoner_ros/test/test_reasoner_node_sigint_shape.py``
(landed in abd594f) and the runtime_node guard from caae96f. ROS 2 Jazzy
installs a SIGINT signal handler in :func:`rclpy.init` that:

1. Shuts down the rclpy context.
2. Raises ``KeyboardInterrupt`` out of :func:`rclpy.spin`.

Before this guard, ``lifecycle_node.main`` wrapped ``rclpy.spin(node)`` in a
bare ``try/finally`` and called plain ``rclpy.shutdown()`` in the finally. On
every operator Ctrl-C during ``openral deploy sim`` that finally then crashed
with::

    rclpy._rclpy_pybind11.RCLError: failed to shutdown:
    rcl_shutdown already called on the given context

which (a) replaced the ``KeyboardInterrupt`` with a confusing traceback and
(b) stalled the launch shutdown supervisor past the grace window, forcing a
SIGKILL of the deploy graph.

This test parses the world_state ``lifecycle_node.py`` as Python and asserts
the *shape* of the SIGINT-handling contract so a future refactor can't
silently revert to the broken pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE = _REPO_ROOT / "packages" / "world_state" / "openral_world_state_ros" / "lifecycle_node.py"


def _parse() -> ast.Module:
    """Parse the world_state lifecycle_node module as Python. Fail loudly if missing."""
    assert _NODE.is_file(), f"world_state lifecycle_node not found at {_NODE}"
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
        f"world_state lifecycle_node must import ExternalShutdownException from "
        f"rclpy.executors. Found imports from rclpy.executors: {sorted(imported)}"
    )


def test_no_bare_rclpy_shutdown_call() -> None:
    """``rclpy.shutdown()`` may not be called anywhere in lifecycle_node.

    All shutdown sites must use :func:`rclpy.try_shutdown`, which is
    idempotent and a no-op when the context is already shut down.
    """
    bare_calls: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "shutdown"):
            bare_calls.append(call.lineno)
    assert not bare_calls, (
        f"world_state lifecycle_node must use rclpy.try_shutdown() (idempotent); "
        f"found bare rclpy.shutdown() at lines: {bare_calls}. "
        f"See docstring for why this breaks SIGINT teardown."
    )


def test_uses_try_shutdown() -> None:
    """At least one ``rclpy.try_shutdown()`` call must exist."""
    try_shutdown_lines: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "try_shutdown"):
            try_shutdown_lines.append(call.lineno)
    assert try_shutdown_lines, (
        "world_state lifecycle_node must call rclpy.try_shutdown() on the teardown path."
    )


def test_spin_wrapped_in_sigint_except() -> None:
    """``rclpy.spin(node)`` must be inside ``try / except (KI, ESE) / finally``."""
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
    assert spin_trys, (
        "no `try:` block wraps a `.spin()` call in world_state lifecycle_node — "
        "the SIGINT teardown contract requires one (see this test's docstring)."
    )

    qualifying: list[ast.Try] = []
    for try_node in spin_trys:
        names_in_excepts: set[str] = set()
        for handler in try_node.handlers:
            exc = handler.type
            for sub in ast.walk(exc) if exc is not None else ():
                if isinstance(sub, ast.Name):
                    names_in_excepts.add(sub.id)
        needs = {"KeyboardInterrupt", "ExternalShutdownException"}
        if needs.issubset(names_in_excepts):
            qualifying.append(try_node)

    assert qualifying, (
        "the try-block wrapping rclpy.spin() must `except "
        "(KeyboardInterrupt, ExternalShutdownException)` so the SIGINT "
        "teardown path doesn't print a traceback. Found "
        f"{len(spin_trys)} spin-wrapping Try block(s), none catch both."
    )

    assert any(t.finalbody for t in spin_trys), (
        "a `finally:` clause must run cleanup (destroy_node / "
        "try_shutdown) on both normal and interrupted spin exits."
    )
