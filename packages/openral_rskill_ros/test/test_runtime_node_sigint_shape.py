"""runtime_node SIGINT teardown contract — structural regression guard.

ROS 2 Jazzy's ``rclpy.init`` SIGINT handler shuts down the rclpy context and raises
``KeyboardInterrupt`` out of ``Executor.spin``. Before this guard, ``runtime_node``
wrapped ``executor.spin()`` in a bare ``try/finally`` calling plain ``rclpy.shutdown()`` in
``finally``, so every SIGINT crashed with::

    rclpy._rclpy_pybind11.RCLError: failed to shutdown:
    rcl_shutdown already called on the given context

— replacing ``KeyboardInterrupt`` with a confusing stderr traceback and dragging the launch
parent's wait-for-children out to the full ``shutdown_grace_s`` before SIGKILL: the
``fail-timeout`` rows ``tools/audit_sim_configs.py`` showed after the OTLP `--no-dashboard`
fix landed (``outputs/audit_deploy_postfix3.json``).

Structural counterpart to that behavioural audit: parses ``runtime_node`` as Python and
asserts the SIGINT-handling contract's *shape*, so a refactor can't silently revert it.

Empirically validated by ``just sim-audit --deploy-alive-grace 10 --deploy-shutdown-grace 5``
on a real deploy scene.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_NODE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "scripts" / "runtime_node"
_COMPOSE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "openral_rskill_ros" / "compose.py"


def _parse() -> ast.Module:
    """Parse the runtime_node script as Python. Fail loudly if missing."""
    assert _RUNTIME_NODE.is_file(), f"runtime_node not found at {_RUNTIME_NODE}"
    return ast.parse(_RUNTIME_NODE.read_text(), filename=str(_RUNTIME_NODE))


def _spin_helper() -> ast.FunctionDef:
    """``spin_until_shutdown`` in ``openral_rskill_ros/compose.py``: the spin runtime_node uses."""
    tree = ast.parse(_COMPOSE.read_text(), filename=str(_COMPOSE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "spin_until_shutdown":
            return node
    raise AssertionError(f"spin_until_shutdown not found in {_COMPOSE}")


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
    """The spin helper must import ``rclpy.executors.ExternalShutdownException``.

    ``runtime_node`` spins through ``openral_rskill_ros.compose.spin_until_shutdown``,
    whose ``except`` around ``executor.spin()`` catches it alongside
    ``KeyboardInterrupt`` — the executor raises ``ExternalShutdownException`` when
    another thread calls ``rclpy.shutdown()`` while spin is blocked (the rclpy SIGINT
    handler is one such caller).
    """
    imported: set[str] = set()
    for node in ast.walk(_spin_helper()):
        if isinstance(node, ast.ImportFrom) and node.module == "rclpy.executors":
            imported.update(alias.name for alias in node.names)
    assert "ExternalShutdownException" in imported, (
        f"spin_until_shutdown must import ExternalShutdownException from rclpy.executors. "
        f"Found imports from rclpy.executors: {sorted(imported)}"
    )


def test_no_bare_rclpy_shutdown_call() -> None:
    """``rclpy.shutdown()`` may not be called anywhere in runtime_node.

    All shutdown sites must use ``rclpy.try_shutdown``, which is
    idempotent and a no-op when the context is already shut down.
    Bare ``rclpy.shutdown()`` raises ``RCLError`` if SIGINT has
    already torn the context down — guaranteed on every operator
    Ctrl-C in production and on every SIGINT-driven audit probe in
    ``_run_one_deploy`` (tools/audit_sim_configs.py).
    """
    bare_calls: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "shutdown"):
            bare_calls.append(call.lineno)
    assert not bare_calls, (
        f"runtime_node must use rclpy.try_shutdown() (idempotent); "
        f"found bare rclpy.shutdown() at lines: {bare_calls}. "
        f"See docstring for why this breaks SIGINT teardown."
    )


def test_uses_try_shutdown() -> None:
    """At least one ``rclpy.try_shutdown()`` call must exist.

    Cheap sanity check: if a refactor accidentally dropped *all*
    shutdown calls (instead of converting them), the bare-call test
    above would pass vacuously. This catches that regression.
    """
    try_shutdown_lines: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "try_shutdown"):
            try_shutdown_lines.append(call.lineno)
    assert try_shutdown_lines, (
        "runtime_node must call rclpy.try_shutdown() on every exit path "
        "(spin teardown + every early-return error branch)."
    )


def test_spin_wrapped_in_sigint_except() -> None:
    """The spin must end quietly on SIGINT and still run ``runtime_node``'s cleanup.

    1. ``spin_until_shutdown`` wraps ``executor.spin()`` in a ``try`` whose handlers
       name both ``KeyboardInterrupt`` and ``ExternalShutdownException``.
    2. ``runtime_node`` calls ``spin_until_shutdown`` inside a ``try`` with a
       ``finally:`` clause, so cleanup runs on both normal and interrupted exits.
    """
    needs = {"KeyboardInterrupt", "ExternalShutdownException"}
    caught: set[str] = set()
    for node in ast.walk(_spin_helper()):
        if isinstance(node, ast.Try) and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "spin"
            for stmt in node.body
            for sub in ast.walk(stmt)
        ):
            for handler in node.handlers:
                if handler.type is not None:
                    caught.update(n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name))
    assert needs <= caught, (
        f"spin_until_shutdown must wrap executor.spin() in "
        f"`except (KeyboardInterrupt, ExternalShutdownException)`; caught: {sorted(caught)}"
    )

    wrapped = [
        node
        for node in ast.walk(_parse())
        if isinstance(node, ast.Try)
        and node.finalbody
        and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "spin_until_shutdown"
            for stmt in node.body
            for sub in ast.walk(stmt)
        )
    ]
    assert wrapped, (
        "runtime_node must call spin_until_shutdown(executor) inside a `try:` with a "
        "`finally:` clause that runs cleanup on both normal and interrupted exits."
    )
