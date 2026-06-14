"""forwarder_node SIGINT teardown contract — structural regression guard.

Mirrors ``packages/openral_reasoner_ros/test/test_reasoner_node_sigint_shape.py``
(the canonical reference for this pattern across the deploy graph). ROS 2
Jazzy installs a SIGINT signal handler in :func:`rclpy.init` that:

1. Shuts down the rclpy context.
2. Raises ``KeyboardInterrupt`` out of :func:`rclpy.spin`.

Before issue #290, ``forwarder_node.main`` wrapped ``rclpy.spin(node)`` in a
bare ``try/finally`` and called plain ``rclpy.shutdown()`` in the outer
``finally`` block. On every operator Ctrl-C during ``openral deploy sim``
the finally then crashed with::

    rclpy._rclpy_pybind11.RCLError: failed to shutdown:
    rcl_shutdown already called on the given context

This is a safety-path node (Layer 6 — ADR-0018 §5 bullet 2). The structural
contract here additionally proves that the ``except`` clause is scoped to
teardown-signal-only exceptions (``KeyboardInterrupt`` /
``ExternalShutdownException``) — it does NOT catch ``Exception``,
``ROSError``, or ``ROSSafetyViolation``, so an E-stop condition or a
safety-path failure cannot be silently swallowed at shutdown entry.

These nodes are shutdown *entry-points* for the process, not actuation
control loops, so this ``except`` cannot leave motors energised: by the time
``main()`` is exiting the forwarder has already published its estop on
``/openral/estop`` and the C++ safety kernel (ADR-0020) owns the actuation
gate independently.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE = (
    _REPO_ROOT / "packages" / "openral_human_estop" / "openral_human_estop" / "forwarder_node.py"
)


def _parse() -> ast.Module:
    """Parse forwarder_node as Python. Fail loudly if missing."""
    assert _NODE.is_file(), f"forwarder_node not found at {_NODE}"
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
    """rclpy.executors.ExternalShutdownException must be imported.

    Required so the ``except`` around ``rclpy.spin(node)`` can catch it
    alongside ``KeyboardInterrupt``.
    """
    tree = _parse()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "rclpy.executors":
            imported.update(alias.name for alias in node.names)
    assert "ExternalShutdownException" in imported, (
        "forwarder_node must import ExternalShutdownException from rclpy.executors. "
        f"Found imports from rclpy.executors: {sorted(imported)}"
    )


def test_no_bare_rclpy_shutdown_call() -> None:
    """``rclpy.shutdown()`` may not be called anywhere in forwarder_node.

    All shutdown sites must use :func:`rclpy.try_shutdown`, which is
    idempotent and a no-op when the context is already shut down.
    """
    bare_calls: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "shutdown"):
            bare_calls.append(call.lineno)
    assert not bare_calls, (
        "forwarder_node must use rclpy.try_shutdown() (idempotent); "
        f"found bare rclpy.shutdown() at lines: {bare_calls}."
    )


def test_uses_try_shutdown() -> None:
    """At least one ``rclpy.try_shutdown()`` call must exist."""
    try_shutdown_lines: list[int] = []
    for call in _walk_calls(_parse()):
        if _is_rclpy_attr(call.func, "try_shutdown"):
            try_shutdown_lines.append(call.lineno)
    assert try_shutdown_lines, "forwarder_node must call rclpy.try_shutdown() on the teardown path."


def test_spin_wrapped_in_sigint_except() -> None:
    """``rclpy.spin(node)`` must be inside ``try / except (KI, ESE) / finally``.

    Three independent assertions:

    1. There's at least one ``Try`` node whose body calls ``.spin()``.
    2. That Try has an ``except`` clause mentioning both
       ``KeyboardInterrupt`` and ``ExternalShutdownException``.
    3. There is a ``finalbody`` somewhere on the spin teardown path.
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
    assert spin_trys, (
        "no `try:` block wraps a `.spin()` call in forwarder_node — the "
        "SIGINT teardown contract requires one."
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


def test_except_does_not_catch_broad_exceptions() -> None:
    """The spin ``except`` must NOT catch ``Exception``, ``ROSError``, or ``ROSSafetyViolation``.

    This is the structural proof that the teardown path cannot silence an
    E-stop condition or a safety-path failure. Only teardown-signal-only
    exceptions (``KeyboardInterrupt`` / ``ExternalShutdownException``) are
    caught; all others propagate normally.
    """
    tree = _parse()
    forbidden = {"Exception", "ROSError", "ROSSafetyViolation"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        wraps_spin = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "spin"
            for stmt in node.body
            for sub in ast.walk(stmt)
        )
        if not wraps_spin:
            continue
        for handler in node.handlers:
            exc = handler.type
            names: set[str] = set()
            for sub in ast.walk(exc) if exc is not None else ():
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
            caught_forbidden = forbidden & names
            assert not caught_forbidden, (
                f"spin-wrapping except must NOT catch {caught_forbidden} — "
                "doing so would mask E-stop conditions on shutdown. "
                "Only (KeyboardInterrupt, ExternalShutdownException) are allowed."
            )
