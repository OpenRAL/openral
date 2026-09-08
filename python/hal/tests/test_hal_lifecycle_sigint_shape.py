"""HAL lifecycle node SIGINT teardown contract — structural regression guard.

Mirrors ``packages/openral_reasoner_ros/test/test_reasoner_node_sigint_shape.py``
(abd594f) and the runtime_node guard (caae96f). ROS 2 Jazzy's SIGINT handler
(``rclpy.init``) shuts down the rclpy context and raises
``KeyboardInterrupt`` out of ``rclpy.spin``; a bare ``try/finally`` with
plain ``rclpy.shutdown()`` then crashes with ``RCLError: rcl_shutdown already
called on the given context`` on every Ctrl-C, masking the
``KeyboardInterrupt`` and stalling the launch shutdown supervisor past the
30 s ``shutdown_grace`` window (SIGKILL, ``ros2 launch`` exit 250).

Both ``main()`` factories in ``openral_hal.lifecycle``
(``make_lifecycle_main``, ``make_lifecycle_main_from_manifest``) must
wrap every spin in ``try/except (KeyboardInterrupt, ExternalShutdownException)
/finally``; this module parses ``lifecycle.py`` as Python and asserts that
*shape* (structural, not behavioural). The HAL lifecycle node is robot
bring-up, not the safety kernel — it only subscribes to ``/openral/estop``
defensively; this guard touches only the spin/shutdown wrapper, never the
estop latch.
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

    All shutdown sites must use ``rclpy.try_shutdown``, which is
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


def test_spin_finally_disconnects_the_hal() -> None:
    """Every spin ``finally`` must call ``node.shutdown_hal()``.

    SIGINT raises out of ``spin`` without requesting the lifecycle
    ``shutdown`` transition, so ``on_shutdown``/``on_cleanup`` — and with
    them ``HAL.disconnect`` — never run. The ``finally`` is the only place on
    the signal path that can reach it.
    """
    tree = _parse()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        spins = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "spin"
            for stmt in node.body
            for sub in ast.walk(stmt)
        )
        if not spins:
            continue
        teardown = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "shutdown_hal"
            for stmt in node.finalbody
            for sub in ast.walk(stmt)
        )
        assert teardown, (
            "the spin Try at line "
            f"{node.lineno} must call node.shutdown_hal() in its `finally:` — "
            "otherwise a SIGINT teardown never disconnects the HAL and the "
            "terminal task-success verdict is never emitted."
        )
