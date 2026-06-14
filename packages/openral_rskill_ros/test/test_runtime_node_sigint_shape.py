"""runtime_node SIGINT teardown contract — structural regression guard.

ROS 2 Jazzy installs a SIGINT signal handler in :func:`rclpy.init` that:

1. Shuts down the rclpy context.
2. Raises ``KeyboardInterrupt`` out of :py:meth:`Executor.spin`.

Before this guard, ``runtime_node`` wrapped ``executor.spin()`` in a
bare ``try/finally`` and called plain ``rclpy.shutdown()`` in the
``finally`` block. On every SIGINT the finally then crashed with::

    rclpy._rclpy_pybind11.RCLError: failed to shutdown:
    rcl_shutdown already called on the given context

which (a) replaced the ``KeyboardInterrupt`` with a confusing traceback
in stderr and (b) caused the launch parent's wait-for-children to drag
out for the full ``shutdown_grace_s`` window before SIGKILL — the exact
symptom that surfaced as ``fail-timeout`` rows in
``tools/audit_sim_configs.py`` after the OTLP `--no-dashboard` fix
landed (see ``outputs/audit_deploy_postfix3.json``).

This test is the structural counterpart to the behavioural audit: it
parses ``runtime_node`` as Python and asserts the *shape* of the
SIGINT-handling contract, so a future refactor can't silently revert
to the broken pattern.

The empirical validation lives in
``just sim-audit --deploy-alive-grace 10 --deploy-shutdown-grace 5``
on a real deploy scene — see this PR's description.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_NODE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "scripts" / "runtime_node"


def _parse() -> ast.Module:
    """Parse the runtime_node script as Python. Fail loudly if missing."""
    assert _RUNTIME_NODE.is_file(), f"runtime_node not found at {_RUNTIME_NODE}"
    return ast.parse(_RUNTIME_NODE.read_text(), filename=str(_RUNTIME_NODE))


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

    Required so the ``except`` around ``executor.spin()`` can catch it
    alongside ``KeyboardInterrupt`` — the executor raises
    ``ExternalShutdownException`` when another thread calls
    ``rclpy.shutdown()`` while spin is blocked, which happens in any
    launch graph where a sibling lifecycle node triggers shutdown
    cooperatively (the rclpy SIGINT handler is one such caller).
    """
    tree = _parse()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "rclpy.executors":
            imported.update(alias.name for alias in node.names)
    assert "ExternalShutdownException" in imported, (
        f"runtime_node must import ExternalShutdownException from rclpy.executors. "
        f"Found imports from rclpy.executors: {sorted(imported)}"
    )


def test_no_bare_rclpy_shutdown_call() -> None:
    """``rclpy.shutdown()`` may not be called anywhere in runtime_node.

    All shutdown sites must use :func:`rclpy.try_shutdown`, which is
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
    """``executor.spin()`` must be inside ``try / except (KI, ESE) / finally``.

    Three independent assertions:

    1. There's at least one ``Try`` node whose body calls
       ``<x>.spin()`` (we don't constrain the receiver name — could
       be ``executor`` or some renamed variable).
    2. That Try has at least one ``except`` clause whose exception
       expression mentions both ``KeyboardInterrupt`` and
       ``ExternalShutdownException``.
    3. That Try has a ``finalbody`` (so cleanup still runs on
       exceptional paths).
    """
    tree = _parse()
    candidate_trys: list[ast.Try] = []
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
                    candidate_trys.append(node)
                    break
            else:
                continue
            break
    assert candidate_trys, (
        "no `try:` block wraps an `.spin()` call in runtime_node — the "
        "SIGINT teardown contract requires one (see this test's docstring)."
    )

    # Among Try blocks that wrap spin(), at least one must catch
    # the right SIGINT exceptions AND have a finalbody.
    qualifying: list[ast.Try] = []
    for try_node in candidate_trys:
        names_in_excepts: set[str] = set()
        for handler in try_node.handlers:
            exc = handler.type
            # Pull every ``Name`` out of the except expression — handles
            # both ``except KeyboardInterrupt`` and
            # ``except (KeyboardInterrupt, ExternalShutdownException)``.
            for sub in ast.walk(exc) if exc is not None else ():
                if isinstance(sub, ast.Name):
                    names_in_excepts.add(sub.id)
        needs = {"KeyboardInterrupt", "ExternalShutdownException"}
        if needs.issubset(names_in_excepts) and try_node.finalbody:
            qualifying.append(try_node)

    assert qualifying, (
        f"the try-block wrapping executor.spin() must (a) `except "
        f"(KeyboardInterrupt, ExternalShutdownException)` AND (b) have a "
        f"`finally:` clause that runs cleanup on both normal and "
        f"interrupted exits. Found {len(candidate_trys)} spin-wrapping "
        f"Try block(s), none satisfy both criteria."
    )
