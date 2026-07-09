"""Pytest-wide setup: deterministic, color-free console output.

The CLI uses Rich's ``Console()``, whose color/highlight behavior depends on
ambient env (``COLORTERM``, ``FORCE_COLOR``, terminal detection, ...). That
makes JSON-output and substring-matching CLI tests flaky across machines —
ANSI escapes leak into ``CliRunner.output`` and break ``json.loads`` and
``in`` checks. We pin the env and force any already-imported Console
instances into plain mode before tests run.
"""

from __future__ import annotations

import os

os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
for _var in ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE", "COLORTERM"):
    os.environ.pop(_var, None)


def _neuter_rich_console() -> None:
    from rich.console import Console

    original_init = Console.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("force_terminal", False)
        kwargs.setdefault("force_interactive", False)
        kwargs.setdefault("no_color", True)
        kwargs.setdefault("highlight", False)
        kwargs.setdefault("color_system", None)
        original_init(self, *args, **kwargs)

    Console.__init__ = patched_init  # type: ignore[method-assign]


_neuter_rich_console()
