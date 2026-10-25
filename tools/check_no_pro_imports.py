"""Fail when any tracked Python file imports an ``openral_pro*`` module.

The public repo is uniformly Apache-2.0 and must never name a private OpenRAL
Pro module path in code (``docs/decisions.md`` §Licensing & commercial
boundary). Pro plugs in only through entry-point groups resolved by
``openral_rskill.backend_registry``. This check refuses, by AST:

* ``import openral_pro_trt`` / ``import openral_pro.x``
* ``from openral_pro_trt.nvbufsurface import ...``
* ``import_module("openral_pro_trt...")`` / ``importlib.import_module(...)`` /
  ``__import__(...)`` with a string literal naming such a module.

The *name* in strings, comments and docs is allowed: error messages say which
package to install (CLAUDE.md §1.4) and tests assert them. The match needs a
``_``, ``.`` or the end after ``openral_pro``, so ``openral_prompt_router`` is
not a hit.

Usage::

    uv run python tools/check_no_pro_imports.py           # all tracked .py files
    uv run python tools/check_no_pro_imports.py a.py b.py # just these files

Not part of the runtime; CI-adjacent lint only.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

_PRO_MODULE = re.compile(r"^openral_pro(?:$|[._])")
_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def find_pro_imports(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """Return ``(line, module)`` for every ``openral_pro*`` import in *source*.

    Args:
        source: Python source text.
        filename: Name used in a ``SyntaxError``.

    Returns:
        The offending imports in source order; empty when clean.

    Example:
        >>> find_pro_imports("from openral_pro_trt.nvbufsurface import load")
        [(1, 'openral_pro_trt.nvbufsurface')]
        >>> find_pro_imports("import openral_prompt_router")
        []
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            hits.extend((node.lineno, a.name) for a in node.names if _PRO_MODULE.match(a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _PRO_MODULE.match(node.module):
                hits.append((node.lineno, node.module))
        elif (
            isinstance(node, ast.Call)
            and _call_name(node.func) in _IMPORT_CALLS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _PRO_MODULE.match(node.args[0].value)
        ):
            hits.append((node.lineno, node.args[0].value))
    return sorted(hits)


def _tracked_python_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [root / p for p in out.decode().split("\0") if p]


def main(argv: list[str] | None = None) -> int:
    """Scan files and print each hit; exit 1 on any."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="default: every tracked .py")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    files = args.files or _tracked_python_files(root)

    failed = False
    for path in files:
        for line, module in find_pro_imports(path.read_text(encoding="utf-8"), str(path)):
            print(
                f"{path}:{line}: imports {module!r}; resolve it through an "
                "openral_rskill.backend_registry entry-point seam instead"
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
