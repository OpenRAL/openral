"""`docs/methods` markers that name their own module must be tracked, not skipped.

A `### `packages/foo/`` heading names a package directory rather than one module, so its
bullets carry the file inline: `` `SYMBOL` (bucket2_markers.py L58) ``. Before this was
handled, `refresh_file` set no file context for those bullets and silently skipped them —
nine of eleven such markers in the inventory had rotted to wrong line numbers.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from refresh_methods_linenos import (  # noqa: E402  # reason: needs the sys.path insert above
    _DIR_HEADING_RE,
    _MARKER_RE,
    _resolve_inline_path,
    refresh_file,
)

_ENTRY_RE = re.compile(r"`([^`]+)`[^\n]*?\((?:([\w./-]+\.py) )?L(\d+)(?:[–-]\d+)?\)")


def _definition_lines(path: Path) -> dict[int, str]:
    """Map every module-level definition or assignment line to its name."""
    table: dict[int, str] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            table[node.lineno] = node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    table.setdefault(node.lineno, target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            table.setdefault(node.lineno, node.target.id)
    return table


def _inline_entries() -> list[tuple[str, int, str, int, str, str]]:
    """Every `(module.py LNN)` marker, as (doc, line, rel, lineno, span, section_dir).

    ``section_dir`` is the package-directory heading the bullet sits under, which is how
    the tool resolves a bare module name like ``gpu.py``.
    """
    out = []
    for md in sorted((REPO_ROOT / "docs" / "methods").glob("*.md")):
        section_dir = ""
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            heading = _DIR_HEADING_RE.match(line)
            if heading:
                section_dir = heading.group(1)
                continue
            if line.startswith("#"):
                section_dir = ""
                continue
            for match in _ENTRY_RE.finditer(line):
                span, rel, cited = match.group(1), match.group(2), int(match.group(3))
                if rel:
                    out.append((md.name, lineno, rel, cited, span, section_dir))
    return out


def test_marker_regex_accepts_both_marker_shapes() -> None:
    assert _MARKER_RE.search("(L193)").groups() == (None, "193", None)
    assert _MARKER_RE.search("(L153–160)").groups() == (None, "153", "160")
    assert _MARKER_RE.search("(gpu.py L193)").groups() == ("gpu.py", "193", None)
    assert _MARKER_RE.search("(tools/demo_publisher.py L30)").groups() == (
        "tools/demo_publisher.py",
        "30",
        None,
    )


def test_resolve_inline_path_finds_module_under_its_package() -> None:
    base = REPO_ROOT / "packages/openral_foxglove_bringup/"
    assert _resolve_inline_path("bucket2_markers.py", base) is not None
    assert _resolve_inline_path("tools/demo_publisher.py", base) is not None
    assert _resolve_inline_path("no_such_module.py", base) is None


def test_the_inventory_has_inline_path_markers_to_police() -> None:
    """Guards the test itself: if the shape disappears, this file is dead weight."""
    assert len(_inline_entries()) >= 10


@pytest.mark.parametrize("doc,doc_line,rel,cited,span,section_dir", _inline_entries())
def test_inline_path_marker_points_at_its_definition(
    doc: str, doc_line: int, rel: str, cited: int, span: str, section_dir: str
) -> None:
    base = REPO_ROOT / section_dir if section_dir else None
    source = _resolve_inline_path(rel, base)
    assert source is not None, f"{doc}:{doc_line} cites a missing module: {rel}"
    name = re.sub(r"^(class |@dataclass\s+class |def )", "", span).split("(")[0]
    name = name.split(":")[0].strip()
    table = _definition_lines(source)
    assert table.get(cited) == name, (
        f"{doc}:{doc_line} cites {rel} L{cited} for `{name}`, but L{cited} defines "
        f"{table.get(cited)!r}. Run `python tools/refresh_methods_linenos.py`."
    )


def test_refresh_is_idempotent_on_the_real_inventory() -> None:
    """`--check` must be clean, which also proves inline markers survive a rewrite."""
    for md in sorted((REPO_ROOT / "docs" / "methods").glob("*.md")):
        changed, unresolved = refresh_file(md, check=True)
        assert changed == 0, f"{md.name} has {changed} stale marker(s)"
        assert not unresolved, f"{md.name}: {unresolved}"
