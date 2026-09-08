"""Refresh the ``(LNN)`` line citations in the ``docs/methods/`` inventory.

The public-symbol inventory (``docs/METHODS.md`` index + ``docs/methods/*.md``)
is hand-curated, but its ``(LNN)`` source-line markers rot every time code
moves. This script re-derives each marker from the current source tree:

* A ``###``/``####`` heading whose backticked token is a ``.py`` path under
  ``python/``, ``packages/``, ``tools/`` or ``tests/`` sets the "current file"
  for the bullet entries that follow.
* Each bullet ending in ``(LNN)`` or ``(LNN–MM)`` names a symbol in its first
  backticked code span (``class X(...)``, ``def``-style ``name(args) -> ret``,
  ``@dataclass X``, ``const X: ...``, ``prop a, b, c``). Indented bullets
  resolve inside the enclosing top-level ``class`` entry's scope.
* Symbols are located with ``ast``; unresolved entries are reported and
  left untouched (a stale entry is a defect to fix by hand, not to guess).

Usage::

    uv run python tools/refresh_methods_linenos.py            # rewrite in place
    uv run python tools/refresh_methods_linenos.py --check    # report drift, exit 1

Not part of the runtime; CI-adjacent doc tooling only.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "docs" / "methods"

_HEADING_RE = re.compile(r"^#{3,4} .*?`((?:python|packages|tools|tests)/[^`]+\.py)`")
_MARKER_RE = re.compile(r"\((?:`?([\w./-]+\.py)`? )?L(\d+)(?:[–-](\d+))?\)")
# A `### `packages/foo/`` heading names a package directory rather than one module, so
# its bullets carry the file inline: `` `SYMBOL` (bucket2_markers.py L58) ``. Without
# this the tool never sets a file context for them and skips them silently.
_DIR_HEADING_RE = re.compile(r"^#{3,4} .*?`((?:python|packages|tools|tests)/[^`]+/)`")
_BULLET_RE = re.compile(r"^(\s*)- (.*)$")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _index_python_file(path: Path) -> tuple[dict[str, int], set[str]]:
    """Map module-level and class-member symbol names to definition lines.

    Returns both bare names (``rSkill``, ``from_pretrained``) and qualified
    ``Class.member`` keys, plus the set of names that are merely imports —
    those satisfy an exact primary-symbol lookup (re-export entries cite the
    import line) but are excluded from the loose identifier fallback so a
    span's ``-> Path`` return annotation can never "resolve" a stale entry.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    index: dict[str, int] = {}
    import_names: set[str] = set()

    def add(name: str, lineno: int) -> None:
        index.setdefault(name, lineno)

    def target_names(node: ast.stmt) -> list[str]:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.append(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.append(alias.asname or alias.name)
                import_names.add(alias.asname or alias.name)
        return names

    def add_class(node: ast.ClassDef) -> None:
        add(node.name, node.lineno)
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(f"{node.name}.{member.name}", member.lineno)
                add(member.name, member.lineno)
                # `self.attr = ...` assignments (typically in __init__) so the
                # inventory can cite instance attributes.
                for sub in ast.walk(member):
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if (
                                isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                            ):
                                add(f"{node.name}.{tgt.attr}", sub.lineno)
                                add(tgt.attr, sub.lineno)
            else:
                for name in target_names(member):
                    add(f"{node.name}.{name}", member.lineno)
                    add(name, member.lineno)

    def visit(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.ClassDef):
                add_class(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node.name, node.lineno)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            else:
                for name in target_names(node):
                    add(name, node.lineno)

    visit(tree.body)
    return index, import_names


def _symbols_from_span(span: str) -> list[str]:
    """Extract the symbol name(s) an inventory entry's code span refers to."""
    text = span.strip()
    text = re.sub(r"^@dataclass(\([^)]*\))?\s+", "", text)
    for prefix in ("Protocol", "class", "async def", "def", "const", "prop"):
        if text.startswith(prefix + " "):
            rest = text[len(prefix) + 1 :].strip()
            if prefix == "prop":
                return [p.strip().strip("`") for p in rest.split(",") if p.strip()]
            text = rest
            break
    match = _IDENT_RE.match(text)
    return [match.group(0)] if match else []


def _resolve(symbols: list[str], scope: str | None, index: dict[str, int]) -> list[int] | None:
    """Resolve symbol names to line numbers, preferring the enclosing class scope."""
    lines: list[int] = []
    for sym in symbols:
        candidates = [sym] if "." in sym else ([f"{scope}.{sym}"] if scope else []) + [sym]
        for cand in candidates:
            if cand in index:
                lines.append(index[cand])
                break
        else:
            return None
    return lines


def _resolve_inline_path(rel: str, base_dir: Path | None) -> Path | None:
    """Locate a module named inline in a bullet, e.g. ``(bucket2_markers.py L58)``.

    Tried in order: relative to the section's package directory, then the
    repo root, then a search by basename within the package directory.
    Returns ``None`` when nothing matches.
    """
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / rel)
    candidates.append(REPO_ROOT / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if base_dir is not None:
        matches = sorted(base_dir.rglob(Path(rel).name))
        if len(matches) == 1:
            return matches[0]
    # Last resort: a heading may name a dotted module (`openral_runner.backends.reward`)
    # rather than a path, so there is no base directory. Accept a repo-wide suffix match
    # only when it is unique.
    skip = {".git", ".venv", "build", "install", "log", "site", "__pycache__", ".claude"}
    suffix = tuple(Path(rel).parts)
    hits = [
        candidate
        for candidate in REPO_ROOT.rglob(Path(rel).name)
        if not skip & set(candidate.parts) and candidate.parts[-len(suffix) :] == suffix
    ]
    return hits[0] if len(hits) == 1 else None


def refresh_file(md_path: Path, *, check: bool) -> tuple[int, list[str]]:
    """Rewrite the markers in one inventory file.

    Returns (number of changed markers, list of unresolved-entry descriptions).
    """
    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_file: Path | None = None
    base_dir: Path | None = None
    file_index: dict[str, int] = {}
    import_names: set[str] = set()
    scope: str | None = None
    changed = 0
    unresolved: list[str] = []

    for i, line in enumerate(lines):
        dir_heading = _DIR_HEADING_RE.match(line)
        if dir_heading and not _HEADING_RE.match(line):
            candidate_dir = REPO_ROOT / dir_heading.group(1)
            base_dir = candidate_dir if candidate_dir.is_dir() else None
            if base_dir is None:
                unresolved.append(
                    f"{md_path.name}:{i + 1}: package directory missing: {dir_heading.group(1)}"
                )
            current_file = None
            scope = None
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            base_dir = None
            rel = Path(heading.group(1))
            candidate = REPO_ROOT / rel
            if candidate.exists():
                current_file = candidate
                file_index, import_names = _index_python_file(candidate)
            else:
                current_file = None
                unresolved.append(f"{md_path.name}:{i + 1}: source file missing: {rel}")
            scope = None
            continue
        if line.startswith("#"):
            current_file = None
            base_dir = None
            scope = None
            continue

        bullet = _BULLET_RE.match(line)
        if not bullet:
            continue
        indent, body = bullet.groups()
        span_match = _CODE_SPAN_RE.search(body)
        if not indent and span_match:
            head = span_match.group(1).strip()
            scope = None
            for prefix in ("class ", "@dataclass "):
                if head.startswith(prefix):
                    name_match = _IDENT_RE.match(head[len(prefix) :].strip())
                    if name_match:
                        scope = name_match.group(0)

        markers = list(_MARKER_RE.finditer(line))
        if not markers or span_match is None:
            continue

        # A bullet under a package-directory heading names its own module inline.
        inline_rel = markers[-1].group(1)
        bullet_file, bullet_index, bullet_imports = current_file, file_index, import_names
        if inline_rel:
            resolved_path = _resolve_inline_path(inline_rel, base_dir)
            if resolved_path is None:
                unresolved.append(f"{md_path.name}:{i + 1}: source file missing: {inline_rel}")
                continue
            bullet_file = resolved_path
            bullet_index, bullet_imports = _index_python_file(resolved_path)
        if bullet_file is None:
            continue
        file_index_for_line, import_names_for_line = bullet_index, bullet_imports

        symbols = _symbols_from_span(span_match.group(1))
        resolved = _resolve(symbols, scope if indent else None, file_index_for_line)
        if not resolved:
            # Fallback for spans like `SCENES.register("x")(_build_x_scene)`:
            # try the other identifiers in the span (return annotation stripped)
            # until one resolves to a definition. Import names only count when
            # the span IS an import statement (re-export entries).
            span_text = span_match.group(1)
            allow_imports = span_text.lstrip().startswith(("from ", "import "))
            for ident in _IDENT_RE.findall(span_text.split("->")[0]):
                if "." in ident or (ident in import_names_for_line and not allow_imports):
                    continue
                if ident in file_index_for_line:
                    resolved = [file_index_for_line[ident]]
                    break
        if not resolved:
            unresolved.append(f"{md_path.name}:{i + 1}: cannot locate `{span_match.group(1)}`")
            continue
        lo, hi = min(resolved), max(resolved)
        # Reproduce the entry's own quoting: some bullets write the path in backticks.
        if inline_rel:
            quoted = f"`{inline_rel}`" in line
            prefix = f"`{inline_rel}` " if quoted else f"{inline_rel} "
        else:
            prefix = ""
        span_text_marker = f"L{lo}–{hi}" if (len(resolved) > 1 and hi != lo) else f"L{lo}"
        new_marker = f"({prefix}{span_text_marker})"
        marker = markers[-1]
        if marker.group(0) != new_marker:
            changed += 1
            lines[i] = line[: marker.start()] + new_marker + line[marker.end() :]

    if changed and not check:
        md_path.write_text("".join(lines), encoding="utf-8")
    return changed, unresolved


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report drift without writing; exit 1 on drift"
    )
    args = parser.parse_args(argv)

    total_changed = 0
    all_unresolved: list[str] = []
    for md_path in sorted(METHODS_DIR.glob("*.md")):
        changed, unresolved = refresh_file(md_path, check=args.check)
        total_changed += changed
        all_unresolved.extend(unresolved)
        if changed:
            verb = "drifted" if args.check else "rewrote"
            print(f"{md_path.name}: {verb} {changed} marker(s)")

    if all_unresolved:
        print(
            f"\n{len(all_unresolved)} entr(ies) could not be resolved (fix by hand):",
            file=sys.stderr,
        )
        for item in all_unresolved:
            print(f"  {item}", file=sys.stderr)
    if args.check and total_changed:
        print(
            f"\n{total_changed} marker(s) stale — run `uv run python tools/refresh_methods_linenos.py`."
        )
        return 1
    if not args.check:
        print(f"\nDone: {total_changed} marker(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
