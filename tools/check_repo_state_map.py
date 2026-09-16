"""Check ``docs/architecture/repo-state-map.html`` against the actual tree.

The map is hand-edited (CLAUDE.md §4.3), so it drifts in two mechanically
detectable ways. Both have shipped to master before:

* **Dead ``pkg:`` pointers.** A card names a file or directory that no longer
  exists — or never did. The "Examples" card sat at ``status: green`` against
  an ``examples/`` directory with no git history in this repo.
* **Stale asserted counts.** A card's ``desc`` opens with ``"<N> files."`` or
  ``"<N> manifests:"`` and the number rots as the tree grows. All five such
  counts were wrong at once (unit 190 vs 379, sim 26 vs 54, integration 18 vs
  42, HIL 9 vs 12, manifests 49 vs 48). Counts are held to within
  ``COUNT_TOLERANCE``, not to the digit: the map is a dated snapshot, and a
  hook that went red every time somebody added one test would be turned off
  long before it caught the next 190-vs-379.

Everything else on the map is prose a human has to judge; this only pins the
part a machine can. A clean run is not a statement that the map is *accurate*,
only that it does not point at missing files or carry an arithmetic lie.

Usage::

    uv run python tools/check_repo_state_map.py            # report, exit 1 on drift
    uv run python tools/check_repo_state_map.py --quiet    # exit code only

Not part of the runtime; CI-adjacent doc tooling only.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = REPO_ROOT / "docs" / "architecture" / "repo-state-map.html"

# A JS string literal, tolerating the escaped quotes the longer descs carry.
_STRING = r'"((?:[^"\\]|\\.)*)"'
_PKG_RE = re.compile(rf"pkg:\s*{_STRING}")
_DESC_RE = re.compile(rf"desc:\s*{_STRING}")
_COUNT_RE = re.compile(r"^(\d+)\s+(files|manifests)\b")
COUNT_TOLERANCE = 0.10

# A parenthesised status marker means the token is a plan, not a path on disk.
_STATUS_MARKERS = (
    "(planned",
    "(private",
    "(separate",
    "(banned",
    "(in dev",
    "(in-progress",
    "(intentionally",
)
_SEARCH_EXCLUDES = (".git", ".venv", "site", "node_modules", "__pycache__")


@cache
def _repo_py_basenames() -> frozenset[str]:
    """Every ``.py`` basename in the tree, for the "does it live anywhere" fallback."""
    names: set[str] = set()
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SEARCH_EXCLUDES for part in path.parts):
            continue
        names.add(path.name)
    return frozenset(names)


def iter_cards(html: str) -> list[tuple[str, str]]:
    """Pair each ``pkg:`` value with the ``desc:`` that follows it in the same card."""
    cards: list[tuple[str, str]] = []
    for pkg_match in _PKG_RE.finditer(html):
        desc_match = _DESC_RE.search(html, pkg_match.end())
        cards.append((pkg_match.group(1), desc_match.group(1) if desc_match else ""))
    return cards


def _resolves(head: str, token: str) -> bool:
    """Is ``token`` a real path, read the way a reader of this card would read it?

    Three spellings are all in use on the map and all count as resolved:
    straight from the repo root (``packages/openral_hal_so100``), relative to
    the card's leading package under the workspace src layout (``policies/
    pi05.py`` under a ``python/sim`` head), or a bare module the card names
    without a directory at all (``openarm.py``).
    """
    if (REPO_ROOT / token).exists():
        return True
    base = Path(head)
    if (
        base.parts[:1] == ("python",)
        and len(base.parts) >= 2
        and (REPO_ROOT / base / "src" / f"openral_{base.parts[1]}" / token).exists()
    ):
        return True
    return token.endswith(".py") and Path(token).name in _repo_py_basenames()


def check_paths(cards: list[tuple[str, str]]) -> list[str]:
    """Report ``pkg:`` tokens that name nothing on disk."""
    problems: list[str] = []
    for pkg, _ in cards:
        if any(marker in pkg for marker in _STATUS_MARKERS):
            continue
        tokens = [t.strip() for chunk in pkg.split(" · ") for t in chunk.split(" + ")]
        head = tokens[0].split(" ")[0]
        for raw in tokens:
            token = raw.split(" ")[0]
            # Brace globs stand for several files; bare prose ("all packages") is not a
            # path, and only a token that looks like one is worth resolving at all.
            if not token or "{" in token or ("/" not in token and not token.endswith(".py")):
                continue
            if not _resolves(head, token):
                problems.append(f"{pkg}\n      dead path: {token}")
    return problems


def check_counts(cards: list[tuple[str, str]]) -> list[str]:
    """Report ``desc`` counts that have drifted past ``COUNT_TOLERANCE``."""
    problems: list[str] = []
    for pkg, desc in cards:
        match = _COUNT_RE.match(desc)
        directory = REPO_ROOT / pkg.rstrip("/")
        if match is None or not directory.is_dir():
            continue
        claimed, noun = int(match.group(1)), match.group(2)
        pattern = "test_*.py" if noun == "files" else "*/rskill.yaml"
        actual = len(list(directory.glob(pattern)))
        if abs(claimed - actual) > max(actual, 1) * COUNT_TOLERANCE:
            problems.append(
                f"{pkg}\n      claims {claimed} {noun}, found {actual} ({pattern}) — "
                f"past the {COUNT_TOLERANCE:.0%} tolerance"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--quiet", action="store_true", help="exit code only, no report")
    args = parser.parse_args(argv)

    cards = iter_cards(MAP_PATH.read_text(encoding="utf-8"))
    problems = check_paths(cards) + check_counts(cards)

    if not problems:
        if not args.quiet:
            print(f"repo-state-map: {len(cards)} cards checked, no drift.")
        return 0
    if not args.quiet:
        print(f"repo-state-map: {len(problems)} drift(s) in {len(cards)} cards:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nFix the map (or the tree) — see CLAUDE.md §4.3.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
