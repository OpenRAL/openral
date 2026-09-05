"""Guard: every RoboCasa-gated sim test is listed in robocasa_sim_tests.sh.

The RoboCasa suite has **no CI surface and cannot be given one** — see
``scripts/robocasa_sim_tests.sh`` for the asset arithmetic and for why a
self-hosted runner was evaluated and rejected. That makes
``scripts/robocasa_sim_tests.sh`` the only definition of what the suite *is*,
and this test the only thing keeping it honest: a gated file missing from
TARGETS is a test nobody will ever run again, and nothing else would say so.

That is not hypothetical. ``test_kernel_fridge_layout_pin_start_state.py`` was
added by #224 and sat outside any list until #232 went looking, which is how
the gap was found at all.

Sibling of :mod:`tests.unit.test_ros_live_targets`, same reasoning, different
suite — except that one guards a list with a CI lane behind it and this one
guards a list with only a human behind it, which makes it matter more, not
less.

Run with:
    uv run pytest tests/unit/test_robocasa_sim_targets.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "robocasa_sim_tests.sh"
_SIM = _REPO_ROOT / "tests" / "sim"

#: The gate every RoboCasa test carries. robocasa (robosuite >=1.5) and libero
#: (robosuite 1.4) are mutually exclusive dependency groups, so the backend is
#: never merely optional — it is either provisioned or the test skips.
_GATE = 'importorskip("robocasa")'


def _script_targets() -> set[str]:
    """The repo-relative paths inside the TARGETS=( … ) block."""
    body = _SCRIPT.read_text(encoding="utf-8")
    # Anchor the closing paren to its own line so a parenthesis inside an
    # array comment can never truncate the parse.
    block = re.search(r"TARGETS=\((.*?)^\)$", body, flags=re.DOTALL | re.MULTILINE)
    assert block is not None, f"no TARGETS=() block in {_SCRIPT}"
    return {
        stripped
        for line in block.group(1).splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }


def _gated_tests() -> set[str]:
    """Every RoboCasa-gated test file under ``tests/sim/``, repo-relative."""
    return {
        str(path.relative_to(_REPO_ROOT))
        for path in _SIM.rglob("test_*.py")
        if _GATE in path.read_text(encoding="utf-8")
    }


def test_every_robocasa_gated_sim_test_is_in_targets() -> None:
    gated = _gated_tests()
    assert gated, f"no {_GATE} tests found under tests/sim/ — glob or gate string moved?"
    missing = gated - _script_targets()
    assert not missing, (
        f"RoboCasa-gated tests missing from {_SCRIPT.relative_to(_REPO_ROOT)} TARGETS "
        f"(they will run on NO CI surface): {sorted(missing)}"
    )


def test_targets_only_lists_files_that_exist() -> None:
    stale = {t for t in _script_targets() if not (_REPO_ROOT / t).exists()}
    assert not stale, f"TARGETS lists files that do not exist: {sorted(stale)}"
