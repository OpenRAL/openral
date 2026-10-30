"""Installed-package boundary fake for ``importlib.metadata.entry_points``.

CLAUDE.md §1.11 boundary double. Entry points come from the installed
distributions' metadata, a process/packaging boundary the unit tier cannot
change without installing a package into the test venv. This fake writes a
real module onto ``sys.path`` and returns a real ``EntryPoint`` pointing at it,
so ``EntryPoint.load()`` really imports and runs that module's
``@POLICIES.register`` calls. Only the metadata lookup is replaced.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path

import pytest


def install_policy_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    *ids: str,
) -> list[str]:
    """Publish ``ids`` in ``openral.policies``, each registering the real ``zero`` adapter.

    ``target`` is the module whose ``entry_points`` name is replaced
    (``openral_sim.registry``). Returns the groups that were queried, in order.
    """
    lines = ["from openral_sim.registry import POLICIES", "from openral_sim.policies import mock"]
    lines += [f"POLICIES.register({i!r})(mock._build_zero_policy)" for i in ids]
    module = f"fake_wam_{tmp_path.name}"  # unique: each test re-imports and re-registers
    (tmp_path / f"{module}.py").write_text("\n".join(lines) + "\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls: list[str] = []

    def entry_points(group: str) -> tuple[EntryPoint, ...]:
        calls.append(group)
        return (EntryPoint(name=ids[0], value=module, group=group),)

    monkeypatch.setattr(target, "entry_points", entry_points)
    return calls
