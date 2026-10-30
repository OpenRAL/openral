"""``POLICIES`` resolves ids from the ``openral.policies`` entry-point group.

This is how an installed package (OpenRAL Pro's WAM adapters) adds a policy
without open importing it. No mocks of the registry itself: a real module on
``sys.path`` registers through a real ``EntryPoint``.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim import registry


def test_unknown_policy_id_loads_entry_points_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fake_wam_pkg.py").write_text(
        "from openral_sim.registry import POLICIES\n"
        "@POLICIES.register('fake_wam')\n"
        "def build(env_cfg):\n"
        "    return 'adapter'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    fresh: registry._Registry[str] = registry._Registry(
        "policy", entry_point_group="openral.policies"
    )
    monkeypatch.setattr(registry, "POLICIES", fresh)
    calls: list[str] = []

    def fake_entry_points(group: str) -> tuple[EntryPoint, ...]:
        calls.append(group)
        return (EntryPoint(name="fake_wam", value="fake_wam_pkg", group=group),)

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)

    assert fresh.get("fake_wam")(None) == "adapter"
    assert "fake_wam" in fresh
    with pytest.raises(ROSConfigError, match="unknown policy id 'nope'"):
        fresh.get("nope")
    assert calls == ["openral.policies"]
