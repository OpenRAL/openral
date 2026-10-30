"""``POLICIES`` resolves ids from the ``openral.policies`` entry-point group.

This is how an installed package (OpenRAL Pro's WAM adapters) adds a policy
without open importing it. A real module on ``sys.path`` registers the real
``zero`` policy adapter through a real ``EntryPoint``. Only the entry-point
enumeration (installed-package metadata, a process boundary) is substituted.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError
from openral_sim import registry
from openral_sim.policies import mock
from openral_sim.policy import PolicyAdapter


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *ids: str) -> list[str]:
    lines = ["from openral_sim.registry import POLICIES", "from openral_sim.policies import mock"]
    lines += [f"POLICIES.register({i!r})(mock._build_zero_policy)" for i in ids]
    module = f"fake_wam_{tmp_path.name}"  # unique: each test re-imports and re-registers
    (tmp_path / f"{module}.py").write_text("\n".join(lines) + "\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls: list[str] = []

    def fake_entry_points(group: str) -> tuple[EntryPoint, ...]:
        calls.append(group)
        return (EntryPoint(name=ids[0], value=module, group=group),)

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)
    return calls


def test_unknown_policy_id_loads_entry_points_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh: registry._Registry[PolicyAdapter] = registry._Registry(
        "policy", entry_point_group="openral.policies"
    )
    monkeypatch.setattr(registry, "POLICIES", fresh)
    calls = _install(tmp_path, monkeypatch, "fake_wam")

    spec = VLASpec(id="fake_wam", weights_uri="hf://none/none", extra={"action_dim": 7})
    env_cfg = SimpleNamespace(vla=spec, scene=SimpleNamespace(backend_options={}))
    adapter = fresh.get("fake_wam")(env_cfg)
    adapter.reset()
    np.testing.assert_array_equal(adapter.step({}, "pick"), np.zeros(7, np.float32))
    assert "fake_wam" in fresh
    with pytest.raises(ROSConfigError, match="unknown policy id 'nope'"):
        fresh.get("nope")
    assert calls == ["openral.policies"]


def test_an_installed_exact_id_wins_over_a_builtin_family_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh: registry._Registry[PolicyAdapter] = registry._Registry(
        "policy", entry_point_group="openral.policies"
    )
    fresh.register("zero")(mock._build_random_policy)  # the built-in family prefix
    monkeypatch.setattr(registry, "POLICIES", fresh)
    _install(tmp_path, monkeypatch, "zero/wam")

    assert fresh.get("zero/wam") is mock._build_zero_policy
    assert fresh.get("zero/other") is mock._build_random_policy
