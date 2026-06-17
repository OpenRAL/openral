"""Auto-discovery of ``robots/<id>/robot.yaml`` into the eval ROBOTS registry.

Replaces the older ``_BUILTIN_ROBOTS`` tuple pattern: dropping a new
``robots/<id>/robot.yaml`` is enough to make ``robot_id: <id>`` resolvable
in any ``SimEnvironment`` config. This test pins the contract.

Coverage
--------
- Every directory under ``robots/`` with a ``robot.yaml`` is registered
  under :data:`openral_sim.ROBOTS`.
- ``$OPENRAL_ROBOTS_DIR`` overrides the in-tree scan (out-of-tree
  manifests).
- A drop-in YAML inside an override directory shows up after the module
  is reloaded.
"""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path

import pytest


def test_in_tree_robots_are_all_registered() -> None:
    """Every directory under ``robots/`` with a ``robot.yaml`` resolves."""
    from openral_sim.registry import ROBOTS

    repo_robots = Path("robots")
    expected = sorted(
        d.name for d in repo_robots.iterdir() if d.is_dir() and (d / "robot.yaml").is_file()
    )
    registered = sorted(ROBOTS.names())
    missing = [r for r in expected if r not in registered]
    assert not missing, f"robots/ subdirectories not auto-registered: {missing}"


def test_drop_in_robot_via_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``$OPENRAL_ROBOTS_DIR`` redirects discovery."""
    robots_dir = tmp_path / "robots"
    drop_in = robots_dir / "test_drop_in_robot"
    drop_in.mkdir(parents=True)
    (drop_in / "robot.yaml").write_text(
        textwrap.dedent("""\
            name: "test_drop_in_robot"
            embodiment_kind: "manipulator"
            joints:
              - name: "j0"
                joint_type: "revolute"
                parent_link: "base"
                child_link: "link_0"
                axis_xyz: [0.0, 0.0, 1.0]
                position_limits: [-1.0, 1.0]
                velocity_limit: 1.0
                effort_limit: 1.0
                actuator_kind: "servo"
            capabilities:
              supported_control_modes:
                - "joint_position"
              embodiment_tags: ["test_drop_in"]
            safety: {}
        """)
    )

    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(robots_dir))

    # Re-import the adapter module so the discover scan runs against the
    # override path. Use a fresh registry to keep state hygienic.
    from openral_sim import registry as registry_module
    from openral_sim.registry import _Registry  # type: ignore[attr-defined]

    fresh_robots = _Registry("robot")
    monkeypatch.setattr(registry_module, "ROBOTS", fresh_robots)
    robots_mod = importlib.import_module("openral_sim.policies.robots")
    importlib.reload(robots_mod)

    assert "test_drop_in_robot" in fresh_robots.names()
    factory = fresh_robots.get("test_drop_in_robot")
    desc = factory()
    assert desc.name == "test_drop_in_robot"


def test_discovery_degrades_when_installed_as_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """No repo root + no override (a ``pip install openral-cli`` wheel) →
    discovery returns ``[]`` instead of raising at import.

    Regression: ``openral_sim.policies.robots`` is imported at CLI startup, so
    when ``_find_repo_root`` raised on a wheel install (no ``robots/`` ancestor)
    it crashed *every* ``openral`` command — including ``--help`` / ``doctor`` —
    before Typer ever ran. It must degrade to zero registered robots instead.
    """
    import openral_sim.policies.robots as robots_mod

    monkeypatch.delenv("OPENRAL_ROBOTS_DIR", raising=False)
    # Simulate the wheel layout: this module has no repo-root ancestor.
    monkeypatch.setattr(robots_mod, "_find_repo_root", lambda: None)

    assert robots_mod._robots_search_dir() is None
    assert robots_mod._discover_robot_ids() == []  # must not raise

    # A real robot lookup still fails loudly, with an actionable message.
    with pytest.raises(Exception, match=r"OPENRAL_ROBOTS_DIR"):
        robots_mod._resolve_manifest("panda_mobile")
