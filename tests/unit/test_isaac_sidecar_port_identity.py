"""Isaac sidecar: per-scene port derivation + identity-checked ping handshake.

Regression guard for the shared-port bug where all Isaac scenes defaulted to ZMQ
port 5757, so ``SidecarClient`` could silently adopt a *lingering* sidecar from a
different scene and serve its wrong layout (wrong action_dim / camera count /
frame size). Two defences, both exercised here without booting Isaac:

1. :func:`_scene_default_port` gives each scene its own stable default port.
2. :meth:`SidecarClient._assert_identity` rejects an existing sidecar whose
   ``ping`` identity contradicts the requested scene.

See ``docs/audit/test-audit-report.md`` §5a.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import (
    _SIDECAR_PORT_MAX,
    _SIDECAR_PORT_MIN,
    _scene_default_port,
)
from openral_sim.sidecar import SidecarClient

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ─── per-scene port derivation ───────────────────────────────────────────────

# Real Isaac scenes that collided on the old hard 5757 default.
_LIFT = ("isaac_sim/lift_cube", "franka_panda", "lift_cube")
_BOWL = ("isaac_sim/bowl_plate", "franka_panda", "bowl_plate")
_MANIFEST = ("isaac_sim/manifest", "panda_mobile", "manifest")


def test_scene_port_is_in_band() -> None:
    for scene in (_LIFT, _BOWL, _MANIFEST):
        port = _scene_default_port(*scene)
        assert _SIDECAR_PORT_MIN <= port < _SIDECAR_PORT_MAX


def test_scene_port_is_deterministic_across_calls() -> None:
    # Stable across processes (hashlib, not the PYTHONHASHSEED-salted builtin
    # hash) — the spawn process and a later client process must agree.
    assert _scene_default_port(*_MANIFEST) == _scene_default_port(*_MANIFEST)


def test_distinct_scenes_get_distinct_ports() -> None:
    ports = {_scene_default_port(*s) for s in (_LIFT, _BOWL, _MANIFEST)}
    assert len(ports) == 3, f"scenes collided on a shared port: {ports}"
    # And none falls back to the legacy shared default.
    assert 5757 not in ports


# ─── shipped scene YAMLs must not re-pin a shared port ───────────────────────


def _shipped_isaac_scenes() -> list[Path]:
    """Every in-tree scene YAML whose backend is the Isaac sidecar."""
    found: list[Path] = []
    for path in sorted((_REPO_ROOT / "scenes").rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and doc.get("scene", {}).get("backend") == "isaacsim":
            found.append(path)
    return found


def test_shipped_isaac_scenes_exist() -> None:
    # Sanity: the glob actually finds the Isaac scenes (else the guards below
    # would vacuously pass).
    assert _shipped_isaac_scenes(), "no isaacsim scenes found — glob/layout changed"


def test_no_shipped_isaac_scene_pins_a_port() -> None:
    """The original bug: every Isaac scene hard-coded ``port: 5757`` in
    backend_options, which *overrides* :func:`_scene_default_port` and made all
    scenes share one endpoint. Ship scenes must leave ``port`` unset so each gets
    its own derived port; an operator can still override per-invocation.
    """
    offenders = [
        str(p.relative_to(_REPO_ROOT))
        for p in _shipped_isaac_scenes()
        if "port" in (yaml.safe_load(p.read_text())["scene"].get("backend_options") or {})
    ]
    assert not offenders, f"Isaac scenes re-pin a shared port (regression): {offenders}"


def test_shipped_isaac_scenes_derive_distinct_ports() -> None:
    """The real shipped scenes (with deploy task-id synthesis applied) must map to
    distinct sidecar ports so two concurrent Isaac scenes never collide.
    """
    from openral_hal.sim_bringup import _synthesise_deploy_task_id

    ports: dict[int, str] = {}
    for path in _shipped_isaac_scenes():
        doc = yaml.safe_load(path.read_text())
        scene = doc["scene"]
        opts = scene.get("backend_options") or {}
        layout = str(opts.get("layout", "lift_cube"))
        robot = doc.get("robot_id") or "franka_panda"
        # deploy scenes (no task:) get a synthesised noop task id, matching the
        # runtime path; sim scenes carry their own task.id.
        task = doc.get("task")
        task_id = task["id"] if task else _synthesise_deploy_task_id(scene["id"])
        port = _scene_default_port(task_id, robot, layout)
        assert port not in ports, f"port {port} collides: {path.name} vs {ports[port]}"
        ports[port] = path.name


# ─── identity-checked ping ───────────────────────────────────────────────────


def _client(expected: dict[str, object] | None) -> SidecarClient:
    return SidecarClient(
        name="isaac",
        host="127.0.0.1",
        port=23456,
        timeout_ms=1000,
        boot_timeout_s=1.0,
        launch_argv=["true"],
        auto_spawn=False,
        expected_identity=expected,
    )


def test_assert_identity_accepts_matching_sidecar() -> None:
    client = _client({"task": "isaac_sim/manifest", "layout": "manifest"})
    # Matching ping (plus extra keys like action_dim) — must not raise.
    client._assert_identity(
        {"ok": True, "task": "isaac_sim/manifest", "layout": "manifest", "action_dim": 11},
        "tcp://127.0.0.1:23456",
    )


def test_assert_identity_rejects_foreign_scene() -> None:
    client = _client({"task": "isaac_sim/manifest", "layout": "manifest"})
    # A lingering lift_cube sidecar answering on our port — must raise loudly.
    with pytest.raises(ROSConfigError, match="already serving a different scene"):
        client._assert_identity(
            {"ok": True, "task": "isaac_sim/lift_cube", "layout": "lift_cube", "action_dim": 8},
            "tcp://127.0.0.1:23456",
        )


def test_assert_identity_tolerates_missing_keys() -> None:
    # Back-compat: an older sidecar that does not report task/layout is adopted
    # (no contradiction to assert on) rather than spuriously rejected.
    client = _client({"task": "isaac_sim/manifest", "layout": "manifest"})
    client._assert_identity({"ok": True, "action_dim": 11}, "tcp://127.0.0.1:23456")


def test_no_expected_identity_is_a_noop() -> None:
    client = _client(None)
    client._assert_identity({"ok": True, "task": "whatever"}, "tcp://127.0.0.1:23456")
