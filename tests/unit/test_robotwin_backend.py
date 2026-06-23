"""RoboTwin 2.0 SAPIEN dual-arm benchmark backend (ADR-0061).

Exercises the openral-side backend, the shipped scene/suite YAMLs, the
``aloha_agilex`` robot manifest, and the ``smolvla-robotwin`` rSkill — all without
booting the SAPIEN sidecar (the heavy py3.10 lerobot-main + RoboTwin venv is
externally provisioned; the sim-tier test skips when it is absent).

Covers:
1. Per-scene ZMQ port derivation (in-band, deterministic, distinct) — same
   shared-port defence as the Isaac backend.
2. ``robotwin/<task>`` → upstream task-name extraction.
3. The typed ``ROSConfigError`` (with the manual recipe) when the sidecar venv is
   unprovisioned and auto-provision is off.
4. Scene + suite YAMLs validate at the official RoboTwin horizon (sapien backend,
   max_steps 300, n_episodes 100).
5. The ``smolvla-robotwin`` rSkill loads and the ADR-0060 task-data gate accepts it
   on every ``robotwin/*`` scene and rejects a foreign scene.
6. The ``aloha_agilex`` robot manifest (14-DoF, 3 cameras) + ``SAPIEN`` enum.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from openral_core import RobotDescription, RSkillManifest
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_core.schemas import BenchmarkScene, PhysicsBackend
from openral_sim.backends.robotwin import (
    _ROBOTWIN_ROBOT_ID,
    _ROBOTWIN_ROOT_ENV,
    _SIDECAR_PORT_MAX,
    _SIDECAR_PORT_MIN,
    _build_robotwin_scene,
    _robotwin_root,
    _robotwin_task_name,
    _scene_default_port,
)
from openral_sim.benchmark import check_benchmark_task_compatibility

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKS = (
    "lift_pot",
    "handover_block",
    "stack_blocks_two",
    "beat_block_hammer",
    "place_empty_cup",
)


# ─── enum ────────────────────────────────────────────────────────────────────


def test_sapien_physics_backend_enum() -> None:
    assert PhysicsBackend.SAPIEN.value == "sapien"


# ─── port derivation ─────────────────────────────────────────────────────────


def test_scene_port_is_in_band() -> None:
    for t in _TASKS:
        port = _scene_default_port(f"robotwin/{t}", _ROBOTWIN_ROBOT_ID)
        assert _SIDECAR_PORT_MIN <= port < _SIDECAR_PORT_MAX


def test_scene_port_is_deterministic_across_calls() -> None:
    a = _scene_default_port("robotwin/lift_pot", _ROBOTWIN_ROBOT_ID)
    b = _scene_default_port("robotwin/lift_pot", _ROBOTWIN_ROBOT_ID)
    assert a == b


def test_distinct_tasks_get_distinct_ports() -> None:
    ports = {_scene_default_port(f"robotwin/{t}", _ROBOTWIN_ROBOT_ID) for t in _TASKS}
    assert len(ports) == len(_TASKS), f"task ports collided: {ports}"
    assert 5757 not in ports  # no fallback to the legacy shared default


# ─── task-name extraction ────────────────────────────────────────────────────


def test_robotwin_task_name_strips_namespace() -> None:
    assert _robotwin_task_name("robotwin/lift_pot") == "lift_pot"
    assert _robotwin_task_name("beat_block_hammer") == "beat_block_hammer"


# ─── unprovisioned sidecar → typed error with recipe ─────────────────────────


def test_sidecar_python_raises_with_recipe_when_unprovisioned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openral_sim.backends import robotwin

    # No override, auto-provision off, and a cache home with no venv.
    monkeypatch.delenv("OPENRAL_ROBOTWIN_SIDECAR_PYTHON", raising=False)
    monkeypatch.setenv("OPENRAL_ROBOTWIN_AUTO_PROVISION", "0")
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SIDECAR_HOME", tmp_path / "robotwin-sidecar")
    with pytest.raises(ROSConfigError, match="RoboTwin sidecar venv not found"):
        robotwin._sidecar_python()


def test_robotwin_root_resolves_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assets = tmp_path / "assets" / "objects" / "objaverse"
    assets.mkdir(parents=True)
    (assets / "list.json").write_text("[]")
    monkeypatch.setenv(_ROBOTWIN_ROOT_ENV, str(tmp_path))

    assert _robotwin_root() == tmp_path.resolve()


def test_robotwin_root_errors_without_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ROBOTWIN_ROOT_ENV, str(tmp_path))

    with pytest.raises(ROSConfigError, match="RoboTwin assets not found"):
        _robotwin_root()


def test_robotwin_sidecar_derives_sapien_sim_time() -> None:
    spec = importlib.util.spec_from_file_location(
        "robotwin_sidecar_for_test",
        _REPO_ROOT / "tools" / "robotwin_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env:
        elapsed_steps = np.array([3], dtype=np.int64)
        control_timestep = 0.05

    assert module._sapien_sim_time_ns(Env()) == 150_000_000


def test_robotwin_launch_argv_passes_checkout_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # _build_robotwin_scene gates on the robotwin wire deps (msgpack/pyzmq), an
    # opt-in group absent from the base test env — skip if unavailable (§1.11).
    pytest.importorskip("msgpack", reason="robotwin wire deps (msgpack/pyzmq) not installed")
    pytest.importorskip("zmq", reason="robotwin wire deps (msgpack/pyzmq) not installed")
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_sim import _deps
    from openral_sim.backends import robotwin

    captured: dict[str, list[str]] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["argv"] = list(cast(dict[str, Any], kwargs)["launch_argv"])

        def connect(self) -> None:
            return None

    monkeypatch.setattr(_deps, "ensure_backend_deps", lambda _name: None)
    monkeypatch.setattr(robotwin, "_sidecar_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(
        robotwin, "_locate_sidecar_script", lambda: tmp_path / "robotwin_sidecar.py"
    )
    monkeypatch.setattr(robotwin, "_robotwin_root", lambda: tmp_path / "RoboTwin")
    monkeypatch.setattr(robotwin, "SidecarClient", FakeClient)

    env_cfg = SimEnvironment(
        robot_id="aloha_agilex",
        scene=SceneSpec(id="robotwin", backend=PhysicsBackend.SAPIEN, cameras=["camera1"]),
        task=TaskSpec(
            id="robotwin/lift_pot",
            scene_id="robotwin",
            instruction="lift the pot",
            success_key="is_success",
            max_steps=300,
        ),
        vla=VLASpec(id="random", weights_uri="none"),
    )

    _build_robotwin_scene(env_cfg)

    argv = captured["argv"]
    assert "--robotwin-root" in argv
    assert argv[argv.index("--robotwin-root") + 1] == str(tmp_path / "RoboTwin")


# ─── shipped scene + suite YAMLs ─────────────────────────────────────────────


def _shipped_robotwin_scenes() -> list[Path]:
    found: list[Path] = []
    for path in sorted((_REPO_ROOT / "scenes").rglob("robotwin_*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and doc.get("scene", {}).get("backend") == "sapien":
            found.append(path)
    return found


def test_shipped_robotwin_scenes_exist() -> None:
    assert len(_shipped_robotwin_scenes()) == len(_TASKS)


def test_robotwin_scenes_validate_at_official_horizon() -> None:
    for path in _shipped_robotwin_scenes():
        scene = BenchmarkScene.from_yaml(str(path))
        assert scene.scene.id == "robotwin"
        assert scene.scene.backend == PhysicsBackend.SAPIEN
        assert scene.robot_id == _ROBOTWIN_ROBOT_ID
        assert scene.task.max_steps == 300  # LeRobot robotwin episode_length
        assert scene.n_episodes == 100  # RoboTwin official protocol
        assert scene.task.success_key == "is_success"


def test_robotwin_scenes_do_not_pin_a_port() -> None:
    offenders = [
        str(p.relative_to(_REPO_ROOT))
        for p in _shipped_robotwin_scenes()
        if "port" in (yaml.safe_load(p.read_text())["scene"].get("backend_options") or {})
    ]
    assert not offenders, f"RoboTwin scenes re-pin a shared port: {offenders}"


def test_robotwin_suite_validates() -> None:
    import openral_core as oc

    suite = oc.load_benchmark_suite(str(_REPO_ROOT / "benchmarks" / "robotwin.yaml"))
    oc.raise_on_invalid_suite(suite, suite_id="robotwin")
    assert [s.task.id for s in suite] == [f"robotwin/{t}" for t in _TASKS]


# ─── robot manifest ──────────────────────────────────────────────────────────


def test_aloha_agilex_manifest() -> None:
    r = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "aloha_agilex" / "robot.yaml"))
    assert r.name == "aloha_agilex"
    assert len(r.joints) == 14
    assert [s.name for s in r.sensors] == ["camera1", "camera2", "camera3"]
    assert r.action_spec.dim == 14
    assert "aloha_agilex" in r.capabilities.embodiment_tags


# ─── rSkill + ADR-0060 task-data gate ────────────────────────────────────────


def _robotwin_rskill() -> RSkillManifest:
    return RSkillManifest.from_yaml(
        str(_REPO_ROOT / "rskills" / "smolvla-robotwin" / "rskill.yaml")
    )


def test_smolvla_robotwin_manifest_loads() -> None:
    m = _robotwin_rskill()
    assert m.model_family == "smolvla"
    assert m.embodiment_tags == ["aloha_agilex"]
    assert m.evaluated_tasks == ["robotwin"]
    assert m.state_contract is not None
    assert m.state_contract.dim == 14
    assert m.action_contract.dim == 14


def test_gate_accepts_rskill_on_every_robotwin_scene() -> None:
    m = _robotwin_rskill()
    for t in _TASKS:
        # Must not raise (scene-id family match on "robotwin").
        check_benchmark_task_compatibility(m, task_id=f"robotwin/{t}", scene_id="robotwin")


def test_gate_rejects_rskill_on_foreign_scene() -> None:
    m = _robotwin_rskill()
    with pytest.raises(ROSCapabilityMismatch):
        check_benchmark_task_compatibility(
            m, task_id="maniskill3/PickCube-v1", scene_id="maniskill3"
        )
