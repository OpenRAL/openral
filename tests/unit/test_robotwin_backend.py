"""RoboTwin 2.0 SAPIEN dual-arm benchmark backend.

Exercises the openral-side backend, the shipped scene/suite YAMLs, the
``aloha_agilex`` robot manifest, and the ``smolvla-robotwin`` rSkill — all without
booting the SAPIEN sidecar (the heavy py3.12 lerobot + RoboTwin venv is
externally provisioned; the sim-tier test skips when it is absent).

Covers:
1. Per-scene ZMQ port derivation (in-band, deterministic, distinct) — same
   shared-port defence as the Isaac backend.
2. ``robotwin/<task>`` → upstream task-name extraction.
3. The typed ``ROSConfigError`` (with the manual recipe) when the sidecar venv is
   unprovisioned and auto-provision is off.
4. Scene + suite YAMLs validate at the official RoboTwin horizon (sapien backend,
   max_steps 300, n_episodes 100).
5. The ``smolvla-robotwin`` rSkill loads and the task-data gate accepts it
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
from numpy.typing import NDArray
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


def test_robotwin_sidecar_derives_wrapped_sapien_sim_time() -> None:
    spec = importlib.util.spec_from_file_location(
        "robotwin_sidecar_wrapped_for_test",
        _REPO_ROOT / "tools" / "robotwin_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Inner:
        elapsed_steps = 4
        control_freq = 20

    class Env:
        _env = Inner()

    assert module._sapien_sim_time_ns(Env()) == 200_000_000


def test_robotwin_sidecar_fallback_time_starts_at_zero_and_steps() -> None:
    spec = importlib.util.spec_from_file_location(
        "robotwin_sidecar_fallback_for_test",
        _REPO_ROOT / "tools" / "robotwin_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env:
        def step(
            self, action: NDArray[np.float32]
        ) -> tuple[dict[str, object], float, bool, bool, dict[str, bool]]:
            return {}, 0.0, False, False, {"is_success": False}

    scene = object.__new__(module._RoboTwinEnv)
    scene._env = Env()
    scene._success_key = "is_success"
    scene._fallback_dt_ns = 40_000_000
    scene._fallback_time_ns = 0
    scene._unwrap_obs = lambda _obs: {"images": {}, "state": np.zeros(14), "task": ""}

    assert scene.sim_time_ns() == 0
    reply = scene.step(np.zeros(14, dtype=np.float32))

    assert reply["sim_time_ns"] == 40_000_000


def test_robotwin_sidecar_builds_lerobot_07_env() -> None:
    # The sidecar constructs lerobot 0.7's RoboTwinEnv (SAPIEN is deferred to the
    # first reset, so no RoboTwin checkout is needed); catches a moved import path.
    pytest.importorskip("lerobot.sim.integrations.robotwin")
    spec = importlib.util.spec_from_file_location(
        "robotwin_sidecar_build_for_test",
        _REPO_ROOT / "tools" / "robotwin_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    env = module._RoboTwinEnv(
        task="lift_pot",
        instruction="lift the pot with both arms",
        cameras=["top", "wrist_left", "wrist_right"],
        obs_height=64,
        obs_width=64,
        episode_length=10,
        success_key="is_success",
        fallback_dt_s=1.0 / 25.0,
    )

    assert type(env._env).__module__ == "lerobot.sim.integrations.robotwin"
    assert env.action_dim == 14


def _fake_default_venv(home: Path, marker: str | None) -> Path:
    py = home / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    if marker is not None:
        (home / ".venv" / ".deps-installed").write_text(marker)
    return py


def test_sidecar_python_refuses_venv_from_superseded_pins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openral_sim._sidecar_common import spec_marker
    from openral_sim.backends import robotwin

    home = tmp_path / "robotwin-sidecar"
    # A venv provisioned before the lock (lerobot 0.6.0 pins) carries another marker.
    old_spec = ("lerobot==0.6.0", "sapien==3.0.3", "mplib==0.2.1")
    _fake_default_venv(home, spec_marker(old_spec))
    monkeypatch.delenv("OPENRAL_ROBOTWIN_SIDECAR_PYTHON", raising=False)
    monkeypatch.setenv("OPENRAL_ROBOTWIN_AUTO_PROVISION", "0")
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SIDECAR_HOME", home)

    with pytest.raises(ROSConfigError, match="built from another lock"):
        robotwin._sidecar_python()


def test_sidecar_python_returns_venv_built_from_current_pins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openral_sim._sidecar_common import spec_marker
    from openral_sim.backends import robotwin

    home = tmp_path / "robotwin-sidecar"
    py = _fake_default_venv(home, spec_marker(robotwin._robotwin_venv_spec()))
    monkeypatch.delenv("OPENRAL_ROBOTWIN_SIDECAR_PYTHON", raising=False)
    monkeypatch.setenv("OPENRAL_ROBOTWIN_AUTO_PROVISION", "0")
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SIDECAR_HOME", home)

    assert robotwin._sidecar_python() == py


def _x86_64_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")


def test_provision_syncs_the_hash_lock_and_writes_its_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ``run_cmd`` is the subprocess boundary (§1.11); ``ensure_pip_venv`` runs for real,
    # so its sentinel check + write are exercised.
    from openral_sim._sidecar_common import spec_marker
    from openral_sim.backends import robotwin

    home = tmp_path / "robotwin-sidecar"
    calls: list[list[str]] = []

    def fake_run_cmd(_label: str, cmd: list[str], **_kw: Any) -> None:
        calls.append(list(cmd))
        if cmd[1:2] == ["venv"]:  # `uv venv <dir> --python 3.12`
            py = Path(cmd[2]) / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text("")

    _x86_64_linux(monkeypatch)
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SIDECAR_HOME", home)
    monkeypatch.setattr("openral_sim._sidecar_common.run_cmd", fake_run_cmd)
    monkeypatch.setattr(robotwin, "run_cmd", fake_run_cmd)
    # An operator-owned root: provisioning returns before cloning or fetching assets.
    monkeypatch.setenv(_ROBOTWIN_ROOT_ENV, str(tmp_path / "own-RoboTwin"))

    py = robotwin._provision_robotwin_venv()

    _venv_cmd, sync = calls
    assert sync[1:3] == ["pip", "sync"]
    assert "--require-hashes" in sync
    assert sync[-1] == str(robotwin._robotwin_lock())
    sentinel = (home / ".venv" / ".deps-installed").read_text(encoding="utf-8")
    assert sentinel == spec_marker(robotwin._robotwin_venv_spec())
    # ... which is exactly what the non-provisioning lookup accepts.
    monkeypatch.delenv("OPENRAL_ROBOTWIN_SIDECAR_PYTHON", raising=False)
    assert robotwin._find_sidecar_python() == py

    # A second provision reuses the venv: no further commands.
    robotwin._provision_robotwin_venv()
    assert len(calls) == 2


def test_provision_refuses_a_non_x86_64_linux_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_sim.backends import robotwin

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")

    with pytest.raises(ROSConfigError, match="x86_64 Linux only"):
        robotwin._provision_robotwin_venv()


def test_lock_is_found_without_the_sidecar_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A relocated sidecar script does not move the lock the venv is checked against.
    from openral_sim.backends import robotwin

    script = tmp_path / "robotwin_sidecar.py"
    script.write_text("")
    monkeypatch.setenv("OPENRAL_ROBOTWIN_SIDECAR_SCRIPT", str(script))

    assert robotwin._robotwin_lock() == (
        _REPO_ROOT / "tools" / "sidecar_requirements" / "robotwin.lock"
    )


def test_fetch_assets_downloads_unzips_and_configures(tmp_path: Path) -> None:
    # The download + config steps run in the sidecar venv (process boundary, doubled);
    # the unzip is real.
    import zipfile

    from openral_sim.backends import robotwin

    root = tmp_path / "RoboTwin"
    (root / "assets").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts" / "update_embodiment_config_path.py").write_text("")
    calls: list[tuple[list[str], Path]] = []

    def fake_run_cmd(_label: str, cmd: list[str], *, cwd: Path, **_kw: Any) -> None:
        calls.append((list(cmd), cwd))
        if "snapshot_download" in cmd[-1]:
            for name in robotwin._ROBOTWIN_ASSET_ZIPS:
                with zipfile.ZipFile(cwd / name, "w") as zf:
                    zf.writestr(f"{name.removesuffix('.zip')}/marker.txt", name)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(robotwin, "run_cmd", fake_run_cmd)
        robotwin._fetch_robotwin_assets(root, Path("/venv/bin/python"))

    (download, download_cwd), (config, config_cwd) = calls
    assert download_cwd == root / "assets"
    # No resume_download: huggingface-hub 2.x (the lock's) rejects it.
    assert "resume_download" not in download[-1]
    assert config == ["/venv/bin/python", "scripts/update_embodiment_config_path.py"]
    assert config_cwd == root
    for name in robotwin._ROBOTWIN_ASSET_ZIPS:
        assert not (root / "assets" / name).exists()
        assert (root / "assets" / name.removesuffix(".zip") / "marker.txt").is_file()


def _lock_pins() -> dict[str, str]:
    lock = (_REPO_ROOT / "tools" / "sidecar_requirements" / "robotwin.lock").read_text()
    return {
        line.split("==")[0]: line.split("==")[1].split()[0]
        for line in lock.splitlines()
        if line and line[0].isalpha() and "==" in line
    }


def test_robotwin_lock_pins_the_sidecar_stack() -> None:
    pins = _lock_pins()
    # numpy is mplib 0.2.1's (robotwin-override.txt).
    assert pins["lerobot"] == "0.7.0"
    assert pins["mplib"] == "0.2.1"
    assert pins["sapien"] == "3.0.3"
    assert pins["numpy"] == "1.26.4"


def test_every_robotwin_lock_pin_is_hashed() -> None:
    # Installed with --require-hashes: each `name==ver` block needs its own hash line.
    lines = (_REPO_ROOT / "tools" / "sidecar_requirements" / "robotwin.lock").read_text()
    blocks: dict[str, list[str]] = {}
    current = ""
    for line in lines.splitlines():
        if line and line[0].isalpha() and "==" in line:
            current = line.split()[0]
            blocks[current] = []
        elif current and line.strip().startswith("--hash=sha256:"):
            blocks[current].append(line)
    unhashed = sorted(pin for pin, hashes in blocks.items() if not hashes)
    assert blocks and not unhashed, unhashed


def test_sidecar_lerobot_matches_the_workspace() -> None:
    # One lerobot across the workspace and the sidecar (openral#716): a workspace bump
    # must recompile robotwin.lock, and test_robotwin_sidecar_builds_lerobot_07_env
    # only checks the sidecar's import path when the two agree.
    from importlib.metadata import PackageNotFoundError, version

    try:
        workspace = version("lerobot")
    except PackageNotFoundError:
        pytest.skip("lerobot not installed in this venv")
    assert _lock_pins()["lerobot"] == workspace


def test_curobo_planner_fallback_fills_the_missing_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A RoboTwin-shaped package whose planner defines CuroboPlanner only when cuRobo
    # imports; the sidecar's import hook supplies the stand-in without editing it.
    import sys

    spec = importlib.util.spec_from_file_location(
        "robotwin_sidecar_curobo_for_test",
        _REPO_ROOT / "tools" / "robotwin_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    robot = tmp_path / "envs" / "robot"
    robot.mkdir(parents=True)
    (tmp_path / "envs" / "__init__.py").write_text("")
    (robot / "__init__.py").write_text("from .robot import *\n")
    (robot / "planner.py").write_text(
        "try:\n    import curobo_not_installed\n\n    class CuroboPlanner: ...\n"
        "except Exception:\n    pass\n"
    )
    (robot / "robot.py").write_text("from .planner import CuroboPlanner\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "envs" or m.startswith("envs.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    module._install_curobo_planner_fallback()
    import envs.robot.robot as robot_mod  # type: ignore[import-not-found]  # reason: package written to tmp_path above

    stub = robot_mod.CuroboPlanner()
    assert robot_mod.CuroboPlanner is module._CuroboPlannerUnavailable
    assert stub.plan_grippers(0.0, 1.0)["num_step"] == 200
    with pytest.raises(RuntimeError, match="cuRobo is not installed"):
        stub.plan_path()


def test_robotwin_launch_argv_passes_checkout_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # _build_robotwin_scene gates on the sidecar wire deps (msgpack/pyzmq), an
    # opt-in group absent from the base test env — skip if unavailable (§1.11).
    pytest.importorskip("msgpack", reason="sidecar-wire deps (msgpack/pyzmq) not installed")
    pytest.importorskip("zmq", reason="sidecar-wire deps (msgpack/pyzmq) not installed")
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
        scene=SceneSpec(id="robotwin", backend=PhysicsBackend.SAPIEN, cameras=["top"]),
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
    assert [s.name for s in r.sensors] == ["top", "wrist_left", "wrist_right"]
    assert r.action_spec is not None and r.action_spec.dim == 14
    assert "aloha_agilex" in r.capabilities.embodiment_tags


# ─── rSkill + task-data gate ─────────────────────────────────────────────────


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
    assert m.action_contract is not None and m.action_contract.dim == 14


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
