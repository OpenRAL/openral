"""Isaac scene: environment USD + spawn pose from the deploy scene.

A deploy scene puts any manifest robot into any Isaac stage with three fields —
``scene.assets_uri`` (environment USD), ``robot_id``, ``base_pose`` (spawn).
These pin the openral side of that contract without a GPU:

* ``_resolve_environment_usd`` / ``_spawn_pose`` validate the two fields;
* the shipped ``isaac_panda_mobile_warehouse.yaml`` goes through the real deploy
  loader (``build_sim_env_from_yaml``) and the real factory to a real sidecar
  subprocess (``tests/unit/fakes/isaac_sidecar_no_kit.py`` — the real
  ``tools/isaac_sidecar.py`` argv parsing + ZMQ serve loop, minus Kit), so the
  test sees exactly the argv Isaac would boot with.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from openral_core import Pose6D
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import _resolve_environment_usd, _spawn_pose

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAREHOUSE_YAML = _REPO_ROOT / "scenes" / "deploy" / "isaac_panda_mobile_warehouse.yaml"
_FAKE_SIDECAR = _REPO_ROOT / "tests" / "unit" / "fakes" / "isaac_sidecar_no_kit.py"


def _pose(quat_xyzw: tuple[float, float, float, float]) -> Pose6D:
    return Pose6D(xyz=(-4.8, 0.0, 0.0), quat_xyzw=quat_xyzw, frame_id="world")


# ── _spawn_pose ───────────────────────────────────────────────────────────────


def test_spawn_pose_defaults_to_the_origin() -> None:
    assert _spawn_pose(None) == (0.0, 0.0, 0.0, 0.0)


def test_spawn_pose_extracts_yaw() -> None:
    s = math.sqrt(0.5)
    assert _spawn_pose(_pose((0.0, 0.0, s, s))) == pytest.approx((-4.8, 0.0, 0.0, math.pi / 2))


def test_spawn_pose_accepts_an_unnormalised_yaw_quaternion() -> None:
    assert _spawn_pose(_pose((0.0, 0.0, 2.0, 2.0)))[3] == pytest.approx(math.pi / 2)


def test_spawn_pose_rejects_a_tilted_robot() -> None:
    s = math.sqrt(0.5)
    with pytest.raises(ROSConfigError, match="only a yaw"):
        _spawn_pose(_pose((s, 0.0, 0.0, s)))  # 90 deg roll


def test_spawn_pose_rejects_the_zero_quaternion() -> None:
    with pytest.raises(ROSConfigError, match="zero quaternion"):
        _spawn_pose(_pose((0.0, 0.0, 0.0, 0.0)))


# ── _resolve_environment_usd ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "uri",
    [
        "isaac:Isaac/Environments/Simple_Warehouse/warehouse.usd",
        "https://example.org/stage.usdc",
        "omniverse://localhost/Projects/stage.usd",
    ],
)
def test_remote_environment_passes_through(uri: str) -> None:
    assert _resolve_environment_usd(uri) == uri


def test_local_environment_resolves_to_an_absolute_path(tmp_path: Path) -> None:
    stage = tmp_path / "cell.usda"
    stage.write_text('#usda 1.0\n(\n    upAxis = "Z"\n)\n')
    assert _resolve_environment_usd(f"file://{stage}") == str(stage)
    assert _resolve_environment_usd(str(stage)) == str(stage)


def test_missing_local_environment_fails_before_boot(tmp_path: Path) -> None:
    with pytest.raises(ROSConfigError, match="no such file"):
        _resolve_environment_usd(str(tmp_path / "typo.usd"))


def test_non_usd_environment_is_rejected() -> None:
    with pytest.raises(ROSConfigError, match="not a USD file"):
        _resolve_environment_usd("scenes/arena.xml")


# ── shipped warehouse scene → real loader → real factory → sidecar argv ──────


def test_warehouse_deploy_scene_boots_the_sidecar_with_stage_and_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
    pytest.importorskip("msgpack", reason="isaacsim group (msgpack) not installed")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    argv_out = tmp_path / "argv.json"
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_SCRIPT", str(_FAKE_SIDECAR))
    monkeypatch.setenv("OPENRAL_TEST_ISAAC_ARGV_OUT", str(argv_out))
    monkeypatch.setenv("OPENRAL_ISAAC_AUTO_SPAWN", "1")

    env, _seed = build_sim_env_from_yaml(str(_WAREHOUSE_YAML))
    try:
        argv = json.loads(argv_out.read_text())
        # The environment + spawn fields imply the manifest layout.
        assert argv[argv.index("--layout") + 1] == "manifest"
        assert argv[argv.index("--robot") + 1] == "panda_mobile"
        assert (
            argv[argv.index("--environment-usd") + 1]
            == "isaac:Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        )
        i = argv.index("--spawn-pose")
        assert [float(v) for v in argv[i + 1 : i + 5]] == pytest.approx(
            [-4.8, 0.0, 0.0, math.pi / 2], abs=1e-6
        )
        # panda_mobile's 11-D contract (7 arm + gripper + 3 base twist) came back
        # over the identity-checked ping.
        assert env.action_dim == 11
    finally:
        env.close()


def test_environment_on_a_hardcoded_layout_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    doc = _WAREHOUSE_YAML.read_text().replace(
        "  backend_options:\n", '  backend_options:\n    layout: "lift_cube"\n'
    )
    yaml_path = tmp_path / "warehouse_lift_cube.yaml"
    yaml_path.write_text(doc)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_SCRIPT", str(_FAKE_SIDECAR))
    with pytest.raises(ROSConfigError, match="need layout 'manifest'"):
        build_sim_env_from_yaml(str(yaml_path))
