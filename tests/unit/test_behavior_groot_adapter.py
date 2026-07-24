"""BEHAVIOR-1K GR00T rSkill and adapter contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest
from openral_sim.policies import behavior_groot

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILL = _REPO_ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"


def test_behavior_groot_rskill_manifest_loads() -> None:
    manifest = load_rskill_manifest(str(_RSKILL))
    assert manifest.model_family == "gr00t"
    assert manifest.license == "unknown"
    assert manifest.state_contract is not None
    assert manifest.state_contract.dim == 61
    assert manifest.action_contract is not None
    assert manifest.action_contract.dim == 23
    assert manifest.policy_extras["implementation"] == "behavior_b1k_sidecar"
    assert manifest.policy_extras["quantization"] == "nf4"
    assert manifest.policy_extras["nf4_min_params"] == 1_000_000


def test_behavior_wire_observation_preserves_official_payload() -> None:
    raw = {
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    wire = behavior_groot._behavior_wire_observation(
        {"behavior_raw": raw},
        instruction="turn on the radio",
    )
    assert wire["robot_r1::proprio"] is raw["robot_r1::proprio"]
    assert wire["openral_instruction"] == "turn on the radio"


def test_behavior_wire_observation_builds_from_canonical_fields() -> None:
    images = {
        "head": np.zeros((8, 8, 3), dtype=np.uint8),
        "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    wire = behavior_groot._behavior_wire_observation(
        {"images": images, "state": np.zeros(61, dtype=np.float32)},
        instruction="turn on the radio",
    )
    assert np.asarray(wire["robot_r1::proprio"]).shape == (61,)
    for key in behavior_groot._CAMERA_KEYS.values():
        assert np.asarray(wire[key]).shape == (8, 8, 3)


def test_behavior_sidecar_python_error_has_setup_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing" / "python"
    monkeypatch.setenv(behavior_groot._SIDECAR_PYTHON_ENV, str(missing))
    with pytest.raises(ROSConfigError, match=r"Python 3\.10"):
        behavior_groot._sidecar_python()


def test_behavior_checkpoint_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = load_rskill_manifest(str(_RSKILL))
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(checkpoint))
    assert behavior_groot._checkpoint_path(manifest) == checkpoint


def test_behavior_policy_ports_are_stable_and_task_specific() -> None:
    a = behavior_groot._policy_default_port("turning_on_radio", "/checkpoint")
    b = behavior_groot._policy_default_port("turning_on_radio", "/checkpoint")
    other = behavior_groot._policy_default_port("cleaning_a_table", "/checkpoint")
    assert a == b
    assert behavior_groot._PORT_MIN <= a < behavior_groot._PORT_MAX
    assert a != other


def test_gr00t_factory_dispatches_behavior_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_sim.policies import gr00t

    sentinel = object()
    monkeypatch.setattr(
        behavior_groot,
        "build_behavior_groot_policy",
        lambda _env, _manifest, _extra: sentinel,
    )
    env = SimpleNamespace(
        vla=VLASpec(id="gr00t", weights_uri=str(_RSKILL)),
        scene=SimpleNamespace(cameras=["head", "left_wrist", "right_wrist"]),
    )
    assert gr00t._build_gr00t(env) is sentinel


def test_gr00t_factory_reports_missing_organizer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openral_sim.policies import gr00t

    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(tmp_path / "missing-checkpoint"))
    monkeypatch.setenv(behavior_groot._AUTO_SPAWN_ENV, "1")
    env = SimpleNamespace(
        vla=VLASpec(id="gr00t", weights_uri=str(_RSKILL)),
        scene=SimpleNamespace(cameras=["head", "left_wrist", "right_wrist"]),
    )
    with pytest.raises(ROSConfigError, match="checkpoint not found"):
        gr00t._build_gr00t(env)
