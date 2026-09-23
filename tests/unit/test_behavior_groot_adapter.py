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
    # Its own registered family (not a hidden `implementation` switch inside
    # `gr00t`), so the dependency probe and the dispatch agree.
    assert manifest.model_family == "gr00t_b1k"
    assert "implementation" not in manifest.policy_extras
    assert manifest.license == "nvidia_open_model"
    assert manifest.is_commercial_use_allowed
    assert manifest.state_contract is not None
    assert manifest.state_contract.dim == 61
    assert manifest.action_contract is not None
    assert manifest.action_contract.dim == 23
    # Packing knobs live under `quantization.extra`, not `policy_extras`: one
    # home shared with the GR00T and RLDX families.
    assert manifest.quantization.extra["quantize_scope"] == "model"
    assert manifest.quantization.extra["nf4_min_params"] == 1_000_000
    # `dtype` is what runs (NF4 at load); the stored precision is recorded.
    assert manifest.quantization.dtype.value == "int4"
    assert manifest.quantization.extra["stored_dtype"] == "bf16"


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


def test_behavior_policy_ports_are_stable_and_config_specific() -> None:
    def port(task: str = "turning_on_radio", quant: str = "nf4") -> int:
        return behavior_groot._policy_default_port(
            task, "/checkpoint", quant, "temporal_ensemble", 1_000_000
        )

    assert port() == port()
    assert behavior_groot._PORT_MIN <= port() < behavior_groot._PORT_MAX
    assert port() != port(task="cleaning_a_table")
    # A quantization change must land on a different port so an A/B rerun can
    # never silently adopt the still-running sidecar quantized the old way.
    assert port() != port(quant="int8")


def test_gr00t_b1k_factory_dispatches_behavior_sidecar(
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
        vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)),
        scene=SimpleNamespace(cameras=["head", "left_wrist", "right_wrist"]),
    )
    assert gr00t._build_gr00t_b1k(env) is sentinel


def test_gr00t_b1k_factory_reports_missing_organizer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openral_sim.policies import gr00t

    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(tmp_path / "missing-checkpoint"))
    monkeypatch.setenv(behavior_groot._AUTO_SPAWN_ENV, "1")
    env = SimpleNamespace(
        vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)),
        scene=SimpleNamespace(cameras=["head", "left_wrist", "right_wrist"]),
    )
    with pytest.raises(ROSConfigError, match="checkpoint not found"):
        gr00t._build_gr00t_b1k(env)


def test_bf16_override_maps_to_the_sidecars_unquantized_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`OPENRAL_QUANTIZATION_DTYPE=bf16` loads the stored bf16 checkpoint.

    The sidecar CLI only accepts ("none", "nf4", "int8") and its "none" mode
    loads the checkpoint as stored (bf16), so bf16 maps onto "none". A raw
    "bf16" reaching argparse would be a `SystemExit(2)`.
    """
    from openral_sim._quantization import QUANTIZATION_DTYPE_ENV

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    sidecar_python = tmp_path / "python"
    sidecar_python.touch()
    monkeypatch.setenv(behavior_groot._CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(behavior_groot._SIDECAR_PYTHON_ENV, str(sidecar_python))
    monkeypatch.setenv(behavior_groot._AUTO_SPAWN_ENV, "1")
    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "bf16")

    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def connect(self) -> None:
            pass

    monkeypatch.setattr(behavior_groot, "SidecarClient", _StubClient)

    manifest = load_rskill_manifest(str(_RSKILL))
    env_cfg = SimpleNamespace(vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)))
    behavior_groot.build_behavior_groot_policy(env_cfg, manifest, {})

    assert "--quantization" in captured["launch_argv"]  # type: ignore[operator]
    argv = list(captured["launch_argv"])  # type: ignore[arg-type]
    assert argv[argv.index("--quantization") + 1] == "none"


def test_precision_the_sidecar_cannot_load_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """fp32 has no sidecar mode; it must fail loudly, not collapse to "none"."""
    from openral_sim._quantization import QUANTIZATION_DTYPE_ENV

    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "fp32")
    manifest = load_rskill_manifest(str(_RSKILL))
    env_cfg = SimpleNamespace(vla=VLASpec(id="gr00t_b1k", weights_uri=str(_RSKILL)))
    with pytest.raises(ROSConfigError, match="gr00t_b1k cannot load dtype 'fp32'"):
        behavior_groot.build_behavior_groot_policy(env_cfg, manifest, {})


def test_sidecar_parses_int8_quantization() -> None:
    # The sidecar CLI is the adapter's argv contract: `quantization` from
    # policy_extras is forwarded verbatim, so every advertised choice must parse.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "behavior_groot_sidecar", _REPO_ROOT / "tools" / "behavior_groot_sidecar.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parse_args(["--checkpoint", str(_RSKILL), "--quantization", "int8"])
    assert args.quantization == "int8"
    assert args.nf4_min_params == 4_000_000
