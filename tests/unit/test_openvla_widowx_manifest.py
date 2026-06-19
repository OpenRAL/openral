"""Manifest + ADR-0060 task-gate tests for the OpenVLA WidowX rSkill.

Validates that ``rskills/openvla-oft-simpler-widowx-nf4`` loads as a real
``openvla`` manifest and that the benchmark task-compatibility gate accepts the
SimplerEnv WidowX put-on-plate tasks it declares while refusing the Panda
PickCube pairing it must never run (the WidowX-vs-Panda honesty contract,
ADR-0061).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openral_core import RSkillEvalResult, RSkillManifest
from openral_core.exceptions import ROSCapabilityMismatch
from openral_core.schemas import ActionRepresentation
from openral_sim.benchmark import check_benchmark_task_compatibility
from openral_sim.cli import _load_or_build_env

_MANIFEST = Path("rskills/openvla-oft-simpler-widowx-nf4/rskill.yaml")
_EVAL = Path("rskills/openvla-oft-simpler-widowx-nf4/eval/simpler_env_widowx.json")
_SCENE = Path("scenes/sim/widowx_carrot_on_plate.yaml")


_DEFAULT_CLI_ARGS: dict[str, Any] = {
    "config": _SCENE,
    "rskill": "rskills/openvla-oft-simpler-widowx-nf4",
    "robot": None,
    "task": None,
    "instruction": None,
    "max_steps": None,
    "n_episodes": None,
    "n_action_steps": None,
    "seed": None,
    "device": None,
    "save_dir": None,
    "save_video": None,
    "view": None,
    "verbose": False,
}


def _load() -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_MANIFEST))


def test_manifest_loads_as_openvla_family() -> None:
    m = _load()
    assert m.model_family == "openvla"
    assert m.kind == "vla"
    assert m.weights_uri == "hf://RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"


def test_manifest_declares_widowx_bridge_tasks() -> None:
    m = _load()
    assert m.evaluated_tasks == ["simpler_env/widowx_carrot_on_plate"]


def test_manifest_records_openvla_policy_extras() -> None:
    m = _load()
    assert m.benchmarks["simpler_env_widowx"] == 0.4
    assert m.policy_extras == {
        "openvla_generation_method": "generate_action_verl",
        "openvla_do_sample": True,
        "openvla_temperature": 0.6,
        "openvla_padding_max_length": 30,
        "openvla_action_scale": 2.0,
        "openvla_binarize_gripper": True,
        "openvla_gripper_threshold": 0.5,
        "openvla_torch_seed": 0,
    }


def test_eval_artifact_records_nonzero_local_success() -> None:
    result = RSkillEvalResult.from_json(str(_EVAL))
    assert result.source.reproduced_locally is True
    assert result.results["simpler_env/widowx_carrot_on_plate_success_rate"] == 0.4
    assert result.results["successes"] == 2


def test_action_contract_is_delta_ee_plus_gripper() -> None:
    m = _load()
    assert m.action_contract is not None
    assert m.action_contract.dim == 7
    assert m.action_contract.representation == ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER


def test_gate_accepts_declared_widowx_task() -> None:
    # No raise == accepted.
    check_benchmark_task_compatibility(
        _load(), task_id="simpler_env/widowx_carrot_on_plate", scene_id="simpler_env"
    )


def test_gate_rejects_panda_pickcube() -> None:
    with pytest.raises(ROSCapabilityMismatch):
        check_benchmark_task_compatibility(
            _load(), task_id="maniskill3/PickCube-v1", scene_id="maniskill3"
        )


def test_sim_cli_threads_manifest_policy_extras() -> None:
    args = SimpleNamespace(**_DEFAULT_CLI_ARGS)
    env = _load_or_build_env(args)
    assert env.vla.id == "openvla"
    assert env.vla.extra["openvla_generation_method"] == "generate_action_verl"
    assert env.vla.extra["openvla_action_scale"] == 2.0


def test_sim_cli_override_preserves_manifest_policy_extras() -> None:
    args = SimpleNamespace(**{**_DEFAULT_CLI_ARGS, "n_action_steps": 4})
    env = _load_or_build_env(args)
    assert env.vla.extra["openvla_generation_method"] == "generate_action_verl"
    assert env.vla.extra["n_action_steps"] == 4
