"""Manifest + ADR-0060 task-gate tests for the OpenVLA WidowX rSkill.

Validates that ``rskills/openvla-oft-simpler-widowx-nf4`` loads as a real
``openvla`` manifest and that the benchmark task-compatibility gate accepts the
SimplerEnv WidowX put-on-plate tasks it declares while refusing the Panda
PickCube pairing it must never run (the WidowX-vs-Panda honesty contract,
ADR-0061).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RSkillManifest
from openral_core.exceptions import ROSCapabilityMismatch
from openral_core.schemas import ActionRepresentation
from openral_sim.benchmark import check_benchmark_task_compatibility

_MANIFEST = Path("rskills/openvla-oft-simpler-widowx-nf4/rskill.yaml")


def _load() -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_MANIFEST))


def test_manifest_loads_as_openvla_family() -> None:
    m = _load()
    assert m.model_family == "openvla"
    assert m.kind == "vla"
    assert m.weights_uri == "hf://RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"


def test_manifest_declares_widowx_bridge_tasks() -> None:
    m = _load()
    assert "simpler_env/widowx_carrot_on_plate" in m.evaluated_tasks
    assert len(m.evaluated_tasks) == 4


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
