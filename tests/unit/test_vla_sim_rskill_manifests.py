"""Manifest contracts of the CUDA-gated VLA sim rSkills, checked without a GPU.

The rollouts in ``tests/sim/test_{aloha_bimanual_act_aloha,pusht_2d_diffusion_pusht,
franka_panda_smolvla_libero}.py`` skip without CUDA, which took their manifest checks
down with them on every CI runner. The sensor feature keys are what the sim adapters
wire camera frames to, so a drift here breaks those rollouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import RSkillManifest

_RSKILLS = Path(__file__).resolve().parents[2] / "rskills"


@pytest.mark.parametrize(
    ("skill_dir", "name", "embodiment_tag", "feature_keys"),
    [
        (
            "act-aloha",
            "OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32",
            "aloha",
            ["observation.images.top"],
        ),
        (
            "diffusion-pusht",
            "OpenRAL/rskill-diffusion-pusht-pusht-fp32",
            "pusht",
            ["observation.images.camera1"],
        ),
        (
            "smolvla-libero",
            "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16",
            "franka_panda",
            ["observation.images.camera1", "observation.images.camera2"],
        ),
    ],
)
def test_vla_sim_rskill_declares_its_rgb_inputs(
    skill_dir: str, name: str, embodiment_tag: str, feature_keys: list[str]
) -> None:
    m = RSkillManifest.from_yaml(str(_RSKILLS / skill_dir / "rskill.yaml"))
    assert m.name == name
    assert embodiment_tag in m.embodiment_tags
    assert m.role == "s1"
    assert {s.modality for s in m.sensors_required} == {"rgb"}
    assert sorted(s.vla_feature_key for s in m.sensors_required) == feature_keys
