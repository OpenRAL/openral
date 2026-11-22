"""Cosmos 3 LIBERO rSkill: OpenMDW-1.1 is commercial-allowed; the adapter refuses non-LIBERO scenes.

Real in-tree manifest (``rskills/cosmos3-edge-libero-bf16``), the real loader license guard
and the real registered factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RSkillLicensePosture
from openral_core.exceptions import ROSCapabilityMismatch
from openral_rskill.loader import load_rskill_manifest, rSkill

_SKILL = Path(__file__).resolve().parents[2] / "rskills" / "cosmos3-edge-libero-bf16"


def test_openmdw_weights_are_commercial_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_rskill_manifest(str(_SKILL))
    assert manifest.license is RSkillLicensePosture.OPENMDW_1_1
    assert manifest.is_commercial_use_allowed is True
    monkeypatch.delenv("OPENRAL_ALLOW_NONCOMMERCIAL", raising=False)
    rSkill._check_license(manifest, commercial_use=True)


def test_the_factory_refuses_a_non_libero_scene() -> None:
    """Refused before torch / lerobot load: the decode targets LIBERO's OSC command."""
    import openral_sim.policies  # noqa: F401 — registers cosmos3
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_sim.registry import POLICIES

    env_cfg = SimEnvironment(
        robot_id="franka_panda",
        scene=SceneSpec(id="robocasa", backend=PhysicsBackend.MUJOCO, cameras=["camera1"]),
        task=TaskSpec(
            id="robocasa/PickPlace",
            scene_id="robocasa",
            instruction="pick the cup",
            success_key="is_success",
            max_steps=10,
        ),
        vla=VLASpec(id="cosmos3", weights_uri=str(_SKILL)),
    )
    with pytest.raises(ROSCapabilityMismatch, match="LIBERO"):
        POLICIES.get("cosmos3")(env_cfg)
