"""XR-1 rSkill contracts pinned from the upstream processor metadata."""

from __future__ import annotations

from pathlib import Path

from openral_core import RSkillManifest

_ROOT = Path(__file__).parents[2]


def test_xr1_checkpoint_contracts() -> None:
    expected = {
        "xr1-robocasa": ("robocasa_mg", 8, 7, 10, 10),
        "xr1-robocasa365": ("robocasa365", 16, 12, 16, 16),
        "xr1-vlabench": ("vlabench_choice", 7, 7, 10, 5),
    }
    for package, contract in expected.items():
        manifest = RSkillManifest.from_yaml(str(_ROOT / "rskills" / package / "rskill.yaml"))
        profile, state_dim, action_dim, chunk_size, replay_steps = contract
        assert manifest.policy_extras["xr1_profile"] == profile
        assert manifest.state_contract is not None
        assert manifest.state_contract.dim == state_dim
        assert manifest.action_contract is not None
        assert manifest.action_contract.dim == action_dim
        assert manifest.chunk_size == chunk_size
        assert manifest.n_action_steps == replay_steps
        assert manifest.quantization.dtype.value == "int4"
