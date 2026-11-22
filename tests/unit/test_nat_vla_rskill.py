"""NaT VLA rSkill: the LFM Open License posture gates commercial use; the vendored module loads.

Real in-tree manifest (``rskills/nat-vla-so101-pretrained-bf16``) and the real loader guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RSkillLicensePosture
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest, rSkill

_SKILL = Path(__file__).resolve().parents[2] / "rskills" / "nat-vla-so101-pretrained-bf16"


def test_lfm_weights_are_not_commercial_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """LFM Open License v1.0: commercial only under $10M revenue, so the guard treats it as NC."""
    manifest = load_rskill_manifest(str(_SKILL))
    assert manifest.license is RSkillLicensePosture.LFM_OPEN_1_0
    assert manifest.is_commercial_use_allowed is False
    monkeypatch.delenv("OPENRAL_ALLOW_NONCOMMERCIAL", raising=False)
    with pytest.raises(ROSConfigError, match=r"lfm1\.0"):
        rSkill._check_license(manifest, commercial_use=True)
    # Research use (or the explicit, logged override) loads.
    rSkill._check_license(manifest, commercial_use=False)
    monkeypatch.setenv("OPENRAL_ALLOW_NONCOMMERCIAL", "1")
    rSkill._check_license(manifest, commercial_use=True)


def test_the_adapter_resolves_a_nat_vla_module() -> None:
    """lerobot's own nat_vla when installed, else the copy vendored at the merge commit."""
    pytest.importorskip("lerobot.policies.pretrained")
    from openral_sim.policies.nat_vla import _nat_vla_module

    config_cls, policy_cls, make_processors, source = _nat_vla_module()
    assert source in {"lerobot", "vendored"}
    assert config_cls.__name__ == "NATVLAConfig"
    assert policy_cls.__name__ == "NATVLAPolicy"
    assert callable(make_processors)
