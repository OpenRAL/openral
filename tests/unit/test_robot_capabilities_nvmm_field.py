"""RobotCapabilities.nvmm_available field — additive, defaults to False.

Pinned by ADR-0013 PR 2/3. The field surfaces whether the L4T
``libnvbufsurface.so`` is available so ``rSkill.check_capabilities``
can refuse a skill that requires the NVMM zero-copy ingest path on a
host that cannot provide it.
"""

from __future__ import annotations

from openral_core import RobotCapabilities


def test_default_is_false() -> None:
    caps = RobotCapabilities()
    assert caps.nvmm_available is False


def test_can_be_set_true() -> None:
    caps = RobotCapabilities(nvmm_available=True)
    assert caps.nvmm_available is True


def test_pre_v0_5_payload_loads_with_default() -> None:
    # Simulate a manifest written before v0.5 — no nvmm_available key.
    payload = {
        "has_vision": True,
        "embodiment_tags": ["so100"],
    }
    caps = RobotCapabilities.model_validate(payload)
    assert caps.nvmm_available is False
