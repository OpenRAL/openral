"""ComputeSpec.nvmm_available field — additive, defaults to False.

Pinned by ADR-0013 PR 2/3. The field surfaces whether the L4T
``libnvbufsurface.so`` is available so ``rSkill.check_capabilities``
can refuse a skill that requires the NVMM zero-copy ingest path on a
host that cannot provide it.

After the ComputeSpec split the field lives on :class:`ComputeSpec`,
not :class:`RobotCapabilities`.
"""

from __future__ import annotations

from openral_core import ComputeSpec


def test_default_is_false() -> None:
    spec = ComputeSpec()
    assert spec.nvmm_available is False


def test_can_be_set_true() -> None:
    spec = ComputeSpec(nvmm_available=True)
    assert spec.nvmm_available is True


def test_pre_compute_spec_payload_loads_with_default() -> None:
    # Simulate a compute spec written without nvmm_available (older manifest).
    payload: dict[str, object] = {
        "gpu_vram_gb": 24.0,
        "cuda_compute_capability": [8, 9],
    }
    spec = ComputeSpec.model_validate(payload)
    assert spec.nvmm_available is False
