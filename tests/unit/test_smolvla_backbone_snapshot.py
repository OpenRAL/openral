"""``_config_with_local_backbone``: SmolVLA's backbone files come from one local snapshot.

Real checkpoint (the SO-101 eraser-place rSkill, pinned SHA) and the real
SmolVLM2 backbone repo; skipped when lerobot or the HF cache/network is
unavailable (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = "OpenRAL/rskill-smolvla-so101-eraser_place-bf16"
_REV = "7a9a8a0684b28fb6470f25d333aa65e31c8b781b"


def test_backbone_resolves_to_a_local_snapshot_and_keeps_the_hub_id() -> None:
    pytest.importorskip("lerobot.policies.smolvla.configuration_smolvla")
    from openral_sim.policies.smolvla import _config_with_local_backbone

    try:
        config, backbone_id = _config_with_local_backbone(_REPO, _REV)
    except OSError as exc:  # no HF cache and no network
        pytest.skip(f"checkpoint unavailable: {exc}")

    assert backbone_id == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    local = Path(config.vlm_model_name)
    assert local.is_dir()
    for name in ("config.json", "processor_config.json", "tokenizer.json"):
        assert (local / name).is_file(), name
