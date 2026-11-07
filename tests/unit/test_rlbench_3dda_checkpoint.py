"""3D Diffuser Actor checkpoint resolution honours the manifest's weights_uri pin."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub.constants as hc
import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import load_rskill_manifest
from openral_sim.policies.rlbench_3dda import _resolve_checkpoint

_SKILL = Path(__file__).resolve().parents[2] / "rskills" / "3d-diffuser-actor-rlbench"


def test_override_wins_over_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "diffuser_actor_peract.pth"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("OPENRAL_3DDA_CHECKPOINT", str(ckpt))
    assert _resolve_checkpoint(load_rskill_manifest(str(_SKILL))) == ckpt


def test_missing_override_file_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_3DDA_CHECKPOINT", str(tmp_path / "absent.pth"))
    with pytest.raises(ROSConfigError, match="OPENRAL_3DDA_CHECKPOINT"):
        _resolve_checkpoint(load_rskill_manifest(str(_SKILL)))


def test_pinned_checkpoint_resolves_from_the_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache-only: proves the pinned revision is the one looked up (no 168 MB fetch)."""
    monkeypatch.delenv("OPENRAL_3DDA_CHECKPOINT", raising=False)
    monkeypatch.setattr(hc, "HF_HUB_OFFLINE", True)
    manifest = load_rskill_manifest(str(_SKILL))
    revision = str(manifest.weights_uri).rpartition("@")[2]
    assert len(revision) == 40
    try:
        path = _resolve_checkpoint(manifest)
    except OSError as exc:  # LocalEntryNotFoundError: checkpoint not provisioned here
        pytest.skip(f"3DDA checkpoint not in the HF cache at the pin: {exc}")
    assert path.parent.name == revision
