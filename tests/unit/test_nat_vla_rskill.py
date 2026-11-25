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


def _norm_stats_or_skip(uri: str) -> str:
    """The real MolmoAct2 ``norm_stats.json`` the manifest pins (HF cache, else the Hub)."""
    pytest.importorskip("huggingface_hub")
    from openral_sim.policies.nat_vla import _hf_file

    try:
        return _hf_file(uri)
    except Exception as exc:  # reason: an unreachable Hub with a cold cache is the skip
        pytest.skip(f"norm_stats.json not reachable: {exc!r}")


def test_manifest_maps_the_new_convention_rest_pose_into_the_pretraining_one() -> None:
    """A range-centred rest pose lands on the old-convention starts the mixture holds.

    makermods/eraser_place episodes start at shoulder_lift -102.7 / elbow 95.1 (median of
    25); old-convention episodes start at 185-198 / ~180. The manifest's joint_affine
    must carry one onto the other, and actions must map back exactly.
    """
    np = pytest.importorskip("numpy")
    from openral_sim.policies.nat_vla import joint_affine

    manifest = load_rskill_manifest(str(_SKILL))
    affine = joint_affine(manifest.policy_extras["joint_affine"], 6, 6)
    assert affine is not None
    scale, offset = affine
    rest_new = np.array([-7.2, -102.7, 95.1, 56.7, 6.2, 1.9], dtype=np.float32)
    rest_model = scale * rest_new + offset
    assert 184.0 <= rest_model[1] <= 198.0  # shoulder_lift, sign-flipped
    assert 178.0 <= rest_model[2] <= 183.0  # elbow_flex
    assert np.allclose(rest_model[[0, 3, 4, 5]], rest_new[[0, 3, 4, 5]])
    assert np.allclose((rest_model - offset) / scale, rest_new)


def test_pooled_stats_hold_the_mapped_pose_and_not_the_raw_one() -> None:
    """With the pooled SO-ARM stats, the mapped reaching pose normalises inside [-1, 1].

    The raw range-centred values sit far outside the mixture's q01/q99 on shoulder_lift:
    the miswiring that swung the real arm backward.
    """
    np = pytest.importorskip("numpy")
    from openral_sim.policies.nat_vla import joint_affine, molmoact2_tag_stats

    manifest = load_rskill_manifest(str(_SKILL))
    extra = manifest.policy_extras
    stats = molmoact2_tag_stats(
        _norm_stats_or_skip(str(extra["norm_stats"])), str(extra["norm_tag"])
    )
    q01 = np.asarray(stats["observation.state"]["q01"])
    q99 = np.asarray(stats["observation.state"]["q99"])
    assert stats["action"]["min"][1] > 0 and stats["action"]["max"][1] < 200  # q01/q99, not ±270
    affine = joint_affine(extra["joint_affine"], 6, 6)
    assert affine is not None
    reach_new = np.array([0.0, -37.4, 36.3, 64.0, 38.0, 5.5], dtype=np.float32)  # makermods mean
    mapped = (2 * (affine[0] * reach_new + affine[1] - q01) / (q99 - q01)) - 1
    raw = (2 * (reach_new - q01) / (q99 - q01)) - 1
    assert np.all(np.abs(mapped[:3]) < 1.0), mapped
    assert raw[1] < -1.5, raw  # shoulder_lift: far below the mixture's q01


@pytest.mark.parametrize(
    "value",
    [[[1.0, 0.0]] * 5, [[0.0, 1.0]] * 6, [["a", 1.0]] * 6, [[float("nan"), 0.0]] * 6],
)
def test_joint_affine_refuses_malformed_pairs(value: object) -> None:
    """One finite, non-zero-scale pair per column, or a typed config error."""
    pytest.importorskip("numpy")
    from openral_sim.policies.nat_vla import joint_affine

    with pytest.raises(ROSConfigError, match="joint_affine"):
        joint_affine(value, 6, 6)


def test_molmoact2_tag_stats_refuses_an_unknown_tag() -> None:
    """A tag the file does not hold names the tags it does."""
    from openral_sim.policies.nat_vla import molmoact2_tag_stats

    manifest = load_rskill_manifest(str(_SKILL))
    path = _norm_stats_or_skip(str(manifest.policy_extras["norm_stats"]))
    with pytest.raises(ROSConfigError, match="so100_so101_molmoact2"):
        molmoact2_tag_stats(path, "libero")
