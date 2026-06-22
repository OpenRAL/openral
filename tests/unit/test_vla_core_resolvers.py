"""Unit tests for the manifest-first resolvers in ``openral_rskill._vla_core``.

These exercise the precedence rule introduced in commit
``feat(skill): _vla_core resolvers (no auto-derive)``:

    spec_extra > manifest field > schema default

There is **no** auto-derive layer below those three; the resolver
deliberately refuses to read from ``policy.config.input_features`` or
to invent ``camera1 -> image`` style maps. The whole point of the
resolver is that missing per-checkpoint hints surface as the schema
default rather than silently changing behaviour.

CLAUDE.md §1.11 -- real schemas; no mocks.
"""

from __future__ import annotations

from openral_core import (
    ImagePreprocessing,
    RSkillLatencyBudget,
    RSkillManifest,
    StateContract,
)
from openral_rskill._vla_core import (
    apply_chunk_replay,
    resolve_camera_keys,
    resolve_image_preprocessing,
    resolve_state_dim,
)


def _manifest(**overrides: object) -> RSkillManifest:
    """Build a minimal valid manifest, layering in test-specific fields."""
    base: dict[str, object] = {
        "name": "test/foo",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "vla",
        "model_family": "smolvla",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": [
            {
                "kind": "joint_position",
                "control_mode_semantics": {"mode": "absolute"},
            }
        ],
        "chunk_size": 1,
        "weights_uri": "hf://test/foo",
        "latency_budget": RSkillLatencyBudget(per_chunk_ms=100.0),
        "processors": {
            "preprocessor_uri": "hf://test/foo/policy_preprocessor.json",
            "postprocessor_uri": "hf://test/foo/policy_postprocessor.json",
        },
        "description": "Resolver-test rSkill fixture.",
        "actions": ["generalist"],
    }
    base.update(overrides)
    return RSkillManifest.model_validate(base)


# ── resolve_image_preprocessing ──────────────────────────────────────────────


class TestResolveImagePreprocessing:
    def test_spec_extra_overrides_manifest(self) -> None:
        """spec_extra.flip_180 wins over manifest.image_preprocessing.flip_180."""
        m = _manifest(image_preprocessing=ImagePreprocessing(flip_180=False, aliases={"a": "b"}))
        ip = resolve_image_preprocessing(m, {"flip_180": True})
        assert ip.flip_180 is True
        # aliases fall through from the manifest since spec_extra didn't carry one
        assert ip.aliases == {"a": "b"}

    def test_manifest_wins_when_extra_silent(self) -> None:
        """Manifest's image_preprocessing applies when spec_extra has nothing."""
        m = _manifest(
            image_preprocessing=ImagePreprocessing(
                flip_180=True,
                input_template="observation.image.{cam}",
                aliases={"front": "agentview"},
            )
        )
        ip = resolve_image_preprocessing(m, {})
        assert ip.flip_180 is True
        assert ip.input_template == "observation.image.{cam}"
        assert ip.aliases == {"front": "agentview"}

    def test_schema_defaults_when_both_silent(self) -> None:
        """Both silent → ImagePreprocessing() schema defaults."""
        m = _manifest()  # no image_preprocessing
        ip = resolve_image_preprocessing(m, {})
        assert ip.flip_180 is False
        assert ip.input_template == "observation.images.{cam}"
        assert ip.aliases == {}

    def test_no_manifest_path(self) -> None:
        """Manifest=None (legacy path) → spec_extra-or-default."""
        ip = resolve_image_preprocessing(None, {"flip_180": True})
        assert ip.flip_180 is True
        assert ip.aliases == {}

    def test_legacy_flip_images_180_alias(self) -> None:
        """Legacy ``flip_images_180`` key still resolves to flip_180."""
        m = _manifest()
        ip = resolve_image_preprocessing(m, {"flip_images_180": True})
        assert ip.flip_180 is True

    def test_norm_tag_propagates_from_manifest(self) -> None:
        """Manifest's norm_tag must survive resolution (regression: MolmoAct2
        SO-100/101 rejected its own checkpoint because the resolver dropped the
        manifest's ``norm_tag`` and the adapter fell back to ``libero``)."""
        m = _manifest(image_preprocessing=ImagePreprocessing(norm_tag="so100_so101_molmoact2"))
        ip = resolve_image_preprocessing(m, {})
        assert ip.norm_tag == "so100_so101_molmoact2"

    def test_norm_tag_none_when_manifest_silent(self) -> None:
        """No manifest norm_tag → schema default ``None`` (adapter applies its own)."""
        assert resolve_image_preprocessing(_manifest(), {}).norm_tag is None
        assert resolve_image_preprocessing(None, {}).norm_tag is None

    def test_image_max_crops_propagates_from_manifest(self) -> None:
        """Manifest's image_max_crops survives resolution (8 GiB activation lever).

        Regression guard for the MolmoAct2 SO-101 8 GiB OOM: the NF4 weights are
        a fixed ~3.5 GiB, so the overflow is the image processor's multi-crop
        activations. The skill pins ``image_max_crops`` in its manifest so a
        fresh rollout fits without a ``vla.extra`` override.
        """
        m = _manifest(image_preprocessing=ImagePreprocessing(image_max_crops=4))
        assert resolve_image_preprocessing(m, {}).image_max_crops == 4

    def test_image_max_crops_extra_overrides_manifest(self) -> None:
        """spec_extra.image_max_crops wins over the manifest's per-checkpoint pin."""
        m = _manifest(image_preprocessing=ImagePreprocessing(image_max_crops=4))
        assert resolve_image_preprocessing(m, {"image_max_crops": 8}).image_max_crops == 8

    def test_image_max_crops_none_when_silent(self) -> None:
        """No pin anywhere → ``None`` (adapter keeps the checkpoint default of 8)."""
        assert resolve_image_preprocessing(_manifest(), {}).image_max_crops is None
        assert resolve_image_preprocessing(None, {}).image_max_crops is None


# ── resolve_state_dim ────────────────────────────────────────────────────────


class TestResolveStateDim:
    def test_spec_extra_overrides_manifest(self) -> None:
        m = _manifest(state_contract=StateContract(dim=16))
        assert resolve_state_dim(m, {"state_dim": 8}) == 8

    def test_manifest_wins_when_extra_silent(self) -> None:
        m = _manifest(state_contract=StateContract(dim=16))
        assert resolve_state_dim(m, {}) == 16

    def test_none_when_both_silent(self) -> None:
        m = _manifest()
        assert resolve_state_dim(m, {}) is None

    def test_invalid_extra_falls_through_to_manifest(self) -> None:
        """Non-int / non-positive spec_extra.state_dim is ignored."""
        m = _manifest(state_contract=StateContract(dim=16))
        assert resolve_state_dim(m, {"state_dim": "not an int"}) == 16
        assert resolve_state_dim(m, {"state_dim": 0}) == 16


# ── resolve_camera_keys ──────────────────────────────────────────────────────


class TestResolveCameraKeys:
    def test_spec_extra_wins(self) -> None:
        assert resolve_camera_keys(
            _manifest(), {"camera_keys": ["x", "y"]}, scene_cameras=["a", "b"]
        ) == ("x", "y")

    def test_scene_cameras_when_extra_silent(self) -> None:
        assert resolve_camera_keys(_manifest(), {}, scene_cameras=["a", "b", "c"]) == (
            "a",
            "b",
            "c",
        )

    def test_default_when_both_silent(self) -> None:
        assert resolve_camera_keys(_manifest(), {}) == ("camera1", "camera2")

    def test_no_manifest_path(self) -> None:
        assert resolve_camera_keys(None, {}, scene_cameras=["a"]) == ("a",)


# ── apply_chunk_replay (manifest-aware) ──────────────────────────────────────


class _FakePolicy:
    """Standin for a lerobot policy that exposes config.chunk_size and config.n_action_steps."""

    class _Cfg:
        chunk_size: int = 50
        n_action_steps: int = 1

    def __init__(self) -> None:
        self.config = _FakePolicy._Cfg()


class TestApplyChunkReplay:
    def test_spec_extra_wins(self) -> None:
        p = _FakePolicy()
        m = _manifest(n_action_steps=10)
        applied = apply_chunk_replay(p, {"n_action_steps": 3}, manifest=m)
        assert applied == 3
        assert p.config.n_action_steps == 3

    def test_manifest_wins_over_default(self) -> None:
        p = _FakePolicy()
        m = _manifest(n_action_steps=10)
        applied = apply_chunk_replay(p, {}, manifest=m, default_n_action_steps=25)
        assert applied == 10

    def test_default_when_manifest_silent(self) -> None:
        p = _FakePolicy()
        m = _manifest()  # no n_action_steps
        applied = apply_chunk_replay(p, {}, manifest=m, default_n_action_steps=25)
        assert applied == 25

    def test_chunk_size_when_all_silent(self) -> None:
        p = _FakePolicy()
        applied = apply_chunk_replay(p, {})
        assert applied == 50  # chunk_size

    def test_clamps_to_chunk_size(self) -> None:
        p = _FakePolicy()
        # spec_extra=999, chunk_size=50 → clamped to 50
        assert apply_chunk_replay(p, {"n_action_steps": 999}) == 50
        # zero / negative clamps to 1
        assert apply_chunk_replay(p, {"n_action_steps": 0}) == 1
