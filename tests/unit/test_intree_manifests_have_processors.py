"""Canary test: every in-tree rSkill manifest declares the new V1 fields.

Closes the "forgotten manifest" failure mode for the rSkill self-containment
audit (Gap 1, Gap 2, Gap 3). For every ``rskills/*/rskill.yaml`` discovered
by ``openral_rskill.loader.discover_intree_rskills`` the test asserts:

- Every entry in ``actuators_required`` carries a ``control_mode_semantics``
  block (Gap 2).
- The manifest declares a ``processors`` block UNLESS it is one of the two
  explicit legacy ACT-ALOHA skills whose upstream checkpoints carry norm
  stats inside ``model.safetensors`` and which the ACT adapter dispatches
  to a different code path.

No mocks, no synthetic fixtures — exercises the real `discover_intree_rskills`
helper and the real manifests on disk (CLAUDE.md §1.11, §5.4).
"""

from __future__ import annotations

from openral_rskill.loader import discover_intree_rskills

# Manifests whose checkpoint carries normalization OUTSIDE a lerobot
# PolicyProcessorPipeline (so `processors is None` is correct, not an
# oversight):
#   - act-aloha / act-aloha-insertion: norm stats live inside
#     model.safetensors; the ACT adapter dispatches on
#     `manifest.processors is None` and uses the snapshot_download path.
#   - gr00t-n17-libero: GR00T checkpoints carry norm stats in their own
#     `experiment_cfg/` metadata, not lerobot processor JSONs, and run
#     out-of-process via the GR00T sidecar (ADR-0046). `gr00t` is
#     deliberately excluded from `_MODERN_PROCESSOR_FAMILIES` for the same
#     reason.
_LEGACY_NO_PROCESSORS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "act-aloha",
        "act-aloha-insertion",
        "gr00t-n17-libero",
        # Non-lerobot VLAs: 3D Diffuser Actor (RLBench — own point-cloud/pose
        # pipeline) and OpenVLA-OFT (own HF-transformers processor) carry no
        # lerobot PolicyProcessorPipeline, so `processors is None` is correct.
        "3d-diffuser-actor-rlbench",
        "openvla-oft-simpler-widowx-nf4",
    }
)


def test_every_intree_manifest_declares_control_mode_semantics() -> None:
    """Every actuator entry must have a ``control_mode_semantics`` block."""
    manifests = list(discover_intree_rskills())
    assert manifests, "no in-tree rskills/*/rskill.yaml manifests discovered"

    missing: list[tuple[str, int]] = []
    for name, manifest in manifests:
        for i, actuator in enumerate(manifest.actuators_required):
            # control_mode_semantics is REQUIRED on the schema; presence
            # here means pydantic loaded it (Gap 2 contract).
            if actuator.control_mode_semantics is None:  # type: ignore[unreachable]
                missing.append((name, i))

    assert not missing, f"manifests missing control_mode_semantics on actuator entry: {missing}"


def test_every_modern_intree_manifest_declares_processors() -> None:
    """Every non-legacy manifest must declare a ``processors`` block.

    Skills without a VLA policy are exempt by KIND, mirroring the
    schema-level contract that forbids them from carrying a ``processors``
    block at all:

    * Wrapped-ROS rSkills (``kind: ros_action`` / ``ros_service`` per
      ADR-0024) carry no policy weights and therefore no preprocessor
      JSONs.
    * Detector rSkills (``kind: detector`` per ADR-0037) are perception
      producers with an exported ONNX/TensorRT engine and no lerobot
      ``PolicyProcessorPipeline`` — ``RSkillManifest._check_kind_consistency``
      FORBIDS ``processors`` for this kind.
    """
    manifests = list(discover_intree_rskills())
    assert manifests, "no in-tree rskills/*/rskill.yaml manifests discovered"

    missing: list[str] = []
    surprise_legacy: list[str] = []

    for name, manifest in manifests:
        # Non-VLA kinds don't carry a lerobot policy; `processors` is
        # `None` by construction and the manifest-level validator forbids
        # it from being set (ADR-0024 wrapped-ROS, ADR-0037 detector,
        # ADR-0047 scene VLM, ADR-0057 reward monitor, ADR-0072 playbook).
        if manifest.kind in {
            "ros_action",
            "ros_service",
            "detector",
            "vlm",
            "reward",
            "playbook",
        }:
            continue
        if manifest.processors is None:
            if name not in _LEGACY_NO_PROCESSORS_ALLOWLIST:
                missing.append(name)
        else:
            # An allowlisted legacy skill that nonetheless declared processors
            # is fine — the modern path is preferred whenever available.
            pass

    # Every entry in the allowlist must still exist on disk (catches an
    # allowlist entry going stale after a skill is renamed / removed).
    discovered_names = {name for name, _ in manifests}
    for legacy in _LEGACY_NO_PROCESSORS_ALLOWLIST:
        if legacy not in discovered_names:
            surprise_legacy.append(legacy)

    assert not missing, (
        "in-tree manifests missing the required `processors` block "
        f"(not on the legacy allowlist): {missing}"
    )
    assert not surprise_legacy, (
        f"legacy allowlist entries no longer present on disk: {surprise_legacy}"
    )


def test_every_processors_block_uses_per_file_uris() -> None:
    """The two URIs must point at distinct per-file artefacts."""
    manifests = list(discover_intree_rskills())
    for name, manifest in manifests:
        # Non-VLA kinds have no processors block (ADR-0024 wrapped-ROS,
        # ADR-0037 detector, ADR-0047 scene VLM).
        if manifest.kind in {"ros_action", "ros_service", "detector", "vlm"}:
            continue
        if manifest.processors is None:
            continue
        pre = manifest.processors.preprocessor_uri
        post = manifest.processors.postprocessor_uri
        assert pre != post, f"{name}: preprocessor_uri == postprocessor_uri ({pre!r})"
        # Pattern already enforces file-tail; double-check the friendly invariant
        # so a regression surfaces with a clearer message than a regex mismatch.
        assert "." in pre.rsplit("/", 1)[-1], (
            f"{name}: preprocessor_uri lacks a file tail with extension: {pre!r}"
        )
        assert "." in post.rsplit("/", 1)[-1], (
            f"{name}: postprocessor_uri lacks a file tail with extension: {post!r}"
        )
