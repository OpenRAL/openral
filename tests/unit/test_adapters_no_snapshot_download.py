"""Canary: SmolVLA adapter no longer calls snapshot_download.

Closes Gap 1 + Gap 3 of the rSkill self-containment audit at the adapter
layer. SmolVLA + modern-ACT adapters must consume ``manifest.processors``
via per-file ``hf_hub_download``; ``snapshot_download`` is the deliberately
replaced implicit-fetch path and must not creep back in.

The ACT adapter's LEGACY branch (``rskills/act-aloha``, norm stats inside
``model.safetensors``, loaded via ``ACTPolicy.from_pretrained``) still
legitimately calls ``snapshot_download`` — documented, out of scope here.
Pins:

- ``policies/smolvla.py``: no ``snapshot_download`` reference anywhere
  (the only HF Hub call goes through ``materialize_processor_dir``).
- ``policies/act.py``: at most one ``snapshot_download`` call site (the
  ACT-specific config+weights snapshot feeding ``_sanitize_act_config_json``
  + ``ACTPolicy.from_pretrained``), not inside the modern-processors branch.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_source(rel_path: str) -> str:
    return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _count_code_occurrences(src: str, symbol: str) -> int:
    """Count ``symbol`` occurrences in non-comment lines only.

    Comments may mention the symbol without tripping the canary.
    """
    count = 0
    for line in src.splitlines():
        # Drop the comment tail; whatever's left is code.
        code_part = line.split("#", 1)[0]
        count += code_part.count(symbol)
    return count


def test_smolvla_adapter_has_no_snapshot_download() -> None:
    """SmolVLA adapter must consume manifest.processors only — no implicit snapshot."""
    src = _read_source("python/sim/src/openral_sim/policies/smolvla.py")
    occurrences = _count_code_occurrences(src, "snapshot_download")
    assert occurrences == 0, (
        f"policies/smolvla.py contains {occurrences} non-comment "
        "snapshot_download reference(s); the adapter must consume "
        "manifest.processors via materialize_processor_dir instead "
        "(rSkill self-containment audit Gap 1+3)."
    )


def test_act_adapter_keeps_only_legacy_snapshot_calls() -> None:
    """ACT keeps ≤ ``max_allowed`` non-comment ``snapshot_download`` refs: the
    legacy weights snapshot (``_sanitize_act_config_json`` +
    ``ACTPolicy.from_pretrained``) and the legacy norm-stats loader
    (``rskills/act-aloha``); the modern path uses ``materialize_processor_dir``.
    """
    src = _read_source("python/sim/src/openral_sim/policies/act.py")
    occurrences = _count_code_occurrences(src, "snapshot_download")
    max_allowed = 4  # 2 imports + 2 calls (config snapshot + legacy norm-stats)
    assert occurrences <= max_allowed, (
        f"policies/act.py has {occurrences} non-comment snapshot_download "
        f"references (allowed <= {max_allowed}); a new implicit snapshot "
        "fetch was introduced. The modern processors path must use "
        "materialize_processor_dir."
    )


def test_act_adapter_dispatches_on_manifest_processors() -> None:
    """The modern branch must be gated on manifest.processors, not a fs probe."""
    src = _read_source("python/sim/src/openral_sim/policies/act.py")
    # Sentinel: the new branch reads manifest.processors; the old branch
    # probed the snapshot directory for policy_preprocessor.json.
    assert "manifest.processors is not None" in src, (
        "policies/act.py must dispatch on `manifest.processors is not None` "
        "for the modern PolicyProcessorPipeline path."
    )
    # Non-comment lines must not call os.path.exists on a preprocessor json —
    # that was the historic fs-probe dispatch.
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if "os.path.exists(" in code and "policy_preprocessor" in code:
            raise AssertionError(
                "policies/act.py still probes the filesystem for "
                "policy_preprocessor.json — that path was replaced by the "
                "manifest-driven dispatch."
            )


def test_smolvla_adapter_uses_materialize_processor_dir() -> None:
    """SmolVLA must wire the new helper into the build function."""
    src = _read_source("python/sim/src/openral_sim/policies/smolvla.py")
    assert "materialize_processor_dir(manifest)" in src, (
        "_build_smolvla should call materialize_processor_dir(manifest) to "
        "feed make_pre_post_processors."
    )


def test_act_adapter_uses_materialize_processor_dir_in_modern_branch() -> None:
    """ACT's modern branch must call materialize_processor_dir."""
    src = _read_source("python/sim/src/openral_sim/policies/act.py")
    assert "materialize_processor_dir(manifest)" in src, (
        "_build_act's modern branch should call materialize_processor_dir(manifest)."
    )


def test_act_close_releases_the_nvmm_executor() -> None:
    """The TRT device executor must be closed on skill swap, not left to GC.

    Its engine + activation workspace live outside torch's caching allocator,
    so release_torch_modules cannot reclaim them; a close() that skips the
    executor leaves VRAM resident and the next skill's load OOMs (same class
    SmolVLA's _nvmm_encoder teardown guards against). Fake stands in for the
    OpenRAL Pro executor at the package boundary (§1.11).
    """
    from openral_core import VLASpec
    from openral_sim.policies.act import _ACTAdapter

    closed: list[bool] = []

    class _FakeExecutor:
        def close(self) -> None:
            closed.append(True)

    adapter = _ACTAdapter(
        spec=VLASpec(id="act", weights_uri="lerobot/act_aloha_sim_transfer_cube_human"),
        device="cpu",
        _policy=object(),
        _preprocessor=None,
        _postprocessor=None,
        _torch=None,
        _nvmm_executor=_FakeExecutor(),
    )
    adapter.close()

    assert closed == [True], "close() must close the NVMM executor exactly once"
    assert adapter._nvmm_executor is None
    assert adapter._policy is None
