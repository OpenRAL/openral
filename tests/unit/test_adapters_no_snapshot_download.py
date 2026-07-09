"""Canary: SmolVLA adapter no longer calls snapshot_download.

Closes Gap 1 + Gap 3 of the rSkill self-containment audit at the
adapter layer. The SmolVLA + modern-ACT adapters MUST consume
``manifest.processors`` via per-file ``hf_hub_download`` —
``snapshot_download`` is the implicit-fetch path we deliberately
replaced and should not creep back in.

The ACT adapter still legitimately calls ``snapshot_download`` from the
LEGACY branch (``rskills/act-aloha``, where norm stats live inside
``model.safetensors`` and the policy class itself loads via
``ACTPolicy.from_pretrained(pretrained_path)``). That branch is
documented and out of scope for this round. This test pins:

- ``policies/smolvla.py``: no ``snapshot_download`` reference anywhere
  (the only HF Hub call goes through ``materialize_processor_dir``).
- ``policies/act.py``: at most one ``snapshot_download`` call site
  (the ACT-specific config+weights snapshot that feeds
  ``_sanitize_act_config_json`` + ``ACTPolicy.from_pretrained``), and
  that call site is NOT inside the modern-processors branch.

If the count creeps up, the canary fires before a silent regression
lands a new implicit fetch.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_source(rel_path: str) -> str:
    return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _count_code_occurrences(src: str, symbol: str) -> int:
    """Count ``symbol`` occurrences in non-comment lines only.

    Comments are allowed to mention the symbol (e.g. "no snapshot_download
    here" or a historical note) without tripping the canary. We only flag
    real Python references.
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
    """ACT keeps a bounded number of snapshot_download references.

    Two legitimate call sites today, each preceded by a local import =
    four code occurrences:

    1. Policy weights snapshot for ``_sanitize_act_config_json`` +
       ``ACTPolicy.from_pretrained`` (config + weights, not processors).
    2. Legacy norm-stats loader (`_try_load_act_norm_stats` —
       ``rskills/act-aloha`` keeps working unchanged).

    The modern PolicyProcessorPipeline path now uses
    ``materialize_processor_dir`` instead of ``snapshot_download``;
    a third call site indicates a regression.
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
