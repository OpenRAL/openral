"""Canary test: every in-tree rSkill manifest carries a canonical repo name.

The ``rskill-<model>-<robot>-<task>-<quantization>`` convention (or
``rskill-playbook-<name>``) was enforced only inside
``tools/rskill_publisher.py``, which runs at publish time. A manifest whose
``name:`` drifted from the convention therefore stayed green in CI and only
failed when somebody tried to publish it — which is exactly how
``rskills/gr00t-n17-b1k-turning-on-radio`` shipped as
``OpenRAL/rskill-gr00t-n17-b1k-turning-on-radio`` (non-canonical ``<model>``
token, no ``<quantization>`` segment) and stayed unpublishable.

This asserts the same predicate the publisher uses,
:func:`openral_core.schemas.repo_name_is_canonical`, against every real
manifest on disk via ``discover_intree_rskills`` — no mocks, no synthetic
fixtures (CLAUDE.md §1.11).
"""

from __future__ import annotations

from openral_core.schemas import expected_repo_name, repo_name_is_canonical
from openral_rskill.loader import discover_intree_rskills

# ``rskills/template/`` is the scaffolding stub that ``rskill_scaffolder.py``
# copies and rewrites; its placeholder name is intentional and it is never
# published (no TEMPLATE_ORG on the Hub).
_PLACEHOLDER_DIRS: frozenset[str] = frozenset({"template"})


def test_every_intree_manifest_name_is_canonical() -> None:
    """A non-canonical ``name:`` is a publish-time failure; catch it in CI."""
    manifests = list(discover_intree_rskills())
    assert manifests, "no in-tree rskills/*/rskill.yaml manifests discovered"

    offenders: list[str] = []
    for dir_name, manifest in manifests:
        if dir_name in _PLACEHOLDER_DIRS:
            continue
        if repo_name_is_canonical(
            manifest.name, kind=manifest.kind, model_family=manifest.model_family
        ):
            continue
        try:
            suggested = expected_repo_name(manifest)
        except ValueError as exc:  # no compliant name derivable from the manifest
            suggested = f"<undeterminable: {exc}>"
        offenders.append(f"{dir_name}: {manifest.name!r} -> suggested {suggested}")

    assert not offenders, (
        "rSkill manifests whose name: violates rskill-<model>-<robot>-<task>-"
        "<quantization> (or rskill-playbook-<name>) and would be refused by "
        "tools/rskill_publisher.py:\n  " + "\n  ".join(offenders)
    )
