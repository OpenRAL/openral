"""Search the HF Hub for installable rSkills (layer 3, ``openral_rskill``).

``openral rskill search`` previously drove the Hub's own ``search=`` query
param directly against ``HfApi.list_models`` (repo-id substring only —
description/scene/action keywords never matched), used ``--limit`` to cap
the number of *org repos inspected* rather than results returned (so once
the org passed the default limit, skills silently vanished), and fetched
every candidate manifest sequentially. This module fixes all three: it
lists the org once via the ``rskill`` model-card tag (no Hub-side
``search``), fetches candidate manifests concurrently, and matches the
free-text query locally against a richer haystack (repo id, manifest
fields, and Hub tags) so a query word that only appears in a skill's
``description`` still matches.

Public surface
--------------
- ``HubRSkillHit``: One matching Hub repo, its validated manifest, and
  whether it is already available locally (in-tree or installed).
- ``HubRSkillSearchResult``: The full result — hits plus bookkeeping
  (``skipped``, ``skipped_repo_ids``, ``inspected``) so callers can surface
  "N repos skipped" (with ids) without this module logging anything itself.
- ``search_hub_rskills``: The search itself.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import structlog
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillManifest
from pydantic import BaseModel, ConfigDict, Field

from openral_rskill.loader import discover_intree_rskills, rSkill

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

log = structlog.get_logger(__name__)


class HubRSkillHit(BaseModel):
    """One Hub repo whose ``rskill.yaml`` validated and matched the query.

    Attributes:
        repo_id: HF Hub repository identifier, e.g.
            ``"OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32"``.
        manifest: The validated ``RSkillManifest`` fetched from the repo.
        local: ``"in-tree"`` when the Hub ``repo_id`` matches the ``name``
            declared by a ``rskills/<dir>/rskill.yaml`` in this checkout
            (compared against the local manifest's name, never the remote
            one — a renamed Hub repo with a stale manifest ``name`` must not
            read as in-tree); ``"installed"`` when the local HF-install
            registry (``rSkill.list_installed()``) has a matching
            ``repo_id``; ``None`` when neither.
        hub_tags: The repo's model-card tags as reported by the Hub
            (includes ``"OpenRAL"`` and ``"rskill"``, plus embodiment/license
            tags set by the publisher).

    Example:
        >>> from openral_rskill.loader import load_rskill_manifest
        >>> m = load_rskill_manifest("act-aloha")
        >>> hit = HubRSkillHit(
        ...     repo_id=m.name,
        ...     manifest=m,
        ...     local="in-tree",
        ...     hub_tags=("OpenRAL", "rskill"),
        ... )
        >>> hit.local
        'in-tree'
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str
    manifest: RSkillManifest
    local: Literal["in-tree", "installed"] | None
    hub_tags: tuple[str, ...] = Field(default_factory=tuple)


class HubRSkillSearchResult(BaseModel):
    """Result of ``search_hub_rskills`` — matching hits plus bookkeeping.

    Attributes:
        hits: Matching repos, sorted by ``repo_id`` and capped by ``limit``.
        skipped: Count of inspected org repos with no valid ``rskill.yaml``
            (excluded from ``hits`` regardless of query/filters). Always
            equal to ``len(skipped_repo_ids)``.
        skipped_repo_ids: The repo ids counted in ``skipped``, sorted. This
            module never logs per-repo skips (the CLI never calls
            ``structlog.configure()``, so structlog's default ``PrintLogger``
            would write straight to stdout and corrupt ``--json`` piping,
            e.g. into ``jq``) — this field is the explicit record instead
            (CLAUDE.md §1.4: explicit beats implicit, nothing silently
            dropped).
        inspected: Total org repos inspected (before query/facet filtering
            and before ``limit`` truncation) — i.e. ``len(hits) + skipped``
            plus any repos that had a valid manifest but did not match.

    Example:
        >>> HubRSkillSearchResult(hits=[], skipped=0, skipped_repo_ids=(), inspected=0)
        HubRSkillSearchResult(hits=[], skipped=0, skipped_repo_ids=(), inspected=0)
    """

    hits: list[HubRSkillHit]
    skipped: int
    skipped_repo_ids: tuple[str, ...] = Field(default_factory=tuple)
    inspected: int


def _manifest_haystack(repo_id: str, manifest: RSkillManifest, hub_tags: tuple[str, ...]) -> str:
    """Build the lower-cased free-text blob a query's tokens are matched against."""
    parts = [
        repo_id,
        manifest.name,
        manifest.description,
        manifest.model_family or "",
        manifest.kind,
        manifest.role,
        *manifest.embodiment_tags,
        *hub_tags,
    ]
    return " ".join(parts).lower()


def _matches_query(query: str, haystack: str) -> bool:
    """Every whitespace-split token of ``query`` must substring-match ``haystack``."""
    tokens = query.lower().split()
    return all(token in haystack for token in tokens)


def _matches_facets(
    m: RSkillManifest, *, kind: str, role: str, embodiment: str, license_: str, family: str
) -> bool:
    """Return whether a manifest passes every non-empty facet filter."""
    if kind and m.kind != kind:
        return False
    if role and m.role != role:
        return False
    if embodiment and embodiment not in m.embodiment_tags:
        return False
    if license_ and m.license.value != license_:
        return False
    return not (family and m.model_family != family)


def _fetch_one(model: ModelInfo) -> tuple[str, RSkillManifest | None, tuple[str, ...], str]:
    """Fetch + validate one Hub repo's ``rskill.yaml``; ``None`` manifest = skip.

    Runs in a worker thread (see ``search_hub_rskills``). A repo with no
    manifest, an unparseable one, or a Hub-side miss is not an rSkill — the
    caller counts and surfaces these rather than failing the whole search.

    Returns the skip reason as the 4th element (``""`` on success) so the
    information is never silently dropped — but this function does NOT log
    it: the CLI never calls ``structlog.configure()``, so structlog's
    default ``PrintLogger`` writes straight to stdout, which would corrupt
    ``openral rskill search --json | jq``. ``search_hub_rskills`` threads
    the reason through for a future verbose surface; today only the repo id
    is exposed, via ``HubRSkillSearchResult.skipped_repo_ids``.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError  # noqa: PLC0415

    repo_id = model.id
    hub_tags = tuple(getattr(model, "tags", None) or ())
    try:
        path = hf_hub_download(repo_id=repo_id, filename="rskill.yaml")
        manifest = RSkillManifest.from_yaml(path)
    except (
        OSError,  # reason: local cache / filesystem failure reading the downloaded file
        ValueError,  # reason: malformed YAML the Pydantic layer can't even parse
        ROSConfigError,  # reason: schema validation failure (RSkillManifest.from_yaml)
        HfHubHTTPError,  # reason: repo has no rskill.yaml at all, or a transient Hub HTTP error
        EntryNotFoundError,  # reason: rskill.yaml missing from an otherwise-valid repo
    ) as exc:
        return repo_id, None, hub_tags, str(exc)
    return repo_id, manifest, hub_tags, ""


def _local_markers() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(in_tree_names, installed_repo_ids)`` for the ``local`` marker.

    ``in_tree_names`` holds the ``name`` field declared by every in-tree
    ``rskills/<dir>/rskill.yaml`` — the caller compares the Hub repo id
    against these, never against the (possibly stale) remote manifest's own
    ``name`` field.

    A registry read failure (corrupt JSON, or a row that fails
    ``InstalledRSkillEntry`` validation) must never fail the search itself —
    but per CLAUDE.md §1.4 (explicit beats implicit) it is surfaced as a
    warning, not silently swallowed.
    """
    in_tree = frozenset(m.name for _, m in discover_intree_rskills())
    installed: frozenset[str] = frozenset()
    try:
        installed = frozenset(e.repo_id for e in rSkill.list_installed())
    except (ROSConfigError, ValueError) as exc:
        # reason: corrupt registry JSON (ROSConfigError) or a row failing
        # InstalledRSkillEntry validation (pydantic ValidationError, a ValueError
        # subclass) — treat as no installed entries but make it visible.
        log.warning("rskill.hub_search.registry_unreadable", error=str(exc))
    return in_tree, installed


def search_hub_rskills(
    query: str = "",
    *,
    kind: str = "",
    role: str = "",
    embodiment: str = "",
    license: str = "",  # reason: matches the CLI's --license facet name
    family: str = "",
    org: str = "OpenRAL",
    limit: int | None = None,
    max_workers: int = 8,
) -> HubRSkillSearchResult:
    """Search an HF Hub org for rSkills matching a free-text query + facets.

    Lists every repo the org tagged ``rskill`` (one Hub call, no Hub-side
    ``search=``), fetches each candidate's ``rskill.yaml`` concurrently, then
    matches ``query`` locally (case-insensitive, every whitespace-split token
    must substring-match repo id / manifest name / description / model
    family / kind / role / embodiment tags / Hub tags) and applies the facet
    filters. Repos without a valid manifest are counted in ``skipped`` and
    excluded from ``hits`` regardless of query or filters.

    Args:
        query: Free-text query; empty matches every valid manifest.
        kind: Exact-match filter on ``manifest.kind`` (e.g. ``"vla"``).
        role: Exact-match filter on ``manifest.role`` (``"s0"``/``"s1"``/``"s2"``).
        embodiment: Require this tag in ``manifest.embodiment_tags``.
        license: Exact-match filter on ``manifest.license.value``.
        family: Exact-match filter on ``manifest.model_family``.
        org: HF Hub org (``author``) to search. Defaults to ``"OpenRAL"``.
        limit: Max hits returned, applied after sorting by ``repo_id``.
            ``None`` (default) returns every match.
        max_workers: Thread pool size for concurrent manifest fetches.

    Returns:
        A ``HubRSkillSearchResult`` with hits sorted by ``repo_id``.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.

    Example:
        >>> # result = search_hub_rskills("aloha", kind="vla")
        >>> # [h.repo_id for h in result.hits]
        >>> # ['OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32']
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    models = list(HfApi().list_models(author=org, filter="rskill", limit=None))
    inspected = len(models)

    skipped_repo_ids: list[str] = []
    candidates: list[tuple[str, RSkillManifest, tuple[str, ...]]] = []
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rskill_hub_search"
    ) as pool:
        for repo_id, manifest, hub_tags, _reason in pool.map(_fetch_one, models):
            if manifest is None:
                skipped_repo_ids.append(repo_id)
                continue
            candidates.append((repo_id, manifest, hub_tags))

    in_tree_names, installed_ids = _local_markers()

    hits: list[HubRSkillHit] = []
    for repo_id, manifest, hub_tags in candidates:
        if not _matches_facets(
            manifest, kind=kind, role=role, embodiment=embodiment, license_=license, family=family
        ):
            continue
        haystack = _manifest_haystack(repo_id, manifest, hub_tags)
        if not _matches_query(query, haystack):
            continue
        local: Literal["in-tree", "installed"] | None
        if repo_id in in_tree_names:
            local = "in-tree"
        elif repo_id in installed_ids:
            local = "installed"
        else:
            local = None
        hits.append(
            HubRSkillHit(repo_id=repo_id, manifest=manifest, local=local, hub_tags=hub_tags)
        )

    hits.sort(key=lambda h: h.repo_id)
    if limit is not None:
        hits = hits[:limit]

    sorted_skipped = tuple(sorted(skipped_repo_ids))
    return HubRSkillSearchResult(
        hits=hits,
        skipped=len(sorted_skipped),
        skipped_repo_ids=sorted_skipped,
        inspected=inspected,
    )
