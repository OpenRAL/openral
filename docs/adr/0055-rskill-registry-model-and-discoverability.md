# ADR-0055 — rSkill registry model + discoverability (`rskill search`)

- **Status:** Proposed 2026-06-12. **Landed:** the two-tier model is already how the
  system works (this ADR names it); the first-party weightless `ros_action` skills are
  on the Hub (ADR-0054 / the `OpenRAL/` namespace normalization); and the registry is
  now uniformly **public** (D6 decided — all `OpenRAL/*` repos public). **Open:** the
  deterministic tag projection (D3), the `rskill search` command (D4), and the
  Collection (D5).
- **Date:** 2026-06-12
- **ADR number:** `0055`. Renumbered from `0053` on merge with `master` (which claimed
  `0050`/`0051`/`0052`); the approach-to-pose ADR is `0053`, the `goal_builder` ADR
  `0054`. The integer is not load-bearing — cross-refs use filenames.
- **Related:**
  - ADR-0024 — `kind: ros_action` / `ros_service`; "the manifest is the artifact"
    for weightless skills.
  - ADR-0006 — HF Hub skill packaging (one repo per skill; provenance **unverified**,
    sigstore not implemented — do **not** describe published skills as "signed").
  - ADR-0021 — Tier-0 curl-bash installer + `openral install` dependency groups
    (sibling of, not the same as, `rskill install`).
  - ADR-0054 — the `goal_builder` skills whose publish surfaced the namespace drift
    that motivated writing this down.

## Context

An rSkill exists in two places, and we have never written down the relationship:

| Place | What it holds | Tooling |
| --- | --- | --- |
| In-repo `rskills/<id>/` | `rskill.yaml` + `README.md` (HF model card) + `eval/*.json`. **No weights** (gitignored). | schema-validated in CI; `discover_intree_rskills()` lists them; `resolve_rskill_local_dir()` resolves a bare/path/Hub URI to the local dir. |
| HF Hub `OpenRAL/rskill-<id>` | The same `rskill.yaml` + `README.md` + `eval/`, **plus** weights (`model.safetensors` / `model.onnx`) for VLA/detector skills. | `tools/rskill_publisher.py … --publish` pushes repo→Hub (always **private**); `openral rskill install <hub_id>` pulls Hub→local, validates, surfaces license, registers in `~/.local/share/openral/rskills.json`. |

So the flow is already one-way **author in-repo → publish to Hub → install from Hub**.
What was missing was (a) a name for it, (b) consistency (the `ros_action` skills used a
lowercase `openral/` namespace and were never published, so the drift stayed invisible
until a publish 403'd — see ADR-0054), and (c) **discovery**: `openral rskill install`
requires the user to *already know* the `OpenRAL/rskill-…` id. There is no
`rskill search`. The Hub itself is a search engine, but we under-use it:

- READMEs already carry model-card `tags:` front-matter, but they are **hand-authored
  and free-form** (`[OpenRAL, rskill, ros2, moveit]` on one skill, a different ad-hoc
  set on the next), so faceted filtering by `kind` / `role` / embodiment / license is
  unreliable.
- Every `OpenRAL/*` repo is **private**, so `HfApi.list_models(author="OpenRAL")`
  returns results only for org members. External users see nothing.

## Decision

### D1 — Name the two-tier model; keep `rskills/` in-repo

`rskills/` is the **first-party authoring source of truth + CI corpus** — not the
user-facing catalog. The HF `OpenRAL/*` org **is** the registry/catalog. Removing
`rskills/` from the repo (a question raised in discussion) is rejected: it is where
manifests are schema-validated in CI, where skill authors find reference examples, and
the only home for weightless skills. Third-party/community skills, by contrast, can
*only* live on the Hub (`their-org/rskill-X`, resolved by URI) — the monorepo cannot
host the ecosystem, which is exactly why the Hub is the catalog and the repo is not.

`tools/rskill_publisher.py` stays the **single** repo→Hub push path; `rskill install`
is the pull. To stop the in-repo/Hub copies from silently diverging (the failure mode
behind ADR-0054's namespace bug), add a **CI drift-check**: for each in-repo manifest
with a published Hub twin, assert the Hub `rskill.yaml` matches (or fail, prompting a
republish). This is the safety net the "manifest in two places" design needs.

### D2 — Publish weightless first-party skills too (done for the 4 `ros_action` skills)

Consistency target: **every first-party rSkill is resolvable by a stable
`OpenRAL/rskill-…` Hub id**, weights or not. A weightless `ros_action` skill's Hub repo
is just its manifest + card — still worth publishing so the reasoner/users resolve it
the same way as a VLA. Already executed for `rskill-moveit-{joints,eef-pose,look-at}`
and `rskill-nav2-navigate-to-pose` (private; ADR-0054).

### D3 — Deterministic model-card tags (a reserved projection of manifest fields)

The `rskill_publisher` doc validator generates **and verifies** a reserved tag block on
each README from the manifest, so Hub faceting is reliable instead of vibes:

```
tags:
  - OpenRAL                 # org marker (constant)
  - rskill                  # type marker (constant)
  - rskill-kind:ros_action  # manifest.kind ∈ {vla, ros_action, ros_service, detector, vlm}
  - rskill-role:s1          # manifest.role ∈ {s1, s2, s0}
  - embodiment:franka_panda # one per RobotCapabilities embodiment tag
  - embodiment:ur5e
  - license:apache-2.0      # manifest.license
```

Free-form descriptive tags (`smolvla`, `lerobot`, `moveit`) stay allowed and additive —
the validator only owns the `rskill-kind:`/`rskill-role:`/`embodiment:`/`license:`
reserved prefixes, and fails publish if they drift from the manifest. No schema change:
this is a README-generation rule in the publisher, not a manifest field.

### D4 — `openral rskill search` (the missing discovery half)

Add one command under the existing `rskill` Typer group, completing the
**search → install** loop (`rskill install` already exists):

```
openral rskill search [QUERY] [--kind ros_action] [--role s1] \
                       [--embodiment franka_panda] [--license apache-2.0]
```

A thin wrapper over `HfApi.list_models(author="OpenRAL", tags=[…])` that maps the flags
to the D3 reserved tags, renders a Rich table (`id`, `kind`, `role`, embodiments,
license), and tells the user to `rskill install <id>`. ~40–60 lines, reuses the
`HfApi` + Rich patterns already in `main.py`. **No bespoke index, embedding store, or
search service** — the Hub is the index.

### D5 — `OpenRAL/rskills` Collection

Curate an HF Collection grouping the published skills for human browsing — the
zero-code complement to the programmatic `rskill search`.

### D6 — Visibility: the registry is **public** (decided)

All `OpenRAL/*` rSkill repos are public, including the ones wrapping non-commercial /
proprietary-weight checkpoints (NVIDIA non-commercial, RLWRLD, gated-research π0.5). The
earlier instinct to keep license-gated skills private was **moot**: a skill repo holds
only the `rskill.yaml` manifest + model card + `eval/` — **never the weights**. The
weights stay in their gated upstream repos, referenced by `weights_uri`, and the
loader's non-commercial guard (`commercial_use` / `OPENRAL_ALLOW_NONCOMMERCIAL`, per
CLAUDE.md §1.9 + ADR-0046) fires at **load time regardless of repo visibility**. So
making a manifest public neither distributes nor relaxes anything license-protected — it
only makes the skill *discoverable*; the protected artifact (weights) and its
enforcement point (the loader) are untouched.

What this means going forward: `tools/rskill_publisher.py` creates repos **private by
default** (a safe default for a half-finished publish), so the publish flow is
**publish → verify → flip public** (`HfApi.update_repo_settings(private=False)`). License
protection lives in the manifest's `license` field + `weights_uri` gating + the loader,
not in repo visibility.

## Consequences

- Discovery stops requiring tribal knowledge of repo ids: `rskill search --embodiment
  franka_panda --kind vla` → `rskill install <id>`.
- The CI drift-check (D1) makes the lowercase-namespace class of bug (ADR-0054) a red
  test instead of a silent divergence found only at publish time.
- Tags become a contract (D3), so third parties publishing `their-org/rskill-X` with the
  same reserved tags are discoverable by the same `rskill search` if we later widen the
  `author=` filter — the model extends past first-party for free.
- The registry is public (D6), so `rskill search` works for everyone, not just org
  members — and license protection stays where it belongs (manifest `license` +
  `weights_uri` gating + the loader's load-time guard), not in repo visibility.

## Alternatives considered

- **Drop `rskills/` from the repo; Hub-only.** Rejected (D1): loses the CI validation
  corpus, reference examples, and the home for weightless skills; couples every manifest
  edit to a network round-trip.
- **Build a dedicated search index / service (embeddings, a hosted catalog API).**
  Rejected (D4): the Hub already indexes models with faceted tag search; a parallel
  index is undifferentiated infrastructure to maintain and keep in sync.
- **Leave tags free-form.** Rejected (D3): unreliable faceting is barely better than no
  tags; the value is in the *contract*, which costs ~one validator rule.

## Implementation plan (phased; each independently testable)

1. **D3 tag projection** in `_rskill_doc_validator.py` (or the publisher): generate +
   verify the reserved tag block; unit-test the manifest→tags mapping; backfill the
   existing 33 cards. Fail publish on reserved-tag drift.
2. **D4 `rskill search`** command + unit test against a recorded `list_models` response
   (per CLAUDE.md §1.11, a recorded Hub fixture under `tests/<tier>/fakes/`, not a mock).
3. **D1 CI drift-check** — a test asserting each published in-repo manifest matches its
   Hub twin (skipped offline; real Hub fetch when network is present).
4. **D5 Collection** — one-time curation (manual / `HfApi.add_collection_item`).
5. **D6 visibility** — **done**: all `OpenRAL/*` repos flipped public via
   `HfApi.update_repo_settings(private=False)`. The publisher still creates repos private
   by default, so future publishes follow publish → verify → flip-public.
