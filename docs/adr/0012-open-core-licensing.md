# ADR-0012: Open-core licensing — Apache-2.0 core + PolyForm Small Business commercial tier

- Status: Proposed
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: CLAUDE.md §1.9 (license lineage), §6.1 (layer discipline),
  §7.9 (docs / ADR discipline)

## Context

Today every package in this monorepo is uniformly **Apache-2.0**: the
root `LICENSE`, every `pyproject.toml`, and the `NOTICE`. CLAUDE.md
§1.9 codifies this: *"Apache-2.0 incoming is the default; copy-left is
rejected without TSC review."* The only non-permissive carve-out
referenced anywhere is the *separate* `openral/cloud` repo, which
is BSL-1.1 and lives outside this monorepo.

We want to keep the project freely usable for researchers and small
organisations so they can adopt and contribute, but require larger
enterprises (target threshold ~100 employees) to purchase a
commercial plan. The restriction must be **permanent** (no
auto-conversion to OSS) and the project should remain
**OSI-approved open source for the core**, with a **source-available
commercial tier** for advanced features. This is a strategy shift,
not a tweak: it requires an ADR per §6.1 and an amendment to §1.9.

The audit covered the 10 Python packages under `python/`, the 6 ROS
packages under `packages/`, and the 9 in-tree rSkill manifests under
`rskills/`. Every Tier-2 package named below is **blue (planned)** in
`docs/architecture/repo-state-map.html` — none of them exist on disk
today. This means the licensing split has **zero migration cost
right now**: we set the posture before the code lands.

The licensing-option comparison and the industry-pattern survey
(Datadog, Grafana, Sentry, Elastic, Honeycomb, New Relic,
OpenTelemetry, Jaeger, Prometheus) that informed this decision were
done out-of-tree as a planning artifact; the conclusion that
permissive SDK + restricted product is the prevailing pattern is
summarised in *Alternatives considered* below.

## Decision

Adopt a **two-tier open-core licensing model**:

| Tier | License | Scope |
|------|---------|-------|
| **Tier 1 — Open Core** | **Apache-2.0** (unchanged) | Contracts, runtime substrate, simulators, HAL/sensors, **observability primitives**, safety. |
| **Tier 2 — Commercial Source-Available** | **PolyForm Small Business License 1.0.0** + **Academic Research Additional Permission** | LLM reasoner, dispatcher, WAM adapters, commercial skill catalog, fleet orchestration. |

### Tier 1 — Apache-2.0 (unchanged)

These define the **contract surface** and the **commodity runtime
substrate**. They MUST remain permissive open or the ecosystem breaks
(third-party HAL adapters, downstream ROS users, academic forks,
conda-forge / Debian packaging, certifiers reading safety and trace
code).

| Package / path | Layer | Why permissive |
|---|---|---|
| `python/core/` (`openral_core`) | Cross-cutting | Normative Pydantic schemas. Same status as protobufs/IDL. |
| `packages/msgs/` | Cross-cutting | ROS 2 IDL — anyone publishing/subscribing needs unrestricted access. |
| `python/cli/` (`ral`) | UX | Driver-only; restricting buys nothing. |
| `python/hal/` + `packages/hal_*/` (SO-100, Franka, UR5e, UR10e) | L0 | Vendor adapters; encourages community contribution. |
| `python/sensors/` | L1 | Community sensor catalog. |
| `python/world_state/` + `packages/world_state/` | L2 | Foundational; wraps tf2. |
| `python/rskill/` (`Skill` ABC + `rSkill` loader + runtime adapters) | L3 | The **packaging format** must be open for the rSkill ecosystem to exist. Skill weights remain governed by their upstream licenses (SmolVLA Apache, π0 research-permissive, GR00T NVIDIA non-commercial, etc.) via the existing license-lineage code per §1.9. |
| `python/sim/` + `python/eval-shim/` | L3-eval | Reproducible benchmarks; dataset-flywheel substrate. |
| **`python/observability/`** (today: `configure_observability`, `shutdown_observability`, `rskill_span`, `inference_span`, `safety_span`, `traced`, `install_structlog_bridge`) | L7 | **Trace primitives stay Apache-2.0.** Two reasons: (a) Traceability is a *safety property* per CLAUDE.md §1.1, §1.8 ("every action that reaches a motor is traceable... replayable from the trace alone") — certifiers and auditors need source-level access to the recording mechanism. (b) Industry-standard pattern: Datadog, New Relic, Honeycomb, Grafana, Sentry, and Elastic all keep their SDKs/agents permissive (Apache-2.0 or MIT) even when the backend is restricted. Restricting ~200 lines of OTel/structlog wrappers gates code anyone could rewrite from upstream in an afternoon. |
| `python/detect/` | UX | Auto-detect; promotes adoption. |
| `packages/safety/` + `cpp/safety_kernel/` (planned, L6) | L6 | **Safety code stays Apache-2.0 even when it lands.** Certifiability requires open auditability. Restricting safety code is an anti-pattern (§7.7). |
| `cpp/rt_bridge/` (planned) | L0 | Real-time plumbing; commodity. |
| `tools/`, `scripts/`, `examples/`, `tests/`, `docs/`, `robots/` | — | Build / fixtures / docs. |
| `rskills/<name>/rskill.yaml` (9 in-tree manifests) | L3 | Manifests reference upstream-licensed weights; manifests themselves stay Apache. |

### Tier 2 — PolyForm Small Business 1.0.0 + Academic Permission

These are the **value-added agentic and orchestration layers** — the
parts an enterprise pays for. All are currently **planned** in the
repo state map, so this split has zero migration cost today: we set
the license posture before the code lands.

| Package / path | Layer | Why commercial tier |
|---|---|---|
| `python/reasoner/` (planned, L4) | L4 | The LLM-driven S2 planner. Tool palette, BT-XML emission, replanning ladder. |
| `python/wam/` (planned, L5) | L5 | World Action Models (Cosmos Predict / IRASim / UnifoLM-WMA-0 adapters). Heavy compute, enterprise differentiator. |
| `python/dispatcher/` (planned) | Cross-cutting | Edge/cloud/split dispatcher. The hosted SaaS half stays BSL-1.1 in `openral/cloud`; the edge/split halves land here under PolyForm. |
| `python/skill_catalog/` (planned) | L3-extension | Curated, validated, support-backed rSkill catalog beyond the 9 free in-tree examples. The base Skill SDK and rSkill format stay Apache; only the *premium catalog* is gated. |
| `python/fleet/` (planned) | Cross-cutting | Multi-robot orchestration. |

The **safety kernel** and **observability primitives** are
deliberately kept in Tier 1. Locking safety or trace recording behind
a commercial license is hostile to certification bodies, to
researchers, and to the project's stated north star
(*"every action that reaches a motor is traceable, typed,
safety-checked, and replayable"*).

### Threshold

PolyForm Small Business 1.0.0's built-in carve-out: **<100
individuals (employees + contractors) AND <$1M annual revenue in the
prior tax year (USD 2019, CPI-adjusted)**. This is permanent — there
is no Change Date or auto-conversion.

### Academic Research Additional Permission

Appended verbatim to the PolyForm SBL `LICENSE` in each Tier-2
package, and stored canonically at
`LICENSES/Academic-Research-Permission.txt`:

> **Additional Permission — Academic and Non-Commercial Research Use.**
> Notwithstanding the size and revenue limits in the *Small Business*
> section of the PolyForm Small Business License 1.0.0, the licensor
> grants any natural person affiliated with a degree-granting academic
> institution, or any registered non-profit research organisation, the
> right to use, modify, and distribute this software for
> **non-commercial research and teaching purposes**, regardless of the
> size or revenue of that institution. Publication of research
> results (including pre-prints, journal articles, and open datasets)
> is permitted. Productisation, commercial deployment, or paid
> services built on the software remain subject to the underlying
> PolyForm Small Business License.

This closes the "universities have >100 employees" gap that would
otherwise block academic adopters.

### Files this ADR establishes

1. `LICENSES/Apache-2.0.txt` — verbatim Apache-2.0 text (copy of the
   root `LICENSE`).
2. `LICENSES/PolyForm-Small-Business-1.0.0.txt` — verbatim PolyForm
   Small Business License 1.0.0 text, mirrored from the canonical
   PolyForm Project source.
3. `LICENSES/Academic-Research-Permission.txt` — the additional
   permission text quoted above.
4. Amended `NOTICE` enumerating the two-tier posture.
5. Amended `README.md` "License" section explaining the model in
   plain language.
6. Amended `CLAUDE.md §1.9` reflecting the dual-tier model.

The ADR deliberately does **not** create empty Tier-2 package
directories, `pyproject.toml` shells, a CI license-check gate, or
extend the (not-yet-existing) `LicensePosture` enum. Those changes
land in the same PR that introduces the first Tier-2 package, where
there is actual code for them to govern. Pre-creating empty
infrastructure is anti-pattern per CLAUDE.md
*"don't add features beyond what the task requires."*

## Consequences

- **No code under `python/` or `packages/` changes license today.**
  Every existing module is Tier 1 and stays Apache-2.0. The split is
  pre-declared policy for future Tier-2 packages.
- **OSI status is preserved for the core.** Debian, Fedora,
  conda-forge, and the FSF can continue to ship Tier 1 packages
  without legal review. They will not ship Tier 2 packages when those
  land — this is the explicit, intended trade.
- **Researchers at large universities are not blocked.** The
  Academic Research Additional Permission grants non-commercial use
  regardless of institution size. The permission is bespoke text, not
  a standard license; a one-page FAQ at `docs/licensing/academic-use.md`
  will land alongside the first Tier-2 package.
- **A Contributor License Agreement (CLA) becomes necessary before
  the first Tier-2 package merges.** Without a CLA, the project
  cannot unilaterally apply PolyForm to a contribution. Recommended
  posture: CLA-assistant bot + Apache-style ICLA. This is called out
  here so it lands with the first Tier-2 PR, not retroactively.
- **No relicensing of historical commits is required**, because every
  Tier-2 package is greenfield. Code already on `main` under
  Apache-2.0 stays Apache-2.0 for those revisions, forever. Waiting
  longer would have made this much more expensive: relicensing
  retroactively requires every contributor's consent.
- **HF Hub does not enforce these licenses.** Skill weights on the
  Hub are gated by *their* upstream licenses; OpenRAL cannot add
  restrictions to third-party weights. The license-lineage code in
  `python/rskill/` already handles this; the docs will state it
  explicitly.
- **The 100-person threshold is imprecise.** If a 100-employee company
  has a 5-person robotics team using OpenRAL for research, they
  are technically over the limit. PolyForm-SBL accepts this
  imprecision in exchange for being a *standard, well-drafted*
  license — a bespoke threshold (e.g. 30 employees) would cost
  $5–15K of legal time and lose the network effect of being a
  recognised license text. Accepted.
- **Sales motion is required.** Open-core revenue depends on a
  sales/support function. This is a strategy decision beyond the
  codebase.

## Alternatives considered

- **BSL 1.1** (HashiCorp, MariaDB, CockroachDB) — rejected. BSL
  *requires* a Change Date (max 4 years) at which the code converts
  to an OSS license. Cannot satisfy the "permanent restriction"
  requirement.
- **FSL (Fair Source License, Sentry)** — rejected. Time-bombed
  (2-year non-compete then converts to MIT/Apache). Same problem as
  BSL.
- **Sourcegraph's original Fair Source License** — rejected. Lets the
  org set a user-count threshold (e.g. 30), but the license has been
  retired by its author, has known drafting gaps, and is no longer
  maintained.
- **Elastic License v2 (ELv2)** — rejected. Restricts
  hosting-as-a-service but has *no* organisation-size carve-out,
  which is the explicit feature we want.
- **SSPL** (MongoDB) — rejected. Hostile to cloud providers via
  copyleft; doesn't gate by company size; rejected by Debian/Fedora;
  reputational baggage in research.
- **Commons Clause + Apache-2.0** — rejected. Prohibits "selling" but
  has no size threshold; widely criticised as ambiguous.
- **AGPL-3.0 + commercial dual** — rejected for the *Tier 2* surface
  but considered as fallback. AGPL is the only way to keep the
  restricted tier *OSI-approved* (large companies typically buy
  commercial to escape network copyleft). The user explicitly stated
  Tier 2 does not need OSI status; PolyForm is cleaner for size
  gating.
- **PolyForm Noncommercial 1.0.0** — rejected. Blocks *all*
  commercial use, including the <100-employee companies we want as
  free users.
- **Custom Fair-Source-style with a 30-employee threshold** — rejected.
  CLAUDE.md §1.13 / §14 favours standard, well-known text over
  bespoke drafting. The 100-employee PolyForm threshold is close
  enough to intent.
- **Restricting `python/observability/` instead of (or in addition to)
  the reasoner / WAM / dispatcher tier** — rejected. Industry
  pattern across every major commercial observability project
  (Datadog, New Relic, Honeycomb, Grafana, Sentry, Elastic,
  OpenTelemetry) is permissive SDK + restricted backend. The trace
  primitives in `python/observability/` today are the SDK
  equivalent; the *product* (fleet aggregation, replay, flywheel) is
  the backend equivalent and lives in the existing BSL-1.1
  `openral/cloud` repo. Bringing observability primitives under
  PolyForm would taint every Tier-1 package that imports them and
  buys us nothing defensible. (A future ADR can introduce an
  `observability_pro/` on-prem product if there is a customer
  demanding on-prem rather than SaaS for the flywheel.)
- **Status quo (pure Apache-2.0)** — rejected by the strategy
  brief. Acceptable engineering posture; incompatible with the
  business goal of charging large organisations for the orchestration
  surface.

## Verification

The PR that introduces this ADR is verified by:

1. **`just docs-build`** (mkdocs `--strict`) succeeds with the new
   ADR rendered.
2. **`just lint`** passes on a clean tree (no code touched).
3. **`just test`** passes (no code touched).
4. The three files in `LICENSES/` are present and match canonical
   sources: `Apache-2.0.txt` is byte-identical to the root `LICENSE`;
   `PolyForm-Small-Business-1.0.0.txt` is mirrored from the upstream
   PolyForm Project repository (`polyformproject/polyform-licenses`);
   `Academic-Research-Permission.txt` matches the text quoted in
   *Decision*.
5. `NOTICE` and `README.md` describe the two-tier model coherently.
6. `CLAUDE.md §1.9` is amended additively per §7.9.

Follow-on verification — landing alongside the *first* Tier-2 package
(separate PR):

- CLA-assistant bot configured on the repository.
- `tools/license_check.py` lands as part of that PR and is wired into
  `just lint`.
- `LicensePosture` enum (in the existing skill license-lineage code
  per `docs/METHODS.md`) extended with `POLYFORM_SMALL_BUSINESS` and
  `APACHE_2_0_TIER1`; schema regenerated via `just schema-export`.
- Per-package `LICENSE` file added in the new Tier-2 package
  directory; `pyproject.toml` declares
  `license = { text = "LicenseRef-PolyForm-Small-Business-1.0.0" }`
  (PolyForm is not on the SPDX license list).
- `docs/architecture/repo-state-map.html` updated per §7.10 to
  surface tier per block.

## Amendments

### 2026-05-17 — Observability primitives explicitly include metrics + propagation

The original Decision lists "observability primitives (e.g. span
builders, OTel helpers)" under the Tier-1 Apache-2.0 scope. The OTel
rollout in `claude/add-otel-robot-tracing-Hxf2R` (see the
2026-05-17 amendment on ADR-0010) lands four additional modules in
`openral_observability/` — `semconv`, `metrics`, `propagation`, and
`cli` — all under Apache-2.0, the same as the package they extend.

This amendment is purely a clarification: **all** OpenRAL OTel
plumbing — trace spans, OTel metric instruments, the W3C
TraceContext propagator that lets the C++ safety kernel parent its
`safety.check` span to the Python `rskill.tick` — is Tier-1
Apache-2.0. Tier-2 (`reasoner`, `wam`, `dispatcher`, `skill_catalog`,
`fleet`) may *consume* these primitives but cannot move them behind
the source-available license. The C++ safety kernel
(`cpp/safety_kernel/`, planned) consumes the same propagator on the
ROS-side and ships under the same Tier-1 license; the kernel's
`opentelemetry-cpp` integration is open-core too.

The only Tier-2 boundary observability touches is the
`openral.observability.export_failures` counter labeled
`signal_kind="dispatch"` — `openral_dispatcher` (Tier-2) increments
that counter when its cloud-side path drops a batch. The counter
instrument itself lives in Tier-1; only the dispatch-side increment
is gated by the cloud license.

No code changes required on this ADR — purely confirms the
licensing posture of the OTel surface that lands on the
observability branch.
