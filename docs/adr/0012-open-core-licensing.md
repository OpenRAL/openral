# ADR-0012: Licensing — uniform Apache-2.0, no commercial tier

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-06-16 (superseded the two-tier open-core model with uniform Apache-2.0; see History below)
- Related: CLAUDE.md §1.9 (license lineage), §3 (layer discipline),
  §4 (docs / ADR discipline), [ADR-0004](0004-monorepo-over-polyrepo.md)

## Context

Every package in this monorepo is **Apache-2.0**: the root `LICENSE`,
every `pyproject.toml`, and the `NOTICE`. CLAUDE.md §1.9 codifies this:
*"OpenRAL's own code is uniformly Apache-2.0 … copy-left is rejected
without TSC review."*

A previously-proposed version of this ADR introduced a **two-tier
open-core model** — Apache-2.0 for the core, plus a *source-available
commercial tier* (PolyForm Small Business License 1.0.0 + a bespoke
Academic Research Additional Permission) for planned agentic /
orchestration layers (`reasoner`, `wam`, `dispatcher`, `skill_catalog`,
`fleet`), with a hosted SaaS half under BSL-1.1 in a separate
`openral/cloud` repo. That model was **never enacted in code** — no
package ever shipped under PolyForm or BSL, and the two license texts
were the only artifacts it produced.

That two-tier model is now **withdrawn**. OpenRAL is, and will remain, a
fully open-source project under a single permissive license. The
agentic and orchestration layers it would have gated (`reasoner`,
`wam`, `dispatcher`, `skill_catalog`, `fleet`) are core to the value of
an open robot-agent harness and must be open for the ecosystem,
certifiers, academic forks, and downstream ROS users — the same
reasoning that already kept safety and observability open.

## Decision

**License every package in this repository under Apache-2.0. There is
no Tier 2, no source-available tier, and no BSL tier.**

- All Python packages under `python/` (including the planned
  `reasoner`, `wam`, `dispatcher`, `skill_catalog`, `fleet`), all ROS
  packages under `packages/`, the C++ under `cpp/`, and all `tools/`,
  `scripts/`, `examples/`, `tests/`, `docs/`, `robots/`, and
  `rskills/<name>/rskill.yaml` manifests are **Apache-2.0**.
- The only license texts in the tree are `LICENSE` (root) and
  `LICENSES/Apache-2.0.txt` (a verbatim copy). No `PolyForm-*` or
  `Academic-Research-Permission` texts exist.
- `pyproject.toml` in every member declares
  `license = { text = "Apache-2.0" }`.
- No CLA is required for the previous tier-relicensing reason. The DCO
  (see `CONTRIBUTING.md`) remains the contribution mechanism.

### Third-party model weights are a separate matter

Skill **weights** distributed via Hugging Face Hub keep their *own*
upstream licenses (SmolVLA Apache-2.0, π0 research-permissive, GR00T
NVIDIA non-commercial / NVIDIA Open Model, OpenVLA MIT, ACT MIT,
Diffusion Policy MIT, …). The license-lineage code in `python/rskill/`
surfaces each weight's posture at install time and the loader enforces
the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) per
[ADR-0046](0046-nvidia-gr00t-backend.md). **This is compliance for
models OpenRAL does not own — it does not affect the Apache-2.0 license
of OpenRAL's own code, and it is unchanged by this ADR.** OpenRAL cannot
add or remove restrictions on third-party weights; it only surfaces the
posture. Closed third-party vendor SDKs (e.g. API-only VLAs) are never
bundled — they stay behind the license guard and an env var.

## Consequences

- **No code under `python/`, `packages/`, or `cpp/` changes license.**
  Everything is and stays Apache-2.0; the planned orchestration packages
  inherit Apache-2.0 when they land.
- **OSI status holds for the whole repo.** Debian, Fedora, conda-forge,
  and the FSF can ship every package without legal review.
- **No CLA, no license-check gate, no `LicensePosture` *code* tier
  enum.** The earlier follow-on work to add `POLYFORM_SMALL_BUSINESS` /
  `APACHE_2_0_TIER1` enum values, a `tools/license_check.py`
  source-tree gate, and per-directory `LICENSE` files is **cancelled** —
  there is nothing to gate. (The `RSkillLicensePosture` enum for
  third-party *weights* is unrelated and stays.)
- **No per-subtree licensing discipline** is needed for reviewers
  (cf. ADR-0004): the whole tree is one license.
- **Revenue strategy is out of scope of the codebase.** Any future
  commercial offering (hosted service, support) does not require
  relicensing OpenRAL source; if one is ever proposed it lands as its
  own ADR and TSC decision, not as a silent license change here.

## History — the withdrawn two-tier model

The original (2026-05-24, *Proposed*) decision adopted a two-tier
open-core model: Apache-2.0 core + a PolyForm Small Business 1.0.0
"Commercial Source-Available" tier (with an Academic Research Additional
Permission) for `reasoner` / `wam` / `dispatcher` / `skill_catalog` /
`fleet`, and a BSL-1.1 hosted half in `openral/cloud`. The intent was to
let researchers and small organisations (<100 employees, <$1M revenue)
use everything free while charging larger enterprises for the
orchestration surface. The survey behind it (Datadog, Grafana, Sentry,
Elastic, Honeycomb, New Relic, OpenTelemetry, Jaeger, Prometheus) found
"permissive SDK + restricted product" to be the prevailing commercial
pattern, and considered and rejected BSL, FSL, Elastic License v2, SSPL,
Commons Clause, AGPL dual-licensing, and PolyForm Noncommercial.

That model never produced a single licensed line of code. It is
withdrawn in favour of a uniformly Apache-2.0, fully open-source
project; the two PolyForm/Academic license texts and the NOTICE/README
two-tier wording it introduced were removed in the same change as this
amendment. This section is retained so the history and the rejected
alternatives have a paper trail (CLAUDE.md §4).

## Verification

1. **`just docs-build`** (mkdocs `--strict`) renders this ADR.
2. **`just lint`** / **`just test`** pass (no code touched).
3. `LICENSES/` contains only `Apache-2.0.txt`; no `PolyForm-*` or
   `Academic-Research-Permission` files remain.
4. `NOTICE`, `README.md`, `GOVERNANCE.md`, and `CLAUDE.md §1.9` describe
   a single Apache-2.0 license with no commercial / source-available /
   BSL tier.
5. `grep -rniE 'polyform|source-available|two-tier|BSL tier'` over the
   tree returns no live OpenRAL-licensing claims (third-party weight /
   graph-DB dependency mentions are unrelated).
