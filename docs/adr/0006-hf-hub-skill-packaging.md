# ADR-0006: Hugging Face Hub as the rSkill packaging substrate

- Status: Accepted
- Date: 2026-05-24 (retroactive — documents the Week-3 packaging decision already in code)
- Amended: 2026-05-24 (see Amendments below)

## Context

OpenRAL's atomic unit of robot behaviour is the **rSkill** (CLAUDE.md
§6.4): one signed, capability-tagged, license-tracked package
containing:

- `rskill.yaml` — manifest (Pydantic `RSkillManifest`).
- weights (`model.safetensors` or sharded).
- optional `engine.plan` — per-arch TRT engine.
- optional `Dockerfile` — cloud-side runtime.
- `README.md` — runnable example + model card.
- `eval/<benchmark>.json` — per-benchmark `RSkillEvalResult` (CLAUDE.md
  §6.4 + ADR-0009 PR D).

rSkills are downloaded at install time, verified at load time, and
license-gated at install + deploy time. They need a substrate that:

1. Has **versioned blob storage** with content-addressing.
2. Has a **per-asset license surface** the loader can read at install.
3. Has **gated / private repos** so we can ship non-commercial weights
   behind a token without rebuilding distribution.
4. Is **idiomatic in the ML ecosystem** — every VLA-relevant team
   (LeRobot, Hugging Face, NVIDIA, Physical Intelligence, Google
   DeepMind via Gemini Robotics, Skild AI) already publishes to HF.
5. Has **a stable Python client** with token + revision pinning.
6. Supports **sigstore / cryptographic signing** of artifacts.

The candidates:

| Substrate | Versioned blobs | License surface | Private/gated | Python client | Signing | ML-idiomatic |
|---|---|---|---|---|---|---|
| **Hugging Face Hub** | Yes (git-lfs + revisions) | Yes (`README.md` license tag + `LICENSE` file) | Yes (private, gated) | `huggingface_hub` | sigstore via opt-in | **Yes** |
| PyPI | Wheels are not really blobs | Trove classifiers only | No private PyPI without infra | `pip` | sigstore (PEP 740) | No — wheels aren't models |
| Docker Hub / GHCR | Yes (digest-addressed) | OCI annotations | Yes | Various | Cosign | Partial (engines yes, weights no) |
| S3 + signed URLs | Yes | Up to us | Up to us | Up to us | Up to us | No — we'd be reinventing HF Hub |
| Self-hosted Artifactory | Yes | Configurable | Yes | Up to us | Configurable | No |
| Git-LFS on GitHub | Yes (rate-limited) | LICENSE file | Yes | `git` | sigstore | No — LFS quotas hurt |

The roadmap (`docs/roadmap/index.md`) already calls out
`tools/rskill_publisher.py` and the first published rSkill at
`OpenRAL/rskill-smolvla-libero` as shipped — the decision is in
code. This ADR records why.

## Decision

**Every rSkill is one Hugging Face Hub repo** (`huggingface.co/openral/rskill-<name>`).
The loader is `rSkill.from_pretrained("openral/rskill-<name>")`.

Concrete rules:

1. **One HF Hub repo per rSkill.** Repo name is `rskill-<slug>`. The
   `RSkillManifest` is the repo's `rskill.yaml`. The model card
   (`README.md`) is the repo's landing page.
2. **`huggingface_hub` is the only client.**
   `python/rskill/src/openral_rskill/` (workspace member
   `openral-rskill`) uses `HfApi` / `snapshot_download` / per-file
   `hf_hub_download` (ADR-0013 manifests can declare per-file URIs
   instead of full snapshot).
3. **License surface is enforced at install.** The loader reads the
   manifest's `license` field, cross-checks against the `LICENSE` file
   in the repo, and surfaces the posture to the user. Non-commercial
   licenses (GR00T) refuse a commercial deployment without an
   explicit env var (CLAUDE.md §1.9 / ADR-0012 — referenced, separate
   ADR for the licensing tiers).
4. **Revision pinning is contractual.** Every loaded skill records a
   revision SHA in the trace. CLAUDE.md §8 "reproducibility over
   speed" — a trace must replay from the SHA alone.
5. **Sigstore verification is the path forward — NOT yet implemented.**
   As of this writing `openral-rskill` performs **no** signature
   verification: there is no verify step in `rSkill.from_pretrained`
   and **no `signature` field on `RSkillManifest`**. Weights are trusted
   on HF Hub transport security alone. A future ADR adds both the
   manifest `signature` field and the sign-on-publish step inside
   `tools/rskill_publisher.py`, plus the verify-before-activate step in
   the loader. Until then the loader emits an `rskill.unverified_provenance`
   warning and supports `OPENRAL_REQUIRE_SIGNED_SKILLS=1` to fail closed;
   `*.pt` weights additionally require `OPENRAL_ALLOW_UNSAFE_PICKLE=1`.
   Do not describe skills as "signed/verified" until this lands (CLAUDE.md §1.2).
6. **Private / gated repos for license-restricted weights.** The first
   published rSkill (`OpenRAL/rskill-smolvla-libero`) ships
   `private=True`-gated; commercial-restricted weights ship under a
   gated org repo.
7. **Datasets follow the same pattern** under
   `huggingface.co/openral/dataset-<name>` (LeRobotDataset v3).
   Skill ↔ dataset linkage is a manifest field, not a separate
   registry.
8. **HF Hub is not the runtime substrate.** The Hub provides
   distribution + revisions + license metadata; runtime concerns
   (engine cache, quantization registry, lifecycle node) live in
   `openral-rskill` and `openral-runner`. The Hub never sees a
   `WorldState`.

## Consequences

- **Pros**
  - Zero distribution infrastructure to operate — HF Hub handles
    storage, CDN, ACLs, gated access, revisions.
  - The ML-idiomatic path. Every model author working with LeRobot /
    Pi / NVIDIA already publishes to HF; an OpenRAL rSkill looks like
    a normal HF model repo plus an `rskill.yaml` and an `eval/`
    directory.
  - License posture is a **per-asset** property already surfaced by HF
    metadata; the loader extends it, doesn't reinvent it.
  - The same `huggingface_hub` library handles datasets, weights, and
    engines, so the skill ↔ dataset linkage in the manifest needs no
    extra client.
  - Per-file download (ADR-0013 `manifest.processors`) reduces the
    blast radius of a malformed weight — we never have to fetch a
    full snapshot to discover that the processor JSON is malformed.

- **Cons**
  - We are a **tenant** of HF Hub for the open-core distribution
    path. If HF Hub changes its policy, our distribution story
    follows. Mitigated by: (a) the manifest is a flat YAML the user
    can mirror to S3 or GHCR if needed; (b) `tools/rskill_publisher.py`
    is intentionally substrate-agnostic at the schema level.
  - Sigstore signing is not yet end-to-end. The verify-only stub
    documents the gap; the next ADR closes it.
  - Engine artifacts (TRT plans) are per-arch and large; HF Hub
    handles them fine, but cache discipline lives in
    `~/.cache/openral/engines/` (per-host), not in HF's LFS.
  - The `openral/` HF org needs governance — token rotation, gated-repo
    membership, abuse handling. Listed in the roadmap "Org / publishing"
    bucket.

## Alternatives considered

- **PyPI wheels containing the weights.** Rejected — wheels are
  resolution-time artifacts, not model-revision-time artifacts.
  Pinning a SHA per skill would conflate `pyproject.toml` resolution
  with model versioning, and PyPI's trove classifiers are too coarse
  to express the license posture we need (`code Apache-2.0; weights
  NVIDIA AI Foundation non-commercial`).
- **Docker images (GHCR) as the primary substrate.** Considered for
  cloud-runtime skills. Decided: images are an **optional**
  per-skill artifact (`Dockerfile` in the HF Hub repo), not the
  primary substrate. The cloud runtime story (ADR-0012 tier 2) layers
  on top of HF distribution, not under it.
- **Self-host on S3 + Cloudfront.** Possible, but reinvents what HF
  Hub gives free. We'd still want the manifest schema; we'd lose the
  ecosystem affinity.
- **Git-LFS on GitHub.** Quota economics break at the multi-GB
  weights scale, and GitHub has no first-class license-on-asset
  surface.
- **Defer packaging entirely; ship skills as in-tree code.** Rejected
  — every published rSkill is a third-party trust boundary
  (different team, different license, different revision cadence).
  Committing weights into the openral monorepo would conflate
  contract (this repo, Apache-2.0) with content (per-skill,
  per-license) and balloon the clone size.

## Why this ADR is retroactive

The decision is encoded in `python/rskill/src/openral_rskill/`
(`rSkill.from_pretrained`, license surface, sigstore verify-only stub),
in `tools/rskill_publisher.py`, in CLAUDE.md §6.4 / §1.9 / §7.4, and
in the live published example `OpenRAL/rskill-smolvla-libero`.
This ADR records the reasoning so future "should we move to GHCR /
PyPI / self-hosted" proposals have a paper trail to push against
(CLAUDE.md §7.9).

## References

- CLAUDE.md §6.4 (rSkill packaging), §1.9 (license lineage),
  §7.4 (working with VLAs).
- `python/rskill/src/openral_rskill/` — `rSkill.from_pretrained`, the
  manifest loader, sigstore verify-only stub.
- `tools/rskill_publisher.py` — packaging tool with privacy gate.
- `OpenRAL/rskill-smolvla-libero` — first published example
  (private-gated).
- ADR-0013 — `manifest.processors` per-file download (the rSkill
  contract evolution that builds on this substrate choice).
- ADR-0012 — open-core licensing tiers (cross-references this ADR for
  the open-tier distribution path).
