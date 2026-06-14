# ADR-0003: Pydantic v2 over `@dataclass` for all schemas and contracts

- Status: Accepted
- Date: 2026-05-24 (retroactive — documents a Week-1 decision already in code)
- Amended: 2026-05-24 (see Amendments below)

## Context

OpenRAL has typed contracts at every layer boundary: `RobotDescription`
manifests (`robots/<id>/robot.yaml`), `RSkillManifest` (`rskills/<id>/rskill.yaml`),
`SensorSpec` / `SensorBundle`, `WorldState`, `Action`, `SimEnvironment`,
`BenchmarkSpec`, `RSkillEvalResult`, and the `ROSError` exception
hierarchy. CLAUDE.md §1.3 declares these "normative" — Python code
outside the contracts is implementation, code inside is API.

Python ships **three** plausible contract substrates:

| Substrate | Validation | JSON Schema export | Round-trip | YAML/TOML I/O | Discriminated unions | Stable v2 |
|---|---|---|---|---|---|---|
| `@dataclass` (stdlib) | None — fields are accepted as-is | None | Manual `asdict` + `from_dict` | Manual | None | Yes |
| `attrs` | Optional via validators | None native | Manual | Manual | None | Yes |
| **Pydantic v2** | Eager + composable | First-class (`model_json_schema()`) | First-class (`model_validate`, `model_dump`) | First-class (via `ruamel`/`pyyaml` glue) | First-class (`Field(..., discriminator=...)`) | Yes (since 2023) |

The schemas live at network boundaries (HF Hub manifests, ROS 2
message conversion, sim YAML configs, OTLP attribute payloads, CLI JSON
output) and at trust boundaries (sigstore-verified rSkill packages,
license-gated weights). Validation must be **eager**, not deferred — a
mistyped `embodiment_tags` entry should fail at load, not at runtime
when a Skill activates and the loader can no longer surface a sensible
error message.

## Decision

**All schemas, configs, manifests, and external-interface types use
Pydantic v2 `BaseModel`.** Plain `@dataclass` is allowed **only** inside
a single module that does not cross a layer boundary.

Concrete rules:

1. The `openral_core.schemas` module is the home for every normative
   model. New fields land there; downstream packages import.
2. Every public model has hypothesis fuzz tests that exercise
   **generation → serialization round-trip → JSON Schema validation**
   (CLAUDE.md §5.4).
3. `tools/schema_export.py` regenerates `docs/reference/schemas/*.json`;
   CI fails on drift (`just schema-export` is idempotent).
4. SemVer applies to `openral-core`. While the schemas are pre-publish
   (`schema_version: "0.1"`), the surface evolves in place without
   migrators — a real bump is reserved for the first post-1.0 shape
   change. See CLAUDE.md §1.6.
5. Discriminated unions are preferred to `isinstance` checks when a
   field can carry multiple shapes (e.g., `PhysicsBackend`,
   `RuntimeKind`).
6. `Field(..., description="…")` is mandatory on every public field —
   the descriptions are surfaced in JSON Schema and in `ral` CLI help
   text.

## Consequences

- **Pros**
  - JSON Schema export is free and used by the docs site and the
    `rSkill.from_yaml` loader's error messages.
  - Validation is eager — a malformed `rskill.yaml` fails at install,
    not at first inference.
  - One contract substrate across the workspace makes the rules
    learnable in one sitting; new contributors do not have to memorise
    "schemas use Pydantic, configs use dataclasses, errors use
    `TypedDict`".
  - Discriminated unions give us the same shape `ROS 2 IDL` enforces in
    the C++ world, so the Pydantic ↔ IDL bridge in `packages/msgs/`
    stays mechanical.

- **Cons**
  - One more workspace dependency (`pydantic>=2.5`).
  - Slightly higher import cost than `@dataclass`; mitigated by lazy
    imports at module boundaries.
  - Pydantic v2's `model_config` syntax differs from v1; the workspace
    is v2-only by lock-file pin, so v1 examples online require
    translation.

## Alternatives considered

- **`@dataclass` everywhere.** Rejected — no validation, no JSON Schema
  export, no round-trip helpers. We'd reinvent half of Pydantic by
  hand at every layer boundary.
- **`attrs` with `cattrs` for serialisation.** Rejected — equivalent
  feature set on paper, smaller ecosystem in 2026, no first-class JSON
  Schema export. The `cattrs` round-trip story is also less mature for
  discriminated unions.
- **`msgspec`.** Compelling for pure speed (zero-cost
  deserialisation), but the validation surface is smaller and the JSON
  Schema export is via a separate library. The serialiser-vs-validator
  split would force us to maintain two pictures of every contract.
- **Mix `@dataclass` for internal modules and Pydantic for public
  contracts.** The rule above effectively *is* this — but the dividing
  line is "crosses a layer boundary", not "lives in a specific
  package". The default is Pydantic; `@dataclass` requires
  justification.

## Why this ADR is retroactive

The decision was made in Week 1 of the kickoff and has been in the
code since the first `openral_core` commit; CLAUDE.md §1.3 and §5.1
already encode it normatively. This ADR records the reasoning so a
future contributor proposing `msgspec` or `attrs` has a paper trail to
push against (CLAUDE.md §7.9).

## References

- CLAUDE.md §1.3, §5.1, §5.4
- `python/core/src/openral_core/schemas.py` — the 41-model canonical
  schema module.
- `docs/reference/schemas/` — generated JSON Schema artifacts.
- `tools/schema_export.py` — drift-checking exporter.
- ADR-0002 (eval/sim environments) — first use of discriminated unions
  (`PhysicsBackend`).
- ADR-0013 (rSkill manifest actuators + processors) — the most recent
  schema evolution under this rule.
