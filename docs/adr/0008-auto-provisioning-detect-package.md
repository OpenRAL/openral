# ADR-0008: Auto-provisioning — `python/detect/` as a new top-level package

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

Today, bringing up a fresh robot host is hand-rolled: `ral init`
produces a 3-field stub `robot.yaml` and the operator hand-authors the
full `RobotDescription` (joints, sensors, intrinsics, capabilities,
safety envelope, compute) before any skill will load.  The schema-level
match function `rSkill.check_compatibility` already exists in
`python/rskill/src/openral_rskill/loader.py:454-480` but is never
invoked during provisioning.

We want a single `openral detect` command that probes the host (USB / DDS /
GPU / V4L2 / RealSense / network), assembles a complete
`RobotDescription` with **real** sensor attributes pulled from the
sensor catalog and **real** GPU caps so the matcher can evaluate skill
runtime / quantization compatibility, and a sibling `ral skill check`
that prints the resulting compatibility table.

Adding a new top-level Python workspace package is flagged by
CLAUDE.md §12 ("Add a new top-level package — STOP. Propose it in a
discussion or ADR first.").  This ADR is that proposal.

## Decision

Introduce `python/detect/` (`openral-detect`) as a new
top-level workspace member that owns the auto-provisioning machinery:

- Probes (`probes/{usb,dds,gpu,cameras,realsense,network}.py`) — pure
  functions, never raise, return empty + typed warning on missing
  hardware or missing optional deps.
- Schema (`report.py`) — Pydantic `DetectionReport` with sub-models
  rich enough for the assembler to populate every new
  `RobotCapabilities` GPU field (commit 2).
- Assembler (`assemble.py`) — identify-then-enrich: load the canonical
  `robots/<name>/robot.yaml` for known rigs, splice catalog-built
  sensor specs, promote GPU caps onto `RobotCapabilities`.
- Compatibility report (`compatibility.py`) — wraps
  `rSkill.check_compatibility` and the `rSkill.list_installed`
  registry; supports `--skills-dir` for in-tree CI walks.
- Two new CLI commands: `openral detect` and `ral skill check`.  `ral init`
  is **removed entirely** in the same release (no backward-compat).

`openral-detect` depends on `openral-core`,
`openral-sensors`, `openral-rskill`, `openral-cli`, and
`openral-observability`.  No runtime layer (HAL, World State,
Skill, Reasoner, Safety) depends on it.  It is bootstrap-only and
does **not** appear in the runtime data-flow strip.

## Alternatives considered

1. **Fold into `openral-cli`.**  Rejected: `cli/` startup must
   not import `pyrealsense2` / `pynvml` / `jetson-stats`; the optional
   probes belong behind a real package boundary so the CLI's
   import-time cost stays small.
2. **Fold into `openral-sensors`.**  Rejected: that package is a
   *static catalog* by charter (no live device probing).  Mixing live
   discovery into it would erase the boundary that lets `openral sensor
   list` work offline.
3. **Skip the package; emit only a JSON detection report and let
   operators hand-edit the robot.yaml.**  Rejected: it does not solve
   the "complete RobotDescription" requirement and forces every
   operator to learn the schema.

## Consequences

- Adds a workspace member; `pyproject.toml` registers it under
  `[tool.uv.sources]`.
- `RobotCapabilities` gains six GPU/runtime/dtype fields (commit 2).
  Lands on the pre-publish baseline (`schema_version: "0.1"`); every
  committed `robots/<name>/robot.yaml` and HAL static description gets
  back-filled with safe defaults in the same commit, and Pydantic
  defaults make the new fields opt-in for hand-edited custom
  manifests.
- `rSkill.check_capabilities` gains runtime + quantization matchers
  (commit 3).  Empty robot lists keep the matcher in the
  "unknown — skip" branch so legacy manifests behave unchanged.
- `SensorCatalogEntry` gains an opt-in `signatures: tuple[
  SensorSignature, ...] = ()` field (commit 5) so probes can
  reverse-look up devices to the canonical catalog id.
- Optional extras `[nvidia]`, `[jetson]`, `[realsense]` keep the
  default install lean.

No layer-boundary crossing in the runtime sense; CLAUDE.md §6.1's eight
layers stay as-is.  `openral_detect` reads `openral_hal`'s
public surface (the canonical `robots/<name>/robot.yaml` files and the
HAL adapters' static descriptions) without any reverse dependency.
