# ADR-0068 - Robot sensor catalog provenance via `SensorSpec.catalog_id`

- **Status:** Accepted 2026-06-22.
- **Date:** 2026-06-22
- **ADR number:** `0068`. The integer is not load-bearing; cross-refs use
  filenames.
- **Related:**
  - ADR-0065 - `SensorSpec.sim_placement` for generic sim camera rigging.
  - `docs/reference/sensors_landscape.md` - sensor catalog and factory roadmap.

## Context

Robot manifests currently inline full `SensorSpec` blocks. That is good for
replayability because each manifest captures the calibrated intrinsics, frames,
VLA feature keys, and sim placement used by a deployment or dataset bridge.

The sensor catalog in `openral_sensors` separately owns nominal factory specs for
known physical devices such as RealSense, Luxonis OAK-D Pro, Logitech C920,
Robotiq FT-300S, and generic USB UVC RGB cameras. Robot-mounted sensors should
record when they derive from one of those catalog entries, but replacing inline
manifest specs with lazy catalog references would hide per-robot calibration and
make traces less self-contained.

Scene cameras are different: many benchmark cameras are simulated viewpoints
(`agentview`, `top`, `corner`, `mujoco_top`) rather than deployable hardware.
They may remain inline without catalog provenance.

## Decision

1. Add optional `SensorSpec.catalog_id: str | None`.
2. Treat `catalog_id` as provenance only. The loader does not resolve or merge
   catalog entries at manifest load time.
3. Keep every deployed robot sensor spec fully materialized in `robot.yaml`.
   Calibration, placement, serials/MXIDs, feature keys, topics, and rates remain
   explicit on the `SensorSpec`.
4. Robot-mounted physical sensors should set `catalog_id` when the device is
   known and represented by `openral_sensors.CATALOG`.
5. Scene-only cameras and simulated benchmark viewpoints do not need catalog ids.

## Consequences

**Positive**

- Robot manifests retain replayable sensor contracts.
- Catalog provenance makes audits and hardware bring-up easier without inventing
  a schema-level lazy resolution system.
- Generic USB wrist cameras can avoid false C920 provenance by using
  `generic/usb_uvc_rgb`.

**Costs**

- Additive core schema field requires JSON Schema export, method docs, and fuzz
  coverage updates.
- The catalog does not guarantee that a manifest's calibrated values still match
  the nominal catalog factory. Validation can compare them later, but this ADR
  intentionally avoids implicit corrections.

## Alternatives considered

- **Use `catalog: <id>` instead of inline specs.** Rejected: it hides
  deployment-specific calibration and placement, and would require merge rules
  for every field.
- **Put catalog ids in `metadata`.** Rejected: provenance is common enough to be
  a typed first-class field, and `metadata` is harder to audit.
- **Require catalog ids for scene sensors.** Rejected: benchmark cameras often
  do not correspond to physical hardware.
