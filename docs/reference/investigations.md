# Investigations and evidence

Studies, measurements and design notes written while building the collision stack,
object tracking and real pick and place. They explain *why* the code looks the way it
does. They are records: none is maintained against master, and each page opens with
its status. For current behaviour read the [robots](robots.md), [rSkills](rskills.md)
and [safety package](https://github.com/OpenRAL/openral/tree/master/packages/openral_safety)
references.

## Safety kernel (current design records)

| Page | What it covers |
|---|---|
| [Staged hull narrow phase](collision-hull-narrow-phase.md) | The shipped 26-DOP → exact-hull narrow phase for arm-link vs world-voxel checks. |
| [Collision geometry review](collision-geometry-review.md) | The one-fitter capsule + hull lowering used for every robot, and the tests that pin it. |
| [External torque observer](external-torque-observer.md) | The per-joint external-torque collision tier in the kernel. |
| [Finger mimic geometry](finger-mimic-geometry.md) | The follower gripper finger as a rigid mimic-joint link. |

## Collision studies (historical)

| Page | Status |
|---|---|
| [Validation evidence ledger](collision-validation-evidence.md) | Historical ledger of validation rounds. |
| [Collision-primitive study](collision-primitive-study.md) | Superseded. |
| [Collision primitives, in pictures](collision-primitive-images.md) | Dated measurement. |
| [Tight link geometry](collision-tight-geometry.md) | Superseded design (implemented). |
| [Alternatives survey](collision-safety-alternatives-survey.md) | Historical survey. |
| [RoboCasa start-state census](robocasa-start-state-census.md) | Dated measurement. |
| [RoboCasa carry survey](robocasa-carry-survey.md) | Dated study. |
| [The live world map, measured](world-map-fidelity.md) | Dated measurement. |
| [G1 and R1 Pro base footprints](g1-r1pro-base-footprints.md) | Source of the manifests' footprints. |

## Manipulation and object tracking (historical)

| Page | Status |
|---|---|
| [Real pick and place: design](real-pick-place-design.md) | Historical design note. |
| [Real pick and place: ADR drafts](real-pick-place-adr-drafts.md) | Historical drafts. |
| [Manipulation-workspace contact](manipulation-workspace-contact.md) | Historical design note. |
| [Object primitives design](object-primitives-design.md) | Superseded design. |
| [Object tracking plan](object-tracking-implementation-plan.md) | Superseded plan. |
| [Object tracking evaluation, 2026-10-19](object-tracking-spark-evaluation-2026-10-19.md) | Dated evaluation. |
| [Object-centric mapping survey](object-centric-mapping-survey.md) | Dated survey. |
| [Isaac-reproducible datasets survey](isaac-reproducible-datasets-survey.md) | Dated survey. |
