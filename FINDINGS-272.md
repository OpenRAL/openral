# #272 — the carried payload trips on occupancy it authored before the grasp

Investigation notes for `fix/272-voxel-backing-evidence`. Not a deliverable;
this file is deleted before the PR, and what survives goes into
`docs/reference/collision-validation-evidence.md`.

## The signature

Nine placing-phase E-stops from the 2026-09-09 ceiling battery, read through
their own `evidence_voxel_backing` records (on the
`sim.estop_ground_truth_evidence` line, **not** the snapshot line):

| run | kernel | certified gap | cell verdict | backing |
| --- | ---: | ---: | --- | --- |
| `fridge/r01` | −14.78 | +0.05 | **`attached_payload`** | gripper hand + finger2, `obj_main/obj_g3` |
| `fridge/r05` | −12.12 | +0.15 | **`attached_payload`** | gripper hand + finger2, `obj_main/obj_g3` |
| `fridge/r06` | −11.67 | −0.67 | **`attached_payload`** | gripper hand + finger2, `obj_main/obj_g3` |
| `fridge/r08` | −15.77 | +3.70 | **`attached_payload`** | gripper hand + finger2, `obj_main/obj_g3` |
| `sink_cup/r09` | −11.17 | −0.22 | **`attached_payload`** | `obj_main/obj_g7`, `obj_g9` — nothing else |
| `sink_cup/r04` | −5.91 | +44.96 | `solid_world` | `island_island_group_top_front_0` |
| `sink_cup/r06` | −0.40 | +9.14 | `solid_world` | `island_island_group_top_left_2` |
| `baguette/r05` | −8.08 | +48.53 | `solid_world` | `counter_1_left_group_top_left_1` |
| `baguette/r10` | −1.12 | −0.12 | `solid_world` | `coffee_machine_left_group_g3` |

Five cells carry **no `solid_world` class at all**.

## What is established

1. **The exclusion is live at stop time.** `obj_main` is classified
   `attached_payload`, which is only assigned from `attached_body_ids` —
   i.e. `read_attached_body_ids()` returned it. The gripper is classified
   `self_occupancy_suspect`, so it is in `_depth_self_bodies` (`robot_bodies=21`
   in the battery's own bridge log). **Neither is missing from the filter.**
2. **The stop is 5–6 s after the grasp**, in all five: `+5.90, +5.49, +6.05,
   +5.48, +4.99 s` from the single `automatic sim attachment revision` marker.
3. **All four `fridge` runs trip on the same cell index, 233816** — same scene
   reset, same location, four independent runs.
4. **The cell sits where the payload started.** Rebuilt at the battery's own
   `layout_id=47 style_id=32`: payload start `[4.0237, −0.6771, 0.9209]`,
   cell centre `[4.0875, −0.7375, 0.9375]` — **89 mm apart**, with the payload's
   own geom bounds reaching within **37 mm** of the cell centre (cell
   half-diagonal 21.65 mm).

## The mechanism this points at

The payload writes occupancy while it rests at its start pose, is added to the
depth exclusion set at the grasp, lifts — and trips on its own pre-grasp
footprint 5–6 s later, before that footprint is cleared.

**Not yet proven.** Points 1–4 are consistent with it and nothing contradicts
it, but authorship needs grid *history*, which no artifact in this battery
carries. A single snapshot cannot separate "the payload wrote this cell" from
"the cell was written by something else and the payload has since moved into
it" — `voxel_backing_record`'s own docstring makes exactly this point about
`self_occupancy_suspect`, and it applies here too.

## Step 1 — what would settle it

Instrument the cell's **first-seen stamp** in the published grid, so a stop can
say whether its cell predates the attach. Cheaper alternative: replay the depth
stream at the fridge start pose and watch cell 233816 appear and fail to clear.

## Why clearing may be failing (hypothesis, untested)

`synthesize_depth_frame` returns a clearing mask for pixels "whose only return
was a self-filtered body **and which found no farther surface**". At the fridge
shelf there is always a farther surface, so those pixels get a real depth at the
shelf rather than a clearing ray. Whether OctoMap's ray traversal then frees the
cells in front of the shelf is the thing to check — it should, which is why this
is a hypothesis and not a finding.

## Candidate fix, if authorship is confirmed

**Invalidate the payload's own footprint at attach time.** The bridge knows the
payload's pose and geometry at the instant of the grasp, so it knows exactly
which cells it could have authored. Clearing precisely those is bounded by a
known volume at a known time.

This is still a **fail-open** direction and wants an ADR before code — but it is
far narrower than ADR-0101's fixture suppression (rejected 2026-09-12): the
volume is the payload's own, not a modelled third party, and the trigger is a
single discrete event rather than a continuous stream.

## Corrections this investigation forced

- **#272 as originally filed was wrong on both claims** and has been retracted
  and re-scoped. `evidence_voxel_backing` was never missing — it is on the
  `estop_ground_truth_evidence` line, 26 of 27 stops.
- **The two wide-gap stops are ordinary #266 box overhang**, not phantom
  occupancy: excess 50.87 / 56.61 mm against #266's measured median of 50.8 mm,
  both cells backed by real collidable world geometry. #267 removes them.
- **The #268 replay's thresholds are not transferable across orientations.** The
  sweep moves the payload's position only, at its home orientation
  (`replay_poses.py:113`); box overhang is orientation-dependent. This was the
  root of the wrong inference, propagated to #259's close and ADR-0101's
  amendment, and corrected in both.
