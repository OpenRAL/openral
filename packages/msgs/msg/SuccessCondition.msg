# What success means for a policy run, as a predicate over world state (issue
# #635); wire form of core.SuccessCondition.
#
# At least `min_count` objects labelled `object_label` rest, released, with their
# box centres inside `region`, and have for `settle_s`. The rSkill runner judges
# it over the object tracker's TrackedObjectSet and reports a core.TaskOutcome on
# the ExecuteRskill result. Advisory: it ends a goal or labels its result,
# commands nothing, and is never a safety-kernel input.

# Label of the tracked objects (TrackedObject.label, compared case-folded).
string object_label
# Box an object's centre must rest inside, in the tracker's grid frame (the
# robot base frame). `region.geometry[]` MUST be empty.
PlaceRegion region
# How many objects must rest in the region (1..32). 0 — an unset field — reads 1.
uint32 min_count
# Seconds of tracker evidence the predicate must hold over (0..60]. 0 — an unset
# field — reads core's default (1.0), never "no settle".
float64 settle_s
# Longest gap between two sets the tracker publishes that the hold may span,
# seconds (0..60]; a longer one restarts settle_s. 0 — an unset field — reads
# core's default (2.0).
float64 max_evidence_gap_s
# End the run once the condition has settled. False — the default — only labels
# the result: the run still ends on its budget, its skill or a completion check.
bool end_on_success
