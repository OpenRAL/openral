# Dispatch's typed statement that a place phase is active for a payload, and
# which target the payload is being placed into (ADR-0097).
#
# The pick-phase support-contact witness (ADR-0092 D6) exempts contact with the
# pick surface only; a NEW contact against any other surface mid-carry is never
# exempt, since nothing else distinguishes "arrived at declared destination"
# from "grazed a wall." This declaration is the ONLY source of that
# distinction — never inferred from motion or simulator introspection.
#
# A declaration alone exempts NOTHING: it only permits a producer to attest a
# second, place-phase SupportContactWitness on fresh bounded contact with the
# declared target, which the kernel then exempts under the identical bounds,
# fail-closed rules and hysteresis the pick witness is under. Measured contact
# without a declaration exempts nothing (unchanged pre-ADR-0097 behaviour).
#
# Scoped to one goal execution. MUST be retracted (active=false) on goal
# end/cancel and E-stop; every consumer MUST also expire it on timeout_s so a
# dispatcher dying mid-goal cannot leave an exemption armed.

# Identity of the declared place target (e.g. "sim:cab_1_left_group_main").
# A producer may attest a place witness only for support contact ON this
# target — the attested support_id names it, or names a surface that is part
# of it. Empty is invalid while active=true.
string target_id

# Payload the declaration is scoped to (an AttachedCollisionObject.object_id).
# Empty means "whichever payload is carried", which is the direct-dispatch
# case where the grasped object's identity is discovered at attach time.
string object_id

# Attributability (HZ-0097-2). A wrong declared target can arm the place
# witness at an unintended surface; these fields are what makes a wrong
# declaration reconstructible from the trace alone after the fact.
string rskill_id
string trace_id

# Backstop expiry window, seconds after stamp_ns. Consumers treat the
# declaration as dead past it, independent of whether a retraction arrived.
float64 timeout_s

# Dispatcher timestamp (ns). With active, this is the whole liveness key.
int64 stamp_ns

# False retracts the declaration. Retraction is explicit and additive to the
# timeout: whichever comes first wins.
bool active

# Producer-supplied bounded region of the declared target (ADR-0097's
# 2026-08-14 amendment). Additive and backward compatible: region_valid false —
# the default, and what dispatch itself always publishes, since dispatch cannot
# measure geometry — means NO approach allowance, i.e. exactly the margins the
# kernel used before the amendment. The evidence producer that resolves the
# target (in sim, the MuJoCo attachment tracker) fills it in and re-publishes the
# declaration on the attachment path the kernel already ingests. The allowance
# dies with the declaration: retraction, timeout, goal end/cancel and E-stop all
# take the region with them, because the region only ever travels inside it.
bool region_valid
PlaceRegion region

# Per-declaration bound on the contact force this place is permitted to reach
# (ADR-0100, survey Path C). Dead with the declaration, same as everything else
# here.
#
# <= 0 — THE DEFAULT — means NO force gate. Above 0, a live ContactForceWitness
# on the declared payload attesting a CALIBRATED magnitude past this value
# refuses the configuration; the gate only ever adds a refusal (ADR-0100 §1).
#
# NOT read from SafetyEnvelope.contact_force_threshold_n (robot-wide, 30.0 N
# default, predates the ISO reading): published per-region energy budgets span
# an order of magnitude (SARA-shield 0.49 J for a constrained-blunt hand vs
# PACS HANDOVER's 0.014 J), so the dispatching skill supplies it per place.
float64 contact_force_threshold_n
