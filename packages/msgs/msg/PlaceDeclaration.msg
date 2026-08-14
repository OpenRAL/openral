# Dispatch's typed statement that a place phase is active for a payload, and
# which target the payload is being placed into (ADR-0097).
#
# The pick-phase support-contact witness (ADR-0092 D6) exempts the contact a
# payload has with the surface it was picked from. Its anti-scope is
# deliberate: a NEW contact against any other surface mid-carry is never
# exempt, because nothing distinguishes "the payload arrived at its declared
# destination" from "the payload grazed a wall." This declaration is that
# distinction, and it is the ONLY thing that supplies it — it is never
# inferred from motion, and never from simulator introspection.
#
# A declaration alone exempts NOTHING. It only permits a producer to attest a
# second, place-phase SupportContactWitness when the payload actually comes
# into fresh bounded contact with the declared target; the safety kernel then
# exempts exactly the patch that witness describes, under the identical bounds,
# fail-closed rules and hysteresis the pick witness is under. Measured contact
# without a declaration exempts nothing either — that is unchanged pre-ADR-0097
# behaviour.
#
# The declaration is scoped to one goal execution. It MUST be retracted
# (active=false) on goal end, goal cancel, and E-stop, and every consumer MUST
# additionally expire it on timeout_s so a dispatcher that dies mid-goal cannot
# leave an exemption armed for a later, unrelated skill execution.

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
