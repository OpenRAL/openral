# Collision object rigidly attached to one robot link.
#
# Must be absent from world occupancy while attached: openral_octomap_bridge
# subscribes this array on /openral/world_state_fast and clears every occupancy
# cell the payload's own primitives explain out of the /openral/world_voxels
# grid, at the payload's live pose, for as long as it appears here. Geometry
# stays collision-active as robot payload; only touch_links may contact it.
# One identity/attach-link/touch-link set may carry several primitives
# (AttachedCollisionPrimitive[]), each posed in the object frame: pose_in_link
# places the object frame in the attach-link frame, then each primitive's
# pose_in_object composes on top.

string object_id
string attach_link
string[] touch_links

# Object pose in the attach-link frame (frame_id == attach_link). Each
# primitive is placed by pose_in_link . primitive.pose_in_object.
geometry_msgs/Pose pose_in_link

# Bounded list of convex collision primitives carried by this object. The
# kernel enforces a fixed total-primitive capacity across all attached objects.
AttachedCollisionPrimitive[] primitives

bool mass_valid
float64 mass_kg
bool center_of_mass_valid
geometry_msgs/Vector3 center_of_mass_m
bool inertia_valid
float64[9] inertia_kg_m2

float32 confidence
string evidence_kind
string evidence_ref
int64 stamp_ns

# Optional support-contact attestation (ADR-0092 D6). Attachment alone never
# licenses payload-vs-world contact; this does, and only within the witness's
# own bounds. False — the default for any producer that cannot measure support
# contact, including the vision attachment producer — means the safety kernel
# exempts nothing, which is the pre-witness behaviour.
bool support_contact_valid
SupportContactWitness support_contact

# Optional contact-force attestation (ADR-0100, survey Path C). Rides here for
# the same reason support_contact does: it is per-payload, optional, and the
# kernel already ingests this object. False — the default, and what every
# producer that cannot measure contact force publishes — means the force gate
# never arms and geometry decides alone, which is the pre-ADR-0100 behaviour.
#
# The gate this feeds only ever ADDS a refusal (ADR-0100 §1). A true value here
# can never license a contact the geometry refuses.
bool contact_force_valid
ContactForceWitness contact_force
