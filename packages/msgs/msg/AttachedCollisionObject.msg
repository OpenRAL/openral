# Collision object rigidly attached to one robot link.
#
# The same object must be absent from world occupancy while attached. Its
# geometry remains collision-active as robot payload; only touch_links may
# contact it. One object keeps a single identity, attach link, and touch-link
# set, but may carry several collision primitives
# (AttachedCollisionPrimitive[]), each posed in the object frame. The object's
# pose_in_link places the object frame in the attach-link frame; every
# primitive's pose_in_object composes on top of it.

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
