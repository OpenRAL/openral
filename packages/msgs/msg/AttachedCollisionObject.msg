# Collision object rigidly attached to one robot link.
#
# The same object must be absent from world occupancy while attached. Its
# geometry remains collision-active as robot payload; only touch_links may
# contact it.

string object_id
string attach_link
string[] touch_links

geometry_msgs/Pose pose_in_link
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
