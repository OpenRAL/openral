# Collision object rigidly attached to one robot link.
#
# The same object must be absent from world occupancy while attached. Its
# geometry remains collision-active as robot payload; only touch_links may
# contact it.

uint8 SHAPE_SPHERE  = 1
uint8 SHAPE_CAPSULE = 2
uint8 SHAPE_BOX     = 3

string object_id
string attach_link
string[] touch_links

# Shape dimensions:
#   sphere  -> [radius]
#   capsule -> [radius, central_segment_length]
#   box     -> [half_extent_x, half_extent_y, half_extent_z]
uint8 shape_type
float64[] shape_dimensions
geometry_msgs/Pose pose_in_link

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
