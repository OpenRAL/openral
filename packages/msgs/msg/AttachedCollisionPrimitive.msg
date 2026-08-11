# One payload collision primitive in its owning object's local frame.

uint8 SHAPE_SPHERE  = 1
uint8 SHAPE_CAPSULE = 2
uint8 SHAPE_BOX     = 3

# Shape dimensions:
#   sphere  -> [radius]
#   capsule -> [radius, central_segment_length]
#   box     -> [half_extent_x, half_extent_y, half_extent_z]
uint8 shape_type
float64[] shape_dimensions
geometry_msgs/Pose pose_in_object
