# One convex collision primitive belonging to an AttachedCollisionObject.
#
# A single attached object keeps one perception/world identity, one attach
# link, and one touch-link set, but may carry several of these primitives.
# Each primitive owns its own shape and a pose expressed in the owning object's
# frame; the kernel places it in the robot base frame by composing
#   FK(attach_link) . object.pose_in_link . primitive.pose_in_object
# and re-checks it as collision-active robot geometry.

uint8 SHAPE_SPHERE  = 1
uint8 SHAPE_CAPSULE = 2
uint8 SHAPE_BOX     = 3

uint8 shape_type
# Shape dimensions:
#   sphere  -> [radius]
#   capsule -> [radius, central_segment_length]
#   box     -> [half_extent_x, half_extent_y, half_extent_z]
float64[] shape_dimensions

# Primitive pose in the owning object's frame (composes on top of the object's
# pose_in_link).
geometry_msgs/Pose pose_in_object
