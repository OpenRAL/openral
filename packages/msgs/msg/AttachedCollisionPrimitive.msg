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

# Optional stage-2 tight geometry refining a SHAPE_BOX primitive, in that box's
# own local frame — the payload-side twin of RobotDescription's
# LinkCollisionGeometry.tight_geometry. Empty means "no refinement": the box IS
# the model, which is every pre-#266 publisher and every primitive whose sim
# geom (sphere / box / capsule) already lowered exactly.
#
# It exists because a MESH geom does not: its only lowering is the local AABB,
# whose corners stand a median 50.78 mm proud of the mesh — 2.3-3.9x the entire
# world-voxel quantisation term, and the dominant term behind the 97 % of
# 15 mm-arm safety stops that were payload-vs-world (issue #266).
#
# The kernel uses these ONLY to refine the payload-vs-world-voxel distance, and
# only after proving (at ingest) that they stay inside the shape_dimensions box
# above: both stages are subsets of that box, so the broad-phase window never
# moves and the refinement can only ever REMOVE a stop, never add one. A
# primitive whose refinement fails that proof is checked as the plain box.
#
# 13 lower and 13 upper 26-DOP slab bounds along openral_core.DOP_AXES /
# kDopAxis, in the box's local frame. Both empty, or both 13 long.
float64[] tight_dop_lo
float64[] tight_dop_hi
# Exact convex-hull vertices of the source mesh, flattened xyz, in the same
# frame. Empty is stage 1 (the DOP) only — the representation a mesh whose hull
# exceeds kMaxTightHullVertices gets, and a cost decision rather than a safety
# one, since the DOP is a containing solid either way.
float64[] tight_hull_vertices
