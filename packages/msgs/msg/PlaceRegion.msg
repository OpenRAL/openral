# The bounded region of a declared place target (ADR-0097's 2026-08-14
# amendment) — the volume inside which the declared payload's world-collision
# margin is reduced by min(one voxel, 25 mm) so it can physically reach the
# support contact the place witness is earned by.
#
# Why a region is needed at all: a payload inserted into an ENCLOSED target
# passes within margin distance of voxel-quantised walls, and at 25 mm cells that
# quantisation inflates thin cabinet geometry enough to close the opening's
# predicted clearance before the payload arrives. Round-6 measured that as an
# E-stop 22-30 mm short of the shelf. Open-top targets never hit it. Without the
# region there is no way to distinguish "approaching the declared receptacle"
# from "approaching anything else", so there would be no way to bound the
# relaxation — which is exactly what makes it acceptable.
#
# PRODUCER-SUPPLIED, and producer-specific: sim computes it from the declared
# body's MuJoCo model subtree; real hardware obtains it from the perception
# stack (a detector/segmenter rSkill) through a seam that is NOT YET
# IMPLEMENTED, so no allowance is applied on real hardware today. The safety
# kernel is producer-agnostic: it consumes this box the same way regardless of
# which producer measured it, and it never derives one itself.
#
# The region is an ORIENTED box, not an axis-aligned one. A receptacle has an
# orientation, and the axis-aligned hull of a rotated one is strictly larger,
# i.e. strictly more permissive.
#
# Absent (PlaceDeclaration.region_valid == false) means NO allowance — the
# pre-amendment margin, unchanged. Degenerate and over-large regions are refused
# by every consumer for the same reason.

# The frame `pose` is expressed in. This MUST be the robot base frame — the same
# frame OccupancyVoxels is published in — and the safety kernel refuses a region
# whose frame_id does not match the occupancy grid's, because a region measured
# in one frame and applied in another is a relaxation aimed at the wrong volume.
string frame_id

# Region box centre pose in `frame_id`.
geometry_msgs/Pose pose

# Region box half-extents along the box's own axes (m). Every component must be
# > 0; a component above 1.5 m, or a box above 8 m^3, is refused — a declaration
# names ONE receptacle, not a room.
geometry_msgs/Vector3 half_extents

# Attributability (HZ-0097-2/4): what the producer measured this box from, e.g.
# "mujoco_body_subtree:cab_1_left_group_main". Free text, never parsed.
string evidence_ref

# Producer timestamp (ns). The region is refreshed at the attachment publication
# rate, so a consumer can tell a live measurement from a stale one.
int64 stamp_ns
