# The bounded region of a declared place target (ADR-0097's 2026-08-14
# amendment) — the volume inside which the declared payload's world-collision
# margin is reduced by min(1.5 x voxel, 40 mm) so it can physically reach the
# support contact the place witness is earned by.
#
# Needed because a payload inserted into an ENCLOSED target passes within
# margin distance of voxel-quantised walls: at 25 mm cells that quantisation
# closes a thin cabinet opening's predicted clearance before the payload
# arrives (round-6 measured an E-stop 22-30 mm short of the shelf; open-top
# targets never hit it). Without the region there is no way to bound the
# relaxation to "approaching the declared receptacle."
#
# PRODUCER-SUPPLIED: sim computes it from the declared body's MuJoCo model
# subtree; real hardware's perception-stack seam is NOT YET IMPLEMENTED, so no
# allowance applies on real hardware today. The kernel is producer-agnostic and
# never derives one itself.
#
# ORIENTED box, not axis-aligned — the axis-aligned hull of a rotated
# receptacle is strictly larger, i.e. strictly more permissive.
#
# Absent (PlaceDeclaration.region_valid == false) means NO allowance, the
# pre-amendment margin unchanged. Degenerate/over-large regions are refused.

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

# The declared target's OWN collision geometry (ADR-0098, collision-safety
# survey Path B), each primitive posed in `frame_id` — same frame as `pose`,
# not relative to the region box. Empty = pre-ADR-0098 behaviour (backward
# compatible default).
#
# Box alone can't discriminate at mm scale against 25 mm voxels and grants the
# same blanket allowance whether or not the declared body is actually there;
# these primitives ARE the declared body, so a cell adjudicates against the
# modelled surface instead. Never more permissive than box-only: the kernel
# uses the modelled distance only within the blanket allowance the cell would
# get anyway, falling back to blanket otherwise.
#
# PRODUCER-SUPPLIED, same terms as the box: sim measures the MuJoCo model
# subtree; real hardware's perception-stack seam is NOT YET IMPLEMENTED. A
# producer that cannot measure geometry publishes an empty list, never a guess.
#
# The kernel refuses the WHOLE region (not a box-only downgrade) when the list
# overflows its primitive cap or carries a malformed shape (#142/#146 posture).
AttachedCollisionPrimitive[] geometry
