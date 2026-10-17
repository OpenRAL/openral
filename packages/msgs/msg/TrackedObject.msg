# One object the Layer-2 tracker grounds, tracks, binds and releases (issue #418);
# wire form of core.TrackedObject. Python proposes the geometry; the safety kernel
# re-verifies cover on its own grid and decides every exemption.

# "trk:<n>", unique for the session, never reused.
string object_id
# +1 at every re-grounding after release; the kernel retires (object_id, epoch).
uint32 epoch
string label
# core.ObjectRole value.
string role
# "<GraspDeclaration.target_id>@<stamp_ns>" that grounded it.
string declaration_key
# core.TrackStatus value: tentative | confirmed | occluded | split | bound |
# released | lost.
string status

# Gravity-aligned box in the grid frame containing the fitted solid (+ half a
# cell). Grasp-region caps: half-extent <= 0.20 m, volume <= 0.03 m^3;
# `box.geometry[]` MUST be empty.
PlaceRegion box
# Convex hull(s) of the same containing set, posed in box.frame_id; each at most
# 64 hull vertices.
AttachedCollisionPrimitive[<=4] tight
# core.FitKind value: silhouette_cone | multi_view_cone | pinch_fallback.
string fit_kind

# Measured support top under the object (grid frame, m) and what it is:
# "static:<collider>" | "trk:<n>" | "map".
float64 support_z
string support_ref
# Highest support top ever seen under it + 1 cell: the identity-clearing floor.
float64 floor_z

# Size-aware cover: cells the surface points fell in at the last fit, and those
# occupied at the last confirmation.
uint32 visible_cells
uint32 occupied_cells

# Source-clock stamps: depth frame of the last fit, grid source_stamp of the
# last confirmation.
int64 last_fit_stamp_ns
int64 last_confirmed_stamp_ns

# While BOUND: gripper joint, attach link and attach-time pose in that link.
# Empty / pose_in_link_valid false otherwise.
string bound_hand
string attach_link
bool pose_in_link_valid
geometry_msgs/Pose pose_in_link

# True when re-association after release/loss was by proximity, not continuity.
bool identity_inferred

# True when the last fit was taken with the robot hiding more of the object than the
# occluder band re-adds: the geometry may serve exemptions and identity clearing only,
# never a payload (issue #418, PROVISIONAL).
bool provisional
