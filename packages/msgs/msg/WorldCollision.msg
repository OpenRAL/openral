# ADR-0030 — bounded set of world obstacles for the safety kernel's
# world-collision check. Each obstacle is a capsule (a sphere is a capsule with
# half_length 0) expressed in `header.frame_id`, which MUST be the robot base
# frame the kernel computes forward kinematics in (no TF transform is applied in
# the kernel hot path). Over-capacity messages (more than the kernel's
# configured maximum) are rejected and the kernel fails closed.
#
# The arrays are parallel and per-obstacle: `radius[i]`, `half_length[i]`, the
# 6-tuple `origin_xyzrpy[6*i .. 6*i+6]` (xyz metres + roll/pitch/yaw radians),
# and optional `object_id[i]` label.

std_msgs/Header header

float64[] radius
float64[] half_length
float64[] origin_xyzrpy
string[] object_id
