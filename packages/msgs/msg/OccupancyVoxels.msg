# Dense, fixed-capacity 3-D occupancy voxel grid for the safety kernel's
# world-collision check. A perception bridge lowers a 3-D world map (e.g. an
# OctoMap) into a bounded local volume around the robot, publishes it in the
# robot base frame (`header.frame_id`) on /openral/world_voxels. The kernel
# copies it into a pre-sized buffer and rasterizes robot link capsules against
# occupied cells; a grid larger than the kernel's configured capacity fails
# closed.
#
# ORIENTED lattice: `origin` + `orientation` are the pose, in `header.frame_id`,
# of voxel (0,0,0)'s minimum corner; cell axes are `orientation`'s axes, NOT
# `header.frame_id`'s. Needed because the source map's lattice is fixed in the
# map/odom frame — re-expressing it on a base-fixed lattice costs a dilation
# (every cell the source voxel enters must be marked), measured at 29-35 mm
# median / 40 mm worst case on 25 mm cells, and held 48% of the live
# start-state E-stops (issue #173). Publishing on the source lattice with the
# rotation carried here removes that cover; consumers apply it, nobody
# re-rasterizes.
#
# `orientation` MUST be a unit quaternion — the all-zero unset default is not
# one, and every consumer refuses it rather than assume identity (a silently
# wrong orientation is a fail-OPEN misread of the world).
#
# `occupancy` is row-major with x fastest: idx = x + size_x*(y + size_y*z),
# length size_x*size_y*size_z. Occupied when non-zero (0 = free; the
# `nav_msgs/OccupancyGrid` 0..100 convention also works).

std_msgs/Header header

geometry_msgs/Point origin
geometry_msgs/Quaternion orientation
float64 resolution
uint32 size_x
uint32 size_y
uint32 size_z
uint8[] occupancy
