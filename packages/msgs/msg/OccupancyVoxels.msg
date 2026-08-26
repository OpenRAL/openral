# A dense, fixed-capacity 3-D occupancy voxel grid for
# the safety kernel's world-collision check. A perception bridge lowers a 3-D
# world map (e.g. an OctoMap) into a bounded local volume around the robot,
# publishes it in the robot base frame (`header.frame_id`) on
# /openral/world_voxels. The kernel copies it into a pre-sized buffer and
# rasterizes robot link capsules against the occupied cells; a grid larger than
# the kernel's configured capacity fails closed.
#
# The grid is an ORIENTED lattice: `origin` + `orientation` together are the
# pose, in `header.frame_id`, of voxel (0,0,0)'s minimum corner, and the cell
# axes are `orientation`'s axes — NOT `header.frame_id`'s.
#
# Why the grid carries an orientation at all. The source map's lattice is fixed
# in the map/odom frame; a grid whose lattice were fixed in `base_frame` would
# have an arbitrary relative phase and (on a mobile base) an arbitrary relative
# yaw, and re-expressing one lattice on the other costs a dilation: every cell
# the source voxel's volume enters must be marked, or occupancy is lost. That
# closure was measured at 29–35 mm median and 40 mm worst case on 25 mm cells,
# and it held 48% of the live start-state E-stops (issue #173). It is not an
# implementation defect — it is the minimum sound cover for a base-aligned
# lattice, so the only way to remove it is to stop re-expressing: publish on the
# source map's own lattice, one cell per source voxel, and carry the rotation
# here. Consumers apply it; nobody re-rasterizes.
#
# `orientation` MUST be a unit quaternion. The all-zero default of an unset
# field is not one, and every consumer refuses it rather than assuming identity:
# a grid silently read at the wrong orientation places obstacles somewhere the
# robot is not, which is a fail-OPEN misread of the world.
#
# `occupancy` is row-major with x fastest: idx = x + size_x*(y + size_y*z),
# length size_x*size_y*size_z, indexed along the grid's own axes. A cell is
# occupied when its value is non-zero (0 = free; the `nav_msgs/OccupancyGrid`
# 0..100 convention also works since the kernel only tests for non-zero).

std_msgs/Header header

geometry_msgs/Point origin
geometry_msgs/Quaternion orientation
float64 resolution
uint32 size_x
uint32 size_y
uint32 size_z
uint8[] occupancy
