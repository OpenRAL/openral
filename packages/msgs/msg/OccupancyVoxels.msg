# ADR-0030 voxel phase — a dense, fixed-capacity 3-D occupancy voxel grid for
# the safety kernel's world-collision check. A perception bridge lowers a 3-D
# world map (e.g. an OctoMap) into a bounded local volume around the robot,
# transforms it into the robot base frame (`header.frame_id`), and publishes it
# here on /openral/world_voxels. The kernel copies it into a pre-sized buffer
# and rasterizes robot link capsules against the occupied cells; a grid larger
# than the kernel's configured capacity fails closed.
#
# `origin` is the base-frame position of voxel (0,0,0)'s minimum corner.
# `occupancy` is row-major with x fastest: idx = x + size_x*(y + size_y*z),
# length size_x*size_y*size_z. A cell is occupied when its value is non-zero
# (0 = free; the `nav_msgs/OccupancyGrid` 0..100 convention also works since the
# kernel only tests for non-zero).

std_msgs/Header header

geometry_msgs/Point origin
float64 resolution
uint32 size_x
uint32 size_y
uint32 size_z
uint8[] occupancy
