# EXPERIMENT (vision + voxel fusion; sim-only, Safety-WG review owed; issue pro#51).
# Per-track voxel cells the instance mask propagator labelled: SAM2.1 video masks of the
# tracker's TARGET / DESTINATION tracks, minus robot pixels, lifted through the aligned depth
# into the lattice of the /openral/world_voxels grid. The object tracker fits those tracks'
# boxes to them (ObjectTracker.apply_instance_cells) and refuses cells on any other lattice.
#
# header.stamp: the depth frame the masks were lifted through. header.frame_id: the lattice
# frame. `origin` and `resolution` repeat the grid's, so a consumer can check the lattice.
std_msgs/Header header
geometry_msgs/Point origin
float64 resolution

# Track object_ids[n] owns cell_counts[n] consecutive (i, j, k) index triples of `cells`,
# in object_ids order: len(cells) == 3 * sum(cell_counts).
string[] object_ids
uint32[] cell_counts
int32[] cells
