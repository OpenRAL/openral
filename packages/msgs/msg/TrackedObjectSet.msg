# The Layer-2 tracker's whole object set, one revision (issue #418); wire form of
# core.TrackedObjectSet on /openral/tracked_objects (RELIABLE, VOLATILE,
# KEEP_LAST 1, >= 2 Hz heartbeat) and inside WorldStateStamped. The kernel
# refuses the WHOLE set on a frame mismatch, a cap breach, a non-monotone
# revision or a future stamp: every object is then ineligible.

# "1".
string schema_version
# Must equal the voxel grid frame; every object's box is in it.
string frame_id
# Producer stamp, source clock.
int64 stamp_ns
# Strictly increasing per publisher.
uint64 revision
# At most 32 tracks, one per object_id.
TrackedObject[<=32] objects
