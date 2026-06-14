# ADR-0018 F2 — typed WorldState publication.
#
# Replaces the JSON ``std_msgs/String`` snapshot the world_state node
# used to publish on ``/world_state``. Carries the same fields as
# :class:`openral_core.WorldState` in a ROS-native, parallel-array
# shape so downstream nodes (reasoner, dashboards, rqt) can consume it
# without parsing JSON.
#
# Published by ``openral_world_state.world_state_node`` on
#   /openral/world_state_fast   (30 Hz, RELIABLE+VOLATILE+KL=1)
#   /openral/world_state_slow   ( 5 Hz, RELIABLE+VOLATILE+KL=1)
# from one in-memory snapshot per fast tick (the slow topic publishes
# the same payload every 6th fast tick).

std_msgs/Header header
string trace_id
int64 stamp_ns

# Joint state — sensor_msgs/JointState carries name/position/velocity/effort.
sensor_msgs/JointState joint_state

# Base pose / twist (mobile robots; optional).
bool base_pose_valid
geometry_msgs/Pose base_pose
bool base_twist_valid
geometry_msgs/Twist base_twist

# End-effector poses — parallel arrays keyed by EE name.
string[] ee_names
geometry_msgs/Pose[] ee_poses

# Sensor image topic refs — parallel arrays keyed by sensor id.
# ``image_topics[i]`` is the ROS topic the sensor reader publishes on.
string[] image_sensor_ids
string[] image_topics

# Per-component diagnostics — parallel arrays. ``diagnostic_statuses[i]``
# is one of the DIAG_* constants below.
uint8 DIAG_OK    = 0
uint8 DIAG_WARN  = 1
uint8 DIAG_STALE = 2
uint8 DIAG_ERROR = 3
string[] diagnostic_keys
uint8[]  diagnostic_statuses

# Per-component staleness in milliseconds — parallel arrays.
# ``staleness_keys[]`` is a subset of ``diagnostic_keys[]`` (only the
# components for which we ever received at least one update; never-seen
# components are flagged ``DIAG_STALE`` but contribute no age).
string[]  staleness_keys
float32[] staleness_ms

# Battery percentage in [0, 100]; valid is false when unknown.
bool battery_valid
float32 battery_pct

# tf2 frames referenced by this snapshot — consumers read /tf themselves
# (ADR-0018 §2 / capability-review F2: tf2 reference-only).
string[] frame_ids

# Detected objects (ADR-0035 object lift) — parallel arrays keyed by index.
# Positions are pose.xyz in `detected_object_frame` (e.g. "map"); track ids are
# -1 when unset. 3D bbox is intentionally omitted from the wire (center + frame
# is what consumers need; bbox_3d stays in the in-process WorldState).
string[]              detected_object_labels
float32[]             detected_object_confidences
geometry_msgs/Point[] detected_object_positions
int32[]               detected_object_track_ids
string                detected_object_frame
