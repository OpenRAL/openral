std_msgs/Header header
uint8 control_mode      # enum mirroring openral_core.ControlMode
uint16 horizon
float64[] flat          # row-major: [horizon][n_dof]
uint8 n_dof
# Per-axis physical range for a normalized CARTESIAN_DELTA controller.
# Predictive safety applies clip(raw, -1, 1) * scale. Empty = identity
# (the policy already emits physical units). Raw `flat` is preserved for HAL.
float64[] cartesian_delta_scale
string ee_name
string frame_id
float32 confidence
string rskill_id
string rskill_revision
string trace_id

# 1-based monotonic inference-tick index, set identically on every
# slot chunk a multi-surface skill emits in one tick. 0 = unset.
# The DatasetRecorderBridge groups chunks by this to reassemble the full
# per-tick action robustly, without relying on the slot-cycle heuristic.
uint32 tick_index
# Number of non-discard slots emitted for this tick. 1 = single-surface.
uint16 tick_group_size

# Robot joint names this chunk's `flat` rows target, in the slot's own
# order (ADR-0102). Empty = the chunk is a whole-vector action in
# `RobotDescription.joints` order, which is what every single-slot skill
# emits and what every pre-0102 producer meant.
#
# A slot-dispatched JOINT_* chunk is padded to full dof (the kernel
# enforces `chunk.n_dof == envelope.n_dof`) with ZEROS at non-target
# joints, so the payload alone cannot say which joints it owns — and 0.0
# is a legal joint target, so it never can. `ee_name` disambiguates
# gripper/Cartesian slots; this does the same for JOINT_* slots, which is
# the case two committed rSkills already hit (the OpenArm v2 bimanual
# restock contract and gr00t-n17-b1k). When non-empty this is
# AUTHORITATIVE over slot arrival order.
string[] joint_names
