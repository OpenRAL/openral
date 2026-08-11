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
