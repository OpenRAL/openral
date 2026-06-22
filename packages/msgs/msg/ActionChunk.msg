std_msgs/Header header
uint8 control_mode      # enum mirroring openral_core.ControlMode
uint16 horizon
float64[] flat          # row-major: [horizon][n_dof]
uint8 n_dof
string ee_name
string frame_id
float32 confidence
string rskill_id
string rskill_revision
string trace_id

# ADR-0019 — 1-based monotonic inference-tick index, set identically on every
# slot chunk a multi-surface skill emits in one tick (ADR-0028b). 0 = unset.
# The DatasetRecorderBridge groups chunks by this to reassemble the full
# per-tick action robustly, without relying on the slot-cycle heuristic.
uint32 tick_index
