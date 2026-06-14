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
