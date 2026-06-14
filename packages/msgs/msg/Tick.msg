# openral_msgs/Tick — per-tick metadata from HardwareRunner (ADR-0019 PR3).
#
# Published once per InferenceRunnerBase tick alongside the per-sensor
# image streams and /joint_states. Carries the fields the dataset
# converter (PR4) needs to reconstruct a LeRobotDataset v3 row:
#
#   * episode_idx / step_idx — frame coordinates within the dataset
#   * reward / terminated / truncated / action_applied — RL-style transition flags
#   * trace_id / span_id — W3C TraceContext-style join key back to Jaeger
#
# Mirrors openral_core.schemas.TickResult (python/core/src/openral_core/schemas.py:2903).
# When that schema grows fields, mirror them here in the same PR — the
# offline converter in openral_dataset.bag.Rosbag2Sink schema_registry
# imports both shapes side-by-side.

builtin_interfaces/Time stamp
uint64 tick_idx
uint32 episode_idx
uint32 step_idx
float32 reward
bool terminated
bool truncated
bool action_applied
string trace_id
string span_id
