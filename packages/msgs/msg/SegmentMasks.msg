# DIAGNOSTIC mirror of the latest openral_msgs/srv/SegmentInView reply.
#
# The segmenter contract is a SERVICE — one shot per attach/detach/regrasp
# event — so an operator has no way to SEE what was masked: the dashboard's
# camera tiles can render masks (openral_observability.dashboard, PR #122) but
# had no producer, and `ros2 topic echo` can't introspect a past service reply.
# The segmenter node re-publishes its latest reply here behind
# `publish_debug_masks` (default FALSE — nothing packed or sent unless asked).
#
# STRICTLY diagnostic: nothing in the safety or attachment path reads this
# topic (the HAL producer uses its own service reply), so dropping every
# message here changes no robot behaviour — hence sensor-class BEST_EFFORT /
# VOLATILE / KEEP_LAST=1 (only the newest mask set is worth drawing).
#
#   header  — stamp is the CALLER's attach instant (copied from the request,
#             not publish time) so a stale overlay is detectable, not a lie;
#             frame_id is the camera's optical frame.
#   camera  — camera id actually queried (SensorSpec NAME, the dashboard tile
#             key).
#   rskill_id — producing kind:"segmenter" rSkill.
#   masks   — exactly the service's full-frame mono8 candidates (255 = in
#             mask), area-ascending order.
#   mask_scores_advisory — parallel to `masks`. ADVISORY ONLY, never used to
#             rank/filter: a mis-aimed prompt was measured returning a
#             59.8%-of-frame mask at this model's TOP score of 0.977.

std_msgs/Header header
string camera
string rskill_id
sensor_msgs/Image[] masks
float32[] mask_scores_advisory
