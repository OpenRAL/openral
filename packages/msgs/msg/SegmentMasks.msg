# DIAGNOSTIC mirror of the latest openral_msgs/srv/SegmentInView reply.
#
# The segmenter contract is a SERVICE — one shot per attach / detach / regrasp
# event — because the HAL needs an answer, not a stream. That leaves an operator
# with no way to SEE what the model masked: the dashboard's camera tiles can
# render segmentation masks (openral_observability.dashboard, PR #122) but had
# no producer to render, and `ros2 topic echo` cannot introspect a service reply
# that already happened.
#
# So the segmenter node re-publishes its latest reply here, behind
# `publish_debug_masks` (default FALSE — nothing is created, nothing is packed
# and nothing goes on the wire unless an operator asks for it).
#
# STRICTLY diagnostic. Nothing in the safety or attachment path reads this
# topic: the HAL producer takes its masks from its own service reply and gates
# them on geometry, and a subscriber to this topic can only change what a human
# sees. Dropping every message on this topic changes no robot behaviour, which
# is why sensor-class BEST_EFFORT / VOLATILE / KEEP_LAST=1 is the right profile
# — the newest mask set is the only one worth drawing.
#
#   header                — stamp is the CALLER's attach instant (copied from
#                           the request, not the publish time), so a stale
#                           overlay is detectable rather than a lie; frame_id is
#                           the camera's optical frame.
#   camera                — the camera id actually queried. A SensorSpec NAME,
#                           the same key the dashboard's camera tiles use, which
#                           is what binds an overlay to its tile.
#   rskill_id             — the producing kind:"segmenter" rSkill, so "whose
#                           masks are these" is never a guess.
#   masks                 — exactly what the service returned: full-frame mono8
#                           sensor_msgs/Image candidates (255 = in mask), in the
#                           service's own area-ascending order.
#   mask_scores_advisory  — parallel to `masks`, same length. ADVISORY ONLY, as
#                           on the service: displayed, never used to rank or
#                           filter. A mis-aimed prompt was measured returning a
#                           59.8%-of-frame mask at this model's TOP score of
#                           0.977.

std_msgs/Header header
string camera
string rskill_id
sensor_msgs/Image[] masks
float32[] mask_scores_advisory
