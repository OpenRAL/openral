# World State's bounded attestation that an attached payload is resting on a
# named environment surface (ADR-0092 D6).
#
# Attachment says what the robot is carrying; it does NOT say the payload is
# free of its environment. A grasped baguette still lies on the counter while
# the gripper closes, and the safety kernel sees that legitimate contact as a
# payload-vs-world penetration. This witness is the attestation that licenses
# exactly that one contact — bounded in extent by patch_radius_m, bounded in
# depth by max_penetration_m, and dead the moment the payload separates.
# No witness means no exemption; the kernel's default is to stop.
#
# Geometry is expressed in the ATTACHED OBJECT's own frame, never in the base
# frame and never as voxel indices. That is deliberate: indices decorrelate as
# the mobile base drives and the occupancy lattice re-phases underneath the
# robot, while the physical contact persists. The object frame moves with the
# payload, so the attested plane keeps describing the same physical support
# face for as long as the contact lasts.

# Identity of the supporting environment surface (e.g. "sim:counter_main").
# Together with stamp_ns this keys the kernel's re-arm check: a witness that
# has already died stays dead until World State attests a NEW contact.
string support_id

# A point on the support surface, in the attached object's frame.
geometry_msgs/Point contact_point_in_object

# Unit outward normal of the support surface, in the attached object's frame.
# It points FROM the supporting solid TOWARD the payload, so the support
# occupies the half-space
#   (x - contact_point_in_object) . contact_normal_in_object <= 0
geometry_msgs/Vector3 contact_normal_in_object

# Lateral radius of the supported patch about contact_point_in_object, measured
# in the support plane (m, > 0). Contact outside this patch — a wall, a fixture,
# a second object mid-carry — is never exempt.
float64 patch_radius_m

# Attested bound on the PHYSICAL penetration of the payload into the support
# surface (m, >= 0) — what the producer actually measured, not what a quantised
# occupancy map reads. Voxel discretisation inflates a 1 mm contact to ~15 mm at
# 25 mm resolution; the kernel accounts for that separately and geometrically,
# so this field must stay physical.
float64 max_penetration_m

float32 confidence

# Source that MEASURED the support contact, from openral_core's
# AttachmentEvidenceKind. The MuJoCo producer attests "sim_geom_distance": a
# signed closest-distance probe, NOT the solver's contact list. That
# distinction is load-bearing — contype/conaffinity suppression empties whole
# geom pairs, so a payload demonstrably resting on a counter can generate no
# contact record at all. A consumer must not read "sim_contact" as proof a
# solver contact existed, nor "sim_geom_distance" as proof none did.
string evidence_kind
string evidence_ref
int64 stamp_ns
