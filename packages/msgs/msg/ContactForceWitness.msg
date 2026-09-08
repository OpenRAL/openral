# Layer 2's bounded attestation that an attached payload is in MEASURED contact
# with its declared place target, and how hard (ADR-0100, survey Path C).
#
# Second observable on the place path: ADR-0097/0098 discriminate on position,
# but at the instant of a place true clearance is zero, so force (the axis ISO
# 10218-2:2025 / ISO-TS-15066 PFL uses) is what separates "set down" from
# "crush against the shelf."
#
# THE GATE THIS FEEDS ONLY ADDS REFUSALS (ADR-0100 §1) — never licenses a
# contact geometry refuses, never widens an ADR-0097 allowance, never moves a
# margin. Do not extend this message in a way that inverts that without a new
# ADR.
#
# The kernel arms the gate only when ALL FOUR hold: a live place declaration
# for this payload with contact_force_threshold_n > 0; a live witness for the
# same payload naming the declared target; and magnitude_calibrated attested.
# Any one absent = no gate, geometry decides as today.
#
# Liveness follows ADR-0092 D6: dies on separation, re-armed only by a FRESH
# attestation keyed on payload id + target_id + stamp_ns.

# Identity of the surface the measured contact is against. The gate arms only
# when this names the live declaration's target_id, or a surface that is part of
# it — the same scoping rule SupportContactWitness.support_id is under. A
# measured contact against anything else attests nothing here.
string target_id

# Unit direction of the contact force, in the ATTACHED OBJECT's own frame,
# pointing FROM the support INTO the payload.
#
# Direction is in the object frame for the same reason the support witness's
# geometry is: base-frame quantities decorrelate as the mobile base drives while
# the physical contact persists. It is also the component with the better
# sim-to-real story — arXiv:2602.14174 argues from MuJoCo's magnitude gap that
# direction is what survives transfer, which is exactly why this field carries no
# calibration caveat and magnitude_n does.
geometry_msgs/Vector3 direction_in_object

# Contact force magnitude. UNITS ARE ONLY NEWTONS WHEN magnitude_calibrated IS
# TRUE; an uncalibrated producer must publish magnitude_calibrated=false and
# the kernel then ignores this field. No published work validates MuJoCo
# contact-force magnitudes against real F/T sensors (survey §21.7); FORGE
# (arXiv:2408.04587) re-tunes its threshold on hardware across >1000 real
# trials. A sim magnitude is a calibration knob, never an equivalence claim
# (CLAUDE.md §1.2).
float64 magnitude_n

# False — THE DEFAULT, and what every uncalibrated producer must publish — means
# magnitude_n is not in Newtons and the force gate does not arm. A stock sim
# deployment arms nothing until an operator explicitly asserts a calibration.
bool magnitude_calibrated

# Names the calibration magnitude_n was produced under, so a trace can answer
# "under which mapping was this judged" without the scene. Required (non-empty)
# whenever magnitude_calibrated is true; the kernel refuses to arm on a
# calibrated witness that cannot name its calibration.
string calibration_ref

float32 confidence

# Producer lineage, same contract as SupportContactWitness. evidence_kind comes
# from openral_core's AttachmentEvidenceKind. The MuJoCo producer attests
# "sim_contact_force": mj_contactForce over the solver's contact list — which
# means it inherits that list's blind spot, since contype/conaffinity exclusions
# can suppress a contact entirely (field-observed: an arm 30 mm inside a freezer
# door with ncon==0). ABSENCE OF A WITNESS IS THEREFORE NOT EVIDENCE OF NO
# CONTACT, and nothing downstream may read it as such. It only ever means the
# gate does not arm, and geometry decides alone.
string evidence_kind
string evidence_ref
int64 stamp_ns
