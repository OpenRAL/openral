# Layer 2's bounded attestation that an attached payload is in MEASURED contact
# with its declared place target, and how hard (ADR-0099, survey Path C).
#
# This is the second observable on the place path. ADR-0097/0098 discriminate on
# position, and at the instant of a place the true clearance IS zero — no
# geometric margin at any resolution passes "set the object down" while stopping
# "crush the object against the shelf." Force is the axis that separates them,
# and the axis every power-and-force-limiting standard uses (ISO 10218-2:2025,
# which absorbed ISO/TS 15066's PFL content).
#
# THE GATE THIS FEEDS ONLY ADDS REFUSALS. A witness never licenses a contact the
# geometry refuses, never widens an ADR-0097 allowance, and never moves a margin.
# It can turn an accepted configuration into a refused one and nothing else.
# That direction is the whole safety argument (ADR-0099 §1) — do not extend this
# message in a way that inverts it without a new ADR.
#
# The kernel arms the gate only when ALL FOUR hold: a place declaration is live
# for this payload; that declaration carries contact_force_threshold_n > 0; a
# witness is live for the same payload and names the declared target; and the
# witness attests magnitude_calibrated. Any one absent means no gate at all —
# geometry decides exactly as it does today. Four independent fail-toward-shipped
# conditions, none satisfiable by accident.
#
# Liveness follows ADR-0092 D6 exactly: the witness dies on separation and is
# re-armed only by a FRESH attestation, keyed on payload id + target_id +
# stamp_ns, so a heartbeated snapshot cannot resurrect a dead contact.

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
# TRUE. A producer that has not been calibrated against a real force-torque
# measurement must publish magnitude_calibrated=false, and the kernel then
# ignores this field entirely rather than reading it as Newtons.
#
# This caveat is not defensive boilerplate. No published work validates MuJoCo
# contact-force MAGNITUDES against real F/T sensors (survey §21.7); MuJoCo's own
# docs frame the contact model as an approximation resting on solref/solimp
# choices; and FORGE (arXiv:2408.04587) re-tunes its threshold on hardware across
# >1000 real trials, which is itself the finding. A sim magnitude is a
# calibration knob, never an equivalence claim (CLAUDE.md §1.2).
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
