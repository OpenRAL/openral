# ADR-0018 F3 — typed FailureTrigger event on the namespaced
# /openral/failure/{hal,sensor,rskill,safety,wam,critic} bus.
# (The rskill suffix replaced the original "skill" on 2026-05-25 for
# consistency with the carried ``rskill_id`` field; see ADR-0018
# amendment 2026-05-25 §5.)
#
# `kind` and `severity` are uint8 enum constants (typo-proof at build
# time). `evidence_json` carries a serialized Pydantic ``FailureEvidence``
# discriminated union from openral_core — consumers parse it with the
# matching discriminator.
#
# This is a hard wire break vs the pre-F3 string-shaped message: no
# version bump, no schema migrator. The earlier string-shaped trigger
# was never persisted on disk (no rosbag2 producers in tree); the only
# affected consumers are live subscribers, which must rebuild.

std_msgs/Header header

# kind — one source layer per topic, but the kind tags the *cause*.
uint8 KIND_TIMEOUT             = 0
uint8 KIND_FORCE               = 1
uint8 KIND_WORKSPACE           = 2
uint8 KIND_PERCEPTION          = 3
uint8 KIND_CRITIC              = 4
uint8 KIND_CONTROLLER          = 5
uint8 KIND_SELFVERIFY          = 6
uint8 KIND_HUMAN               = 7
uint8 KIND_WAM                 = 8
uint8 KIND_REASONER_TIMEOUT    = 9
uint8 KIND_COLLISION           = 10
uint8 KIND_SUPPRESSED_SUMMARY  = 254
uint8 kind

# severity — escalation level. WARN/INFO may be token-bucket-suppressed
# by the publisher helper; FAIL/ABORT always pass through.
uint8 SEVERITY_INFO   = 0
uint8 SEVERITY_WARN   = 1
uint8 SEVERITY_FAIL   = 2
uint8 SEVERITY_ABORT  = 3
uint8 severity

# Serialized Pydantic FailureEvidence (discriminated union, see
# openral_core.FailureEvidence). Empty string is only valid for
# KIND_SUPPRESSED_SUMMARY when no rate-limited events accumulated.
string evidence_json

# Skill id (when known) and OTel trace_id (W3C traceparent format).
string rskill_id
string trace_id
