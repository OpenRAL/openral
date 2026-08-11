# Current safety state on the LATCHED topic /openral/safety_status
# (ADR-0096). QoS: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1 — the
# "description/static" profile from CLAUDE.md §2, deliberately NOT the
# "safety/e-stop" profile /openral/estop and /openral/failure/safety use.
#
# This message answers "what is true right now", not "what just
# happened": a dashboard tab opened mid-mission or a runner reconnecting
# after a crash must see the current state without having witnessed the
# transition, which VOLATILE cannot deliver by construction.
# /openral/failure/safety stays the VOLATILE *event* stream for the
# reasoner's replanning ladder; this topic neither replaces nor
# deprecates it, nor /openral/estop.
#
# Publishers: the C++ safety kernel (cpp/openral_safety_kernel) and
# SafetyPassthroughNode (packages/openral_safety). Both publish on every
# latch transition, every fail-closed drop transition, every
# clear/recovery transition, on every lifecycle activation
# (hazard-log HZ-0096-1 mitigation 1: a restarted publisher must
# overwrite any stale durable value within one activation cycle), and as
# a 1 Hz liveness refresh so `header.stamp` stays evidence that the
# publisher is alive.
#
# Consumers MUST treat a status whose `header.stamp` is older than their
# documented liveness window as UNKNOWN, not as safe (HZ-0096-1
# mitigation 2/4 — the staleness rule fails toward "assume unsafe",
# never toward "assume clear").

std_msgs/Header header

# True while a fault is latched and recovery requires an explicit
# /openral/estop_reset. False during a non-latching fail-closed drop
# (`drop_reason` still names the cause) and when all is clear.
bool latched

# drop_reason — why actions are being suppressed right now.
#
# The KIND_* block below is numerically IDENTICAL to
# openral_msgs/FailureTrigger's KIND_* constants, so one number means
# the same fault everywhere. ROS IDL has no cross-message constant
# reuse, so they are redeclared here rather than imported; the C++
# kernel gtest and tests/unit/test_safety_status_msg.py pin the two
# blocks together.
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

# DROP_* — a disjoint range covering the NON-latching fail-closed drop
# paths, so a consumer can tell "the kernel is fault-latched" from "the
# kernel is dropping because an upstream input is unavailable" from the
# enum value alone, without first reading `latched`. Values start at 100
# and can never collide with a KIND_* value.
uint8 DROP_ENVELOPE_UNCONFIGURED = 100
uint8 DROP_WORLD_UNAVAILABLE     = 101
uint8 DROP_VOXEL_UNAVAILABLE     = 102
uint8 DROP_STATE_UNAVAILABLE     = 103
uint8 DROP_WORLD_OVERFLOW        = 104
uint8 DROP_VOXEL_OVERFLOW        = 105
# An external /openral/estop publication latched us. The publisher of
# that std_msgs/Empty is unknowable from the wire, so this names the
# topic that latched the node rather than claiming a violation kind
# nobody reported (CLAUDE.md §1.2).
uint8 DROP_EXTERNAL_ESTOP        = 106

# ADR-0092 attached payloads. A grasped object the kernel cannot verify
# (never received, stale, or over-capacity / unknown-link) is its own
# fail-closed drop cause: "we do not know what the robot is carrying" is
# a different operator action from "the world model went stale", so it
# gets its own code rather than borrowing DROP_WORLD_*.
uint8 DROP_ATTACHED_UNAVAILABLE   = 107
uint8 DROP_ATTACHED_OVERFLOW      = 108

# Nothing is being dropped. Published explicitly on every clear /
# recovery transition and on lifecycle activation; never left implicit,
# because a default-initialised message would otherwise read as
# KIND_TIMEOUT.
uint8 DROP_NONE                  = 255
uint8 drop_reason

# Free-text elaboration — the role FailureTrigger.evidence_json plays
# there. Plain string, not JSON: the typed `drop_reason` above already
# carries the machine-readable signal.
string detail

# Correlation fields, matching FailureTrigger's. `rskill_id` is the
# in-flight skill from the chunk that caused the transition (empty when
# the transition had no chunk, e.g. activation or external e-stop);
# `trace_id` is a W3C traceparent.
string rskill_id
string trace_id
