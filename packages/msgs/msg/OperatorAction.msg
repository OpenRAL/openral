# An operator acting on the machine through a channel, announced so the compliance
# recorder can attribute it (issue #468; EU Machinery Regulation 2023/1230 Annex III
# §1.1.9: "evidence of a legitimate or illegitimate intervention").
#
# Topic: /openral/operator_action (RELIABLE, VOLATILE, depth 10). Published by the
# channel just BEFORE it acts (`openral estop reset`, the dashboard), except an E-STOP. The recorder
# observes the effect itself — a /openral/safety_status latch clearing is a reset
# whoever caused it — and uses this message only to say who: a reset that nobody
# announced is still recorded, as an unattributed intervention.

std_msgs/Header header

string ACTION_ESTOP_RESET   = "estop_reset"
string ACTION_PARAM_SET     = "param_set"
# The dashboard's other manifest actions (ADR-0129): an operator E-STOP (announced just
# AFTER it fires, so attribution never delays a stop), an operator prompt, and an rSkill
# run (detail: the rSkill id).
string ACTION_ESTOP         = "estop"
string ACTION_PROMPT        = "prompt"
string ACTION_SKILL_EXECUTE = "skill_execute"
string action

# The channel as configured on the publisher: "cli", "dashboard", "pendant", ...
string channel_label

# Pseudonymous operator id: HMAC-SHA256 hex of the operator identity under the
# deployer's key ($OPENRAL_ACTOR_KEY), or "" when the channel cannot identify the
# operator. Never a name (openral_core.Actor). The dashboard's identity is the
# openral.com user id (the sign-in token's `sub`) over the robot's tunnel, else the
# client IP.
string actor_id

# What was acted on, without personal data (e.g. the parameter name for param_set, the
# rSkill id for skill_execute; `jti=<token id>` when the actor signed in).
string detail
