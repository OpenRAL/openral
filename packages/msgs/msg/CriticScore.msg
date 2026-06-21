# openral_msgs/CriticScore — generic per-frame critic / reward score sample
# (ADR-0064). Reward models publish these on /openral/critic/score; the critic
# producer node routes each sample by critic_id through a CriticWatchdogGroup
# (openral_reasoner.critic_watchdog) and emits a Tier-C FailureTrigger on
# /openral/failure/critic when a critic stalls.
#
# The topic is shared: the Robometer reward rSkill today (ADR-0057), a future
# SARM (self-assessment reward model), a success classifier — all publish here,
# distinguished by critic_id. Each sample is self-describing so onboarding a new
# reward model needs no producer-side configuration.

std_msgs/Header header

# Source identifier — which reward model produced this score (e.g. the Robometer
# reward rSkill id, or a future SARM id). The producer keys one watchdog per id.
string critic_id

# Higher-is-better score in the critic's native range (e.g. Robometer progress
# in [0, 1]). Reward sources normalize so that larger == better.
float64 score

# Pass bar in the same range: score >= threshold is acceptable progress (a
# recovery); below it is a candidate stall.
float64 threshold

# OTel trace_id (W3C traceparent format) for correlation back to the execution
# that produced the scored frame; empty when unset.
string trace_id
