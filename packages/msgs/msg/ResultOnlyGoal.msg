# An ExecuteRskill goal that publishes no /openral/safe_action chunks while it runs: a
# wrapped result-only ros_action / ros_service rSkill (Nav2's navigate_to_pose), whose
# server moves the robot itself. Published by the rskill runner when such a goal starts.
# The deadman watchdog bounds that goal by wall_bound_s instead of its first-chunk window
# (ADR-0127 Decision 5); a goal never named here keeps the first-chunk window.
#
# Topic: /openral/result_only_goal (RELIABLE, TRANSIENT_LOCAL, KEEP_LAST 1).

# action_msgs/GoalInfo.goal_id.uuid of the ExecuteRskill goal.
uint8[16] goal_uuid
# Wall-clock (system time) moment the runner started the goal's budget. The watchdog counts
# the bound from here, so a watchdog that (re)joins mid-goal and receives this latched
# message does not start a fresh window.
builtin_interfaces/Time wall_start
# The rSkill the goal runs, for the watchdog's log and FailureTrigger evidence.
string rskill_id
# The runner's own wall-clock bound on the goal, in seconds: its execution budget, times
# the execution_wall_cap_factor backstop when the budget runs on the graph clock.
float64 wall_bound_s
