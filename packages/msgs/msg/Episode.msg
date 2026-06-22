# openral_msgs/Episode — episode-boundary marker (ADR-0019 PR3).
#
# Published TWICE per episode by DeployRunner.episode_start() /
# episode_end():
#
#   phase=0 (PHASE_START) — task_string carries the natural-language
#     instruction; success is meaningless and MUST be ignored.
#   phase=1 (PHASE_END)   — task_string echoes the start value (so the
#     bag is self-describing); success is the resolved outcome.
#
# The dataset converter (PR4) walks Episode messages to segment the
# bag into discrete episodes — `/openral/episode` is the canonical
# segmentation key. A bag without these markers is rejected at conversion
# time (`ROSConfigError("bag has no /openral/episode markers")`).
#
# Two phases (not "open" + "close" topics) keep the schema small and
# let a single subscriber dispatch off `phase` rather than two topics.

uint8 PHASE_START=0
uint8 PHASE_END=1

builtin_interfaces/Time stamp
uint32 episode_idx
string task_string
uint8 phase
bool success
