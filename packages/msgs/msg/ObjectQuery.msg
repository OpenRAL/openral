# A class of objects a prompt names (issue #418); wire form of core.ObjectQuery.
# Never geometry the kernel uses: the Layer-2 object tracker keeps it as a
# standing query for the life of its goal and grounds every instance.

# Open-vocabulary label, e.g. "orange".
string label
# "one" | "all". "one" still tracks every match; the policy chooses.
string count
# "target" | "destination" | "context" (core.ObjectRole).
string role
# Gripper joint name, a hint for the policy/reasoner only. Empty = none.
string hand
# CONFIRMED instances that resolve the query (>= 1; exactly 1 for count "one").
uint32 min_instances
# Optional search hint; seeds prompts only. `seed.geometry[]` MUST be empty.
bool seed_valid
PlaceRegion seed
