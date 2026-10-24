# What the HAL's gripper trigger computed (issue #418), on /openral/grasp_events
# (RELIABLE, VOLATILE, KEEP_LAST 10). The Layer-2 tracker binds and releases
# tracked objects from it; the HAL keeps only this hardware semantics.
# Deliberately IDL-only: openral_hal._grasp_trigger.GraspEvent is the HAL's own
# enum and no core model shares its name.

uint8 KIND_ATTACH = 1
uint8 KIND_REGRASP = 2
uint8 KIND_DETACH = 3
uint8 kind

# Gripper joint (manifest joint name) and the hand link the pinch is measured on.
string gripper_joint
string hand_link

# Jaw position measured and commanded at the event, and the position the trigger
# held at (the stall), in the gripper joint's own units.
float64 jaw_measured
float64 jaw_commanded
float64 hold_position

# Joint-state stamp the trigger evaluated (source clock).
builtin_interfaces/Time stamp
