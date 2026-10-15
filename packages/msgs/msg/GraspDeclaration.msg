# Dispatch's typed statement that a grasp phase is active, and which target the
# gripper is about to close on (real pick-and-place design §2.1; ADR draft
# "Declaration-scoped grasp-target contact for gripper finger links" in
# docs/reference/real-pick-place-adr-drafts.md). Field-for-field a mirror of
# PlaceDeclaration — same discipline: dispatch names, a producer measures, the
# kernel bounds, everything dies with the goal.
#
# A declaration alone exempts NOTHING. Dispatch never supplies a region (the
# runner strips it); only the attachment evidence producer that measured the
# target may fill `region` and re-publish the declaration on the
# AttachmentState -> WorldStateStamped envelope the kernel already ingests. The
# kernel consumer is a separate change and its feature parameter defaults off.
#
# Scoped to one goal execution. MUST be retracted (active=false) on goal
# end/cancel and E-stop; every consumer MUST also expire it on timeout_s so a
# dispatcher dying mid-goal cannot leave an exemption armed.

# Identity of the declared grasp target (e.g. "cell:pick_box"). Empty is
# invalid while active=true.
string target_id

# Payload identity the grasp will produce once attached (an
# AttachedCollisionObject.object_id). Empty means "discovered at attach time".
string object_id

# Gripper contact links the declaration names (e.g.
# "openarm_left_finger_pair"). Non-empty while active=true. A consumer
# intersects these with its own launch-derived allowlist; a name it does not
# allow exempts nothing.
string[] contact_links

# Attributability (HZ-01xx-2): makes a wrong declaration reconstructible from
# the trace alone.
string rskill_id
string trace_id

# Backstop expiry window, seconds after stamp_ns.
float64 timeout_s

# Dispatcher timestamp (ns). With active, this is the whole liveness key.
int64 stamp_ns

# False retracts the declaration. Retraction is explicit and additive to the
# timeout: whichever comes first wins.
bool active

# Producer-measured oriented box around the declared target, reusing
# PlaceRegion. region_valid false — the default, and what dispatch itself
# always publishes — means NO exemption. In v1 `region.geometry[]` MUST be
# empty for a grasp declaration; a consumer refuses the region otherwise.
bool region_valid
PlaceRegion region

# Optional search box (real pick-and-place design §2.2), in the voxel grid's base
# frame or a fixed tf2 frame (`odom`, `map`) the producer re-expresses in the grid's
# frame every tick: where the target producer looks in the occupancy map for the seed it
# prompts the segmenter with. It only SEEDS perception and arms nothing — the
# exemption is always the measured `region` above. Unlike `region`, a scene may
# supply it and dispatch passes it through unchanged. search_box_valid false =
# no search, so no region is ever measured. `search_box.geometry[]` MUST be empty.
bool search_box_valid
PlaceRegion search_box

# Optional manipulation workspace (issue #349, docs/reference/manipulation-workspace-contact.md),
# in the voxel grid's base frame or, like `search_box`, a fixed tf2 frame: the box the task works in (the source tote's interior for a
# pick). Like `search_box`, a scene may DECLARE it and dispatch passes it through unchanged; on
# its own it exempts nothing. workspace_box_valid false = no workspace. `workspace_box.geometry[]`
# MUST be empty.
bool workspace_box_valid
PlaceRegion workspace_box

# The producer-confirmed workspace: `workspace_box` itself (never grown), stamped with the
# source_stamp of the fresh voxel grid that confirmed it holds occupied cells. Only the producer
# may set it — dispatch never does (the runner strips it) and a scene can never supply it. A
# consumer that enables workspace contact lets the robot's HAND links touch occupied cells
# centred inside it while the hand is slow; workspace_region_valid false — the default — means
# NO workspace contact. `workspace_region.geometry[]` MUST be empty.
bool workspace_region_valid
PlaceRegion workspace_region
