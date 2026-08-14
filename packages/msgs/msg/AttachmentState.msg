# Atomic complete attached-object snapshot.
#
# Evidence producers publish the full set with TRANSIENT_LOCAL durability.
# World State replaces its previous attachment set only after the whole message
# validates, so attach/detach never exposes a partial payload model.

std_msgs/Header header
uint64 revision
AttachedCollisionObject[] objects

# The place declaration in force for this payload, as the evidence producer
# resolved it (ADR-0097 + its 2026-08-14 amendment). It rides here rather than on
# a channel of its own because the amendment extends the payload already crossing
# the dispatch -> HAL -> World State boundary; the safety kernel reads it off
# WorldStateStamped, which relays this field.
#
# What it adds over the dispatcher's own /openral/place_declaration publication
# is `region` — dispatch names a target, only the producer can measure where that
# target IS. It sits in the envelope rather than on each AttachedCollisionObject
# because the declaration is per-goal, not per-payload, and because the region is
# expressed in the (base) frame the occupancy grid uses, so it is refreshed on
# every publication while the object records are not.
#
# False means no declaration is in force, which is no witness and no allowance.
bool place_declaration_valid
PlaceDeclaration place_declaration
