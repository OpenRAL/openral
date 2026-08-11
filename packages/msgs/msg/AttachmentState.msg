# Atomic complete attached-object snapshot.
#
# Evidence producers publish the full set with TRANSIENT_LOCAL durability.
# World State replaces its previous attachment set only after the whole message
# validates, so attach/detach never exposes a partial payload model.

std_msgs/Header header
uint64 revision
AttachedCollisionObject[] objects
