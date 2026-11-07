"""``metal_pen_holder`` sim backend — the MakerMods Metal Arm Station pen-holder task.

Importing this package registers the ``metal_pen_holder`` scene against
``makermods_metal_station`` (see ``env``).
"""

from openral_sim.backends.metal_pen_holder.env import build_metal_pen_holder_scene

__all__ = ["build_metal_pen_holder_scene"]
