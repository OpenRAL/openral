"""``metal_pick_place`` sim backend — MakerMods Metal cube-to-marker on the real rig.

Imported by ``openral_sim.backends.__init__`` so ``@SCENES.register`` fires.
"""

from __future__ import annotations

from openral_sim.backends.metal_pick_place.env import build_metal_pick_place_scene

__all__ = ["build_metal_pick_place_scene"]
