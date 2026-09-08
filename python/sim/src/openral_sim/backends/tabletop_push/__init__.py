"""Robot-agnostic ``tabletop_push`` native scene.

Importing this package registers the ``tabletop_push`` scene factory on
``openral_sim.SCENES``. See ``openral_sim.backends.tabletop_push.env``
for the rollout and ``openral_sim.backends.tabletop_push._assets`` for the
MjSpec composer.
"""

from __future__ import annotations

from openral_sim.backends.tabletop_push.env import build_tabletop_push_scene

__all__ = ["build_tabletop_push_scene"]
