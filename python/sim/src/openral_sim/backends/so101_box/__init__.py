"""``so101_box`` sim backend — SO-101 in a configurable box arena.

A parameterised raw-MuJoCo scene whose every dimension, camera pose,
object size and spawn range is driven by ``scene.backend_options`` in
the YAML. No geometry is hard-coded in Python — once registered, new
"box + SO-101" variants are pure YAML.

The package is imported by :mod:`openral_sim.backends.__init__` so the
``@SCENES.register("so101_box")`` decorator fires at package import.

Example:
    >>> from openral_sim.backends.so101_box._assets import compose_so101_box_mjcf
    >>> xml, meshdir = compose_so101_box_mjcf()
    >>> assert "<mujoco" in xml
"""

from __future__ import annotations

from openral_sim.backends.so101_box.env import build_so101_box_scene

__all__ = ["build_so101_box_scene"]
