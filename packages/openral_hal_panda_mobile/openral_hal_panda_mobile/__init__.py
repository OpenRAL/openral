"""ADR-0024 / ADR-0025 — `openral_hal_panda_mobile` package marker.

The ament-Python entry point is the executable script
``lifecycle_node.py`` installed under ``lib/openral_hal_panda_mobile/``.
This package marker exists so ``ament_python_install_package``
treats the directory as a Python package; the runtime code lives in
the sibling ``lifecycle_node.py`` and is invoked via
``ros2 run openral_hal_panda_mobile lifecycle_node`` (after
``colcon build``).
"""

from __future__ import annotations
