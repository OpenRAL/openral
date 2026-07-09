#!/usr/bin/env python3
r"""panda_mobile HAL lifecycle node entry point.

Manifest-driven node: builds its sim HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. panda_mobile is simulation-only (``hal.real``
is null), so ``deploy run`` raises ``ROSCapabilityMismatch``.

Usage::

    ros2 run openral_hal_panda_mobile lifecycle_node \
        --ros-args -p robot_yaml:=robots/panda_mobile/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_panda_mobile")


if __name__ == "__main__":
    main()
