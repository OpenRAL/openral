#!/usr/bin/env python3
r"""OpenArm HAL lifecycle node entry point.

Manifest-driven node: builds its sim HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. OpenArm is simulation-only (``hal.real`` is
null), so ``deploy run`` raises ``ROSCapabilityMismatch``.

Usage::

    ros2 run openral_hal_openarm lifecycle_node \
        --ros-args -p robot_yaml:=robots/openarm/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_openarm")


if __name__ == "__main__":
    main()
