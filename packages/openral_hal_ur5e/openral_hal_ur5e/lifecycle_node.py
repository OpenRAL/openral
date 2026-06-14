#!/usr/bin/env python3
r"""UR5e HAL lifecycle node entry point.

Manifest-driven node (ADR-0032): builds its sim or real HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. ``openral deploy sim`` injects ``hal_mode:=sim``
(→ ``UR5eHAL``); ``openral deploy run`` injects ``hal_mode:=real``
(→ ``UR5eRealHAL``).

Usage (after ``colcon build`` + ``source install/setup.bash``)::

    ros2 run openral_hal_ur5e lifecycle_node \
        --ros-args -p robot_yaml:=robots/ur5e/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_ur5e")


if __name__ == "__main__":
    main()
