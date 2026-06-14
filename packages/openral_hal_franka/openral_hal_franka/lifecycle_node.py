#!/usr/bin/env python3
r"""Franka Panda HAL lifecycle node entry point.

Manifest-driven node (ADR-0032): builds its sim or real HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. ``openral deploy sim`` injects ``hal_mode:=sim``
(→ ``FrankaPandaHAL``); ``openral deploy run`` injects ``hal_mode:=real``
(→ ``FrankaPandaRealHAL``).

Usage::

    ros2 run openral_hal_franka lifecycle_node \
        --ros-args -p robot_yaml:=robots/franka_panda/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_franka")


if __name__ == "__main__":
    main()
