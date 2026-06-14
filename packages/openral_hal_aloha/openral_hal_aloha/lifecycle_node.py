#!/usr/bin/env python3
r"""bimanual ALOHA (14-DoF, leader+follower) HAL lifecycle node entry point.

Manifest-driven node (ADR-0032): builds its sim or real HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. ``openral deploy sim`` injects ``hal_mode:=sim``
(→ ``AlohaMujocoHAL``); ``openral deploy run`` injects ``hal_mode:=real``
(→ ``AlohaHAL``).

Usage::

    ros2 run openral_hal_aloha lifecycle_node.py \
        --ros-args -p robot_yaml:=robots/aloha_bimanual/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_aloha")


if __name__ == "__main__":
    main()
