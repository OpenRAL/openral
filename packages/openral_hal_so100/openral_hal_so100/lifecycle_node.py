#!/usr/bin/env python3
r"""SO-100 / SO-101 HAL lifecycle node entry point.

Manifest-driven node: builds its sim or real HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. A single package serves both the SO-100
(``so_arm100``) and SO-101 (``so101_new_calib``) from their own manifests.

Usage::

    # MuJoCo digital twin (what `openral deploy sim` does)
    ros2 run openral_hal_so100 lifecycle_node \
        --ros-args -p robot_yaml:=robots/so101_follower/robot.yaml -p hal_mode:=sim
    # real hardware
    ros2 run openral_hal_so100 lifecycle_node \
        --ros-args -p robot_yaml:=robots/so100_follower/robot.yaml -p hal_mode:=real
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_so100")


if __name__ == "__main__":
    main()
