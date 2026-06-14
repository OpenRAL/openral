#!/usr/bin/env python3
r"""Unitree G1 humanoid (29-DoF) HAL lifecycle node entry point.

Manifest-driven node (ADR-0032): builds its HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. ``openral deploy sim`` injects ``hal_mode:=sim``
(→ ``G1MujocoHAL``). The G1 is sim-only today (``hal.real`` is null), so
``hal_mode:=real`` raises ``ROSCapabilityMismatch`` until the M2 S0 layer.

Usage::

    ros2 run openral_hal_g1 lifecycle_node.py \
        --ros-args -p robot_yaml:=robots/g1/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_g1")


if __name__ == "__main__":
    main()
