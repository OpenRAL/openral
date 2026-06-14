#!/usr/bin/env python3
r"""Flexiv Rizon 4 7-DoF arm HAL lifecycle node entry point.

Manifest-driven node (ADR-0032): builds its HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. ``openral deploy sim`` injects ``hal_mode:=sim``
(→ ``Rizon4MujocoHAL``). The Rizon 4 is sim-only today (``hal.real`` is
null), so ``hal_mode:=real`` raises ``ROSCapabilityMismatch`` until the
``flexiv_rdk`` wrapper lands.

Usage::

    ros2 run openral_hal_rizon4 lifecycle_node.py \
        --ros-args -p robot_yaml:=robots/rizon4/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_rizon4")


if __name__ == "__main__":
    main()
