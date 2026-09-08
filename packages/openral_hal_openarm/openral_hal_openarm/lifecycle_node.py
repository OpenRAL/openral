#!/usr/bin/env python3
r"""OpenArm HAL lifecycle node entry point.

Manifest-driven node: builds its HAL via
``openral_hal.lifecycle.make_lifecycle_main_from_manifest``, which reads
the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
``openral_hal.build_hal``. ``robots/openarm/robot.yaml`` declares both
``hal.sim`` (``OpenArmMujocoHAL``) and ``hal.real`` (``OpenArmRealHAL``); under
``hal_mode:=real`` the base node attaches a ``RosControlTransport`` to it, and
the ros2_control graph it commands is brought up by this package's
``launch/openarm_real_bringup.launch.py`` (see ``README.md``).

Usage::

    ros2 run openral_hal_openarm lifecycle_node \
        --ros-args -p robot_yaml:=robots/openarm/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_openarm")


if __name__ == "__main__":
    main()
