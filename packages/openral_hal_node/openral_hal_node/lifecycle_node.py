#!/usr/bin/env python3
r"""The one manifest-driven OpenRAL HAL lifecycle node.

Every robot runs this node. Robot identity comes from the ``robot_yaml`` ROS
parameter (a ``robots/<id>/robot.yaml`` manifest), never from the package:
``openral_hal.lifecycle.make_lifecycle_main_from_manifest`` reads
``robot_yaml`` + ``hal_mode`` and routes through ``openral_hal.build_hal``.
``openral deploy sim`` injects ``hal_mode:=sim`` (a derived ``MujocoArmHAL``,
a manifest ``hal.sim`` entrypoint, or ``SimAttachedHAL`` when a
``sim_env_yaml`` scene is attached); ``openral deploy run`` injects
``hal_mode:=real`` (the manifest's ``hal.real`` entrypoint).

The launch overrides the ROS node NAME to ``openral_hal_<robot_id>``; that
name namespaces the node's topics and services and is unrelated to the
package name.

Usage::

    ros2 run openral_hal_node lifecycle_node.py \
        --ros-args -r __node:=openral_hal_franka_panda \
        -p robot_yaml:=robots/franka_panda/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_node")


if __name__ == "__main__":
    main()
