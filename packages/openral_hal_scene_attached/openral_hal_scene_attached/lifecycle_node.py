#!/usr/bin/env python3
"""Generic scene-attached HAL lifecycle node for deploy-sim.

The node is intentionally sim-only in practice: ``openral deploy sim`` injects a
``sim_env_yaml`` parameter and :func:`openral_hal.build_hal` returns
``SimAttachedHAL`` before consulting any robot-specific ``hal.sim`` entrypoint.
It exists for simulator-owned embodiments such as SimplerEnv WidowX and
RoboTwin AgileX, where the simulator sidecar owns the robot model and OpenRAL
only needs the standard ROS lifecycle publishers/subscribers.
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_scene_attached")


if __name__ == "__main__":
    main()
