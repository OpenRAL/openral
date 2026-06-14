#!/usr/bin/env python3
r"""SO-100 / SO-101 HAL lifecycle node entry point.

Manifest-driven node (ADR-0032 / issue #191 Phase 2): builds its sim or real
HAL via :func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`, which
reads the ``robot_yaml`` + ``hal_mode`` ROS parameters and routes through
:func:`openral_hal.build_hal`. The previous bespoke ``_SO100LifecycleNode``
(``port`` / ``calibrate_on_connect`` / ``sim_robot_yaml`` parameters) is gone:

* **sim** (``hal_mode:=sim``) → a bare MuJoCo digital twin derived from the
  manifest's ``sim:`` block (``MujocoArmHAL.from_description``). The SAME node
  serves both the SO-100 (``so_arm100``) and the SO-101 (``so101_new_calib``)
  from their own ``robots/<id>/robot.yaml`` — no dedicated ``openral_hal_so101``
  package. ``openral deploy sim`` injects the resolved manifest (see
  ``openral_cli.deploy_sim._ROBOT_HAL_REGISTRY``: ``manifest_driven=True`` +
  ``bare_twin_sim=True``).
* **real** (``hal_mode:=real``) → ``SO100FollowerHAL`` over the Feetech serial
  bus. The serial ``port`` + ``calibrate_on_connect`` come from the manifest's
  ``hal.parameters.defaults`` (ADR-0029), threaded into the constructor by
  ``build_hal`` — so no per-robot ROS parameter is needed.

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
