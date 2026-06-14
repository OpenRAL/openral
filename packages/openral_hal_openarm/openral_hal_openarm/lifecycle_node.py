#!/usr/bin/env python3
r"""OpenArm HAL lifecycle node entry point.

Manifest-driven node (ADR-0032 / issue #191 Phase 3b): builds its HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`. The previous
bespoke ``_OpenArmLifecycleNode`` is gone — its per-robot logic is now generic
and declarative:

* **Tabletop MJCF scene composition** — declared in the manifest's
  ``scene_defaults.composition`` block; the generic node calls the composer and
  threads the composed MJCF in as the HAL's ``mjcf_path`` (decision: ADR-0029
  blocker #3).
* **Cameras** — ``OpenArmMujocoHAL.read_images()`` renders the manifest's RGB
  ``SensorSpec``s (mapping ``sim_camera_name`` → MJCF camera) and
  :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge` publishes them on
  ``/openral/cameras/<name>/image`` — the same path every scene robot uses.
* **Viewer** — ``SimSensorBridge`` (gated on ``viewer_enabled``).
* **``reset_to_pose`` service** — opened reflectively by the generic node
  (issue #191 Phase 2) at ``/openral/<robot_id>/reset_to_pose``.
* **HAL construction kwargs** (``settle_steps`` / ``gravity_enabled`` /
  ``staleness_limit_s``) — the manifest's ``hal.parameters`` (Phase 1 seam).

``openral deploy sim`` injects ``robot_yaml`` + ``hal_mode=sim``. OpenArm is
simulation-only (``hal.real`` is null), so ``deploy run`` raises
``ROSCapabilityMismatch``.

Usage::

    ros2 run openral_hal_openarm lifecycle_node \
        --ros-args -p robot_yaml:=robots/openarm/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_openarm")


if __name__ == "__main__":
    main()
