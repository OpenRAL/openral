#!/usr/bin/env python3
r"""panda_mobile HAL lifecycle node entry point.

Manifest-driven node (ADR-0032 / issue #191 Phase 3): builds its HAL via
:func:`openral_hal.lifecycle.make_lifecycle_main_from_manifest`. The previous
bespoke ``_PandaMobileLifecycleNode`` is gone — its mobile-base extras are now
generic and gated on the manifest, so adding a wheeled robot needs no subclass:

* **``/odom`` + ``odom->base_link`` TF + ``/cmd_vel``→BODY_TWIST** — handled by
  :class:`openral_hal.mobile_base_bridge.MobileBaseBridge`, attached by the
  generic node whenever the manifest declares ``base_joints``.
* **``/scan``** — handled by
  :class:`openral_hal.sim_sensor_bridge.SimSensorBridge`: live MJCF ray-cast when
  a ``SimAttachedHAL`` is bound (``openral deploy sim --config <scene>``), a
  constant no-hit fan for the in-process ``PandaMobileHAL`` digital twin.
* **cameras / depth / viewer** — ``SimSensorBridge`` (ADR-0034), as for every
  other manifest robot.
* **control modes** — the base ``_on_safe_action`` + ``decode_action_chunk``
  decode every mode; ``PandaMobileHAL`` / ``SimAttachedHAL`` ``send_action`` is
  the per-robot allowlist (no node-level override needed).

``openral deploy sim`` injects ``robot_yaml`` + ``hal_mode=sim`` (+ the scene
``sim_env_yaml`` for scene-attach). panda_mobile is simulation-only (``hal.real``
is null), so ``deploy run`` raises ``ROSCapabilityMismatch``.

Usage::

    ros2 run openral_hal_panda_mobile lifecycle_node \
        --ros-args -p robot_yaml:=robots/panda_mobile/robot.yaml -p hal_mode:=sim
"""

from __future__ import annotations

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_panda_mobile")


if __name__ == "__main__":
    main()
