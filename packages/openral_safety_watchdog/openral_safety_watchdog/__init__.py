"""ROS 2 reasoner + supervisor graph spec §5 defense-in-depth watchdog nodes.

* :mod:`.deadman_watchdog_node` — fires ``/openral/estop`` when
  ``/openral/safe_action`` goes silent past a deadline.
* :mod:`.hardware_estop_node` — bridges a GPIO relay or USB pendant
  onto ``/openral/estop``.

Both are independent of the C++ safety kernel: a kernel
crash still brakes the robot because these nodes run as separate ROS 2
lifecycle processes.
"""
