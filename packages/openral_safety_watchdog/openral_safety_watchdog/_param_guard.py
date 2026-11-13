"""Parameters are fixed once configured: the watchdog nodes' copy of the kernel's guard.

The C++ safety kernel refuses every change to an existing parameter once it has
left UNCONFIGURED, and for ever after a failed configure or a cleanup
(``SafetyKernelLifecycleNode``'s ``param_guard_``, hazard-log Entry 078). The
deadman reads its deadlines on every tick, so without the same guard a
same-host graph peer could ``ros2 param set safe_action_deadline_s 86400`` and
silence the brake (issue #546).

Both nodes here latch at the *start* of ``on_configure`` and in ``on_cleanup``.
That is the kernel's rule in one flag: a configure that succeeds leaves
UNCONFIGURED (refused), one that fails latches, and so does a cleanup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["install_parameter_guard"]


def install_parameter_guard(node: Any, locked: Callable[[], bool]) -> None:
    """Refuse changes to ``node``'s existing parameters while ``locked()`` is true.

    Register after every ``declare_parameter``. Launch overrides arrive at
    declaration and are unaffected. A batch of parameters that are all
    undeclared is a declaration (rclpy runs on-set callbacks before storing a
    new parameter), which adds a parameter and relaxes none, so it passes.

    Args:
        node: The rclpy node whose parameters to guard.
        locked: Whether the node's parameters are fixed now.

    Example:
        >>> import rclpy
        >>> from rclpy.parameter import Parameter
        >>> rclpy.init()
        >>> node = rclpy.create_node("param_guard_doctest")
        >>> _ = node.declare_parameter("deadline_s", 0.2)
        >>> install_parameter_guard(node, lambda: True)
        >>> node.set_parameters([Parameter("deadline_s", value=60.0)])[0].successful
        False
        >>> node.destroy_node()
        >>> rclpy.shutdown()
    """
    from rcl_interfaces.msg import SetParametersResult

    def _guard(params: list[Any]) -> SetParametersResult:
        declarations_only = not any(node.has_parameter(p.name) for p in params)
        if declarations_only or not locked():
            return SetParametersResult(successful=True)
        names = ", ".join(p.name for p in params)
        node.get_logger().warning(f"safety.param_change_refused params=[{names}]")
        return SetParametersResult(
            successful=False,
            reason=f"{node.get_name()} parameters are fixed after configure; "
            "relaunch with new values instead",
        )

    node.add_on_set_parameters_callback(_guard)
