"""openral_safety ROS 2 package — skeleton only.

The :class:`SafetySupervisorNode` here is a placeholder lifecycle node
that opens / closes the standard managed-lifecycle transitions. It
carries no enforcement logic — the real safety enforcer is the C++
kernel under ``cpp/openral_safety_kernel/`` (planned; CLAUDE.md §6.1
Layer 6), running as a separate process so a Python crash cannot leave
motors energised.

This package exists so the lifecycle-node naming + topic surface is
locked in before the kernel lands; PRs that add enforcement require
safety working group review per CLAUDE.md §7.7.
"""
