"""ADR-0025 — openral_slam_bringup.

Per-deployment bringup wrapper around upstream slam_toolbox. The
package ships only launch files and a default parameter YAML; no
Python runtime modules. The Reasoner drives the resulting
``LifecycleNode`` (``/openral/slam_toolbox``) via
:class:`~openral_core.LifecycleTransitionTool`.
"""

__version__ = "0.1.0"
