"""Bucket-1 allowlist + read-only capability set for the Foxglove bridge.

Kept in an importable module (not the launch file) so the safety invariants
can be unit-tested without spawning a ROS graph. See
``launch/foxglove.launch.py`` for how these are applied.
"""

from __future__ import annotations

#: Explicit allowlist of the Bucket-1 topics. ``foxglove_bridge`` applies
#: ``std::regex_match`` against the *full* topic name, so each entry is
#: anchored implicitly. Anything not listed is NOT exposed — notably the
#: safety/e-stop/action topics are absent on purpose.
BUCKET1_TOPIC_WHITELIST: list[str] = [
    r"/openral/cameras/.*/image",  # sensor_msgs/Image  — camera panels
    r"/map",  # nav_msgs/OccupancyGrid — 2D nav map
    r"/octomap_point_cloud_centers",  # sensor_msgs/PointCloud2 — voxels
    r"/scan",  # sensor_msgs/LaserScan — optional 2D laser
    r"/odom",  # nav_msgs/Odometry — optional trajectory
    r"/joint_states",  # sensor_msgs/JointState — joint plot/URDF
    r"/robot_description",  # std_msgs/String (URDF) — 3D robot model
    r"/tf",  # tf2_msgs/TFMessage — frames
    r"/tf_static",  # tf2_msgs/TFMessage — static frames
]

#: Read-only capability set. Omits ``clientPublish``, ``services``,
#: ``parameters``, ``parametersSubscribe`` from the upstream default so the
#: viewer cannot publish, call services, or write params. ``connectionGraph``
#: powers Foxglove's Topic Graph panel; ``assets`` lets the 3D panel fetch
#: ``package://`` URDF meshes (a read-only fetch).
READ_ONLY_CAPABILITIES: list[str] = ["connectionGraph", "assets"]
