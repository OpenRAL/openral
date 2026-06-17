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
    # ADR-0059 decision 4: ``image_transport`` compressed sibling topics.
    # When the opt-in republisher is active these carry sensor_msgs/CompressedImage
    # at ~1/10th the raw bandwidth; Foxglove renders them natively in the Image panel.
    r"/openral/cameras/.*/image/compressed",  # sensor_msgs/CompressedImage
    r"/openral/cameras/.*/image/compressedDepth",  # sensor_msgs/CompressedImage (depth)
    r"/map",  # nav_msgs/OccupancyGrid — 2D nav map
    r"/octomap_point_cloud_centers",  # sensor_msgs/PointCloud2 — voxels
    r"/scan",  # sensor_msgs/LaserScan — optional 2D laser
    r"/odom",  # nav_msgs/Odometry — optional trajectory
    r"/joint_states",  # sensor_msgs/JointState — joint plot/URDF
    r"/robot_description",  # std_msgs/String (URDF) — 3D robot model
    r"/tf",  # tf2_msgs/TFMessage — frames
    r"/tf_static",  # tf2_msgs/TFMessage — static frames
    # ADR-0059 decision 5: Bucket-2 converter outputs — the custom
    # openral_msgs world types re-published as standard viz types by
    # ``bucket2_markers`` (launch/bucket2.launch.py) so Foxglove renders
    # them natively. Read-only viz, not actuation.
    r"/openral/world_collisions_markers",  # visualization_msgs/MarkerArray — capsule obstacles
    r"/openral/world_voxels_cloud",  # sensor_msgs/PointCloud2 — occupied voxel centres
]

#: Read-only capability set. Omits ``clientPublish``, ``services``,
#: ``parameters``, ``parametersSubscribe`` from the upstream default so the
#: viewer cannot publish, call services, or write params. ``connectionGraph``
#: powers Foxglove's Topic Graph panel; ``assets`` lets the 3D panel fetch
#: ``package://`` URDF meshes (a read-only fetch).
READ_ONLY_CAPABILITIES: list[str] = ["connectionGraph", "assets"]
