"""Bucket-1 allowlist + read-only capability set for the Foxglove bridge.

Kept in an importable module (not the launch file) so the safety invariants
can be unit-tested without spawning a ROS graph. See
``launch/foxglove.launch.py`` for how these are applied.

The allowlist is assembled from four named groups so a reader can tell at a
glance *why* a topic is exposed. All four are read-only observation; the
actuation/command plane (``/openral/estop``, ``/openral/safe_action``,
``/openral/candidate_action``, ``/openral/execute_rskill``,
``/openral/prompt``…) is absent from every group on purpose, and
``test/test_foxglove_launch.py`` proves it stays that way.
"""

from __future__ import annotations

#: **Scene** — the geometry a viewer needs to draw the robot in its world.
#: These feed the 3D and Image panels natively, with no converter.
SCENE_TOPICS: list[str] = [
    r"/openral/cameras/.*/image",  # sensor_msgs/Image  — camera panels
    # ``image_transport`` compressed siblings (opt-in republisher), ~1/10th
    # the raw bandwidth; Foxglove renders natively in the Image panel.
    r"/openral/cameras/.*/image/compressed",  # sensor_msgs/CompressedImage
    r"/openral/cameras/.*/image/compressedDepth",  # sensor_msgs/CompressedImage (depth)
    # Intrinsics for the cameras above. Foxglove's Image panel needs a
    # CameraInfo to undistort, and the 3D panel needs one to draw the camera
    # frustum and project the image into the scene — neither works from the
    # Image topic alone.
    r"/openral/cameras/.*/camera_info",  # sensor_msgs/CameraInfo
    r"/map",  # nav_msgs/OccupancyGrid — 2D nav map
    r"/octomap_point_cloud_centers",  # sensor_msgs/PointCloud2 — voxels
    r"/scan",  # sensor_msgs/LaserScan — optional 2D laser
    r"/odom",  # nav_msgs/Odometry — optional trajectory
    r"/joint_states",  # sensor_msgs/JointState — joint plot/URDF
    r"/robot_description",  # std_msgs/String (URDF) — 3D robot model
    # The manifest URDF when a vendor bringup owns `/robot_description`
    # (real deploys — see `deploy_e2e.launch.py`). Same robot, same joint
    # names; it carries the manifest-only frames (`openarm_base`, sensor
    # mounts) the vendor description does not describe.
    r"/openral/robot_description",  # std_msgs/String (URDF)
    r"/tf",  # tf2_msgs/TFMessage — frames
    r"/tf_static",  # tf2_msgs/TFMessage — static frames
]

#: **Depth & reconstruction** — the RGBD / nvblox / visual-SLAM legs. Only
#: published when the matching deploy posture is on (``--enable-octomap``, a
#: ``slam_mono_camera`` scene, an nvblox graph); absent otherwise, in which
#: case the panels sit empty rather than erroring.
DEPTH_TOPICS: list[str] = [
    r"/openral/cameras/.*/depth/image",  # sensor_msgs/Image — per-camera depth
    r"/openral/cameras/.*/depth/camera_info",  # sensor_msgs/CameraInfo
    r"/openral/cameras/.*/points",  # sensor_msgs/PointCloud2 — back-projected RGBD
    r"/openral/depth/image",  # sensor_msgs/Image — DA3 metric-depth sidecar
    r"/openral/depth/camera_info",  # sensor_msgs/CameraInfo
    r"/openral/nvblox/depth_filtered/image",  # sensor_msgs/Image
    r"/openral/nvblox/depth_filtered/camera_info",  # sensor_msgs/CameraInfo
    r"/openral_nvblox/static_esdf_pointcloud",  # sensor_msgs/PointCloud2 — ESDF slice
    r"/openral/nav2/scan",  # sensor_msgs/LaserScan — nav2's remapped scan
    r"/openral/imu",  # sensor_msgs/Imu — cuVSLAM inertial input
    r"/openral/visual_slam/odometry",  # nav_msgs/Odometry — cuVSLAM pose
]

#: **Bucket-2 converter outputs** — the custom ``openral_msgs`` world types
#: re-published as standard viz types by ``bucket2_markers``
#: (``launch/bucket2.launch.py``) so Foxglove renders them natively.
#: Read-only viz, not actuation.
BUCKET2_TOPICS: list[str] = [
    r"/openral/world_collisions_markers",  # visualization_msgs/MarkerArray — capsule obstacles
    r"/openral/world_voxels_cloud",  # sensor_msgs/PointCloud2 — occupied voxel centres
]

#: **Telemetry** — the mission/state plane the OTel dashboard also renders,
#: mirrored here so a Foxglove-only operator is not blind to it. Every entry
#: is an observation topic a node publishes *about itself*; none of them
#: commands anything, and the bridge advertises no ``clientPublish``
#: capability, so a viewer can read them and nothing more.
#:
#: ``/openral/world_state_fast`` is the densest of these: one
#: ``openral_msgs/WorldStateStamped`` carries joint state, base pose/twist,
#: end-effector poses, per-component diagnostics + staleness, battery, and the
#: detected-object list — most of the dashboard's "World state" card in a
#: single message, plottable field by field.
#:
#: Deliberately NOT here: ``/openral/safety_status``. It is read-only status
#: (the latched kernel state the dashboard's "Safety · current state" card
#: reads), so exposing it would leak no actuation — but it lives in the
#: safety plane this package promises never to advertise, so it stays out
#: pending safety-WG sign-off. The dashboard keeps rendering it either way.
TELEMETRY_TOPICS: list[str] = [
    r"/openral/world_state_fast",  # openral_msgs/WorldStateStamped — 30 Hz snapshot
    r"/openral/world_state_slow",  # openral_msgs/WorldStateStamped — 5 Hz snapshot
    r"/openral/policy_state",  # std_msgs/Float32MultiArray — step-locked policy state
    r"/openral/episode",  # openral_msgs/Episode — episode start/end + success
    r"/openral/critic/score",  # openral_msgs/CriticScore — score + threshold
    r"/openral/reward/active_task",  # std_msgs/String — task the reward monitor scores
    r"/openral/perception/objects",  # openral_msgs/PromptStamped — detector ObjectsMetadata
    r"/openral/attachment_state",  # openral_msgs/AttachmentState — grasped-object set
    r"/openral/attachment_state_applied",  # openral_msgs/AttachmentState — kernel's applied copy
    r"/openral/skill_registry_changed",  # std_msgs/Empty — palette-rebuild ping
    r"/diagnostics",  # diagnostic_msgs/DiagnosticArray — Foxglove Diagnostics panels
    r"/rosout",  # rcl_interfaces/Log — Foxglove Log panel (node log stream)
]

#: Explicit allowlist of the Bucket-1 topics — the union of the groups above.
#: ``foxglove_bridge`` applies ``std::regex_match`` against the *full* topic
#: name, so each entry is anchored implicitly. Anything not listed is NOT
#: exposed — notably the safety/e-stop/action topics are absent on purpose.
BUCKET1_TOPIC_WHITELIST: list[str] = [
    *SCENE_TOPICS,
    *DEPTH_TOPICS,
    *BUCKET2_TOPICS,
    *TELEMETRY_TOPICS,
]

#: Read-only capability set. Omits ``clientPublish``, ``services``,
#: ``parameters``, ``parametersSubscribe`` from the upstream default so the
#: viewer cannot publish, call services, or write params. ``connectionGraph``
#: powers Foxglove's Topic Graph panel; ``assets`` lets the 3D panel fetch
#: ``package://`` URDF meshes (a read-only fetch).
READ_ONLY_CAPABILITIES: list[str] = ["connectionGraph", "assets"]
