cmake_minimum_required(VERSION 3.16)
project(openral_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/PromptStamped.msg
  msg/ActionChunk.msg
  msg/FailureTrigger.msg
  # ADR-0096 — latched current safety state on /openral/safety_status.
  msg/SafetyStatus.msg
  # generic per-frame critic/reward score on /openral/critic/score.
  msg/CriticScore.msg
  # bounded world obstacles for the kernel's world-collision check.
  msg/WorldCollision.msg
  # voxel phase — dense fixed-capacity 3-D occupancy grid (octomap path).
  msg/OccupancyVoxels.msg
  # typed WorldState publication.
  msg/AttachedCollisionPrimitive.msg
  msg/SupportContactWitness.msg
  msg/AttachedCollisionObject.msg
  msg/AttachmentState.msg
  msg/WorldStateStamped.msg
  # dispatch's typed place-phase declaration (ADR-0097). Gates the place
  # witness; exempts nothing on its own. Its producer-measured region carries the
  # bounded approach allowance of the 2026-08-14 amendment.
  msg/PlaceRegion.msg
  msg/PlaceDeclaration.msg
  # DIAGNOSTIC-only mirror of the latest SegmentInView reply, so the dashboard's
  # mask overlay has a producer. Off by default on the segmenter node; nothing
  # in the safety or attachment path reads it.
  msg/SegmentMasks.msg
  # rosbag2 ↔ LeRobotDataset bridge.
  msg/Tick.msg
  msg/Episode.msg
  action/ExecuteRskill.action
  # Snap a HAL's MuJoCo qpos to a manifest.starting_pose before the
  # skill_runner's first inference tick (fix for pi0.5 OOD drift).
  srv/ResetToPose.srv
  # reasoner's read-only locate_in_view query against a live VLM detector.
  srv/LocateInView.srv
  # reasoner's read-only query_scene question against a scene VLM (Qwen3.5-4B).
  srv/QueryScene.srv
  # reasoner's read-only query_task_progress against the Robometer reward monitor.
  srv/QueryTaskProgress.srv
  # HAL's vision attachment-evidence producer asking a kind:"segmenter" rSkill
  # (SAM 2.1) which wrist-frame pixels are the grasped payload. Raw mono8 mask.
  srv/SegmentInView.srv
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
