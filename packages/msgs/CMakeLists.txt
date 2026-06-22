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
  # ADR-0064 — generic per-frame critic/reward score on /openral/critic/score.
  msg/CriticScore.msg
  # ADR-0030 — bounded world obstacles for the kernel's world-collision check.
  msg/WorldCollision.msg
  # ADR-0030 voxel phase — dense fixed-capacity 3-D occupancy grid (octomap path).
  msg/OccupancyVoxels.msg
  # ADR-0018 F2 — typed WorldState publication.
  msg/WorldStateStamped.msg
  # ADR-0018 PR3 — rosbag2 ↔ LeRobotDataset bridge.
  msg/Tick.msg
  msg/Episode.msg
  action/ExecuteRskill.action
  # Snap a HAL's MuJoCo qpos to a manifest.starting_pose before the
  # skill_runner's first inference tick (fix for pi0.5 OOD drift).
  srv/ResetToPose.srv
  # ADR-0043 — reasoner's read-only locate_in_view query against a live VLM detector.
  srv/LocateInView.srv
  # ADR-0047 — reasoner's read-only query_scene question against a scene VLM (Qwen3.5-4B).
  srv/QueryScene.srv
  # ADR-0057 — reasoner's read-only query_task_progress against the Robometer reward monitor.
  srv/QueryTaskProgress.srv
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
