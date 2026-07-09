"""Panda MoveIt demo with NVIDIA Isaac ROS cuMotion as a planning pipeline.

This mirrors the upstream ``moveit_resources_panda_moveit_config`` demo launch,
but adds the ``isaac_ros_cumotion`` MoveIt pipeline and starts the cuMotion
action server against NVIDIA's shipped Franka XRDF. It is intentionally kept
under ``tests/integration`` so the live cuMotion e2e can launch it by file path
without adding a new ROS package.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _cumotion_planning_config() -> dict:
    config_path = os.path.join(
        get_package_share_directory("isaac_ros_cumotion_moveit"),
        "config",
        "isaac_ros_cumotion_planning.yaml",
    )
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_launch_description() -> LaunchDescription:
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "chomp", "pilz_industrial_motion_planner", "stomp"])
        .to_moveit_configs()
    )
    moveit_params = moveit_config.to_dict()
    moveit_params["planning_pipelines"] = [
        *moveit_params["planning_pipelines"],
        "isaac_ros_cumotion",
    ]
    moveit_params["default_planning_pipeline"] = "isaac_ros_cumotion"
    moveit_params["isaac_ros_cumotion"] = _cumotion_planning_config()

    cumotion_share = get_package_share_directory("isaac_ros_cumotion")
    cumotion_robot_share = get_package_share_directory("isaac_ros_cumotion_robot_description")
    panda_description_share = get_package_share_directory("moveit_resources_panda_description")

    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cumotion_share, "launch", "isaac_ros_cumotion.launch.py")
        ),
        launch_arguments={
            "cumotion_action_server.urdf_file_path": os.path.join(
                panda_description_share, "urdf", "panda.urdf"
            ),
            "cumotion_action_server.xrdf_file_path": os.path.join(
                cumotion_robot_share, "xrdf", "franka.xrdf"
            ),
            "cumotion_action_server.read_esdf_world": "false",
            "cumotion_action_server.update_esdf_on_request": "false",
            "cumotion_action_server.publish_world_collision_spheres": "false",
            "cumotion_action_server.publish_self_collision_spheres": "false",
        }.items(),
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
        arguments=["--ros-args", "--log-level", "info"],
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )
    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )
    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
    )

    return LaunchDescription(
        [
            cumotion_launch,
            static_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
        ]
    )
