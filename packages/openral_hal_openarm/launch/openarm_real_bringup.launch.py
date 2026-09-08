r"""Real-hardware ros2_control bringup for the OpenArm v2 bimanual arm.

``OpenArmRealHAL`` (``python/hal/src/openral_hal/openarm_real.py``) only speaks to four
already-running ``ros2_control`` controllers over topics — same pattern as every other real-HW
HAL in this repo (``FrankaPandaRealHAL`` assumes ``franka_ros2``'s ``franka.launch.py`` is
already up). It never has and never will start ``controller_manager`` itself: that graph is C++,
runs at 400 Hz+, and belongs under a vendor bringup launch, not under a Python HAL (CLAUDE.md
§1.5).

Before this file, nothing in this repo started that graph on real CAN hardware: an operator had
to bring it up out-of-band from a separate workspace, and OpenRAL simply assumed it was there.
This file is a standalone include of upstream's ``openarm_bringup``, so
``ros2 launch openral_hal_openarm openarm_real_bringup.launch.py`` is sufficient on a provisioned
rig — no other repo's overlay needs to be sourced.

Provisioning ``openarm_bringup`` itself (the vendor package this include resolves via
``FindPackageShare``) is a one-time host setup step, documented in this package's ``README.md``
next to the CAN xacro patch under ``patches/`` — the same "vendor SDK lives outside the repo,
OpenRAL only ships the glue" posture as the Franka/UR real adapters.

Defaults match ``OpenArmRealHAL``'s constants exactly (``_LEFT_CAN_INTERFACE`` /
``_RIGHT_CAN_INTERFACE`` = ``openarm_left`` / ``openarm_right``, the udev names the OpenArm CAN
setup assigns) so a bare ``ros2 launch openral_hal_openarm openarm_real_bringup.launch.py`` and
the HAL agree without any argument overrides; ``tests/hil/test_openarm_bringup_agreement.py``
guards the controller/joint side of that agreement, and
``tests/unit/test_openarm_real_bringup_launch.py`` guards this file's defaults against drifting
from those constants.

Usage::

    ros2 launch openral_hal_openarm openarm_real_bringup.launch.py \
        left_can_interface:=openarm_left right_can_interface:=openarm_right
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# Must match `openral_hal.openarm_real._LEFT_CAN_INTERFACE` / `_RIGHT_CAN_INTERFACE` — the udev
# names the OpenArm CAN setup assigns to the USB CAN-FD adapter's two channels.
_LEFT_CAN_INTERFACE = "openarm_left"
_RIGHT_CAN_INTERFACE = "openarm_right"

# Must match the controller type `OpenArmRealHAL` publishes `JointTrajectory` messages to
# (`left_joint_trajectory_controller` / `right_joint_trajectory_controller`), not
# `forward_position_controller` (which takes `Float64MultiArray`, not `JointTrajectory`).
_ROBOT_CONTROLLER = "joint_trajectory_controller"


def generate_launch_description() -> LaunchDescription:
    """Include upstream `openarm_bringup`'s real-hardware bimanual launch."""
    real_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("openarm_bringup"), "launch", "openarm.bimanual.launch.py"]
            )
        ),
        launch_arguments={
            "use_fake_hardware": "false",
            "right_can_interface": LaunchConfiguration("right_can_interface"),
            "left_can_interface": LaunchConfiguration("left_can_interface"),
            "robot_controller": _ROBOT_CONTROLLER,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_can_interface",
                default_value=_LEFT_CAN_INTERFACE,
                description="SocketCAN interface for the left arm's motor bus.",
            ),
            DeclareLaunchArgument(
                "right_can_interface",
                default_value=_RIGHT_CAN_INTERFACE,
                description="SocketCAN interface for the right arm's motor bus.",
            ),
            real_bringup,
        ]
    )
