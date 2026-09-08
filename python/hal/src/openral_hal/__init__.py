"""openral HAL — Hardware Abstraction Layer public API.

Public surface:
- ``HAL``: structural Protocol every adapter must satisfy (RFC §8.2).
- ``LifecycleEStopHAL`` / ``ResettableLifecycleEStopHAL``: typed lifecycle
  e-stop propagation and recovery contracts.
- ``HALHealthProvider`` / ``HALHealthReport``: cached-diagnostics contract
  for the generic lifecycle heartbeat.
- ``RosControlHAL``: ros2_control-backed adapter.
- ``SO100FollowerHAL`` / ``SO100_DESCRIPTION`` / ``so100_with_sensors``
  (catalog-backed sensor loadout, issue #23); ``SO100DigitalTwin`` /
  ``SO100DigitalTwinConfig`` in-process sim; ``SO100MujocoHAL`` MuJoCo
  twin, ``mujoco_menagerie`` MJCF, same 6-DoF action layout.
- ``UR5eHAL`` / ``UR10eHAL`` (+ ``*_DESCRIPTION``, ``*_with_sensors``);
  real ``UR5eRealHAL`` / ``UR10eRealHAL`` via ``ros2_control`` +
  ``ur_robot_driver`` (URCap / RTDE).
- ``FrankaPandaHAL`` / ``FRANKA_PANDA_DESCRIPTION`` /
  ``franka_panda_with_sensors``; real ``FrankaPandaRealHAL`` over
  franka_ros2 / FCI (issue #56).
- ``SawyerRealHAL`` / ``SAWYER_DESCRIPTION`` / ``SAWYER_REAL_DESCRIPTION``
  over intera_sdk / sawyer_robot (issue #57).
- ``AlohaHAL`` / ``ALOHA_DESCRIPTION`` / ``ALOHA_REAL_DESCRIPTION`` over
  the Trossen Interbotix XS SDK (issue #58); ``AlohaMujocoHAL`` MuJoCo
  twin, gym-aloha's ``bimanual_viperx_transfer_cube.xml``, same 14-DoF
  action layout.
- ``G1MujocoHAL`` / ``G1_DESCRIPTION``: Unitree G1, 29-DoF (ADR-0087 glide
  default, optional ADR-0089 pretrained walking controller in sim);
  real-HW G1 planned M2 (CLAUDE.md §6.2).
- ``H1MujocoHAL`` / ``H1_DESCRIPTION``: Unitree H1, 19-DoF (predecessor to
  G1: 5-DoF/leg, 1-DoF torso, 4-DoF/arm); same validator scope as
  ``G1MujocoHAL``; real-HW also waits on the M2 S0 cerebellum.
- ``Rizon4MujocoHAL`` / ``RIZON4_DESCRIPTION``: Flexiv Rizon 4, 7-DoF
  cobot, whole-body force sensitivity; structurally like the UR/Franka
  sim HALs.
- ``OpenArmMujocoHAL`` / ``OPENARM_DESCRIPTION``: Enactic OpenArm v2
  bimanual, 2 x (7-DoF arm + 1 gripper) = 16-DoF; fresh ``HALBase``
  subclass (bimanual doesn't fit ``MujocoArmHAL``); native
  ``<position>`` actuators write target → ctrl directly; MJCF fetched
  lazily by ``openral_hal._openarm_v2_assets``.
- ``OpenArmRealHAL`` / ``OPENARM_REAL_DESCRIPTION``: real adapter for the
  same arm; commands the four ``openarm_bringup`` ros2_control
  controllers (per-side arm + gripper) driving Damiao CAN FD motor buses
  from C++ at 400 Hz; ``connect()`` refuses a bus that is not up.
- ``AnvilOpenArmV2MujocoHAL`` / ``ANVIL_OPENARM_V2_DESCRIPTION``: MuJoCo
  twin for the Anvil OpenARM 2.0, same 16-DoF surface; differs in J1
  clamp (+/-135 deg) and J6 radial deviation (-45..+70 deg) plus a
  visual-only wrist bracket; MJCF pinned at a SHA from
  ``bensonlee5/anvil-openarm-mujoco`` via
  ``openral_hal._anvil_openarm_v2_assets`` (``openarm:anvil_v2_bimanual``).
- ``SimTransport``: typed in-memory ros2_control transport for unit tests.
- ``GalaxeaA1HAL`` / ``GALAXEA_A1_DESCRIPTION``: real Galaxea A1 via an
  isolated ROS 1 Noetic sidecar (vendor SDK operator-provided).

Where a sim sibling exists, ``*_REAL_DESCRIPTION`` constants derive from
``openral_hal._real_description.make_real_description`` and share its
HAL entrypoints. Real-only platforms (Galaxea A1) publish one
description directly until a licensed digital-twin asset is available.
"""

from openral_hal.aloha import (
    ALOHA_DESCRIPTION,
    ALOHA_REAL_DESCRIPTION,
    AlohaHAL,
    AlohaMujocoHAL,
)
from openral_hal.anvil_openarm_v2 import ANVIL_OPENARM_V2_DESCRIPTION, AnvilOpenArmV2MujocoHAL
from openral_hal.flexiv_rizon4 import RIZON4_DESCRIPTION, Rizon4MujocoHAL
from openral_hal.franka_panda import (
    FRANKA_PANDA_DESCRIPTION,
    FrankaPandaHAL,
    franka_panda_with_sensors,
)
from openral_hal.franka_panda_real import (
    FRANKA_PANDA_REAL_DESCRIPTION,
    FrankaPandaRealHAL,
)
from openral_hal.g1 import G1_DESCRIPTION, G1MujocoHAL
from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION, GalaxeaA1HAL
from openral_hal.h1 import H1_DESCRIPTION, H1MujocoHAL
from openral_hal.openarm import OPENARM_DESCRIPTION, OpenArmMujocoHAL
from openral_hal.openarm_real import OPENARM_REAL_DESCRIPTION, OpenArmRealHAL
from openral_hal.panda_mobile import (
    PANDA_MOBILE_BASE_JOINT_NAMES,
    PANDA_MOBILE_JOINT_NAMES,
    PandaMobileHAL,
)
from openral_hal.protocol import (
    HAL,
    EStopRecovery,
    HALHealthProvider,
    HALHealthReport,
    LifecycleEStopHAL,
    ResettableLifecycleEStopHAL,
)
from openral_hal.resolver import build_hal
from openral_hal.ros_control import RosControlHAL
from openral_hal.sawyer_real import (
    SAWYER_DESCRIPTION,
    SAWYER_REAL_DESCRIPTION,
    SawyerRealHAL,
)
from openral_hal.sim_transport import SimTransport
from openral_hal.so100_follower import (
    SO100_DESCRIPTION,
    SO100FollowerHAL,
    so100_with_sensors,
)
from openral_hal.so100_mujoco import SO100MujocoHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_hal.ur import (
    UR5e_DESCRIPTION,
    UR5eHAL,
    UR10e_DESCRIPTION,
    UR10eHAL,
    ur5e_with_sensors,
    ur10e_with_sensors,
)
from openral_hal.ur_real import (
    UR5e_REAL_DESCRIPTION,
    UR5eRealHAL,
    UR10e_REAL_DESCRIPTION,
    UR10eRealHAL,
)

__all__ = [
    "ALOHA_DESCRIPTION",
    "ALOHA_REAL_DESCRIPTION",
    "ANVIL_OPENARM_V2_DESCRIPTION",
    "FRANKA_PANDA_DESCRIPTION",
    "FRANKA_PANDA_REAL_DESCRIPTION",
    "G1_DESCRIPTION",
    "GALAXEA_A1_DESCRIPTION",
    "H1_DESCRIPTION",
    "HAL",
    "OPENARM_DESCRIPTION",
    "OPENARM_REAL_DESCRIPTION",
    "PANDA_MOBILE_BASE_JOINT_NAMES",
    "PANDA_MOBILE_JOINT_NAMES",
    "RIZON4_DESCRIPTION",
    "SAWYER_DESCRIPTION",
    "SAWYER_REAL_DESCRIPTION",
    "SO100_DESCRIPTION",
    "AlohaHAL",
    "AlohaMujocoHAL",
    "AnvilOpenArmV2MujocoHAL",
    "EStopRecovery",
    "FrankaPandaHAL",
    "FrankaPandaRealHAL",
    "G1MujocoHAL",
    "GalaxeaA1HAL",
    "H1MujocoHAL",
    "HALHealthProvider",
    "HALHealthReport",
    "LifecycleEStopHAL",
    "OpenArmMujocoHAL",
    "OpenArmRealHAL",
    "PandaMobileHAL",
    "ResettableLifecycleEStopHAL",
    "Rizon4MujocoHAL",
    "RosControlHAL",
    "SO100DigitalTwin",
    "SO100DigitalTwinConfig",
    "SO100FollowerHAL",
    "SO100MujocoHAL",
    "SawyerRealHAL",
    "SimTransport",
    "UR5eHAL",
    "UR5eRealHAL",
    "UR5e_DESCRIPTION",
    "UR5e_REAL_DESCRIPTION",
    "UR10eHAL",
    "UR10eRealHAL",
    "UR10e_DESCRIPTION",
    "UR10e_REAL_DESCRIPTION",
    "build_hal",
    "franka_panda_with_sensors",
    "so100_with_sensors",
    "ur5e_with_sensors",
    "ur10e_with_sensors",
]
