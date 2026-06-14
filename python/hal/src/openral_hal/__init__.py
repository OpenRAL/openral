"""openral HAL — Hardware Abstraction Layer public API.

Public surface:
- ``HAL``: structural Protocol every adapter must satisfy (RFC §8.2).
- ``RosControlHAL``: ros2_control-backed adapter.
- ``SO100FollowerHAL``: lerobot SO-100 follower arm adapter.
- ``SO100_DESCRIPTION`` / ``so100_with_sensors``: canonical SO-100 description
  and a catalog-backed factory that resolves a sensor loadout (issue #23).
- ``SO100DigitalTwin`` / ``SO100DigitalTwinConfig``: in-process simulator.
- ``UR5eHAL`` / ``UR5e_DESCRIPTION`` / ``ur5e_with_sensors``.
- ``UR10eHAL`` / ``UR10e_DESCRIPTION`` / ``ur10e_with_sensors``.
- ``UR5eRealHAL`` / ``UR5e_REAL_DESCRIPTION``: real-hardware UR5e via
  ``ros2_control`` + ``ur_robot_driver`` (URCap / RTDE).
- ``UR10eRealHAL`` / ``UR10e_REAL_DESCRIPTION``: real-hardware UR10e via
  the same driver.
- ``FrankaPandaHAL`` / ``FRANKA_PANDA_DESCRIPTION`` / ``franka_panda_with_sensors``.
- ``FrankaPandaRealHAL`` / ``FRANKA_PANDA_REAL_DESCRIPTION``: real-hardware
  adapter over franka_ros2 / FCI (issue #56).
- ``SawyerRealHAL`` / ``SAWYER_DESCRIPTION`` / ``SAWYER_REAL_DESCRIPTION``:
  real-hardware adapter over intera_sdk / sawyer_robot (issue #57).
- ``AlohaHAL`` / ``ALOHA_DESCRIPTION`` / ``ALOHA_REAL_DESCRIPTION``:
  real-hardware adapter over the Trossen Interbotix XS SDK (issue #58).
- ``AlohaMujocoHAL``: MuJoCo-backed digital twin for the bimanual ALOHA,
  driving gym-aloha's ``bimanual_viperx_transfer_cube.xml`` with the same
  14-DoF action layout as ``AlohaHAL``.
- ``SO100MujocoHAL``: MuJoCo-backed digital twin for the SO-100 follower,
  driving the ``mujoco_menagerie`` MJCF with the same 6-DoF action layout
  as ``SO100FollowerHAL``.
- ``G1MujocoHAL`` / ``G1_DESCRIPTION``: MuJoCo-backed digital twin for the
  Unitree G1 humanoid (29-DoF, no S0 cerebellum — contract validator only;
  the robot falls without gravity disabled).  Real-HW G1 HAL is planned
  under the M2 milestone (CLAUDE.md §6.2).
- ``H1MujocoHAL`` / ``H1_DESCRIPTION``: MuJoCo-backed digital twin for the
  Unitree H1 humanoid (19-DoF — predecessor to the G1 with a simpler 5-DoF
  per leg, 1-DoF torso, 4-DoF per arm layout).  Same contract-validator
  scope as ``G1MujocoHAL``; real-HW H1 HAL also waits on the M2 S0
  cerebellum.
- ``Rizon4MujocoHAL`` / ``RIZON4_DESCRIPTION``: MuJoCo-backed digital twin
  for the Flexiv Rizon 4 (7-DoF cobot with whole-body force sensitivity).
  Structurally identical to the UR / Franka sim HALs.
- ``OpenArmMujocoHAL`` / ``OPENARM_DESCRIPTION``: MuJoCo-backed digital
  twin for the Enactic OpenArm v2 bimanual (2 x (7-DoF arm + 1 gripper) =
  16-DoF action).  Fresh ``HALBase`` subclass because the bimanual
  layout doesn't fit ``MujocoArmHAL``, but otherwise trivial — v2's
  native ``<position>`` actuators (per-class PD baked into the MJCF)
  let the HAL just write target → ctrl and step.  The v2 MJCF is
  fetched lazily by ``openral_hal._openarm_v2_assets``; will simplify
  back to ``robot_descriptions`` once upstream bumps its pin.
- ``SimTransport``: typed in-memory ros2_control transport for unit tests.

All ``*_REAL_DESCRIPTION`` constants are derived from their sim siblings via
``openral_hal._real_description.make_real_description``; they share the same
``hal`` entrypoints (``hal.sim`` / ``hal.real``, ADR-0031) and differ only in
``sdk_kind``.
"""

from openral_hal.aloha import (
    ALOHA_DESCRIPTION,
    ALOHA_REAL_DESCRIPTION,
    AlohaHAL,
    AlohaMujocoHAL,
)
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
from openral_hal.h1 import H1_DESCRIPTION, H1MujocoHAL
from openral_hal.openarm import OPENARM_DESCRIPTION, OpenArmMujocoHAL
from openral_hal.panda_mobile import (
    PANDA_MOBILE_BASE_JOINT_NAMES,
    PANDA_MOBILE_JOINT_NAMES,
    PandaMobileHAL,
)
from openral_hal.protocol import HAL
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
    "FRANKA_PANDA_DESCRIPTION",
    "FRANKA_PANDA_REAL_DESCRIPTION",
    "G1_DESCRIPTION",
    "H1_DESCRIPTION",
    "HAL",
    "OPENARM_DESCRIPTION",
    "PANDA_MOBILE_BASE_JOINT_NAMES",
    "PANDA_MOBILE_JOINT_NAMES",
    "RIZON4_DESCRIPTION",
    "SAWYER_DESCRIPTION",
    "SAWYER_REAL_DESCRIPTION",
    "SO100_DESCRIPTION",
    "AlohaHAL",
    "AlohaMujocoHAL",
    "FrankaPandaHAL",
    "FrankaPandaRealHAL",
    "G1MujocoHAL",
    "H1MujocoHAL",
    "OpenArmMujocoHAL",
    "PandaMobileHAL",
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
