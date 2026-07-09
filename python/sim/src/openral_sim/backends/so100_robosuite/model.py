"""robosuite-side robot + gripper models for the Hugging Face SO-100 follower.

The SO-100 is a 5-DOF revolute arm + 1-DOF parallel jaw, built around
Feetech servos. This module wires it into robosuite's
``ManipulatorModel`` / ``GripperModel`` plumbing so any robosuite env
(``Lift``, ``PickPlace``, ``Stack``, …) can use it via
``robots=["SO100"]`` exactly like any other arm in the registry.

The MJCF used here is generated from the upstream DeepMind
``mujoco_menagerie`` SO-100 description by :mod:`._assets` — see that
module for the rewrite details (actuator type, body hierarchy, eef
sites).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

from openral_sim.backends.so100_robosuite._assets import ensure_so100_assets

__all__ = ["SO100", "SO100Gripper"]


class SO100Gripper(GripperModel):  # type: ignore[misc]  # reason: robosuite has no type stubs
    """Parallel-jaw gripper for the Hugging Face SO-100.

    1 DOF (``Jaw``), driven by a single torque actuator. The
    :class:`robosuite.controllers.parts.gripper.simple_grip.SimpleGripController`
    (registered as ``"GRIP"``) maps a scalar in [-1, 1] to ± full
    actuator torque; positive closes the jaw, negative opens it.
    """

    def __init__(self, idn: int | str = 0) -> None:
        assets = ensure_so100_assets()
        super().__init__(str(assets.gripper_xml), idn=idn)

    def format_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Pass through the single scalar gripper command.

        The composite controller already routed [-1, 1] → torque via
        SimpleGripController; we only need to clip to that nominal
        range.
        """
        assert len(action) == self.dof, (
            f"SO100Gripper expects {self.dof} command(s), got {len(action)}"
        )
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    @property
    def init_qpos(self) -> NDArray[np.float64]:
        """Jaw starts wide open (~0.05 rad past the lower limit)."""
        return np.array([0.0])

    @property
    def speed(self) -> float:
        return 0.05

    @property
    def dof(self) -> int:
        return 1

    @property
    def _important_geoms(self) -> dict[str, list[str]]:
        """Fingerpad geom groups used by ``_check_grasp`` in the env.

        Both jaws live in the gripper XML (see :func:`_assets._write_gripper_xml`
        for why the static fixed jaw moved out of the arm), so both
        fingerpad groups resolve cleanly under the gripper's naming
        prefix.
        """
        return {
            "left_finger": [
                "fixed_jaw_pad_1",
                "fixed_jaw_pad_2",
                "fixed_jaw_pad_3",
                "fixed_jaw_pad_4",
            ],
            "right_finger": [
                "moving_jaw_pad_1",
                "moving_jaw_pad_2",
                "moving_jaw_pad_3",
                "moving_jaw_pad_4",
            ],
            "left_fingerpad": [
                "fixed_jaw_pad_3",
                "fixed_jaw_pad_4",
            ],
            "right_fingerpad": [
                "moving_jaw_pad_3",
                "moving_jaw_pad_4",
            ],
        }


# Register so the gripper_factory can find it by name (e.g. when
# ``robots=["SO100"]`` triggers a load via ``default_gripper``).
register_gripper(SO100Gripper)


class SO100(ManipulatorModel):  # type: ignore[misc]  # reason: robosuite has no type stubs
    """Hugging Face SO-100 follower — 5-DOF revolute arm.

    The 6-DOF jaw motor moves to :class:`SO100Gripper`. The arm itself
    exposes 5 motor actuators (Rotation, Pitch, Elbow, Wrist_Pitch,
    Wrist_Roll) and a ``right_hand`` body inside the menagerie's
    ``Fixed_Jaw`` link, where the gripper merges.

    Args:
        idn: Robot-instance identifier; passed through to ``ManipulatorModel``.
    """

    arms = ["right"]

    def __init__(self, idn: int | str = 0) -> None:
        assets = ensure_so100_assets()
        super().__init__(str(assets.robot_xml), idn=idn)

    @property
    def default_base(self) -> str:
        """No physical mount — the SO-100 bolts directly to the desk."""
        return "NullMount"

    @property
    def default_gripper(self) -> dict[str, str]:
        return {"right": "SO100Gripper"}

    @property
    def default_controller_config(self) -> dict[str, str]:
        # We ship our own composite controller in :mod:`.env`; this just
        # keeps the base class happy if someone instantiates the model
        # standalone.
        return {"right": "default_so100"}

    @property
    def init_qpos(self) -> NDArray[np.float64]:
        """Home pose — arm folded compactly, tip near the table.

        Matches the menagerie's ``home`` keyframe minus the Jaw column:
        ``[0, -1.57, 1.57, 1.57, -1.57]``.
        """
        return np.array([0.0, -1.57, 1.57, 1.57, -1.57])

    @property
    def base_xpos_offset(self) -> dict[str, object]:
        """Robot base offsets per arena (world coords).

        The SO-100 has only ~25 cm of horizontal reach (much smaller than
        the Panda's ~85 cm). The Panda's stock table offset puts the
        robot 16 cm behind the table edge — that's already 56 cm from a
        40 cm-table's centre, well past the SO-100's workspace. We
        instead sit it **at** the back edge of the table so a cube at
        the table centre is ≈ 20 cm away.
        """

        def _table(table_length: float) -> tuple[float, float, float]:
            return (-table_length / 2.0, 0.0, 0.0)

        return {
            "bins": (-0.2, 0.0, 0.0),
            "empty": (-0.1, 0.0, 0.0),
            "table": _table,
        }

    @property
    def top_offset(self) -> NDArray[np.float64]:
        return np.array([0.0, 0.0, 0.25])

    @property
    def _horizontal_radius(self) -> float:
        return 0.28

    @property
    def arm_type(self) -> str:
        return "single"


# Make the model discoverable via ``robosuite.make(robots=["SO100"])``.
# The ManipulatorModel metaclass already calls register_robot in
# ``__init_subclass__``, but we call it explicitly so a re-import (e.g.
# under pytest re-collection) is idempotent.
register_robot(SO100)

# robosuite's ``ROBOT_CLASS_MAPPING`` is a hand-maintained dict mapping
# robot-model name → high-level Robot subclass (FixedBaseRobot,
# WheeledRobot, etc.). The manipulation env's ``_check_robot_configuration``
# looks the SO-100 up here, so we record it as a fixed-base manipulator.
ROBOT_CLASS_MAPPING.setdefault("SO100", FixedBaseRobot)
