"""Scripted reach-and-grasp policy for the SO-100 lift env.

Used by the end-to-end test under ``tests/sim/`` and the
``examples/so100_robosuite_lift.py`` demo to prove the full robosuite
plumbing — robot model, gripper model, OSC controller wiring, env
physics — is correct. The policy is intentionally simple (it just
emits Cartesian deltas for robosuite's stock OSC_POSITION controller
to track), NOT a VLA: this lets the test assert on "attempts to
grasp" without depending on a trained checkpoint or GPU.

The state machine has four phases:

1. **Approach**: drive the eef toward a pre-grasp pose ~4 cm above
   the block, gripper open.
2. **Descend**: drop straight down onto the block centre.
3. **Close**: hold pose, ramp the gripper command to fully closed.
4. **Lift**: command an upward Cartesian delta with the gripper held
   closed.

All four phases use the same primitive — clip ``(target - eef) /
output_max`` to [-1, 1] and hand it to OSC_POSITION, which does the
Cartesian-to-joint inverse dynamics. No grid search, no Jacobian
glue: robosuite's controller already owns that surface.

Phase transitions are time-bounded: each phase has a max-step budget
so a stuck phase still hands control to the next one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["PolicyTelemetry", "ScriptedPickPolicy"]


@dataclass
class PolicyTelemetry:
    """Per-step diagnostics emitted by :meth:`ScriptedPickPolicy.step`.

    Useful for tests to assert on intermediate progress without
    re-deriving FK from the observation dict.

    Attributes:
        phase: Current state-machine label.
        eef_to_cube_distance_m: L2 distance from gripper site to cube
            centre at the start of this step.
        gripper_command: Action value sent to the gripper actuator.
        cartesian_delta: Per-step Cartesian command sent to OSC
            (normalised to [-1, 1] per axis).
        cube_height_m: Cube z above the table surface (positive when
            lifted).
    """

    phase: str
    eef_to_cube_distance_m: float
    gripper_command: float
    cartesian_delta: NDArray[np.float64]
    cube_height_m: float


@dataclass
class ScriptedPickPolicy:
    """Hand-tuned Cartesian reach-and-grasp for the SO-100 lift env.

    Args:
        approach_height_m: Vertical offset above the block for the
            pre-grasp pose. 4 cm clears the block + gripper geometry.
        lift_height_m: Vertical offset above the block's resting pose
            for the lift target. 6 cm gives clear separation from the
            table once grasped.
        approach_steps: Step budget for phase 1 (approach).
        descend_steps: Step budget for phase 2 (descend).
        close_steps: Step budget for phase 3 (gripper close).
        lift_steps: Step budget for phase 4 (lift).
        cartesian_step_m: Maximum unscaled Cartesian step per OSC
            cycle, in metres. Must match the env's ``output_max`` so
            the normalised action lands in [-1, 1]; deviating only
            scales the controller gain, which the SO-100 doesn't
            tolerate well.
    """

    approach_height_m: float = 0.05
    lift_height_m: float = 0.06
    approach_steps: int = 60
    descend_steps: int = 60
    close_steps: int = 40
    lift_steps: int = 80
    cartesian_step_m: float = 0.01

    _phase: str = field(default="approach", init=False)
    _phase_step: int = field(default=0, init=False)
    _gripper_cmd: float = field(default=1.0, init=False)
    _initial_cube_pos: NDArray[np.float64] | None = field(default=None, init=False)

    def reset(self) -> None:
        """Re-arm the state machine for a fresh episode."""
        self._phase = "approach"
        self._phase_step = 0
        self._gripper_cmd = 1.0
        self._initial_cube_pos = None

    def step(self, env: Any, obs: dict[str, Any]) -> tuple[NDArray[np.float64], PolicyTelemetry]:
        """Produce one ``env.action_dim`` command + telemetry.

        Args:
            env: The :class:`openral_sim.backends.so100_robosuite.env._So100Lift`
                instance. Read for table geometry (telemetry only — the
                action itself is purely a function of ``obs``).
            obs: The robosuite observation dict from the previous
                ``env.step``. Reads ``cube_pos`` and ``robot0_eef_pos``.

        Returns:
            Tuple of ``(action, telemetry)`` where ``action`` is a
            ``(3 + 1,)`` numpy array — three OSC_POSITION Cartesian
            deltas (normalised to [-1, 1]) + one gripper command —
            and ``telemetry`` records the policy's current phase plus
            a few derived scalars.
        """
        cube_pos = np.asarray(obs["cube_pos"], dtype=np.float64)
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        table_z = float(env.model.mujoco_arena.table_offset[2])
        cube_height = float(cube_pos[2] - table_z - env.table_full_size[2] / 2.0)

        # Latch the cube pose on the first step so the lift target
        # stays fixed even after we've nudged the block during the
        # close phase.
        if self._initial_cube_pos is None:
            self._initial_cube_pos = cube_pos.copy()

        # Gripper convention note: the SO-100's Jaw joint range is
        # ``[-0.174, 0.5]`` where the LOWER bound is closed (jaws
        # touching) and the upper bound is open. SimpleGripController
        # ("GRIP") scales the gripper command linearly to motor torque
        # around 0, so a **positive** gripper command opens the jaw
        # and a **negative** one closes it.
        open_cmd = 1.0
        closed_cmd = -1.0

        if self._phase == "approach":
            # Hover the eef above the (latched) block centre with the
            # jaw open.
            target = self._initial_cube_pos + np.array([0.0, 0.0, self.approach_height_m])
            self._gripper_cmd = open_cmd
            if self._phase_step >= self.approach_steps:
                self._advance("descend")
        elif self._phase == "descend":
            # Drop straight down to block centre.
            target = self._initial_cube_pos.copy()
            self._gripper_cmd = open_cmd
            if self._phase_step >= self.descend_steps:
                self._advance("close")
        elif self._phase == "close":
            # Hold pose, ramp the gripper closed over the first half
            # of the window so contact builds gradually.
            target = self._initial_cube_pos.copy()
            t = min(1.0, self._phase_step / max(1, self.close_steps // 2))
            self._gripper_cmd = open_cmd + (closed_cmd - open_cmd) * t
            if self._phase_step >= self.close_steps:
                self._advance("lift")
        else:  # "lift"
            target = self._initial_cube_pos + np.array([0.0, 0.0, self.lift_height_m])
            self._gripper_cmd = closed_cmd

        # Cartesian delta normalised to the controller's [-1, 1] input
        # range. OSC_POSITION's ``output_max`` (set in
        # :func:`so100_osc_controller_config`) maps the normalised
        # command back to metres.
        raw_delta = target - eef_pos
        normalised = np.clip(raw_delta / self.cartesian_step_m, -1.0, 1.0)
        action = np.concatenate([normalised.astype(np.float64), [self._gripper_cmd]])

        self._phase_step += 1
        telemetry = PolicyTelemetry(
            phase=self._phase,
            eef_to_cube_distance_m=float(np.linalg.norm(eef_pos - cube_pos)),
            gripper_command=float(self._gripper_cmd),
            cartesian_delta=normalised,
            cube_height_m=cube_height,
        )
        return action, telemetry

    def _advance(self, next_phase: str) -> None:
        self._phase = next_phase
        self._phase_step = 0
