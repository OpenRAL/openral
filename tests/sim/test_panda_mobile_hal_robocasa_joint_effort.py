"""Deploy-sim ``/joint_states.effort`` carries the real actuator torque (#373).

``SimAttachedHAL.read_state`` published ``effort=[0.0]*n`` on the RoboCasa
deploy path even though it holds MuJoCo's ``(model, data)``. A torque observer
(the #373 force/contact layer) reading that sees a weightless arm. Physical
check: an arm the controller holds still balances gravity, so its effort must
equal MuJoCo's gravity/bias load ``qfrc_bias`` on every arm joint.

No mocks (CLAUDE.md §1.11): the real deploy scene, real bring-up, a real kitchen.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openral_sim")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)


@pytest.mark.sim
def test_held_arm_effort_balances_gravity() -> None:
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml("scenes/deploy/robocasa_pnp.yaml")
    try:
        desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
        hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
        hal.connect()
        zero = np.zeros(env.action_dim, dtype=np.float32)
        for _ in range(10):  # let the OSC hold settle
            env.step(zero)
        state = hal.read_state()
        model, data = hal.mujoco_handles()
        arm = [i for i, j in enumerate(desc.joints) if j.name.startswith("panda_joint")]
        dofs = [
            int(
                model.jnt_dofadr[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, desc.joints[i].sim_joint_name
                    )
                ]
            )
            for i in arm
        ]
        effort = np.asarray(state.effort)[arm]
        np.testing.assert_allclose(effort, data.qfrc_bias[dofs], atol=0.1)
        # Not trivially zero: the elbow (joint 4) carries the forearm (~17 N*m here).
        assert np.max(np.abs(effort)) > 5.0
    finally:
        env.close()
