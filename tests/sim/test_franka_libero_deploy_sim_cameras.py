"""LIBERO scene loads via the deploy-sim HAL path; cameras + joint state real (ADR-0034).

Resolves spec risk #1 (the robosuite ``robot0_joint*`` mapping) and the camera-key
fix (scene obs keyed ``camera1``/``camera2``, mapped to the ``agentview``/``wrist``
topics by ``SimSensorBridge`` at publish time) on a REAL LIBERO SimScene. Drives
raw HAL reads (no VLA) so it is GPU-independent for inference.

Environment gate: the LIBERO backend pulls ``lerobot[libero]`` → robosuite 1.4.0,
which conflicts with RoboCasa's robosuite>=1.5 (ADR-0011) — the two cannot coexist
in one venv. So this test ``importorskip``s ``libero`` and runs only on a
LIBERO-provisioned env / CI, never alongside the robocasa group.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("libero")  # ADR-0011: libero (robosuite 1.4) ⊥ robocasa (robosuite >=1.5)

from openral_core import RobotDescription
from openral_hal import build_hal

_FRANKA = "robots/franka_panda/robot.yaml"
_LIBERO_SIM = "scenes/sim/libero_spatial.yaml"
# A standard LIBERO suite scene (routes to openral_sim.backends.libero._LiberoSim).
# Regression guard for the `_LiberoSim.action_dim` probe gap that broke
# SimAttachedHAL.connect() on suite scenes.
_SUITE = _LIBERO_SIM


def test_franka_libero_scene_attach_state_and_images() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim", sim_env_yaml=_LIBERO_SIM)
    hal.connect()
    try:
        # Joint state: 8 DoF (7 arm + gripper). The LIBERO robosuite model names
        # joints ``robot0_joint*``; franka's sim_joint_name=``joint*`` + the
        # robosuite-prefix-strip fallback resolves them — so the arm positions are
        # real (non-zero after reset), not the all-zero "joint not found" fallback.
        state = hal.read_state()
        assert len(state.position) == 8
        assert any(abs(p) > 1e-9 for p in state.position[:7]), (
            "all arm joints read 0.0 — robosuite robot0_ joint mapping failed"
        )

        # Camera frames: the LIBERO backend keys obs['images'] by the VLA camera
        # slot (camera1/camera2), NOT the manifest sensor name. SimSensorBridge maps
        # franka's agentview->camera1 / wrist->camera2 (vla_feature_key) at publish
        # time; here at the HAL level we assert the scene obs keys + shape.
        images = hal.read_images()
        assert {"camera1", "camera2"} <= set(images), f"missing camera frames: {sorted(images)}"
        assert images["camera1"].shape == (256, 256, 3)
        assert images["camera2"].shape == (256, 256, 3)
    finally:
        hal.disconnect()


def test_franka_libero_suite_scene_attach_resolves_action_dim() -> None:
    """A standard LIBERO suite scene attaches via the deploy-sim HAL path.

    Reproduces the `openral deploy sim --config scenes/sim/libero_spatial.yaml`
    failure: `SimAttachedHAL.connect()` probes `env.action_dim` to size the
    cartesian action packer, and the suite backend (`_LiberoSim`) exposed no
    such property — so connect raised `ROSConfigError` and the franka HAL never
    configured. This guards the suite backend so LIBERO resolves the OSC_POSE
    width (7).
    """
    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim", sim_env_yaml=_SUITE)
    hal.connect()  # raised ROSConfigError("cannot resolve the env action width") before the fix
    try:
        # 7 = LIBERO single-Panda OSC_POSE (6-D end-effector delta + gripper).
        assert hal._env_action_dim == 7, f"expected LIBERO OSC width 7, got {hal._env_action_dim}"
        state = hal.read_state()
        assert len(state.position) == 8  # 7 arm + gripper
    finally:
        hal.disconnect()
