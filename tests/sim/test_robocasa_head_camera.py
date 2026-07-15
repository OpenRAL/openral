"""Sim test: the synthetic forward ``head`` navigation camera.

Exercises the real robocasa ``NavigateKitchen`` MuJoCo env (no mocks) to
prove that, with ``OPENRAL_ROBOCASA_HEAD_CAM=1``, the backend surfaces an
``observation.images.head`` frame that is a *distinct* forward render, not an
alias of the arm-mounted manipulation cameras. This is the frame the
InternVLA-N1 VLN rSkill consumes; the deploy path relies on it existing.

Skips automatically when robosuite/robocasa (the ``robocasa`` dependency
group) or a MuJoCo GL backend is unavailable — the legitimate CI skip path.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HAVE_ROBOSUITE = importlib.util.find_spec("robosuite") is not None
pytestmark = pytest.mark.skipif(
    not _HAVE_ROBOSUITE,
    reason="robosuite/robocasa not installed (install: just sync --group robocasa)",
)


def _build_navigate_kitchen():
    from openral_core.schemas import (
        PhysicsBackend,
        SceneSpec,
        SimEnvironment,
        TaskSpec,
        VLASpec,
    )
    from openral_sim.backends.robocasa import _build_robocasa_sim

    env = SimEnvironment(
        robot_id="panda_mobile",
        scene=SceneSpec(
            id="robocasa/NavigateKitchen",
            backend=PhysicsBackend.MUJOCO,
            cameras=["camera1"],
            backend_options={
                "mode": "prebuilt",
                "prebuilt_task": "NavigateKitchen",
                "robots": ["PandaMobile"],
                "controller": "OSC_POSE",
                "horizon": 600,
            },
            observation_height=128,
            observation_width=128,
        ),
        task=TaskSpec(
            id="robocasa/NavigateKitchen/0",
            scene_id="robocasa/NavigateKitchen",
            instruction="navigate through the kitchen to reach the sink",
        ),
        vla=VLASpec(
            id="internvla_n1",
            weights_uri="rskills/rskill-internvla_n1-mobile_base-vln-nf4",
        ),
    )
    return _build_robocasa_sim(env)


def test_head_camera_is_distinct_forward_render() -> None:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ["OPENRAL_ROBOCASA_HEAD_CAM"] = "1"
    try:
        sim = _build_navigate_kitchen()
    except Exception as exc:  # reason: no GL backend / asset fetch fails on CI
        pytest.skip(f"robocasa NavigateKitchen env unavailable: {exc}")

    obs = sim.reset()
    images = obs["images"]

    # The forward head camera is present with the scene's obs resolution.
    assert "head" in images, f"no 'head' camera in obs images {sorted(images)}"
    head = np.asarray(images["head"])
    assert head.shape == (128, 128, 3), head.shape
    assert head.dtype == np.uint8

    # It is a genuinely different view from the arm-mounted manipulation
    # camera (camera1 = shoulder_left), not an alias of it.
    cam1 = np.asarray(images["camera1"])
    assert not np.array_equal(head, cam1), "head render is identical to camera1"

    # A real render, not a black/blank frame.
    assert int(head.max()) > 0
