"""LIBERO-only policy guards: a non-LIBERO sim scene is refused, the runtime scene stub passes.

The deploy / rSkill-runner path builds its policy from ``_SimpleEnvCfg``, whose scene carries
only ``cameras``. The guard read ``scene.id`` unconditionally, so dispatching a LIBERO skill on
a robot raised an untyped AttributeError instead of loading. Real runtime stub, real schemas.
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
from openral_core.exceptions import ROSCapabilityMismatch
from openral_core.schemas import PhysicsBackend
from openral_sim.backends.libero import libero_canonical_state, require_libero_scene


def _sim_env(scene_id: str) -> SimEnvironment:
    return SimEnvironment(
        robot_id="franka_panda",
        scene=SceneSpec(id=scene_id, backend=PhysicsBackend.MUJOCO, cameras=["camera1"]),
        task=TaskSpec(
            id=f"{scene_id}/0",
            scene_id=scene_id,
            instruction="pick the bowl",
            success_key="is_success",
            max_steps=10,
        ),
        vla=VLASpec(id="xvla", weights_uri="rskills/xvla-libero"),
    )


def test_a_libero_scene_passes() -> None:
    require_libero_scene(_sim_env("libero_spatial"), "xvla")


def test_a_non_libero_scene_is_refused() -> None:
    with pytest.raises(ROSCapabilityMismatch, match="robocasa"):
        require_libero_scene(_sim_env("robocasa"), "xvla")


def test_the_runtime_scene_stub_passes() -> None:
    runner = pytest.importorskip("openral_rskill_ros.rskill_runner_node")
    stub = runner._SimpleEnvCfg(
        vla=VLASpec(id="cosmos3", weights_uri="rskills/cosmos3-edge-libero-bf16"),
        cameras=("camera1", "camera2"),
    )
    require_libero_scene(stub, "cosmos3")


def test_the_canonical_state_is_width_checked() -> None:
    state = libero_canonical_state({"state": np.arange(8)}, "cosmos3")
    assert state.dtype == np.float32
    assert state.shape == (8,)
    with pytest.raises(ROSCapabilityMismatch, match="7-D"):
        libero_canonical_state({"state": np.zeros(7)}, "cosmos3")
    with pytest.raises(ROSCapabilityMismatch, match="provides none"):
        libero_canonical_state({}, "cosmos3")
