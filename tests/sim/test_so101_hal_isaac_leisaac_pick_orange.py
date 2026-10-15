"""Sim test: LeIsaac's PickOrange kitchen on OpenRAL's Isaac Sim 6.x backend.

Boots the shipped ``scenes/sim/isaac_leisaac_pick_orange.yaml`` through the
real factory on a real Isaac Sim: the pinned LeIsaac v0.1.0 stage (fetched
into the openral cache on first use), the SO-101 imported from its URDF at
LeIsaac's spawn, the stage's own oranges and plate as graded scene objects.

* At t0 the four stage objects are found where the USD authors them, the
  robot stays pinned at its spawn, and the episode grades ``fail`` (the oranges
  start beside the plate, not on it).
* Teleporting the oranges onto the plate (``set_object_position``) grades
  ``complete`` once they rest, and the episode terminates as a success.

``OPENRAL_TEST_ARTIFACT_DIR`` (optional) receives the front-camera frames
(``leisaac_front_t0.png``, ``leisaac_front_placed.png``) and an MP4 of the run.

Skip policy: as ``test_openarm_hal_isaac_warehouse`` (no Isaac sidecar or no
pyzmq/msgpack -> skip); the first run also needs github.com for the stage.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.sim.conftest import _repo_root, _sidecar_python_available

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]

_SPAWN = np.array([2.2, -0.61, 0.89])
_ORANGES = ("orange_1", "orange_2", "orange_3")
_FRAMES: list[NDArray[np.uint8]] = []


@pytest.fixture(scope="module")
def env() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_core import SimEnvironment, SimScene, VLASpec, load_scene_strict
    from openral_sim.factory import make_env

    scene = load_scene_strict(
        str(_repo_root() / "scenes/sim/isaac_leisaac_pick_orange.yaml"), SimScene
    )
    sim_env = make_env(
        SimEnvironment(
            robot_id=scene.robot_id or "so101_follower",
            scene=scene.scene,
            task=scene.task,
            base_pose=scene.base_pose,
            vla=VLASpec(id="zero", weights_uri="placeholder"),
        )
    )
    yield sim_env
    _write_video()
    sim_env.close()


def _hold(env: Any) -> Any:
    result = env.step(np.full(env.action_dim, np.nan, np.float32))
    _FRAMES.append(np.asarray(result.observation["images"]["camera1"], dtype=np.uint8))
    return result


def _artifact(name: str) -> Path | None:
    out = os.environ.get("OPENRAL_TEST_ARTIFACT_DIR")
    if not out:
        return None
    Path(out).mkdir(parents=True, exist_ok=True)
    return Path(out) / name


def _save_png(name: str, image: NDArray[np.uint8]) -> None:
    path = _artifact(name)
    if path is not None:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        print(f"front camera frame: {path}")


def _write_video() -> None:
    path = _artifact("leisaac_front.mp4")
    if path is None or not _FRAMES or importlib.util.find_spec("imageio") is None:
        return
    import imageio.v2 as imageio

    imageio.mimwrite(path, list(np.stack(_FRAMES)), fps=30)
    print(f"front camera video: {path}")


def test_t0_finds_the_stage_objects_and_grades_fail(env: Any) -> None:
    env.reset(seed=0)
    for _ in range(30):
        result = _hold(env)
    info = result.info
    assert set(info["object_positions"]) == {*_ORANGES, "plate"}
    plate = np.asarray(info["object_positions"]["plate"])
    for name in _ORANGES:
        orange = np.asarray(info["object_positions"][name])
        assert np.all(np.isfinite(orange))
        # On the counter beside the plate: off its +-0.10 m x/y band.
        assert np.max(np.abs(orange[:2] - plate[:2])) > 0.10, (name, orange, plate)
    assert info["placement_grade"] == "fail"
    assert result.terminated is False
    # Fixed base, pinned at LeIsaac's spawn.
    assert np.linalg.norm(np.asarray(info["robot_position"]) - _SPAWN) < 1e-3
    image = result.observation["images"]["camera1"]
    assert image.shape == (480, 640, 3)
    assert float(np.asarray(image).mean()) > 20.0  # lit: the stage ships no lights
    _save_png("leisaac_front_t0.png", image)
    _save_png("leisaac_wrist_t0.png", result.observation["images"]["camera2"])


def test_oranges_teleported_onto_the_plate_grade_success(env: Any) -> None:
    plate = np.asarray(_hold(env).info["object_positions"]["plate"])
    for name, (dx, dy) in zip(_ORANGES, [(-0.04, 0.0), (0.03, 0.035), (0.03, -0.035)], strict=True):
        target = plate + np.array([dx, dy, 0.06])
        moved = env.set_object_position(name, tuple(target))
        assert np.allclose(moved["position"], target, atol=1e-4)
    for _ in range(120):  # fall, settle, then settle_s 0.5 s at rest (30 Hz)
        result = _hold(env)
        if result.terminated:
            break
    assert result.info["placement_grade"] == "complete", result.info["object_positions"]
    assert result.terminated is True
    assert result.reward == 1.0
    assert result.info[env.task.success_key or "is_success"] is True
    _save_png("leisaac_front_placed.png", result.observation["images"]["camera1"])
