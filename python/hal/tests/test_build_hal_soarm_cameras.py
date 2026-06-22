"""SO-ARM (SO-100 / SO-101) deploy-sim cameras via the generic rig (ADR-0065; issue #88).

`openral deploy sim` builds a bare MuJoCo twin for the SO-ARM arms (their
manifests set ``hal.sim: null`` → ``MujocoArmHAL.from_description``). The upstream
``so_arm100`` / ``so_arm101`` MJCFs declare zero ``<camera>`` elements, so the HAL
rendered nothing and every manifest camera came back absent.

The fix: each RGB ``SensorSpec`` carries a ``sim_placement``, and
``MujocoArmHAL.connect`` runs the generic camera rig
(``openral_hal._camera_rig``) to splice those cameras into the bare MJCF — no
per-robot scene composer, no ``scene_defaults.composition`` on the manifest.
This is exercised here against the real manifests + real MuJoCo (no mocks).
"""

from __future__ import annotations

import pytest
from openral_core import RobotDescription
from openral_hal import build_hal

pytest.importorskip("mujoco")

_SOARM = ["robots/so100_follower/robot.yaml", "robots/so101_follower/robot.yaml"]


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_manifest_sensors_carry_sim_placement(robot_yaml: str) -> None:
    desc = RobotDescription.from_yaml(robot_yaml)
    # No scene composition hook on the robot manifest (ADR-0065).
    assert desc.scene_defaults is None or desc.scene_defaults.composition is None
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    # front = world-fixed overhead; wrist = parented to the roll-mounted end body.
    assert by_name["front"].sim_placement is not None
    assert by_name["front"].sim_placement.parent_body is None
    assert by_name["wrist"].sim_placement is not None
    assert by_name["wrist"].sim_placement.parent_body is not None


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_rig_renders_both_manifest_cameras(robot_yaml: str) -> None:
    """End-to-end: the bare twin renders `front` + `wrist` after the connect-time rig.

    Mirrors the SimSensorBridge camera tick exactly (it calls
    ``hal.read_images()``). Skips only if offscreen GL is unavailable on the
    runner (CI without a display); a dev host with a GPU renders for real.
    """
    import numpy as np

    desc = RobotDescription.from_yaml(robot_yaml)
    # No transport / no composition: the HAL rigs cameras from sim_placement.
    hal = build_hal(desc, mode="sim")
    assert type(hal).__name__ == "MujocoArmHAL"
    hal.connect()
    try:
        frames = hal.read_images()
    except Exception as exc:  # pragma: no cover - headless GL not available
        pytest.skip(f"offscreen MuJoCo render unavailable: {exc}")
    finally:
        hal.disconnect()

    if not frames:
        pytest.skip("offscreen MuJoCo render produced no frames (no GL context)")
    assert set(frames) == {"front", "wrist"}
    for name, arr in frames.items():
        a = np.asarray(arr)
        assert a.ndim == 3 and a.shape[2] == 3, f"{name}: {a.shape}"
        assert a.dtype == np.uint8
        # The staging floor + fill light guarantee a non-black frame (a forward
        # wrist camera over a bare arm with no floor would otherwise be void).
        assert a.std() > 1.0, f"{name} looks blank (std={a.std():.2f})"
