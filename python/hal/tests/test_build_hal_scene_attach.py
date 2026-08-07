# python/hal/tests/test_build_hal_scene_attach.py
"""build_hal scene-attach path: sim_env_yaml -> SimAttachedHAL."""

from __future__ import annotations

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal import build_hal

pytest.importorskip("openral_sim")  # scene backends are an optional group
pytest.importorskip("mujoco")

_FRANKA = "robots/franka_panda/robot.yaml"
_SCENE = "scenes/sim/tabletop_cube_push.yaml"  # native MjSpec scene: mujoco-only, no robosuite/GPU


def test_sim_env_yaml_returns_sim_attached_hal() -> None:
    from openral_hal.sim_attached import SimAttachedHAL

    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim", sim_env_yaml=_SCENE)
    assert isinstance(hal, SimAttachedHAL)
    assert hal.description.name == "franka_panda"


def test_sim_without_scene_still_builds_bare_twin() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim")  # no sim_env_yaml
    assert type(hal).__name__ != "SimAttachedHAL"


def test_real_mode_with_sim_env_yaml_raises() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    with pytest.raises(ROSConfigError, match="sim_env_yaml"):
        build_hal(desc, mode="real", sim_env_yaml=_SCENE)


def test_franka_sim_joint_names_match_native_mjcf() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    sim_names = {j.sim_joint_name or j.name for j in desc.joints if j.role != "gripper"}
    assert {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"} <= sim_names


def test_derived_twin_honours_a_composed_scene_mjcf(tmp_path) -> None:
    """`deploy sim`'s composed-scene MJCF reaches a ``hal.sim: null`` robot.

    Regression: the derived-twin branch called ``from_description`` with the
    description ALONE, dropping the transport that
    ``openral_hal.lifecycle._compose_scene_mjcf`` threads in as ``mjcf_path``.
    Every ``DeployScene.composition`` on so100 / so101 was therefore ignored —
    the stack booted a bare arm on an empty plane while the launch log still
    reported that it had composed the scene. Caught on a real ``deploy sim``
    run of ``scenes/deploy/so101_eraser.yaml``, where the policy's overview
    camera showed no desk, no eraser and no tape.
    """
    from openral_sim.backends.so101_eraser import compose_so101_eraser_deploy_mjcf

    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    assert desc.hal.sim is None, "so101 must stay on the derived-twin path for this test"

    xml, meshdir = compose_so101_eraser_deploy_mjcf()
    composed = meshdir.parent / "so101_eraser_transport_test.xml"
    composed.write_text(xml)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": str(composed)})
    assert str(composed) == str(
        hal._mjcf_path
    )  # reason: the composed-scene path has no public accessor

    # And the scene really is in there: the arena the composer adds, not just
    # the bare arm the manifest points at.
    assert "eraser" in xml and "tape" in xml
