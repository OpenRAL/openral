"""Visual-SLAM impl selection + stereo-rig wiring in ``sim_e2e.launch.py``.

Hermetic (no live ROS graph). Asserts:

* ``_stereo_camera_topics`` maps a ``"<left>,<right>"`` scene rig to the four
  ``/openral/cameras/<name>/…`` topics (and stays out of the way when unset).
* ``_build_visual_slam_includes`` composes the right launch file per
  ``slam_visual_impl`` and remaps each impl's own camera arg names, plus nvblox
  only when navigating.
* The two new launch args are declared with the documented defaults.

Same import/skip pattern as ``test_sim_e2e_clock_domain`` — the module's heavy
imports are deferred, so only ``launch`` / ``launch_ros`` are needed to load it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "sim_e2e.launch.py"


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("openral_foxglove_bringup.topics")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")
    spec = importlib.util.spec_from_file_location("openral_sim_e2e_vslam_under_test", _LAUNCH_FILE)
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(subs: object) -> str:
    if isinstance(subs, str):
        return subs
    if isinstance(subs, (list, tuple)):
        return "".join(getattr(s, "text", "") for s in subs)
    return getattr(subs, "text", str(subs))  # a single TextSubstitution


def _source_file(include: object) -> str:
    """Render an IncludeLaunchDescription's source path.

    ``PythonLaunchDescriptionSource.location`` only resolves after the source is
    executed; the path is stored as a substitution list, which we perform in a
    throwaway context to recover the launch-file path pre-execution.
    """
    from launch import LaunchContext

    src = include.launch_description_source  # type: ignore[attr-defined]
    subs = getattr(src, "_LaunchDescriptionSource__location")  # noqa: B009
    ctx = LaunchContext()
    return "".join(s.perform(ctx) for s in subs)


def test_stereo_camera_topics_maps_names_to_bus_topics() -> None:
    mod = _import_launch_module()
    assert mod._stereo_camera_topics("front_left, front_right") == (
        "/openral/cameras/front_left/image",
        "/openral/cameras/front_left/camera_info",
        "/openral/cameras/front_right/image",
        "/openral/cameras/front_right/camera_info",
    )


@pytest.mark.parametrize("csv", ["", "solo", "a,b,c"])
def test_stereo_camera_topics_none_when_not_a_pair(csv: str) -> None:
    """Unset / malformed rig → None so the impl keeps its own default topics."""
    mod = _import_launch_module()
    assert mod._stereo_camera_topics(csv) is None


def test_visual_includes_pycuvslam_with_stereo_rig() -> None:
    """pycuvslam impl composes the wheel launch and remaps its left/right args."""
    mod = _import_launch_module()
    includes = mod._build_visual_slam_includes(
        "/slam_share",
        visual_impl="pycuvslam",
        use_sim_time=True,
        stereo_cameras_csv="cam_l,cam_r",
        enable_nav2=False,
        robot_yaml="/robots/x/robot.yaml",
    )
    assert len(includes) == 1  # no nvblox without nav2
    assert _source_file(includes[0]).endswith("pycuvslam.launch.py")
    rendered = {_render(k): _render(v) for k, v in includes[0].launch_arguments}
    assert rendered["use_sim_time"] == "true"
    assert rendered["left_image_topic"] == "/openral/cameras/cam_l/image"
    assert rendered["right_image_topic"] == "/openral/cameras/cam_r/image"
    assert rendered["left_camera_info_topic"] == "/openral/cameras/cam_l/camera_info"


def test_visual_includes_isaac_ros_default_and_nvblox_on_nav2() -> None:
    """Default impl composes cuvslam.launch.py; nav2 adds nvblox with no rig override."""
    mod = _import_launch_module()
    includes = mod._build_visual_slam_includes(
        "/slam_share",
        visual_impl="isaac_ros",
        use_sim_time=False,
        stereo_cameras_csv="",
        enable_nav2=True,
        robot_yaml="/robots/x/robot.yaml",
    )
    assert len(includes) == 2
    files = [_source_file(inc) for inc in includes]
    assert files[0].endswith("cuvslam.launch.py")
    assert files[1].endswith("nvblox.launch.py")
    # No scene rig → no camera-topic overrides (the impl's own defaults stand).
    cuvslam_args = {_render(k): _render(v) for k, v in includes[0].launch_arguments}
    assert cuvslam_args == {"use_sim_time": "false"}
    nvblox_args = {_render(k): _render(v) for k, v in includes[1].launch_arguments}
    assert nvblox_args["robot_yaml"] == "/robots/x/robot.yaml"


def test_visual_slam_launch_args_declared_with_defaults() -> None:
    mod = _import_launch_module()
    from launch.actions import DeclareLaunchArgument

    desc = mod.generate_launch_description()
    args = {a.name: a for a in desc.describe_sub_entities() if isinstance(a, DeclareLaunchArgument)}
    assert _render(args["slam_visual_impl"].default_value) == "isaac_ros"
    assert _render(args["slam_stereo_cameras"].default_value) == ""
