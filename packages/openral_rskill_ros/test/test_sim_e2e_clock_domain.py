"""Regression guards for deploy-sim's single clock-domain flag.

Root cause of the nav-collision bug (``fix/nav-collision``): the Nav2
stack was launched with a hardcoded ``use_sim_time:=true`` before
deploy-sim had a coherent ``/clock`` publisher. Every Nav2 node's clock
then pinned at 0 while the HAL stamped ``/scan`` + TF on wall-clock, so
the local costmap rejected the "future" scans, stayed empty, and the base
drove straight through obstacles (controller logged "loop rate inf Hz").
``octomap_server`` / ``ros_image_detector`` already worked around the
missing clock with ``use_sim_time:=false``; Nav2 + slam_toolbox +
robot_state_publisher did not — a scattered-literal disagreement.

These tests pin the fix: a single ``clock_origin`` launch arg is the
OpenRAL ClockAuthority source for the whole graph's clock domain, so a node
can never again silently disagree.

Hermetic (no live ROS graph). The module's heavy ``openral_core`` /
``mujoco`` imports are deferred inside ``compose_runtime_graph``; only
``launch`` / ``launch_ros`` plus launch-time package imports are needed to
import the file and inspect its declared args + the Nav2 include.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "sim_e2e.launch.py"


def _import_launch_module() -> ModuleType:
    """Import ``sim_e2e.launch.py`` by absolute path.

    Same pattern as ``test_nav2_launch`` / ``test_slam_toolbox_launch``:
    load via ``spec_from_file_location`` and skip when the ROS 2 launch
    machinery isn't importable (no sourced overlay / interpreter
    mismatch) — the legitimate CLAUDE.md §1.11 skip path, never a fake.
    """
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("openral_foxglove_bringup.topics")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location("openral_sim_e2e_under_test", _LAUNCH_FILE)
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(subs: object) -> str:
    """Render a (possibly-normalized) launch substitution list to text.

    ``DeclareLaunchArgument.default_value`` and an
    ``IncludeLaunchDescription``'s launch-argument keys/values are stored
    as lists of ``TextSubstitution`` once normalized; for the plain
    string literals used here that means a single ``.text`` field.
    """
    if isinstance(subs, str):
        return subs
    return "".join(getattr(s, "text", "") for s in subs)  # type: ignore[union-attr]


def test_clock_origin_arg_declared_and_defaults_host_wall() -> None:
    """The single clock-origin arg exists and defaults to host wall time.

    The CLI normally resolves this to ``simulation`` for capable sim backends;
    the launch-file default stays host_wall so a direct ros2 launch cannot
    accidentally pin every node's clock at zero.
    """
    mod = _import_launch_module()
    from launch.actions import DeclareLaunchArgument

    desc = mod.generate_launch_description()
    args = {a.name: a for a in desc.describe_sub_entities() if isinstance(a, DeclareLaunchArgument)}
    assert "clock_origin" in args, sorted(args)
    assert _render(args["clock_origin"].default_value) == "host_wall", (
        "clock_origin must default to host_wall — direct launch without the CLI "
        "must not set ROS use_sim_time=true without a live /clock."
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("host_wall", "host_wall"),
        ("host-wall", "host_wall"),
        ("simulation", "simulation"),
        ("SIMULATION", "simulation"),
    ],
)
def test_resolve_clock_origin_semantics(value: str, expected: str) -> None:
    """The launch accepts only explicit OpenRAL clock-authority origins."""
    mod = _import_launch_module()
    assert mod._resolve_clock_origin(value) == expected


def test_resolve_clock_origin_rejects_truthy_legacy_values() -> None:
    """The old enable/disable flag semantics are not a clock authority."""
    mod = _import_launch_module()
    with pytest.raises(ValueError, match="clock_origin"):
        mod._resolve_clock_origin("true")


@pytest.mark.parametrize(("use_sim_time", "expected"), [(False, "false"), (True, "true")])
def test_nav2_include_threads_use_sim_time(
    use_sim_time: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_build_nav2_include`` forwards the graph flag — never a hardcoded value.

    This is the exact site of the bug: the include previously pinned
    ``use_sim_time:=true``. The package-share lookup is stubbed so this
    runs even without a built ``openral_nav2_bringup`` overlay — the
    ``PythonLaunchDescriptionSource`` path is parsed lazily at launch
    time, never at construction, so a dummy path is inert for the
    ``launch_arguments`` inspection we do here.
    """
    mod = _import_launch_module()
    import ament_index_python.packages as ament_pkgs

    monkeypatch.setattr(ament_pkgs, "get_package_share_directory", lambda _pkg: "/nonexistent")

    robot_yaml = str(Path(__file__).resolve().parents[3] / "robots" / "panda_mobile" / "robot.yaml")
    include = mod._build_nav2_include(robot_yaml, use_sim_time=use_sim_time)

    rendered = {_render(key): _render(val) for key, val in include.launch_arguments}
    assert rendered.get("use_sim_time") == expected, rendered
    assert rendered.get("robot_yaml") == robot_yaml, rendered
