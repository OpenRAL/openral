"""Regression guards for deploy-sim's single clock-domain flag.

Root cause of the nav-collision bug (``fix/nav-collision``): the Nav2
stack was launched with a hardcoded ``use_sim_time:=true`` while
deploy-sim publishes **no** ``/clock``. Every Nav2 node's clock then
pinned at 0 while the HAL stamped ``/scan`` + TF on wall-clock, so the
local costmap rejected the "future" scans, stayed empty, and the base
drove straight through obstacles (controller logged "loop rate inf Hz").
``octomap_server`` / ``ros_image_detector`` already worked around the
missing clock with ``use_sim_time:=false``; Nav2 + slam_toolbox +
robot_state_publisher did not — a scattered-literal disagreement.

These tests pin the fix: a single ``enable_sim_clock`` launch arg
(default ``false``) is the *one* source of truth for the whole graph's
clock domain, so a node can never again silently disagree.

Hermetic (no live ROS graph). The module's heavy ``openral_core`` /
``mujoco`` imports are deferred inside ``compose_runtime_graph``; only
``launch`` / ``launch_ros`` are needed to import the file and inspect
its declared args + the Nav2 include.
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


def test_enable_sim_clock_arg_declared_and_defaults_false() -> None:
    """The single clock-domain flag exists and defaults to wall-clock.

    Default ``false`` is the whole bugfix: deploy-sim has no ``/clock``
    publisher, so the graph must run on wall-clock to match the HAL.
    """
    mod = _import_launch_module()
    from launch.actions import DeclareLaunchArgument

    desc = mod.generate_launch_description()
    args = {a.name: a for a in desc.describe_sub_entities() if isinstance(a, DeclareLaunchArgument)}
    assert "enable_sim_clock" in args, sorted(args)
    assert _render(args["enable_sim_clock"].default_value) == "false", (
        "enable_sim_clock must default to false — deploy-sim has no /clock "
        "publisher, so use_sim_time=true would pin every node's clock at 0 "
        "(the nav-collision root cause)."
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("false", False),
        ("0", False),
        ("FALSE", False),
        ("", False),
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_resolve_sim_clock_semantics(value: str, expected: bool) -> None:
    """The flag resolver matches the other ``enable_*`` arg idioms."""
    mod = _import_launch_module()
    assert mod._resolve_sim_clock(value) is expected


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
