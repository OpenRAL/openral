"""Hermetic checks on ``nav2.launch.py``.

These do NOT spawn a real Nav2 graph (that's the integration tier).
They assert the launch file's structural pieces remain intact and
that the panda_mobile-tuned config keeps its critical knobs.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_PKG_NAME = "openral_nav2_bringup"
_PKG_DIR = Path(__file__).resolve().parent.parent
_LAUNCH_DIR = _PKG_DIR / "launch"
_CMAKELISTS = _PKG_DIR / "CMakeLists.txt"
_LAUNCH_FILE = _LAUNCH_DIR / "nav2.launch.py"
_CONFIG_PATH = _PKG_DIR / "config" / "nav2_panda_mobile.yaml"
_VISUAL_CONFIG_PATH = _PKG_DIR / "config" / "nav2_visual.yaml"


def _import_launch_module() -> ModuleType:
    """Import ``nav2.launch.py`` by file path.

    Mirrors the slam_toolbox test pattern: bypass the ROS 2 Python
    package shadowing by loading via ``spec_from_file_location``.
    """
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_nav2_bringup_launch_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_file_exists_and_parses() -> None:
    """The panda_mobile-tuned YAML ships in-tree and is valid."""
    assert _CONFIG_PATH.is_file(), _CONFIG_PATH
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    assert isinstance(data, dict)
    # MPPI uses Omni motion model — required for the holonomic base.
    controller_params = data["controller_server"]["ros__parameters"]
    follow_path = controller_params["FollowPath"]
    assert follow_path["motion_model"] == "Omni", (
        "panda_mobile is holonomic — MPPI motion_model must be 'Omni' "
        "(DiffDrive ignores lateral velocity commands and the base "
        "won't strafe). See nav2_panda_mobile.yaml."
    )
    # Symmetric lateral velocity bound — DiffDrive default is 0.0.
    assert follow_path.get("vy_min", 0.0) < 0.0, (
        "holonomic base needs symmetric vy bounds (vy_min < 0); "
        "see nav2_panda_mobile.yaml FollowPath.vy_min."
    )
    # Robot radius widened to panda_mobile's chassis envelope.
    global_costmap = data["global_costmap"]["global_costmap"]["ros__parameters"]
    local_costmap = data["local_costmap"]["local_costmap"]["ros__parameters"]
    assert global_costmap["robot_radius"] >= 0.30, global_costmap["robot_radius"]
    assert local_costmap["robot_radius"] >= 0.30, local_costmap["robot_radius"]


def test_launch_module_includes_upstream_navigation_launch() -> None:
    """The launch description includes the upstream Nav2 navigation_launch.py."""
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    try:
        desc = mod.generate_launch_description()
    except PackageNotFoundError as exc:
        # ``openral_nav2_bringup`` or upstream ``nav2_bringup`` not on
        # the ament index — legitimate skip path per CLAUDE.md §1.11
        # (the operator hasn't ``source install/setup.bash`` or
        # apt-installed Nav2). No fakes; just skip.
        pytest.skip(f"ament package missing (overlay not sourced?): {exc}")
    actions = desc.describe_sub_entities()
    # The upstream nav2 IncludeLaunchDescription is built at
    # launch time inside an OpaqueFunction (it RewrittenYaml-rewrites the
    # base params with this robot's `nav2_param_overrides()`), so it is
    # NOT a static top-level entity. Assert the single deferred function
    # plus the `robot_yaml` arg that drives the per-robot override.
    opaque = [a for a in actions if isinstance(a, OpaqueFunction)]
    assert len(opaque) == 1, f"expected exactly 1 deferred OpaqueFunction; got {len(opaque)}"
    arg_names = {a.name for a in actions if isinstance(a, DeclareLaunchArgument)}
    assert "robot_yaml" in arg_names, arg_names
    # The SLAM-backend selector arg drives the costmap profile.
    assert "slam_backend" in arg_names, arg_names
    default_path = getattr(mod, "DEFAULT_PARAMS_PATH")  # noqa: B009
    assert Path(default_path).is_file(), default_path


def test_slam_backend_selects_costmap_profile() -> None:
    """`_params_path_for_backend` maps the backend to the right config."""
    mod = _import_launch_module()
    visual = Path(mod._params_path_for_backend("visual"))
    lidar = Path(mod._params_path_for_backend("lidar"))
    assert visual.name == "nav2_visual.yaml", visual
    assert lidar.name == "nav2_panda_mobile.yaml", lidar
    # default / unknown backends fall back to the lidar (base) profile.
    assert Path(mod._params_path_for_backend("none")).name == "nav2_panda_mobile.yaml"
    # case-insensitive
    assert Path(mod._params_path_for_backend("VISUAL")).name == "nav2_visual.yaml"


def test_visual_profile_consumes_map_not_scan() -> None:
    """The visual profile's costmaps read `/map` via static_layer (no /scan).

    This is what makes Nav2 backend-agnostic: a lidar-less robot (cuVSLAM+nvblox)
    plans off the same `/map` interface slam_toolbox publishes, with no `/scan`.
    """
    assert _VISUAL_CONFIG_PATH.is_file(), _VISUAL_CONFIG_PATH
    data = yaml.safe_load(_VISUAL_CONFIG_PATH.read_text())
    for scope in ("global_costmap", "local_costmap"):
        cm = data[scope][scope]["ros__parameters"]
        assert "static_layer" in cm["plugins"], (scope, cm["plugins"])
        assert "obstacle_layer" not in cm["plugins"], (scope, cm["plugins"])
        assert "voxel_layer" not in cm["plugins"], (scope, cm["plugins"])
        sl = cm["static_layer"]
        assert sl["map_topic"] == "/map", sl
        # nvblox publishes /map RELIABLE+VOLATILE (not latched) — the static_layer
        # must NOT request transient_local or the QoS mismatches.
        assert sl["map_subscribe_transient_local"] is False, sl
    # collision_monitor must not depend on /scan (lidar-less): scan source disabled.
    cm = data["collision_monitor"]["ros__parameters"]
    assert cm["scan"]["enabled"] is False, cm["scan"]
    # geometry/planner still mirror the base (planner threads unknown space).
    assert data["planner_server"]["ros__parameters"]["GridBased"]["allow_unknown"] is True


def test_costmaps_declare_a_polygon_footprint_not_only_a_radius() -> None:
    """A circle cannot express a carried payload.

    `nav2_costmap_2d` only takes runtime footprint updates seriously as a
    polygon: `robot_radius` alone leaves the costmap on its circular path with
    nothing for the payload publisher to grow. Both profiles must ship one.
    """
    for path in (_CONFIG_PATH, _VISUAL_CONFIG_PATH):
        data = yaml.safe_load(path.read_text())
        for scope in ("global_costmap", "local_costmap"):
            footprint = data[scope][scope]["ros__parameters"].get("footprint", "")
            assert footprint not in ("", "[]"), (
                f"{path.name}:{scope} has no polygon footprint — Nav2 reads "
                f'"" and "[]" as "use robot_radius", which no payload can grow.'
            )


def test_mppi_does_not_yet_consider_the_footprint() -> None:
    """`consider_footprint` is deliberately `false`, and pinned so it stays a decision.

    Flipping it is a navigation-behaviour change, not a tuning nudge, and only
    half its price is known. The flag's OWN cost is measured against the real
    Jazzy ``libnav2_costmap_2d_core`` in
    ``benchmark/cost_critic_footprint_bench.cpp``: on an i5-8600K, 56000
    ``footprintCostAtPose`` calls per 20 Hz iteration cost **+8.1 ms** with the
    bare chassis and **+9.7 ms** while carrying — 17-20 % of the 50 ms budget,
    and unconditional rather than data-dependent, because ``inflation_radius``
    0.40 m sits below both polygons' circumscribed radii (0.444 m / 0.863 m) so
    ``findCircumscribedCost`` returns 0.0. What is NOT measured is the rest of
    the MPPI loop around CostCritic, which needs the composite scene issue #108
    still lacks — nothing in the repo drives the base at 20 Hz mid-carry.

    So the value is asserted rather than left incidental: the dynamic footprint
    reaches every other consumer today (behaviours, docking, the collision
    monitor's approach polygon, ``IsPathValid``), and MPPI's own scoring waits
    for a measurement of the whole loop. Flip this test and
    ``config/nav2_panda_mobile.yaml``'s comment together, never separately.
    """
    for path in (_CONFIG_PATH, _VISUAL_CONFIG_PATH):
        data = yaml.safe_load(path.read_text())
        cost_critic = data["controller_server"]["ros__parameters"]["FollowPath"]["CostCritic"]
        assert cost_critic["consider_footprint"] is False, path.name


def test_payload_nodes_ride_with_nav2_on_the_right_backends() -> None:
    """The footprint publisher needs a manifest; the scan filter needs a `/scan`.

    Without a `robot_yaml` there is no measured outline to publish and inventing
    one would put a made-up robot shape on Nav2's collision surface; on the
    `visual` backend there is no scan to filter at all.
    """
    mod = _import_launch_module()

    both = mod._payload_footprint_nodes(
        robot_yaml="robots/panda_mobile/robot.yaml", slam_backend="lidar", use_sim_time=True
    )
    visual_only = mod._payload_footprint_nodes(
        robot_yaml="robots/panda_mobile/robot.yaml", slam_backend="visual", use_sim_time=True
    )
    no_manifest = mod._payload_footprint_nodes(
        robot_yaml="", slam_backend="lidar", use_sim_time=True
    )

    assert len(both) == 2, both
    assert len(visual_only) == 1, "visual has no /scan to filter"
    assert len(no_manifest) == 1, "no manifest means no nominal footprint to publish"


def _launch_referenced_executables() -> set[str]:
    """Every ``executable=`` this package's launch files ask of *this* package.

    Read with ``ast`` instead of by importing the launch files: the executable
    bit has to be checkable on a bare checkout — no ROS 2 overlay, no
    ``launch_ros``, nothing built. Discovery is by parse, not by name, so a
    node added tomorrow is covered without touching this test.
    """
    found: set[str] = set()
    for launch_file in sorted(_LAUNCH_DIR.glob("*.launch.py")):
        tree = ast.parse(launch_file.read_text(encoding="utf-8"), str(launch_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            package, executable = kwargs.get("package"), kwargs.get("executable")
            if not isinstance(package, ast.Constant) or not isinstance(executable, ast.Constant):
                continue  # a computed package/executable — nothing to resolve statically
            if package.value == _PKG_NAME:
                found.add(str(executable.value))
    return found


def _installed_programs() -> dict[str, Path]:
    """``install(PROGRAMS …)`` sources, keyed by the name launch resolves.

    ``install(PROGRAMS a/b.py DESTINATION lib/<pkg>)`` lands the file at
    ``lib/<pkg>/b.py``, so the source basename *is* the ``executable=`` name.
    """
    # Strip comments first — the block below is preceded by prose explaining
    # the pattern, and an unstripped scan would pick words out of it.
    body = re.sub(r"(?m)#.*$", "", _CMAKELISTS.read_text(encoding="utf-8"))
    programs: dict[str, Path] = {}
    for block in re.finditer(r"install\s*\(\s*PROGRAMS(.*?)DESTINATION", body, re.DOTALL):
        for token in block.group(1).split():
            src = _PKG_DIR / token.replace("${PROJECT_NAME}", _PKG_NAME)
            programs[src.name] = src
    return programs


def test_every_launch_referenced_node_is_installed_and_executable() -> None:
    """A launch-referenced node that is not +x kills the WHOLE launch description.

    ``just ros2-build`` runs colcon with ``--symlink-install``, so
    ``install(PROGRAMS …)`` *symlinks* the source file into
    ``lib/openral_nav2_bringup/`` rather than copying it — and CMake's
    "add +x on install" never happens, because there is no copy to chmod.
    ``launch_ros`` then resolves the executable with ``shutil.which()`` over
    that libexec dir, gets ``None`` for a 0644 file, and raises. launch does
    not skip the one node: it abandons the entire launch description, which is
    how a mode bit took ``openral deploy sim`` (and with it
    ``/openral/execute_rskill``) down completely.

    The bit therefore has to be on the *source* file, and this test asserts it
    for every node the launch files reference — by parse, not by name.
    """
    referenced = _launch_referenced_executables()
    assert referenced, f"no first-party executable= found under {_LAUNCH_DIR} — did they move?"
    programs = _installed_programs()
    for name in sorted(referenced):
        source = programs.get(name)
        assert source is not None, (
            f"launch asks for executable={name!r} from {_PKG_NAME}, but no "
            f"install(PROGRAMS …) in CMakeLists.txt puts it in lib/{_PKG_NAME}/. "
            f"Installed: {sorted(programs)}"
        )
        assert source.is_file(), source
        assert os.access(source, os.X_OK), (
            f"{source.relative_to(_PKG_DIR)} is not executable, so with "
            f"--symlink-install lib/{_PKG_NAME}/{name} is a symlink to a 0644 file "
            f"and launch_ros aborts with \"executable '{name}' not found on the "
            f'libexec directory". Fix with: git update-index --chmod=+x '
            f"packages/{_PKG_NAME}/{source.relative_to(_PKG_DIR)}"
        )


def test_launch_referenced_nodes_are_committed_executable() -> None:
    """+x in the working tree is not enough — the mode has to reach the index.

    A bare ``chmod +x`` fixes the developer's own build and still commits a
    100644 blob, so every fresh clone and every CI runner walks straight back
    into the same launch abort. Assert the committed mode, not just the
    on-disk one.
    """
    if shutil.which("git") is None:
        pytest.skip("git not on PATH — cannot read the committed file mode")
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", str(_PKG_DIR)],
        capture_output=True,
        text=True,
        cwd=_PKG_DIR,
        check=False,  # a non-git checkout is a skip, not a failure
    )
    if proc.returncode != 0:
        pytest.skip(f"not a git work tree (installed copy?): {proc.stderr.strip()}")
    modes = {
        Path(line.split("\t", 1)[1]).name: line.split(" ", 1)[0]
        for line in proc.stdout.splitlines()
        if "\t" in line
    }
    programs = _installed_programs()
    for name in sorted(_launch_referenced_executables()):
        source = programs[name]
        mode = modes.get(source.name)
        assert mode is not None, f"{source} is not tracked by git"
        assert mode == "100755", (
            f"{source.name} is committed {mode}, not 100755 — a local chmod +x "
            f"will not survive a fresh clone. Fix with: git update-index --chmod=+x "
            f"packages/{_PKG_NAME}/{source.relative_to(_PKG_DIR)}"
        )


def test_launch_ros_resolves_every_node_in_the_built_overlay() -> None:
    """The real resolution path: ``ExecutableInPackage`` over the built libexec dir.

    This is the exact substitution ``launch_ros.actions.Node`` performs at
    launch time, so it sees the ``--symlink-install`` tree as launch does —
    the source-level tests above are the ones that run everywhere, this one
    is the end-to-end confirmation when an overlay happens to be sourced.
    """
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix
    from launch import LaunchContext
    from launch.substitutions.substitution_failure import SubstitutionFailure
    from launch_ros.substitutions import ExecutableInPackage

    try:
        get_package_prefix(_PKG_NAME)
    except PackageNotFoundError as exc:
        pytest.skip(f"overlay not sourced ({exc}) — `source install/setup.bash` to run this")

    context = LaunchContext()
    for name in sorted(_launch_referenced_executables()):
        try:
            resolved = ExecutableInPackage(package=_PKG_NAME, executable=name).perform(context)
        except SubstitutionFailure as exc:  # pragma: no cover - the bug this guards
            pytest.fail(
                f"launch_ros cannot resolve executable={name!r}: {exc}. With "
                "--symlink-install the libexec entry is a symlink to the source "
                "file, so the source file itself must be mode 100755."
            )
        assert os.access(resolved, os.X_OK), resolved


def test_the_payload_footprint_leg_can_be_turned_off() -> None:
    """`payload_footprint` is a declared launch arg, not a hardcoded leg."""
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError
    from launch.actions import DeclareLaunchArgument

    try:
        desc = mod.generate_launch_description()
    except PackageNotFoundError as exc:
        pytest.skip(f"ament package missing (overlay not sourced?): {exc}")
    arg_names = {
        a.name for a in desc.describe_sub_entities() if isinstance(a, DeclareLaunchArgument)
    }
    assert "payload_footprint" in arg_names, arg_names
