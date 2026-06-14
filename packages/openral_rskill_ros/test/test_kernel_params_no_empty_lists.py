"""Regression test: ``sim_e2e.launch.py`` produces no empty-list ROS params.

Asserts that :func:`sim_e2e.launch.compose_runtime_graph` builds the
``openral_safety_kernel`` LifecycleNode's parameter dict with NO empty
list (or empty tuple) values for any robot in the in-tree catalogue.

Why this matters: ``launch_ros.utilities.evaluate_parameters`` collapses
an empty Python list to ``()`` and falls through to
``ensure_argument_type(value, (float, int, str, bool, bytes), 'value')``,
which raises::

    Expected 'value' to be one of [<class 'float'>, <class 'int'>, ...],
    but got '()' of type '<class 'tuple'>'

That error fires at launch-time, **before any node logs**, so it
manifests as an opaque "deploy sim crashed instantly" with no traceback
visible without ``ros2 launch --debug``. The historical incident:
``e591374`` (ADR-0040 Phase 1+2) added ``collision_base_dofs`` as an
unconditional ROS param; the list is empty for every fixed-base arm
(openarm, so101, franka_panda, ur5e, ur10e, sawyer, rizon4, …), which
broke ``openral deploy sim`` for the majority of in-tree robots until
the omit-when-empty guard at ``sim_e2e.launch.py:397`` was added.

Per CLAUDE.md §1.11: no mocks. Real
:class:`openral_core.RobotDescription` loaded from a real
``robots/<robot>/robot.yaml``; real ``LaunchContext`` exercising the
real ``compose_runtime_graph`` opaque function; real
``launch_ros.utilities.evaluate_parameters`` so the assertion exercises
the exact code path ``ros2 launch`` would.

Run::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    MUJOCO_GL=egl uv run pytest \
        packages/openral_rskill_ros/test/test_kernel_params_no_empty_lists.py -v
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# ── Guards ───────────────────────────────────────────────────────────────────

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_core")
pytest.importorskip("openral_safety")
pytest.importorskip("mujoco")

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

# parents[3]: test/ → openral_rskill_ros/ → packages/ → <repo-root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "sim_e2e.launch.py"

# Fixed-base arms (no ``base_joints`` in robot.yaml) exhibit the bug; mobile
# bases (panda_mobile) do not. Cover at least one of each so a future
# regression that flips the contract for either path is caught.
_FIXED_BASE_ROBOTS = ["openarm", "so101_follower", "franka_panda"]
_MOBILE_BASE_ROBOTS = ["panda_mobile"]


def _import_launch_module() -> object:
    """Load ``sim_e2e.launch.py`` as a Python module via importlib.

    The launch file lives outside the package's importable Python tree (it
    is installed to ``share/openral_rskill_ros/launch/`` by ament_python),
    so a normal ``from openral_rskill_ros.launch.sim_e2e import …`` is not
    available. Load the source file directly — this is the same pattern
    ``test_franka_scene_attach.launch.py`` follows for its launch-side
    imports.
    """
    spec = importlib.util.spec_from_file_location("sim_e2e_launch", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_launch_context(robot_yaml: Path) -> object:
    """Return a :class:`launch.LaunchContext` pre-populated with the defaults.

    Mirrors the ``DeclareLaunchArgument(default_value=…)`` block at the
    bottom of ``sim_e2e.launch.py`` so ``compose_runtime_graph`` can
    resolve every ``LaunchConfiguration(...).perform(context)``. Only
    ``robot_yaml`` / ``hal_*`` lack defaults (they are CLI-required for
    real launches) so we set them by hand to a representative HAL.
    """
    from launch import LaunchContext

    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(robot_yaml)
    # Use the openarm HAL string regardless of the robot — compose_runtime_graph
    # never imports the HAL package at parse time; only the executable
    # name reaches the LifecycleNode constructor (a launch-arg-validated
    # string).
    cfg["hal_package"] = "openral_hal_openarm"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = "openral_hal_test"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    # Defaults that ship with the launch file.
    cfg["reset_to_pose_service"] = ""
    cfg["dashboard_port"] = "4318"
    cfg["reasoner_provider"] = "ollama"
    cfg["reasoner_model"] = "gemma4:31b-cloud"
    cfg["spatial_memory_path"] = ""
    cfg["spatial_memory_ingest"] = "false"
    cfg["hal_mode"] = "sim"
    cfg["enable_slam"] = "false"
    cfg["enable_nav2"] = "false"
    cfg["enable_octomap"] = "false"
    cfg["octomap_cloud_topic"] = "/openral/cameras/front_depth/points"
    cfg["enable_object_detector"] = "false"
    # Detector path is only read when enable_object_detector is true, but
    # the LaunchConfiguration must resolve regardless.
    cfg["object_detector_onnx"] = str(_REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "model.onnx")
    cfg["enable_dashboard"] = "false"
    return ctx


def _safety_kernel_params(robot_id: str) -> dict[str, object]:
    """Compose the launch graph and return the safety_kernel's evaluated params.

    Drives ``compose_runtime_graph`` end-to-end and resolves the
    safety_kernel LifecycleNode's parameter dict through the same
    ``launch_ros.utilities.evaluate_parameters`` path ``ros2 launch``
    invokes — so an empty list collapsing to ``()`` raises here exactly
    as it would in production.
    """
    from launch_ros.actions import LifecycleNode
    from launch_ros.utilities import evaluate_parameters

    module = _import_launch_module()
    ctx = _make_launch_context(_REPO_ROOT / "robots" / robot_id / "robot.yaml")
    entities = module.compose_runtime_graph(ctx)  # type: ignore[attr-defined]

    # ``LifecycleNode.node_name`` only resolves after the action has
    # been executed by the launch service. We instead match by the
    # private ``_Node__package`` attribute (the literal string the
    # launch file passes to the constructor), which is unique enough:
    # ``openral_safety_kernel`` only appears once in the graph.
    kernel_nodes = [
        e
        for e in entities
        if isinstance(e, LifecycleNode)
        and getattr(e, "_Node__package", None) == "openral_safety_kernel"
    ]
    assert len(kernel_nodes) == 1, (
        f"expected exactly one LifecycleNode(package='openral_safety_kernel') in the "
        f"launch graph, got {len(kernel_nodes)}"
    )
    (kernel_node,) = kernel_nodes

    evaluated = evaluate_parameters(ctx, kernel_node._Node__parameters)
    # evaluate_parameters returns a list-of-dicts (one per parameters=
    # entry). The kernel passes a single dict.
    assert len(evaluated) == 1, f"safety_kernel passed {len(evaluated)} parameter dicts, expected 1"
    assert isinstance(evaluated[0], dict), (
        f"safety_kernel passed a non-dict parameter set: {type(evaluated[0]).__name__}"
    )
    return dict(evaluated[0])


@pytest.mark.parametrize("robot_id", _FIXED_BASE_ROBOTS)
def test_fixed_base_arm_kernel_params_have_no_empty_lists(robot_id: str) -> None:
    """Fixed-base arms: compose_runtime_graph must not emit empty-list ROS params.

    The historical regression: ``collision_base_dofs = []`` for every
    fixed-base arm crashed launch_ros's ``evaluate_parameter_dict``
    with ``"Expected 'value' to be one of [float,int,str,bool,bytes], "
    "but got '()' of type 'tuple'"``. Asserting "no empty list in
    kernel_params" pins the omit-when-empty contract from
    ``sim_e2e.launch.py:397`` (mirrors ``lifecycle_peer_node_ids``
    guard at line 482 and ``workspace_box_min_xyz`` omission in
    ``kernel_params_from_envelope``).
    """
    params = _safety_kernel_params(robot_id)
    empties = {k: v for k, v in params.items() if isinstance(v, list | tuple) and len(v) == 0}
    assert not empties, (
        f"safety_kernel parameter dict for {robot_id!r} contains empty list/tuple "
        f"values which launch_ros would collapse to '()' and reject:\n"
        + "\n".join(f"  {k}: {v!r}" for k, v in empties.items())
    )
    # Sanity floor: at least the scalar contract from kernel_params_from_envelope
    # made it through — otherwise the test would silently pass on an empty dict.
    assert "n_dof" in params
    assert "joint_position_min" in params
    assert "joint_position_max" in params
    # The omitted parameter is the actual regression target.
    assert "collision_base_dofs" not in params, (
        f"{robot_id!r} declares no ``base_joints`` in robot.yaml so "
        "``collision_base_dofs`` must be omitted (not empty)."
    )


@pytest.mark.parametrize("robot_id", _MOBILE_BASE_ROBOTS)
def test_mobile_base_arm_kernel_params_have_collision_base_dofs(robot_id: str) -> None:
    """Mobile-base robots: ``collision_base_dofs`` is present and non-empty.

    Pairs with :func:`test_fixed_base_arm_kernel_params_have_no_empty_lists`
    so the symmetric "omit-when-empty, include-when-populated" contract is
    pinned end-to-end. panda_mobile declares ``base_joints`` in its
    manifest; the param must reach the kernel so the FK can zero the
    base dofs (ADR-0040 mobile-base self-collision correctness).
    """
    params = _safety_kernel_params(robot_id)
    assert "collision_base_dofs" in params, (
        f"{robot_id!r} declares ``base_joints`` in robot.yaml; "
        "``collision_base_dofs`` MUST reach the kernel."
    )
    base_dofs = params["collision_base_dofs"]
    assert isinstance(base_dofs, list | tuple) and len(base_dofs) > 0, (
        f"{robot_id!r} ``collision_base_dofs`` must be non-empty; got {base_dofs!r}"
    )
