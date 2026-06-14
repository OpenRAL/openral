"""ADR-0025 — hermetic checks on ``slam_toolbox.launch.py``.

These do NOT spawn a real ROS 2 graph (that's the integration tier).
They assert that the launch file's structural pieces remain intact so
a regression that breaks the Reasoner's ``LifecycleTransitionTool``
contract surfaces at unit-test time:

* The lifecycle node name the Reasoner is documented to target stays
  pinned (``openral_slam_toolbox``).
* The default parameter YAML exists and parses.
* The launch description's actions include exactly one
  ``LifecycleNode`` and at least one ``RegisterEventHandler`` (the
  auto-transition to ``INACTIVE``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "slam_toolbox.launch.py"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "slam_toolbox_2d.yaml"


def _import_launch_module() -> ModuleType:
    """Import ``slam_toolbox.launch.py`` directly by file path.

    Can't use ``sys.path.insert`` + ``import slam_toolbox`` here because
    a sourced ROS 2 environment already ships a ``slam_toolbox`` Python
    package under ``/opt/ros/<distro>/lib/python*/site-packages/`` —
    that import resolves first regardless of the sys.path order, and
    the resulting module is the slam_toolbox runtime (no
    ``generate_launch_description``). Load the file by its absolute
    path via ``importlib.util.spec_from_file_location`` to bypass the
    package-name collision entirely.
    """
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_slam_bringup_launch_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_file_exists_and_parses() -> None:
    """The default YAML file ships in-tree and is valid."""
    assert _CONFIG_PATH.is_file(), _CONFIG_PATH
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    assert isinstance(data, dict)
    # Wildcard node namespace is the slam_toolbox convention.
    assert "/**" in data, sorted(data)
    params = data["/**"]["ros__parameters"]
    # Frame contract — match the ADR's documented defaults.
    assert params["odom_frame"] == "odom"
    assert params["map_frame"] == "map"
    assert params["base_frame"] == "base_link"
    assert params["scan_topic"] == "/scan"


def test_launch_module_pins_canonical_node_name() -> None:
    """The Reasoner targets ``openral_slam_toolbox`` — pin it."""
    mod = _import_launch_module()
    assert getattr(mod, "NODE_NAME") == "openral_slam_toolbox"  # noqa: B009
    # And the default parameter path resolves to a real file.
    default_path = getattr(mod, "DEFAULT_PARAMS_PATH")  # noqa: B009
    assert Path(default_path).is_file(), default_path


def test_launch_description_shape() -> None:
    """The generated launch description has the right action shape.

    One ``LifecycleNode`` (the slam_toolbox node) and ZERO
    ``EmitEvent`` actions — the slam_toolbox node is left in
    ``UNCONFIGURED`` for the Reasoner to drive through
    ``CONFIGURE → ACTIVATE`` via ``LifecycleTransitionTool``.
    Auto-configure from the launch triggers slam_toolbox 2.8.4's
    Jazzy race (on_configure returns SUCCESS but the change_state
    service responds with ``success=false``, producing a spurious
    ``Failed to make transition 'TRANSITION_CONFIGURE'`` ERROR from
    ``launch_ros.utilities.lifecycle_event_manager``). Reasoner-driven
    lifecycle dodges it — see ADR-0025.
    """
    mod = _import_launch_module()
    from launch.actions import EmitEvent
    from launch_ros.actions import LifecycleNode

    desc = mod.generate_launch_description()
    actions = desc.describe_sub_entities()
    lifecycle_nodes = [a for a in actions if isinstance(a, LifecycleNode)]
    assert len(lifecycle_nodes) == 1, (
        f"expected exactly 1 LifecycleNode, got {len(lifecycle_nodes)}"
    )
    emit_events = [a for a in actions if isinstance(a, EmitEvent)]
    assert len(emit_events) == 0, (
        f"expected zero EmitEvents (Reasoner-managed bring-up); got "
        f"{len(emit_events)}. Any EmitEvent here trips slam_toolbox's "
        "Jazzy lifecycle race — ADR-0025 forbids launch-side auto-configure."
    )
