"""world_state camera-image subscription QoS contract — structural guard.

Mirrors the AST-shape pattern of ``test_lifecycle_node_sigint_shape.py``.

ROS 2 QoS request-vs-offered matching is asymmetric: a BEST_EFFORT
subscription receives from BOTH a RELIABLE publisher (the sim HAL
bridges) and a BEST_EFFORT publisher (the real-mode GStreamer
``ros_tee`` / ``SensorRosPublisher`` camera readers, which follow the
CLAUDE.md §2 sensor-stream profile). A RELIABLE subscription only
matches RELIABLE publishers — with the real-camera publishers it
silently never connects, so WorldState receives zero frames and the
policy is starved of images on real hardware (`openral deploy run`).

This test parses ``lifecycle_node.py`` and asserts the per-camera image
``QoSProfile`` requests BEST_EFFORT reliability so a future refactor
can't silently reintroduce the mismatch.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE = _REPO_ROOT / "packages" / "world_state" / "openral_world_state_ros" / "lifecycle_node.py"


def _parse() -> ast.Module:
    """Parse the world_state lifecycle_node module as Python. Fail loudly if missing."""
    assert _NODE.is_file(), f"world_state lifecycle_node not found at {_NODE}"
    return ast.parse(_NODE.read_text(), filename=str(_NODE))


def _qos_profile_assignments(tree: ast.Module) -> list[ast.Call]:
    """Every ``<name> = QoSProfile(...)`` call in the module, by AST walk."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "QoSProfile"
        ):
            calls.append(node)
    return calls


def _reliability_attr(call: ast.Call) -> str | None:
    """The ``QoSReliabilityPolicy.<X>`` attribute passed as ``reliability=``, if any."""
    for kw in call.keywords:
        if kw.arg == "reliability" and isinstance(kw.value, ast.Attribute):
            return kw.value.attr
    return None


def test_image_subscription_qos_is_best_effort() -> None:
    """The image QoSProfile (assigned to ``image_qos``) must request BEST_EFFORT."""
    tree = _parse()
    image_qos_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "image_qos" in targets and isinstance(node.value, ast.Call):
                image_qos_calls.append(node.value)
    assert image_qos_calls, "expected an `image_qos = QoSProfile(...)` assignment"
    for call in image_qos_calls:
        reliability = _reliability_attr(call)
        assert reliability == "BEST_EFFORT", (
            f"image_qos requests QoSReliabilityPolicy.{reliability}; a RELIABLE "
            "subscription never matches the BEST_EFFORT real-camera publishers "
            "(GStreamer ros_tee / SensorRosPublisher) — WorldState would receive "
            "zero camera frames on `openral deploy run`."
        )
