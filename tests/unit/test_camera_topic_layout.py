"""Lint: the canonical camera topic layout is spelled only by ``openral_core.camera_topic``.

ADR-0108 decision 4. Every producer and consumer builds ``/openral/cameras/<name>/<kind>``
through :func:`openral_core.camera_topic` (or ``CAMERA_TOPIC_PREFIX`` for a regex), so a
hand-built string is the drift the static ROS graph kept finding after it shipped. This test
parses the real sources and fails on any non-docstring string literal or f-string piece that
spells the layout. Docstrings and comments may still describe it.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("python", "packages", "tools")
EXCLUDED_PARTS = {".venv", "__pycache__", "vendor", "tests", "test", "openral_core"}
LAYOUT = "/openral/cameras"
#: ``(path, literal)`` pairs allowed to spell the layout, each with the reason it survives.
ALLOWED: set[tuple[str, str]] = set()


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Bare string statements: module/class/function docstrings and attribute docstrings."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }


def _hits(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and LAYOUT in node.value
        and id(node) not in docstrings
    ]


def _sources() -> list[Path]:
    return [
        p
        for root in ROOTS
        for p in sorted((REPO / root).rglob("*.py"))
        if not EXCLUDED_PARTS.intersection(p.relative_to(REPO).parts)
    ]


def test_scan_covers_the_known_producers() -> None:
    """Guard against a vacuous lint: the walk must reach the sites ADR-0108 moved."""
    scanned = {p.relative_to(REPO).as_posix() for p in _sources()}
    assert "python/hal/src/openral_hal/sim_sensor_bridge.py" in scanned
    assert "packages/openral_rskill_ros/launch/deploy_e2e.launch.py" in scanned
    assert "packages/world_state/openral_world_state_ros/lifecycle_node.py" in scanned


def test_detector_flags_a_hand_built_topic() -> None:
    """The detector fires on a literal and an f-string piece, not on a docstring."""
    source = (
        '"""Publishes /openral/cameras/<name>/image."""\n'
        'A = "/openral/cameras/top/image"\n'
        'B = f"/openral/cameras/{A}/points"\n'
    )
    assert sorted(_hits(source)) == [(2, "/openral/cameras/top/image"), (3, "/openral/cameras/")]


def test_no_hand_built_camera_topics() -> None:
    offenders = [
        f"{p.relative_to(REPO)}:{line}: {text!r}"
        for p in _sources()
        for line, text in _hits(p.read_text(encoding="utf-8"))
        if (p.relative_to(REPO).as_posix(), text) not in ALLOWED
    ]
    assert not offenders, (
        "Build camera topics with openral_core.camera_topic(name, CameraTopicKind.X) "
        "(or re.escape(CAMERA_TOPIC_PREFIX) for a pattern) instead of spelling "
        "the layout:\n" + "\n".join(offenders)
    )
