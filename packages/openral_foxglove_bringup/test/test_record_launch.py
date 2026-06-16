"""Hermetic guards for the opt-in MCAP recorder launch.

Does NOT spawn a ROS graph, mirroring ``test_foxglove_launch.py``.  The test
verifies three invariants:

1. ``record.launch.py`` builds a ``LaunchDescription`` containing an
   ``ExecuteProcess`` whose command is ``ros2 bag record -s mcap …``.
2. The recorded scope references the Bucket-1 patterns and does NOT include
   any forbidden topic literals.
3. The edited ``config/openral_layout.json`` is valid JSON whose
   ``/openral/…`` and navigation topics all belong to an expected-allowed set.

Note: ``/openral/world_collisions_markers`` and
``/openral/world_voxels_cloud`` are the two Phase-3 Bucket-2 topics added to
the layout by this phase.  They are expected to be added to
``BUCKET1_TOPIC_WHITELIST`` in ``topics.py`` by the coordinator (Phase 4
wiring).  Until that merge, the test asserts them against an explicit
``_EXPECTED_ALLOWED`` set that already includes them so the test reflects
intent.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parent.parent
_LAUNCH_FILE = _PKG_DIR / "launch" / "record.launch.py"
_LAYOUT = _PKG_DIR / "config" / "openral_layout.json"

# ---------------------------------------------------------------------------
# Load topics.py by path (ament package not pip-installed under plain pytest).
# ---------------------------------------------------------------------------

def _load_topics() -> object:
    """Load ``topics.py`` by path so the test runs without the ament package installed."""
    spec = importlib.util.spec_from_file_location(
        "_fxbringup_topics_rec", _PKG_DIR / "openral_foxglove_bringup" / "topics.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_topics = _load_topics()
BUCKET1_TOPIC_WHITELIST: list[str] = _topics.BUCKET1_TOPIC_WHITELIST

# Topics that MUST NOT appear in the recorder's scope.  Their absence keeps
# the recorded MCAP free of safety/actuation data (ADR-0059 decision 5).
_FORBIDDEN_TOPIC_LITERALS = [
    "/openral/estop",
    "/openral/human_estop",
    "/openral/safe_action",
    "/openral/candidate_action",
    "/openral/failure/safety",
    "/openral/prompt_in/dashboard",
]

# Topics the layout is allowed to reference: the current Bucket-1 whitelist
# PLUS the two Phase-3 Bucket-2 topics that the coordinator adds to
# ``topics.py`` as part of Phase 4.  Using an explicit set here lets the test
# express intent without depending on the coordinator's merge order.
_EXPECTED_ALLOWED: set[str] = {
    "/openral/cameras/0/image",
    "/openral/world_collisions_markers",  # Phase-3 Bucket-2 — coordinator adds to whitelist
    "/openral/world_voxels_cloud",        # Phase-3 Bucket-2 — coordinator adds to whitelist
    "/map",
    "/octomap_point_cloud_centers",
    "/scan",
    "/odom",
    "/joint_states",
    "/robot_description",
    "/tf",
    "/tf_static",
}


# ---------------------------------------------------------------------------
# Load record.launch.py by path (same pattern as topics.py above).
# ---------------------------------------------------------------------------

def _load_record_launch() -> object:
    """Import record.launch.py by path, injecting the bringup package shim."""
    # Ensure topics.py is importable under the name the launch file imports.
    import sys
    import types

    bringup_pkg = sys.modules.get("openral_foxglove_bringup")
    if bringup_pkg is None:
        bringup_pkg = types.ModuleType("openral_foxglove_bringup")
        sys.modules["openral_foxglove_bringup"] = bringup_pkg
    bringup_pkg.topics = _topics  # type: ignore[attr-defined]
    # Expose the names the launch file does ``from openral_foxglove_bringup.topics import …``.
    topics_mod = types.ModuleType("openral_foxglove_bringup.topics")
    topics_mod.BUCKET1_TOPIC_WHITELIST = _topics.BUCKET1_TOPIC_WHITELIST  # type: ignore[attr-defined]
    topics_mod.READ_ONLY_CAPABILITIES = _topics.READ_ONLY_CAPABILITIES  # type: ignore[attr-defined]
    sys.modules["openral_foxglove_bringup.topics"] = topics_mod

    spec = importlib.util.spec_from_file_location("_record_launch", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_record_launch_has_execute_process() -> None:
    """``generate_launch_description()`` must contain an ``ExecuteProcess``."""
    launch_mod = _load_record_launch()
    ld = launch_mod.generate_launch_description()  # type: ignore[attr-defined]
    # Import lazily so the test file itself doesn't require ROS on the path
    # at collection time; the import is deferred to execution.
    from launch.actions import ExecuteProcess  # type: ignore[import]

    execute_actions = [a for a in ld.entities if isinstance(a, ExecuteProcess)]
    assert execute_actions, "No ExecuteProcess found in the launch description"


def _flatten_cmd(ep: object) -> list[str]:
    """Flatten ``ExecuteProcess.cmd`` to literal string tokens.

    ``launch`` wraps each token in a ``list[Substitution]``.  ``TextSubstitution``
    items carry their literal text in ``.text``; ``LaunchConfiguration`` items are
    runtime-resolved and are skipped here (we only need to inspect literals).
    """
    from launch.substitutions import TextSubstitution  # type: ignore[import]

    tokens: list[str] = []
    for sublist in ep.cmd:  # type: ignore[union-attr]
        parts: list[str] = []
        for item in sublist:
            if isinstance(item, TextSubstitution):
                parts.append(item.text)
        if parts:
            tokens.append("".join(parts))
    return tokens


def test_record_command_is_ros2_bag_record_mcap() -> None:
    """The ``ExecuteProcess`` command must include ``ros2 bag record --storage mcap``."""
    launch_mod = _load_record_launch()
    ld = launch_mod.generate_launch_description()  # type: ignore[attr-defined]
    from launch.actions import ExecuteProcess  # type: ignore[import]

    ep: ExecuteProcess = next(a for a in ld.entities if isinstance(a, ExecuteProcess))
    cmd_tokens = _flatten_cmd(ep)

    assert "ros2" in cmd_tokens, f"'ros2' not in cmd: {cmd_tokens}"
    assert "bag" in cmd_tokens, f"'bag' not in cmd: {cmd_tokens}"
    assert "record" in cmd_tokens, f"'record' not in cmd: {cmd_tokens}"
    assert "mcap" in cmd_tokens, f"'mcap' not in cmd: {cmd_tokens}"
    assert any(t in cmd_tokens for t in ("-s", "--storage")), (
        f"Neither '-s' nor '--storage' found in cmd: {cmd_tokens}"
    )


def test_record_scope_references_bucket1_patterns() -> None:
    """The regex alternation fed to --regex must contain the Bucket-1 patterns."""
    launch_mod = _load_record_launch()
    ld = launch_mod.generate_launch_description()  # type: ignore[attr-defined]
    from launch.actions import ExecuteProcess  # type: ignore[import]

    ep: ExecuteProcess = next(a for a in ld.entities if isinstance(a, ExecuteProcess))
    cmd_tokens = _flatten_cmd(ep)
    regex_idx = next(
        (i for i, t in enumerate(cmd_tokens) if t in ("--regex", "-e")), None
    )
    assert regex_idx is not None, "--regex / -e flag not found in cmd"
    # The alternation string is the very next token after the flag.
    regex_value = cmd_tokens[regex_idx + 1]
    for pat in BUCKET1_TOPIC_WHITELIST:
        assert pat in regex_value, (
            f"Bucket-1 pattern {pat!r} not found in --regex value: {regex_value!r}"
        )


@pytest.mark.parametrize("topic", _FORBIDDEN_TOPIC_LITERALS)
def test_record_scope_excludes_forbidden_topics(topic: str) -> None:
    """No forbidden topic literal should be matched by any Bucket-1 pattern."""
    assert not any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST), (
        f"Forbidden topic {topic} is reachable through the Bucket-1 patterns"
    )


def test_layout_is_valid_json() -> None:
    """The layout file must parse as valid JSON."""
    layout = json.loads(_LAYOUT.read_text())
    assert isinstance(layout, dict)


def test_layout_bucket2_panel_present() -> None:
    """The layout must contain the Phase-3 Bucket-2 3D panel."""
    layout = json.loads(_LAYOUT.read_text())
    assert "3D!bucket2" in layout["configById"], (
        "Bucket-2 panel '3D!bucket2' not found in configById"
    )


def test_layout_bucket2_topics_referenced() -> None:
    """The Bucket-2 panel must reference both Phase-3 converter topics."""
    layout = json.loads(_LAYOUT.read_text())
    panel_topics: dict = layout["configById"]["3D!bucket2"]["topics"]
    assert "/openral/world_collisions_markers" in panel_topics, (
        "/openral/world_collisions_markers not in 3D!bucket2 topics"
    )
    assert "/openral/world_voxels_cloud" in panel_topics, (
        "/openral/world_voxels_cloud not in 3D!bucket2 topics"
    )


def test_layout_openral_topics_are_allowed() -> None:
    """Every /openral/… and known nav topic in the layout must be in the allowed set."""
    layout = json.loads(_LAYOUT.read_text())
    blob = json.dumps(layout)
    for topic in re.findall(r"/[\w/]+", blob):
        if topic.startswith("/openral") or topic in {"/map", "/joint_states", "/odom", "/scan"}:
            assert topic in _EXPECTED_ALLOWED, (
                f"layout references {topic!r} which is not in the expected-allowed set; "
                "add it to _EXPECTED_ALLOWED or ensure the coordinator whitelists it in topics.py"
            )
