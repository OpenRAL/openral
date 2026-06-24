"""Hermetic guards for the read-only Foxglove bridge spike.

These do NOT spawn a ROS graph (mirroring
``openral_slam_bringup/test/test_slam_toolbox_launch.py``). They assert the
two safety-relevant invariants of the launch module: the surface is
read-only, and the safety/actuation topics are never exposed.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parent.parent
_LAYOUT = _PKG_DIR / "config" / "openral_layout.json"


def _load_topics() -> object:
    """Load ``topics.py`` by path so the test runs without the ament package installed.

    This package is an ament_cmake ROS package (not a uv/pip workspace member), so
    ``import openral_foxglove_bringup`` is unavailable under the plain pytest run —
    mirroring how ``openral_slam_bringup``'s launch test loads its target by path.
    """
    spec = importlib.util.spec_from_file_location(
        "_fxbringup_topics", _PKG_DIR / "openral_foxglove_bringup" / "topics.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_topics = _load_topics()
BUCKET1_TOPIC_WHITELIST = _topics.BUCKET1_TOPIC_WHITELIST
READ_ONLY_CAPABILITIES = _topics.READ_ONLY_CAPABILITIES

# Topics that MUST NOT be reachable through the bridge: any of these matching
# the allowlist would let a viewer see (and, if capabilities regressed,
# poke at) the actuation/safety plane.
_FORBIDDEN_TOPICS = [
    "/openral/estop",
    "/openral/human_estop",
    "/openral/safe_action",
    "/openral/candidate_action",
    "/openral/failure/safety",
    "/openral/prompt_in/dashboard",
    # Compressed patterns must not accidentally widen to safety topics either.
    "/openral/cameras/base/image/compressed/estop",
    "/openral/estop/compressed",
    "/openral/safe_action/compressed",
]


def test_capabilities_are_read_only() -> None:
    """No client-publish or service capability — the surface cannot actuate."""
    assert "clientPublish" not in READ_ONLY_CAPABILITIES
    assert "services" not in READ_ONLY_CAPABILITIES
    # Only the read-only viz capabilities are advertised.
    assert set(READ_ONLY_CAPABILITIES) <= {"connectionGraph", "assets"}


@pytest.mark.parametrize("topic", _FORBIDDEN_TOPICS)
def test_safety_topics_not_whitelisted(topic: str) -> None:
    """The Bucket-1 allowlist must not match any safety/actuation topic."""
    assert not any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST), (
        f"{topic} is reachable through the bridge allowlist"
    )


@pytest.mark.parametrize(
    "topic",
    [
        "/openral/cameras/top/image",
        "/map",
        "/octomap_point_cloud_centers",
        "/joint_states",
        "/tf",
        "/tf_static",
        "/robot_description",
    ],
)
def test_bucket1_topics_are_whitelisted(topic: str) -> None:
    """Every native panel's source topic is actually exposed."""
    assert any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST), (
        f"{topic} is NOT reachable — its panel would be empty"
    )


@pytest.mark.parametrize(
    "topic",
    [
        "/openral/cameras/top/image/compressed",
        "/openral/cameras/base/image/compressed",
        "/openral/cameras/left_wrist/image/compressed",
        "/openral/cameras/right_wrist/image/compressed",
    ],
)
def test_compressed_camera_topics_are_whitelisted(topic: str) -> None:
    """image_transport compressed sibling topics are exposed (ADR-0059 decision 4)."""
    assert any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST), (
        f"{topic} is NOT whitelisted — compressed panel would be empty"
    )


def test_layout_is_valid_json_and_references_real_topics() -> None:
    """The shipped layout parses and only references whitelisted topics."""
    layout = json.loads(_LAYOUT.read_text())
    blob = json.dumps(layout)
    for topic in re.findall(r"/[\w/]+", blob):
        # Only check things that look like our topics, skip TF frame ids etc.
        if topic.startswith("/openral") or topic in {"/map", "/joint_states", "/odom", "/scan"}:
            assert any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST), (
                f"layout references {topic} which the bridge does not expose"
            )
