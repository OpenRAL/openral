"""Hermetic guards for the read-only Foxglove bridge spike.

These do NOT spawn a ROS graph (mirroring
``openral_slam_bringup/test/test_slam_toolbox_launch.py``). They assert the
two safety-relevant invariants of the launch module: the surface is
read-only, and the safety/actuation topics are never exposed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openral_foxglove_bringup.topics import (
    BUCKET1_TOPIC_WHITELIST,
    READ_ONLY_CAPABILITIES,
)

_LAYOUT = Path(__file__).resolve().parent.parent / "config" / "openral_layout.json"

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
        "/openral/cameras/0/image",
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
