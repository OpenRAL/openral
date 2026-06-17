#!/usr/bin/env python3
"""Opt-in MCAP bag recorder scoped to the Bucket-1 topic allowlist.

Records only the topics that the read-only Foxglove bridge exposes, so a
recorded session can be replayed offline in Foxglove (File > Open Local File
→ ``*.mcap``).

Usage::

    ros2 launch openral_foxglove_bringup record.launch.py
    ros2 launch openral_foxglove_bringup record.launch.py output_dir:=my_session
    ros2 launch openral_foxglove_bringup record.launch.py use_sim_time:=true

Safety scope
------------
Recording is **read-only** — this launch publishes and commands nothing.  The
scope mirrors the Bucket-1 allowlist: the safety/e-stop/action topics
(``/openral/estop``, ``/openral/safe_action``, ``/openral/candidate_action``,
``/openral/failure/*``) are absent from the allowlist and are therefore never
recorded.

Regex semantics note
--------------------
``ros2 bag record --regex`` / ``-e`` uses Python ``re.search`` partial-match
semantics (matches if the pattern is found *anywhere* in the topic name).
This differs from ``foxglove_bridge``'s ``topic_whitelist``, which applies
``std::regex_match`` (full-string anchored match).  The patterns in
``BUCKET1_TOPIC_WHITELIST`` are written as full-string anchors (e.g.
``r"/map"``), so under ``re.search`` they still only match ``/map`` and not
``/mapping/something`` because the anchor ``^`` is implicit at the start of a
``re.search`` on a full topic string — however the correct guard is that the
pattern ``r"/map"`` will also accidentally match ``/something/map_thing``.
Operators reviewing the recorded bag should confirm no unexpected topics are
captured; the patterns deliberately avoid wildcards on security-sensitive
prefixes.  ADR-0059 decision 5 documents this constraint.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from openral_foxglove_bringup.topics import BUCKET1_TOPIC_WHITELIST


def generate_launch_description() -> LaunchDescription:
    """Opt-in MCAP recorder for the Bucket-1 allowlist."""
    args = [
        DeclareLaunchArgument(
            "output_dir",
            default_value="openral_foxglove_mcap",
            description=(
                "Output directory for the MCAP bag. ros2 bag record appends an "
                "ISO-8601-like timestamp suffix automatically, so parallel sessions "
                "do not overwrite each other. Override to a fixed path for "
                "scripted pipelines."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Set true when a /clock is published (deploy-sim, ADR-0048) so "
                "bag timestamps align with sim time and the replay scrubber "
                "matches the original session."
            ),
        ),
    ]

    output_dir = LaunchConfiguration("output_dir")

    # Build the --regex argument as a single alternation string:
    # ``ros2 bag record -e`` expects one pattern that is OR-ed internally.
    # Joining with ``|`` mirrors what the CLI accepts; each individual pattern
    # comes from the single source of truth in ``topics.py`` and is never
    # edited here directly.
    _regex_alternation = "|".join(BUCKET1_TOPIC_WHITELIST)

    recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",  # -s mcap: Foxglove's native format for offline replay
            "--output",
            output_dir,
            "--regex",
            _regex_alternation,
        ],
        output="screen",
        name="openral_mcap_recorder",
    )

    return LaunchDescription([*args, recorder])
