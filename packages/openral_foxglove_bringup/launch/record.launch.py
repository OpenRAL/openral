#!/usr/bin/env python3
"""Opt-in MCAP bag recorder scoped to the Bucket-1 topic allowlist.

Records only the topics that the read-only Foxglove bridge exposes, so a
recorded session can be replayed offline in Foxglove (File > Open Local File
→ ``*.mcap``).

Usage::

    ros2 launch openral_foxglove_bringup record.launch.py
    ros2 launch openral_foxglove_bringup record.launch.py output_dir:=my_session
    ros2 launch openral_foxglove_bringup record.launch.py use_sim_time:=true

Read-only — publishes/commands nothing. Scope mirrors the Bucket-1 allowlist:
safety/e-stop/action topics (``/openral/estop``, ``/openral/safe_action``,
``/openral/candidate_action``, ``/openral/failure/*``) are absent and never
recorded.

Regex semantics: ``ros2 bag record -e`` uses Python ``re.search`` (partial
match), while ``foxglove_bridge``'s ``topic_whitelist`` uses
``std::regex_match`` (full-string). A pattern like ``r"/map"`` can therefore
also match ``/something/map_thing`` under ``-e`` — verify the recorded bag
captures no unexpected topics. Patterns avoid wildcards on security-sensitive
prefixes.
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
                "Set true when a /clock is published (deploy-sim) so "
                "bag timestamps align with sim time and the replay scrubber "
                "matches the original session."
            ),
        ),
    ]

    output_dir = LaunchConfiguration("output_dir")

    # ``ros2 bag record -e`` expects one OR-ed pattern; patterns come from the
    # single source of truth in ``topics.py``, never edited here.
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
