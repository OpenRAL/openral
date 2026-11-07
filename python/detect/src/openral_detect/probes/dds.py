"""ROS 2 / DDS topic probe — thin wrapper around ``openral_cli.autodetect``.

Adds RMW + domain-id capture and ``/tf`` / ``/robot_description`` presence
on top of the existing ``scan_dds_topics`` + ``infer_robot_from_topics``
helpers, and a warning naming the candidates when Unitree topics are seen but
``unitree_candidates_from_topics`` cannot narrow them to one model.
"""

from __future__ import annotations

import os

from openral_cli.autodetect import (
    DdsTopic,
    infer_robot_from_topics,
    scan_dds_topics,
    unitree_candidates_from_topics,
)

from openral_detect.report import DdsTopicRecord, Ros2TopologyResult

__all__ = ["probe_dds", "unitree_ambiguity_warning"]


def unitree_ambiguity_warning(topics: list[DdsTopic]) -> str | None:
    """Explain why a Unitree topology could not be resolved to one model.

    Truth over plausibility: when the topic list says "a Unitree robot" but its
    message family leaves more than one model, the candidates are surfaced
    instead of guessing one.

    Args:
        topics: Output of ``scan_dds_topics``.

    Returns:
        A ``dds:`` warning naming the candidate models and the ``--robot``
        override, or ``None`` when the topology resolves to one model or holds
        no Unitree topic.

    Example:
        >>> from openral_cli.autodetect import DdsTopic
        >>> msg = unitree_ambiguity_warning([DdsTopic("/lowstate", "unitree_hg/msg/LowState")])
        >>> "candidates: g1, h1_2, h2" in msg and "--robot" in msg
        True
        >>> unitree_ambiguity_warning([]) is None
        True
    """
    if infer_robot_from_topics(topics) is not None:
        return None
    unitree = unitree_candidates_from_topics(topics)
    if len(unitree) <= 1:
        return None
    return (
        "dds: Unitree low-level topics seen but the model is ambiguous "
        f"(candidates: {', '.join(unitree)}); pass `openral detect --robot <id>`."
    )


def probe_dds(
    *,
    timeout_s: float = 5.0,
    warnings: list[str] | None = None,
) -> Ros2TopologyResult:
    """Run a bounded DDS topic scan and capture the ROS 2 topology.

    Args:
        timeout_s: Wall-clock timeout passed to ``ros2 topic list -t``.
        warnings: Optional list to append non-fatal probe issues to.

    Returns:
        A populated ``Ros2TopologyResult``.  Empty when ``ros2``
        is not on ``$PATH`` or no topics are visible within ``timeout_s``.
    """
    try:
        topics = scan_dds_topics(timeout_s=timeout_s)
    except Exception as exc:
        if warnings is not None:
            warnings.append(f"dds: scan_dds_topics failed: {exc!r}")
        topics = []

    topic_records = [DdsTopicRecord(name=t.name, type_name=t.type_name) for t in topics]
    inferred = infer_robot_from_topics(topics) if topics else None
    ambiguity = unitree_ambiguity_warning(topics)
    if ambiguity is not None and warnings is not None:
        warnings.append(ambiguity)

    has_robot_description = any(t.name == "/robot_description" for t in topics)
    has_tf = any(t.name in ("/tf", "/tf_static") for t in topics)

    rmw = os.environ.get("RMW_IMPLEMENTATION", "")
    try:
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    except ValueError:
        domain_id = 0

    if not topics and warnings is not None:
        warnings.append("dds: no topics discovered (ros2 not sourced or no nodes running).")

    return Ros2TopologyResult(
        topics=topic_records,
        inferred_robot_type=inferred,
        has_robot_description=has_robot_description,
        has_tf=has_tf,
        nodes=[],  # populated by a future `ros2 node list` follow-up probe
        rmw_implementation=rmw,
        domain_id=domain_id,
    )
