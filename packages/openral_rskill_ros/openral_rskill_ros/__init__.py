"""ADR-0018 F1 — `openral_rskill_ros` lifecycle node package.

Exports the public symbols (``RskillRunnerNode``, ``compose_runtime``,
``compose_so100_runtime``, ``main``) the rest of the OpenRAL graph
uses to bring up the in-process runner that consumes the shared
``WorldStateAggregator`` (single subscriber of ``/joint_states`` per
ADR-0018 §3) and publishes ``openral_msgs/ActionChunk`` on
``/openral/candidate_action``.
"""

from __future__ import annotations

from openral_rskill_ros.compose import (
    ComposedRuntime,
    compose_runtime,
    compose_so100_runtime,
)
from openral_rskill_ros.rskill_runner_node import (
    RskillRunnerNode,
    main,
    make_default_skill_resolver,
    make_local_skill_resolver,
)

__all__ = [
    "ComposedRuntime",
    "RskillRunnerNode",
    "compose_runtime",
    "compose_so100_runtime",
    "main",
    "make_default_skill_resolver",
    "make_local_skill_resolver",
]
