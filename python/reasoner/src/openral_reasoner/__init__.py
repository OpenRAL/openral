"""OpenRAL S2 reasoner — typed LLM tool dispatch.

The reasoner is the slow planning loop (CLAUDE.md §6.2 — S2;
event-driven with a 0.2 Hz heartbeat since the 2026-05-25 amendment)
that consumes a :class:`~openral_core.WorldState` snapshot, a rolling
buffer of :class:`FailureEventRecord` / :class:`PerceptionEventRecord`
/ :class:`PromptRecord`, and emits **exactly one** typed
:data:`~openral_core.ReasonerToolCall` per tick via the LLM's
structured tool-use API. The ROS-side ``reasoner_node`` (in
``packages/openral_reasoner_ros``) wraps this core with rclpy
subscriptions and dispatch plumbing.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from openral_reasoner.active_search import (
    SearchBudget,
    SearchCandidate,
    SearchProgress,
    format_search_frontier,
    plan_active_search,
)
from openral_reasoner.context import (
    ContextRenderer,
    FailureEventRecord,
    PerceptionEventRecord,
    PromptRecord,
)
from openral_reasoner.core import PreparedTick, ReasonerCore, ReasonerTickResult
from openral_reasoner.cosmos3 import (
    COSMOS3_BASE_URL,
    DEFAULT_COSMOS3_MODEL,
    Cosmos3ToolUseClient,
    find_cosmos3_sidecar_script,
)
from openral_reasoner.critic_watchdog import CriticWatchdog, CriticWatchdogGroup
from openral_reasoner.memory import MemoryEntry, MemoryStore
from openral_reasoner.mission import (
    DEFAULT_MAX_ATTEMPTS,
    MissionState,
    TaskState,
    evaluate_task_verdict,
)
from openral_reasoner.node_policy import (
    CASCADE_PROMPT_SOURCES,
    should_rebuild_mission,
)
from openral_reasoner.palette import ToolPalette, build_tool_palette
from openral_reasoner.persistence import (
    ReasonerLadderState,
    load_ladder_state,
    save_ladder_state,
)
from openral_reasoner.spatial_query import (
    SpatialMemoryQuerier,
    SpatialQueryOutcome,
    SpatialQueryTool,
    format_recall_object_result,
    format_resolve_place_result,
    recall_object_tool_to_query,
    resolve_place_tool_to_query,
    run_spatial_query,
    run_spatial_query_detailed,
)
from openral_reasoner.tool_use import (
    DEFAULT_SYSTEM_PROMPT,
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT_ENV_VAR,
    AnthropicToolUseClient,
    OpenAICompatibleToolUseClient,
    ToolUseClient,
    build_tool_use_client_from_env,
    render_robot_context_prompt,
    resolve_reasoner_system_prompt,
)

__all__ = [
    "CASCADE_PROMPT_SOURCES",
    "COSMOS3_BASE_URL",
    "DEFAULT_COSMOS3_MODEL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SYSTEM_PROMPT",
    "OPENROUTER_BASE_URL",
    "SYSTEM_PROMPT_ENV_VAR",
    "AnthropicToolUseClient",
    "ContextRenderer",
    "Cosmos3ToolUseClient",
    "CriticWatchdog",
    "CriticWatchdogGroup",
    "FailureEventRecord",
    "MemoryEntry",
    "MemoryStore",
    "MissionState",
    "OpenAICompatibleToolUseClient",
    "PerceptionEventRecord",
    "PreparedTick",
    "PromptRecord",
    "ReasonerCore",
    "ReasonerLadderState",
    "ReasonerTickResult",
    "SearchBudget",
    "SearchCandidate",
    "SearchProgress",
    "SpatialMemoryQuerier",
    "SpatialQueryOutcome",
    "SpatialQueryTool",
    "TaskState",
    "ToolPalette",
    "ToolUseClient",
    "build_tool_palette",
    "build_tool_use_client_from_env",
    "evaluate_task_verdict",
    "find_cosmos3_sidecar_script",
    "format_recall_object_result",
    "format_resolve_place_result",
    "format_search_frontier",
    "load_ladder_state",
    "plan_active_search",
    "recall_object_tool_to_query",
    "render_robot_context_prompt",
    "resolve_place_tool_to_query",
    "resolve_reasoner_system_prompt",
    "run_spatial_query",
    "run_spatial_query_detailed",
    "save_ladder_state",
    "should_rebuild_mission",
]
__version__ = _pkg_version("openral-reasoner")
