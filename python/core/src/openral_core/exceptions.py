"""openral exception hierarchy — use these, do not invent new base classes.

All exceptions derive from ``ROSError``. ``ROSSafetyViolation`` and its
subclasses must **never** be caught silently; they are only caught at the safety
supervisor boundary where they trigger an E-stop and a structured incident log.

Example:
    >>> try:
    ...     raise ROSConfigError("missing URDF")
    ... except ROSError as exc:
    ...     print(type(exc).__name__, str(exc))
    ROSConfigError missing URDF
"""

__all__ = [
    "ROSBTValidationError",
    "ROSCapabilityMismatch",
    "ROSCollisionImminent",
    "ROSConfigError",
    "ROSDeadlineMissed",
    "ROSDispatchUnavailable",
    "ROSEStopRequested",
    "ROSError",
    "ROSFleetError",
    "ROSForceLimitExceeded",
    "ROSGPUMemoryError",
    "ROSInferenceTimeout",
    "ROSObjectNotInMemory",
    "ROSPerceptionStale",
    "ROSPlanningError",
    "ROSQuantizationError",
    "ROSReasonerInvalidPlan",
    "ROSRskillGoalSatisfied",
    "ROSRuntimeError",
    "ROSSafetyViolation",
    "ROSWorkspaceViolation",
]


# ─── Base ──────────────────────────────────────────────────────────────────────


class ROSError(Exception):
    """Base class for all OpenRAL errors."""


# ─── Configuration / capability ────────────────────────────────────────────────


class ROSConfigError(ROSError):
    """Bad manifest, missing weights, or invalid YAML / URDF."""


class ROSCapabilityMismatch(ROSError):
    """A skill requires a capability the target robot does not have."""


# ─── Runtime ───────────────────────────────────────────────────────────────────


class ROSRuntimeError(ROSError):
    """General runtime failure during skill execution or HAL operation."""


class ROSInferenceTimeout(ROSRuntimeError):
    """VLA inference did not complete within its latency budget."""


class ROSQuantizationError(ROSRuntimeError):
    """Quantization failed or produced an incompatible engine."""


class ROSGPUMemoryError(ROSRuntimeError):
    """Out of GPU memory; use a smaller skill variant or quantize further."""


# ─── Safety ────────────────────────────────────────────────────────────────────


class ROSSafetyViolation(ROSError):
    """A safety constraint was violated.

    NEVER catch this silently. Catch only at the safety supervisor boundary,
    trigger an E-stop, and emit a structured incident log entry.
    """


class ROSWorkspaceViolation(ROSSafetyViolation):
    """An action would move the robot outside its allowed workspace."""


class ROSForceLimitExceeded(ROSSafetyViolation):
    """Measured or predicted contact force exceeds the configured limit."""


class ROSCollisionImminent(ROSSafetyViolation):
    """A proposed motion would self-collide or strike a world obstacle.

    Raised on the safety path when geometric checking (ADR-0030) finds a
    chunk whose forward-kinematic sweep brings a robot link within its
    clearance of another link or a world primitive. Like every
    :class:`ROSSafetyViolation`, it is caught only at the safety supervisor
    boundary, where it triggers an E-stop and a structured incident log.
    """


class ROSEStopRequested(ROSSafetyViolation):
    """An emergency stop was requested; all actuation must cease immediately."""


# ─── Perception ────────────────────────────────────────────────────────────────


class ROSPerceptionStale(ROSError):
    """A sensor reading is older than the configured staleness deadline."""


class ROSObjectNotInMemory(ROSPerceptionStale):
    """A spatial-memory query matched no node, or only nodes that are stale.

    Raised by the ADR-0038 scene-graph query surface when a ``RecallObjectQuery``
    / ``ResolvePlaceQuery`` cannot be satisfied. The caller degrades by treating
    the target as unknown (and may trigger active search) — it never fabricates
    a pose (CLAUDE.md §1.2, §1.4).
    """


# ─── Planning ──────────────────────────────────────────────────────────────────


class ROSPlanningError(ROSError):
    """The reasoner failed to produce a valid plan."""


class ROSReasonerInvalidPlan(ROSPlanningError):
    """The LLM returned a plan that failed schema or capability validation."""


class ROSBTValidationError(ROSPlanningError):
    """The emitted BehaviorTree XML failed BT.CPP v4 validation."""


# ─── Fleet / dispatch ──────────────────────────────────────────────────────────


class ROSFleetError(ROSError):
    """A fleet-level or dispatch error."""


class ROSDispatchUnavailable(ROSFleetError):
    """No dispatcher (edge or cloud) is available for the requested skill."""


class ROSDeadlineMissed(ROSFleetError):
    """Cloud RTT exceeded the skill's deadline; fallback was not configured."""


# ─── Control-flow completion signal (NOT an error) ─────────────────────────────


class ROSRskillGoalSatisfied(ROSError):
    """A wrapped-ROS rSkill has finished its goal successfully.

    Used as a typed completion signal raised by
    :meth:`openral_rskill.ros_action_rskill.ROSActionRskill._step_impl` after
    the last waypoint of a one-shot planner (e.g. MoveIt) has been emitted,
    or after a result-only wrapped action (e.g. Nav2 ``NavigateToPose``)
    reports success. The ``ExecuteSkill`` action server catches this
    specifically and closes the goal with ``success=True``.

    This is NOT an error — it inherits :class:`ROSError` only to stay
    inside the OpenRAL exception surface so it is greppable and discoverable
    via the standard hierarchy. It must be caught ONLY at the
    ``rskill_runner_node`` execute-callback boundary; everywhere else it is
    a programming bug to raise it.
    """
