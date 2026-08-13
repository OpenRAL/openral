"""``openral_msgs/SafetyStatus`` constant contract (ADR-0096).

``SafetyStatus.drop_reason`` reuses ``FailureTrigger``'s ``KIND_*`` numbers so
one number means the same fault everywhere on the graph. ROS IDL has no
cross-message constant reuse, so the block is *redeclared* in
``SafetyStatus.msg`` — which means nothing but a test stops the two files from
drifting apart. That drift would be silent and wrong in the worst way: a
dashboard or reasoner would render (or switch on) the wrong fault kind for a
real safety stop.

This also pins the second half of the contract: the ``DROP_*`` range must stay
**disjoint** from every ``KIND_*`` value, because ADR-0096's consumers are
allowed to tell "the kernel is fault-latched" from "the kernel is dropping
because an upstream input is unavailable" from ``drop_reason`` alone, without
first reading ``latched``.

Needs the colcon ``openral_msgs`` overlay (``just ros2-build`` + ``source
install/setup.bash``); skips without it (CLAUDE.md §1.11 — never faked).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openral_msgs")

from openral_msgs.msg import (  # noqa: E402  # reason: import follows the overlay gate
    FailureTrigger,
    SafetyStatus,
)

# Every FailureTrigger kind SafetyStatus redeclares. KIND_SUPPRESSED_SUMMARY
# (254) is deliberately absent: it is a rate-limiter roll-up marker on the
# failure *bus*, never a reason actions are being suppressed right now.
_SHARED_KINDS = (
    "KIND_TIMEOUT",
    "KIND_FORCE",
    "KIND_WORKSPACE",
    "KIND_PERCEPTION",
    "KIND_CRITIC",
    "KIND_SELFVERIFY",
    "KIND_HUMAN",
    "KIND_WAM",
    "KIND_REASONER_TIMEOUT",
    "KIND_CONTROLLER",
    "KIND_COLLISION",
)

_DROP_CONSTANTS = (
    "DROP_ENVELOPE_UNCONFIGURED",
    "DROP_WORLD_UNAVAILABLE",
    "DROP_VOXEL_UNAVAILABLE",
    "DROP_STATE_UNAVAILABLE",
    "DROP_WORLD_OVERFLOW",
    "DROP_VOXEL_OVERFLOW",
    "DROP_EXTERNAL_ESTOP",
    "DROP_NONE",
)


@pytest.mark.parametrize("name", _SHARED_KINDS)
def test_kind_constants_match_failure_trigger(name: str) -> None:
    """Each redeclared ``KIND_*`` carries FailureTrigger's exact number."""
    assert getattr(SafetyStatus, name) == getattr(FailureTrigger, name), (
        f"SafetyStatus.{name} drifted from FailureTrigger.{name}; one number "
        "must mean the same fault on both topics"
    )


def test_drop_constants_are_disjoint_from_kind_constants() -> None:
    """No ``DROP_*`` value may collide with any ``KIND_*`` value."""
    kinds = {getattr(SafetyStatus, name) for name in _SHARED_KINDS}
    kinds.add(FailureTrigger.KIND_SUPPRESSED_SUMMARY)
    drops = {getattr(SafetyStatus, name) for name in _DROP_CONSTANTS}
    assert kinds.isdisjoint(drops), f"DROP_*/KIND_* overlap: {sorted(kinds & drops)}"
    assert len(drops) == len(_DROP_CONSTANTS), "two DROP_* constants share a value"


def test_drop_none_is_not_the_default_wire_value() -> None:
    """``DROP_NONE`` must be an explicit value, never the zero default.

    A default-initialised ``SafetyStatus`` decodes ``drop_reason == 0``, which
    is ``KIND_TIMEOUT``. Publishers therefore have to *set* ``DROP_NONE`` on
    every clear/recovery transition; a consumer must never read "all clear"
    out of an unset field.
    """
    assert SafetyStatus.DROP_NONE != 0
    assert SafetyStatus().drop_reason == 0
    assert SafetyStatus().latched is False


def test_message_carries_the_adr_field_set() -> None:
    """Fields exist with the ADR-0096 names and types, header included."""
    msg = SafetyStatus()
    msg.latched = True
    msg.drop_reason = SafetyStatus.DROP_STATE_UNAVAILABLE
    msg.detail = "measured joint-state seed stale"
    msg.rskill_id = "rskills/rskill-smolvla-so100"
    msg.trace_id = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    msg.header.stamp.sec = 12
    assert msg.latched is True
    assert msg.drop_reason == SafetyStatus.DROP_STATE_UNAVAILABLE
    assert msg.header.stamp.sec == 12
