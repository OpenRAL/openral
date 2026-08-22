"""``failure_bus`` ↔ ``openral_msgs/FailureTrigger`` constant-mirror contract.

``openral_observability.failure_bus`` restates the ``FailureTrigger`` ``KIND_*``
/ ``SEVERITY_*`` numbers as plain Python ints so a caller can publish (or read)
a typed failure event **without** a sourced ROS install — the ``openral`` CLI,
the sim runner, and every ROS-free unit test reach for that mirror rather than
the generated IDL.

Nothing pinned it until now, and it had already drifted: the mirror stopped at
``KIND_REASONER_TIMEOUT = 9`` and never grew ``KIND_COLLISION = 10``, which
``FailureTrigger.msg``, ``SafetyStatus.msg``, the safety kernel's
``ViolationKind`` and its ``publish_collision_failure`` all carry. The whole
collision stack therefore produced the one kind no ROS-free caller could name,
and the callers that needed it hard-coded the literal ``10``.

So this reads the **generated** constants off the colcon-built message class and
compares them against the module both ways round:

* every ``KIND_*`` / ``SEVERITY_*`` on the IDL exists in the mirror with the
  same value — the next constant added to the ``.msg`` fails here;
* every ``KIND_*`` / ``SEVERITY_*`` in the mirror still exists on the IDL — a
  constant *removed* from (or renamed in) the ``.msg`` fails here too, instead
  of leaving a Python name that silently means nothing on the wire;
* every mirrored constant is exported in ``__all__`` — the second half of "a
  caller can name it", and the half a hand-added constant forgets.

Reading the IDL by name is the same technique the reasoner's
``_failure_kind_value`` uses for ``ExecuteRskill.Result``: the test cannot
redeclare the numbers it is supposed to be checking.

Needs the colcon ``openral_msgs`` overlay (``just ros2-build`` + ``source
install/setup.bash``); skips without it (CLAUDE.md §1.11 — never faked). It is
listed in ``scripts/ros_live_tests.sh`` so the docker image — the only CI
surface that has the overlay — actually runs it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openral_msgs")

from openral_msgs.msg import FailureTrigger
from openral_observability import failure_bus

_PREFIXES = ("KIND_", "SEVERITY_")


def _idl_constants() -> dict[str, int]:
    """Every ``KIND_*`` / ``SEVERITY_*`` the generated message class declares."""
    return {
        name: int(getattr(FailureTrigger, name))
        for name in dir(FailureTrigger)
        if name.startswith(_PREFIXES)
    }


def _mirror_constants() -> dict[str, int]:
    """Every ``KIND_*`` / ``SEVERITY_*`` the Python mirror declares."""
    return {
        name: value
        for name in dir(failure_bus)
        if name.startswith(_PREFIXES) and isinstance(value := getattr(failure_bus, name), int)
    }


def test_the_idl_declares_the_constants_this_contract_is_about() -> None:
    """Guard the guard: an empty scrape would make every assertion below vacuous."""
    idl = _idl_constants()
    assert len(idl) >= 12, f"scraped only {sorted(idl)} off FailureTrigger — prefix moved?"
    assert "KIND_COLLISION" in idl, "FailureTrigger lost KIND_COLLISION"


def test_every_idl_constant_is_mirrored() -> None:
    """A constant added to ``FailureTrigger.msg`` must reach the Python mirror.

    This is the assertion that would have caught ``KIND_COLLISION`` on the PR
    that added it to the IDL, instead of a docs audit months later.
    """
    missing = sorted(_idl_constants().keys() - _mirror_constants().keys())
    assert not missing, (
        f"openral_msgs/FailureTrigger declares {missing} but "
        f"openral_observability.failure_bus does not — a ROS-free caller cannot "
        f"name {missing[0]} and has to hard-code its number"
    )


def test_the_mirror_declares_nothing_the_idl_does_not() -> None:
    """A mirrored name with no IDL counterpart means nothing on the wire."""
    extra = sorted(_mirror_constants().keys() - _idl_constants().keys())
    assert not extra, (
        f"openral_observability.failure_bus declares {extra}, absent from "
        "openral_msgs/FailureTrigger — a caller publishing it would put an "
        "unknown kind on the bus"
    )


@pytest.mark.parametrize("name", sorted(_idl_constants()))
def test_mirrored_value_matches_the_idl(name: str) -> None:
    """Each mirrored constant carries the generated message's exact number."""
    mirrored = _mirror_constants().get(name)
    # A missing name gets its own, better-worded failure above; here it is
    # enough that ``None`` never equals the IDL's number.
    assert mirrored == getattr(FailureTrigger, name), (
        f"failure_bus.{name} = {mirrored} drifted from "
        f"FailureTrigger.{name} = {getattr(FailureTrigger, name)}"
    )


def test_every_mirrored_constant_is_exported() -> None:
    """``__all__`` carries them all — an unexported constant is not API."""
    unexported = sorted(_mirror_constants().keys() - set(failure_bus.__all__))
    assert not unexported, f"failure_bus.__all__ omits {unexported}"


def test_kind_collision_is_nameable_without_a_sourced_ros_install() -> None:
    """Regression for the drift this file was written for.

    ``KIND_COLLISION`` is the kind the whole collision stack emits — the safety
    kernel's ``publish_collision_failure`` stamps it on every self/world
    collision stop — so it is precisely the one a ROS-free consumer of
    ``/openral/failure/safety`` needs to recognise.
    """
    assert failure_bus.KIND_COLLISION == FailureTrigger.KIND_COLLISION == 10
