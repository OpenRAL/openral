"""Reconcile a scene's requested ``robot_uids`` against a ManiSkill3 task's
``SUPPORTED_ROBOTS`` allowlist.

MS3's per-task allowlist enumerates *base* agent uids only — it does not list
registered camera-variant subclasses (e.g. ``panda_wristcam``, a ``Panda``
subclass that only adds a ``hand_camera``). So MS3 logs a false
"not in the task's list of supported robots" warning for a genuinely-usable
robot. The adapter reconciles by walking the agent's MRO: a variant whose base
uid is supported is accepted; a genuinely-unsupported robot raises a typed
``ROSCapabilityMismatch`` at the boundary instead of MS3's vague warning +
downstream crash.

Registry-only (no env construction / GPU); skipped when maniskill3 is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mani_skill")

import logging

from openral_core.exceptions import ROSCapabilityMismatch
from openral_sim.backends.maniskill3 import (
    _reconcile_robot_uids,
    _suppress_unsupported_robot_warning,
)


def test_reconcile_accepts_directly_supported_robot() -> None:
    # 'panda' is in PickCube-v1's SUPPORTED_ROBOTS — accepted, no raise.
    _reconcile_robot_uids("PickCube-v1", "panda")


def test_reconcile_accepts_camera_variant_of_supported_base() -> None:
    # 'panda_wristcam' is a registered Panda subclass (MRO base uid 'panda',
    # which IS supported) — accepted despite not being enumerated directly.
    _reconcile_robot_uids("PickCube-v1", "panda_wristcam")


def test_reconcile_raises_typed_error_for_unsupported_robot() -> None:
    with pytest.raises(ROSCapabilityMismatch):
        _reconcile_robot_uids("PickCube-v1", "totally_not_a_registered_robot")


def test_suppress_drops_only_the_false_unsupported_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("mani_skill")
    with (
        _suppress_unsupported_robot_warning(),
        caplog.at_level(logging.WARNING, logger="mani_skill"),
    ):
        log.warning(
            "panda_wristcam is not in the task's list of supported robots. "
            "Code may not run as intended"
        )
        log.warning("a genuinely important mani_skill warning")
    messages = [r.getMessage() for r in caplog.records]
    assert not any("not in the task's list of supported robots" in m for m in messages)
    assert any("genuinely important" in m for m in messages)


def test_suppress_filter_is_removed_on_exit() -> None:
    log = logging.getLogger("mani_skill")
    before = list(log.filters)
    with _suppress_unsupported_robot_warning():
        pass
    assert list(log.filters) == before  # no leaked filter
