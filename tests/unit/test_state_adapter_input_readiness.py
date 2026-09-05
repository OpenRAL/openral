# SPDX-License-Identifier: Apache-2.0
"""A state layout must say *why* it cannot be assembled, not raise ``KeyError``.

Every layout assembler indexed ``joint_positions[...]`` unguarded. When the
bound joint was absent it died with a bare ``KeyError`` from inside a layout
file, which the skill runner's catch-all converted into ``deadline-no-grasp`` —
a crash on the first policy step that reads in the artifacts exactly like a
policy timeout, with no hint of what was actually in the frame.

What was actually in the frame, on 2026-09-05, was **another robot**: a sim on
``spark`` with ``ROS_DOMAIN_ID`` unset joined DDS domain 0, multicast discovery
reached a live bimanual OpenArm on a different host, ``/joint_states`` had two
publishers, and the assembler read ``openarm_left_joint1 … openarm_right_joint7``
where it wanted ``panda_gripper``. ``KeyError: 'panda_gripper'`` said none of
that. The typed error below did — it prints what the robot *does* publish, and
that line is what identified the foreign robot (#227).

So the guard has to tell two faults apart, and the message for the second has
to name the joints it saw:

* an **empty** frame — nothing has arrived — is ``ROSPerceptionStale``;
* a **populated** frame that lacks the bound joint is ``ROSConfigError``: either
  the manifest is wrong for this robot, or the frame is some other robot's.

The first case is kept distinct on principle (it says nothing about the
manifest); it has not been observed live. A readiness wait built on the
assumption that it *was* the observed case was removed after ten clean rounds
showed it never fired.

No mocks (CLAUDE.md §1.11) — the real registry, the real layout assembler, and
the bindings read from the real shipped rSkill manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import StateContract
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale

pytest.importorskip("numpy")

from openral_state_adapter import assemble_state, registered_layouts

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "rskills" / "xr1-robocasa365" / "rskill.yaml"


def _shipped_contract() -> StateContract:
    """The real ``xr1-robocasa365`` state contract — the one that hit this live."""
    raw = yaml.safe_load(_MANIFEST.read_text())
    contract = StateContract.model_validate(raw["state_contract"])
    assert contract.bindings is not None, f"{_MANIFEST.name} must declare bindings"
    assert contract.layout in registered_layouts(), (
        f"{contract.layout!r} has no registered assembler; this test would be vacuous"
    )
    return contract


def _tf_lookup(_target: str, _source: str) -> Any:
    """A TF lookup that must never be reached — the joint guard precedes it."""
    raise AssertionError("tf_lookup must not be consulted when joints are missing")


def test_the_shipped_manifest_still_binds_a_gripper_joint() -> None:
    """Fixture precondition: without a bound joint neither case below can fire."""
    contract = _shipped_contract()
    assert contract.bindings is not None
    assert contract.bindings.gripper_qpos_joints, (
        f"{_MANIFEST.name} no longer binds gripper_qpos_joints; re-derive this test"
    )


def test_no_joint_frame_yet_is_a_typed_staleness_not_a_keyerror() -> None:
    """An empty frame: nothing has arrived, which says nothing about the manifest.

    Kept as its own typed outcome so a caller can tell "no data yet" from "wrong
    data". A ``KeyError`` here collapses both into one uninformative abort.
    """
    contract = _shipped_contract()
    assert contract.layout is not None and contract.bindings is not None

    with pytest.raises(ROSPerceptionStale) as excinfo:
        assemble_state(contract.layout, contract.bindings, {}, _tf_lookup)

    message = str(excinfo.value)
    assert "no joint state has arrived yet" in message
    for name in contract.bindings.gripper_qpos_joints:
        assert name in message, "the error must name the joints it was waiting for"


def test_a_joint_the_robot_never_publishes_is_a_config_error() -> None:
    """The observed case: frames ARE arriving and the bound joint is not among them.

    This is what the OpenArm cross-talk looked like from inside the assembler.
    The message must name what the robot does publish — that is the only part
    of the error that carried diagnostic weight on 2026-09-05.
    """
    contract = _shipped_contract()
    assert contract.layout is not None and contract.bindings is not None

    published = {"panda_joint1": 0.0, "panda_joint2": -1.0, "base_x": 0.0}
    with pytest.raises(ROSConfigError) as excinfo:
        assemble_state(contract.layout, contract.bindings, published, _tf_lookup)

    message = str(excinfo.value)
    assert "does not" in message and "publish" in message
    assert "panda_joint1" in message, "the error must say what the robot DOES publish"


def test_the_guard_covers_every_registered_layout_not_just_this_one() -> None:
    """The check lives in ``assemble_state``, so a new layout inherits it.

    Pinning this is the difference between one guard and one-guard-per-layout:
    the next layout added must not have to remember to re-implement it.
    """
    contract = _shipped_contract()
    assert contract.bindings is not None
    bindings = contract.bindings

    for layout in sorted(registered_layouts()):
        if not bindings.gripper_qpos_joints:
            continue
        with pytest.raises((ROSPerceptionStale, ROSConfigError)):
            assemble_state(layout, bindings, {}, _tf_lookup)
