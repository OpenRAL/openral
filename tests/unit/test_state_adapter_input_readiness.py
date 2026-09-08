# SPDX-License-Identifier: Apache-2.0
"""A state layout must say *why* it cannot be assembled, not raise ``KeyError``.

Regression for #227: on 2026-09-05, a sim on ``spark`` with ``ROS_DOMAIN_ID``
unset joined DDS domain 0 and picked up ``/joint_states`` from a live bimanual
OpenArm on another host. The unguarded ``joint_positions[...]`` index died
with a bare ``KeyError: 'panda_gripper'``, which the skill runner's catch-all
turned into an uninformative ``deadline-no-grasp``. The typed error below
prints what the robot *does* publish, which is what identified the foreign
robot.

Two faults, told apart:
* an **empty** frame (nothing arrived yet) → ``ROSPerceptionStale``;
* a **populated** frame lacking the bound joint → ``ROSConfigError`` naming
  the joints it saw (wrong manifest, or another robot's frame).

The empty-frame case has not been observed live but is kept distinct on
principle; a readiness wait built on the assumption it *was* the observed
case was removed after ten clean rounds never fired it.

No mocks (CLAUDE.md §1.11) — real registry, real layout assembler, bindings
read from the real shipped rSkill manifest.
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
