"""Invariant: an rskill claiming ``joint_position``-only actuators must not
emit more action dimensions than its target robot has declared joints
(ADR-0028a).

Pre-ADR-0028a, the only way the runner could discover that a checkpoint's
action vector exceeded the robot's joint count was at *runtime* — the
safety supervisor's ``n_dof`` envelope check would fire, the HAL would
E-stop, and the reasoner would spin in retry loops (see the trace at the
top of ``docs/adr/0028-rskill-action-contract-slots.md``).

This test pins the invariant at fixture load. For every
``rskills/*/rskill.yaml``:

* If the manifest declares ``actuators_required`` of pure
  ``ControlMode.JOINT_POSITION`` AND either no
  ``action_contract.representation`` or
  ``ActionRepresentation.JOINT_POSITIONS``: the manifest is *claiming*
  the action vector is straight joint targets.
* Then for every ``embodiment_tag`` that resolves to a registered
  ``RobotDescription`` (i.e. the tag is also a robot name under
  ``robots/``): ``action_contract.dim <= len(robot.joints)``.

``dim < len(joints)`` is permitted (the checkpoint doesn't command the
trailing joints — e.g. a LIBERO 7-D action on a Franka with a declared
gripper joint; the gripper stays put). ``dim > len(joints)`` is the
failure case — the action vector contains channels that aren't joints
(RoboCasa pi0.5 / rldx1's [arm_osc(6) + gripper(2) + base(3) + torso(1)]
layout, GR-1's full-body joints + hand fingers). Those rskills must
either honestly declare a non-joint ``representation`` or — once
ADR-0028b lands — carry an ``action_contract.slots`` block describing
the layout. Until 0028b, they are tracked here as ``xfail`` with a
pointer to the follow-up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import ActionRepresentation, ControlMode, RSkillManifest
from openral_sim.registry import ROBOTS

_RSKILLS_ROOT = Path("rskills")


def _is_pure_joint_position(manifest: RSkillManifest) -> bool:
    """True iff the manifest claims its action vector is straight joint targets.

    Excludes ADR-0028b manifests that declare an
    ``action_contract.slots`` block — those carry per-slice control
    modes whose typed contract is enforced by the slot validator at
    fixture load, not by the dim-vs-joints heuristic this test
    encodes.
    """
    actuators = manifest.actuators_required or []
    if not actuators:
        return False
    if not all(ar.kind is ControlMode.JOINT_POSITION for ar in actuators):
        return False
    if manifest.action_contract is None:
        return False
    # ADR-0028b: slot-bearing manifests are exempt — the ActionSlot
    # cross-validator already proves coverage + per-mode field
    # requirements; the dim<=joints check would mis-fire on the
    # RoboCasa OSC layout (dim=12 vs panda_mobile 11 joints) even
    # though the slot dispatcher routes correctly.
    if manifest.action_contract.slots:
        return False
    rep = manifest.action_contract.representation
    return rep is None or rep is ActionRepresentation.JOINT_POSITIONS


def _collect_check_cases() -> list[tuple[str, str, int, int]]:
    """Enumerate (rskill_name, robot_name, action_dim, robot_joint_count) tuples.

    One entry per (rskill × matching embodiment_tag). Embodiment tags
    that don't resolve to a registered robot are skipped — they're
    capability tags (``mobile_base``, ``franka``) rather than specific
    embodiments.
    """
    robot_names = set(ROBOTS.names())
    cases: list[tuple[str, str, int, int]] = []
    for manifest_path in sorted(_RSKILLS_ROOT.glob("*/rskill.yaml")):
        manifest = RSkillManifest.from_yaml(str(manifest_path))
        if not _is_pure_joint_position(manifest):
            continue
        if manifest.action_contract is None:  # narrowing
            continue
        for tag in manifest.embodiment_tags or []:
            if tag not in robot_names:
                continue
            robot = ROBOTS.get(tag)()
            cases.append(
                (manifest_path.parent.name, tag, manifest.action_contract.dim, len(robot.joints))
            )
    return cases


# Known mis-declared manifests pending follow-up work. Each pair is a
# (rskill_name, robot_name) tuple; the value is the human-readable
# reason. The three RoboCasa OSC manifests previously pinned here
# resolved when ADR-0028b step 7 added their ``action_contract.slots``
# blocks — ``_is_pure_joint_position`` now exempts slot-bearing
# manifests structurally.
_PENDING_SLOT_LAYOUT: dict[tuple[str, str], str] = {
    ("rldx1-ft-gr1-nf4", "gr1"): (
        "29-D action includes hand finger joints not declared in robots/gr1/robot.yaml; "
        "ADR-0028a out-of-scope (dexterous_hand finger joints). Separate ADR will "
        "add GR-1's per-finger DoFs to its robot.yaml."
    ),
}


def _param_id(case: tuple[str, str, int, int]) -> str:
    rskill, robot, dim, njoints = case
    return f"{rskill}__on__{robot}__dim{dim}_vs_joints{njoints}"


def _maybe_xfail(case: tuple[str, str, int, int]) -> pytest.param:
    rskill, robot, *_ = case
    reason = _PENDING_SLOT_LAYOUT.get((rskill, robot))
    if reason is None:
        return pytest.param(case, id=_param_id(case))
    return pytest.param(
        case, id=_param_id(case), marks=pytest.mark.xfail(reason=reason, strict=True)
    )


@pytest.mark.parametrize("case", [_maybe_xfail(c) for c in _collect_check_cases()])
def test_joint_position_rskill_action_dim_within_robot_joints(
    case: tuple[str, str, int, int],
) -> None:
    """``action_contract.dim`` must not exceed declared joint count.

    See module docstring for the full invariant statement and the
    pending-slot-layout exemption list.
    """
    rskill, robot, action_dim, joint_count = case
    assert action_dim <= joint_count, (
        f"rskill {rskill!r} claims joint_position actuators with action_contract.dim={action_dim}, "
        f"but robot {robot!r} has only {joint_count} joints declared. "
        f"Either the manifest mis-declares the actuator kind (it's emitting non-joint channels — "
        f"declare action_contract.representation or wait for ADR-0028b's slots), or the robot "
        f"under-declares its actuators (missing gripper / finger joints in "
        f"robots/{robot}/robot.yaml). "
        f"See ADR-0028 for the structural fix."
    )
