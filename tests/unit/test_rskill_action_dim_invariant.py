"""Invariant: a joint-like rSkill must fit the robot's declared controllable DoFs.

Before this invariant was enforced at fixture load, the only way the runner
could discover that a checkpoint's action vector exceeded the robot's joint
count was at *runtime* — the safety supervisor's ``n_dof`` envelope check
would fire, the HAL would E-stop, and the reasoner would spin in retry loops.

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

``dim < controllable_dofs`` is permitted (the checkpoint doesn't command the
trailing joints — e.g. a LIBERO 7-D action on a Franka with a declared
gripper joint; the gripper stays put). ``dim > controllable_dofs`` is the
failure case — the action vector contains channels that aren't declared on the
robot. Dexterous-hand fingers count via ``RobotDescription.end_effectors[*].n_dof``;
GR-1 declares 17 body joints + 12 Fourier-hand DoFs that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import ActionRepresentation, ControlMode, RSkillManifest
from openral_sim.registry import ROBOTS

_RSKILLS_ROOT = Path("rskills")


def _is_pure_joint_position(manifest: RSkillManifest) -> bool:
    """True iff the manifest claims its action vector is straight joint targets.

    Excludes manifests that declare an ``action_contract.slots`` block —
    those carry per-slice control modes whose typed contract is enforced
    by the slot validator at fixture load, not by the dim-vs-joints
    heuristic this test encodes.
    """
    actuators = manifest.actuators_required or []
    if not actuators:
        return False
    if not all(ar.kind is ControlMode.JOINT_POSITION for ar in actuators):
        return False
    if manifest.action_contract is None:
        return False
    # Slot-bearing manifests are exempt — the ActionSlot
    # cross-validator already proves coverage + per-mode field
    # requirements; the dim<=joints check would mis-fire on the
    # RoboCasa OSC layout (dim=12 vs panda_mobile 11 joints) even
    # though the slot dispatcher routes correctly.
    if manifest.action_contract.slots:
        return False
    rep = manifest.action_contract.representation
    return rep is None or rep is ActionRepresentation.JOINT_POSITIONS


def _controllable_dofs(robot_name: str) -> int:
    robot = ROBOTS.get(robot_name)()
    dex_hand_dofs = sum(
        int(ee.n_dof)
        for ee in robot.end_effectors
        if ee.kind == "dexterous_hand" and getattr(ee, "actuated", False)
    )
    return len(robot.joints) + dex_hand_dofs


def _collect_check_cases() -> list[tuple[str, str, int, int]]:
    """Enumerate (rskill_name, robot_name, action_dim, controllable_dofs) tuples.

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
            cases.append(
                (
                    manifest_path.parent.name,
                    tag,
                    manifest.action_contract.dim,
                    _controllable_dofs(tag),
                )
            )
    return cases


def _param_id(case: tuple[str, str, int, int]) -> str:
    rskill, robot, dim, dofs = case
    return f"{rskill}__on__{robot}__dim{dim}_vs_dofs{dofs}"


@pytest.mark.parametrize("case", [pytest.param(c, id=_param_id(c)) for c in _collect_check_cases()])
def test_joint_position_rskill_action_dim_within_robot_joints(
    case: tuple[str, str, int, int],
) -> None:
    """``action_contract.dim`` must not exceed declared joint count.

    See module docstring for the full invariant statement and the
    pending-slot-layout exemption list.
    """
    rskill, robot, action_dim, dofs = case
    assert action_dim <= dofs, (
        f"rskill {rskill!r} claims joint_position actuators with action_contract.dim={action_dim}, "
        f"but robot {robot!r} has only {dofs} controllable DoFs declared. "
        f"Either the manifest mis-declares the actuator kind (it's emitting non-joint channels — "
        f"declare action_contract.representation or wait for per-slot action-contract support), "
        f"or the robot under-declares its actuators (missing joints / hand DoFs in "
        f"robots/{robot}/robot.yaml)."
    )
