"""Repo-wide safeguard: every VLA rSkill's ``action_contract`` is executable
on each declared embodiment (ADR-0036 §4).

This is the regression net behind the worked ADR-0036 sweeps
(``test_libero_action_contracts.py`` / ``test_cartesian_rskill_contracts.py``):
those assert the *specific* skills that were fixed; this walks **every**
``rskills/*/rskill.yaml`` so a *future* cartesian rSkill cannot silently ship a
bare ``action_contract.dim`` (no ``representation`` / ``slots``) that the
skill_runner would mis-dispatch as ``JOINT_POSITION`` on a non-matching robot —
the exact bug ADR-0036 fixed for the LIBERO class.

The check is pure ``openral_core`` (representation / slots / dim vs the robot's
joints + a sim-executable control-mode set). It deliberately does **not** import
``openral_reasoner_ros.reasoner_node`` (that pulls rclpy + openral_msgs); the
small ADR-0036 gate rule is re-derived here so the validator runs under a bare
``uv run --no-sync`` without a ROS environment. It mirrors
``reasoner_node._required_control_modes`` and uses the canonical
``openral_core.SIM_EXECUTABLE_CONTROL_MODES`` (ADR-0036 amendment 2026-06-04 —
single source of truth shared by the reasoner gate and the HAL packers, so this
validator can no longer drift from what the deploy-sim path actually executes).

The per-(skill, robot) rule, in order of contract specificity:

* ``action_contract is None`` → no action constraint → pass.
* ``representation`` set → ``control_modes_for_representation(representation)``;
  additionally ``canonical_slots_for_representation`` must resolve without error
  against the robot (proves the cartesian/gripper layout binds to a real
  end-effector and the dim is wide enough).
* ``slots`` set → ``{s.control_mode for s in slots if s.control_mode}`` (the
  ActionSlot cross-validator already proved coverage + per-mode fields at load).
* Bare ``dim`` only → implicitly joint-space (``{JOINT_POSITION}``); additionally
  ``dim`` must equal the robot's actuated-joint count. A mismatch means a
  cartesian (or otherwise non-joint) skill is under-declared — it would default
  the whole vector to ``JOINT_POSITION`` and trip the joint-space envelope. The
  one principled relaxation: robots with actuated *dexterous hands* carry
  actuated DoF (finger joints) that ``robots/<id>/robot.yaml`` does not enumerate
  in ``joints``, so the strict equality becomes ``dim >= len(joints)`` for them
  (still rejects a too-small / cartesian-masquerading vector).

For declared (representation / slots) contracts the required modes must also be
a subset of the deploy-sim-executable set
``openral_core.SIM_EXECUTABLE_CONTROL_MODES`` ({JOINT_POSITION, JOINT_VELOCITY,
CARTESIAN_DELTA, GRIPPER_POSITION, BODY_TWIST, COMPOSITE_MODE}) — i.e. the
deploy-sim OSC / composite path can actually pack and execute them. Real-mode
executability against a specific robot's ``supported_control_modes`` is the
reasoner gate's runtime job; this test guards declaration sanity,
sim-executability, and the bare-dim==joints invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    SIM_EXECUTABLE_CONTROL_MODES,
    ControlMode,
    RobotDescription,
    RSkillManifest,
)
from openral_core.schemas import (
    canonical_slots_for_representation,
    control_modes_for_representation,
)

_REPO = Path(__file__).resolve().parents[2]
_RSKILLS_ROOT = _REPO / "rskills"
_ROBOTS_ROOT = _REPO / "robots"

# Deploy-sim executable control modes are the canonical
# ``openral_core.SIM_EXECUTABLE_CONTROL_MODES`` (ADR-0036 amendment 2026-06-04):
# the exact set the default sim HAL action-packers
# (``openral_hal.sim_attached``) can pack + execute via the robosuite
# OSC / composite controller in the MuJoCo twin, pinned to the packers by
# ``tests/unit/test_sim_executable_modes_match_packers.py``. ``COMPOSITE_MODE`` is
# included (the robosuite-composite multiplexer flag, ADR-0028d); the modes that
# no packer implements (JOINT_TORQUE / JOINT_TRAJECTORY / CARTESIAN_POSE /
# GRIPPER_BINARY) are excluded — admitting them here would let an
# unexecutable-in-sim contract pass declaration validation, the very kind of
# latent false-admit ADR-0036 removed from the gate.

# Embodiment-tag aliases → canonical ``robots/<dir>`` fixture name. The closed
# ``EmbodimentTag`` vocabulary already uses ``franka_panda`` directly, but tags
# can carry framework / dataset aliases that name the same physical robot. Tags
# with no matching ``robots/<tag>/robot.yaml`` and no alias here are sim-benchmark
# aliases (e.g. ``mobile_base`` class tag), not robot fixtures, and are skipped.
_TAG_ALIASES: dict[str, str] = {
    "franka": "franka_panda",
    "panda": "franka_panda",
    "libero": "franka_panda",
    # The closed ``EmbodimentTag`` vocabulary uses short class names for two
    # robots whose fixture dir carries a longer/suffixed name.
    "aloha": "aloha_bimanual",
    "pusht": "pusht_2d",
}


def _robot_fixture_names() -> set[str]:
    """Robot fixture dir names that carry a ``robot.yaml`` under ``robots/``."""
    return {p.parent.name for p in _ROBOTS_ROOT.glob("*/robot.yaml")}


def _resolve_tag_to_fixture(tag: str, fixtures: set[str]) -> str | None:
    """Map an embodiment tag to an in-tree robot fixture dir, or ``None``.

    Direct ``robots/<tag>/robot.yaml`` match wins; otherwise an alias in
    :data:`_TAG_ALIASES` that itself resolves to a fixture; otherwise ``None``
    (the tag is a sim-benchmark / class alias with no robot fixture).
    """
    if tag in fixtures:
        return tag
    alias = _TAG_ALIASES.get(tag)
    if alias is not None and alias in fixtures:
        return alias
    return None


# Known deferred exception, keyed by the manifest's ``name`` field. MetaWorld
# emits a 3-D end-effector delta + 1-D gripper (dim=4); that representation has
# no ``ActionRepresentation`` / ``ControlMode`` yet (it needs a future ADR-0036
# follow-up ``DELTA_EE_6D``-style ``DELTA_EE_3D_PLUS_GRIPPER``). So its bare
# ``dim=4`` contract on the sawyer fixture (8 joints) genuinely fails the
# bare-dim==joints rule. The test SKIPS validation for these names but ALSO
# asserts each is *still* failing (below) so the exception self-removes the
# moment the skill is fixed.
_KNOWN_DEFERRED: dict[str, str] = {
    "OpenRAL/rskill-smolvla-metaworld": (
        "MetaWorld 3-D EE delta+gripper needs ADR-0036 follow-up "
        "DELTA_EE_3D_PLUS_GRIPPER; tracked separately"
    ),
    "OpenRAL/rskill-3d-diffuser-actor-rlbench": (
        "3D Diffuser Actor emits end-effector cartesian_pose trajectories; the "
        "deploy-sim OSC path executes delta/joint modes only (ADR-0036). RLBench "
        "runs it via its own Mover, not the deploy-sim OSC path; tracked separately"
    ),
}


def _has_actuated_dexterous_hand(robot: RobotDescription) -> bool:
    """Whether the robot carries actuated dexterous-hand end-effectors.

    Such robots (e.g. GR-1) have actuated finger DoF that
    ``robots/<id>/robot.yaml`` does not enumerate in ``joints``, so a bare-dim
    joint-space contract legitimately exceeds ``len(joints)``.
    """
    return any(ee.kind == "dexterous_hand" and ee.actuated for ee in robot.end_effectors)


def _required_control_modes(manifest: RSkillManifest) -> set[ControlMode]:
    """Re-derived ``reasoner_node._required_control_modes`` (ROS-free).

    No ``action_contract`` → no constraint (empty set). ``representation`` →
    :func:`control_modes_for_representation`. ``slots`` → each non-discard slot's
    mode. Bare ``dim`` → ``{JOINT_POSITION}``.
    """
    contract = manifest.action_contract
    if contract is None:
        return set()
    if contract.representation is not None:
        return control_modes_for_representation(contract.representation)
    if contract.slots is not None:
        return {s.control_mode for s in contract.slots if s.control_mode is not None}
    return {ControlMode.JOINT_POSITION}


def _check_action_contract_executable(manifest: RSkillManifest, robot: RobotDescription) -> None:
    """Assert ``manifest``'s action contract is executable on ``robot``.

    Raises ``AssertionError`` (via ``pytest``) on violation; returns ``None`` on
    a valid contract. See the module docstring for the full rule.
    """
    contract = manifest.action_contract
    if contract is None:
        return  # no action constraint declared

    required = _required_control_modes(manifest)

    # Bare-dim contract → implicitly joint-space; dim must match the robot's
    # actuated-joint count (relaxed for dexterous-hand robots whose finger DoF
    # are not enumerated in the fixture).
    if contract.representation is None and contract.slots is None:
        assert required == {ControlMode.JOINT_POSITION}  # invariant of the bare-dim branch
        n_joints = len(robot.joints)
        if _has_actuated_dexterous_hand(robot):
            assert contract.dim >= n_joints, (
                f"rSkill {manifest.name!r}: bare action_contract.dim={contract.dim} is below the "
                f"dexterous-hand robot {robot.name!r}'s {n_joints} declared body joints; "
                f"a joint-space vector cannot be narrower than the body chain."
            )
        else:
            assert contract.dim == n_joints, (
                f"rSkill {manifest.name!r}: bare action_contract.dim={contract.dim} (no "
                f"representation/slots) is implicitly JOINT_POSITION, but robot {robot.name!r} has "
                f"{n_joints} actuated joints. A bare-dim contract is joint-space; this mismatch "
                f"means a non-joint (likely cartesian) skill is under-declared and would be "
                f"mis-dispatched as JOINT_POSITION. Declare action_contract.representation "
                f"(e.g. delta_ee_6d_plus_gripper) or explicit slots. See ADR-0036."
            )
        return

    # Declared representation → the canonical slot layout must bind to a real
    # end-effector and be wide enough for ``dim`` (raises ROSConfigError otherwise).
    if contract.representation is not None:
        canonical_slots_for_representation(
            contract.representation,
            dim=contract.dim,
            description=robot,
        )

    # Declared (representation or slots) → required modes must be deploy-sim
    # executable (the OSC / composite controller can synthesise them).
    unexecutable = {ControlMode(m) for m in required} - SIM_EXECUTABLE_CONTROL_MODES
    assert not unexecutable, (
        f"rSkill {manifest.name!r} on robot {robot.name!r} requires control modes "
        f"{sorted(m.value for m in unexecutable)} that the deploy-sim OSC path cannot execute. "
        f"Executable set: {sorted(m.value for m in SIM_EXECUTABLE_CONTROL_MODES)}. See ADR-0036."
    )


def _collect_cases() -> list[tuple[Path, str]]:
    """Enumerate (manifest_path, robot_fixture_dir) pairs for every VLA rSkill.

    One pair per (VLA skill × embodiment tag that resolves to a robot fixture).
    Non-VLA skills are skipped (no action_contract semantics here); a skill whose
    tags resolve to *no* fixture contributes no pair (it cannot be validated
    without a robot).
    """
    fixtures = _robot_fixture_names()
    cases: list[tuple[Path, str]] = []
    for manifest_path in sorted(_RSKILLS_ROOT.glob("*/rskill.yaml")):
        manifest = RSkillManifest.from_yaml(str(manifest_path))
        if manifest.kind != "vla":
            continue
        seen: set[str] = set()
        for tag in manifest.embodiment_tags or []:
            fixture = _resolve_tag_to_fixture(tag, fixtures)
            if fixture is None or fixture in seen:
                continue
            seen.add(fixture)
            cases.append((manifest_path, fixture))
    return cases


def _case_id(case: tuple[Path, str]) -> str:
    manifest_path, fixture = case
    return f"{manifest_path.parent.name}__on__{fixture}"


_CASES = _collect_cases()


def test_at_least_one_case_collected() -> None:
    """Guard against the parametrize list silently collecting nothing."""
    assert _CASES, "no (VLA rSkill × robot fixture) pairs collected — glob/mapping broke"


@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_vla_action_contract_executable_on_embodiment(case: tuple[Path, str]) -> None:
    """Every VLA rSkill's action contract is executable on each declared robot.

    Skips names in :data:`_KNOWN_DEFERRED`; the companion self-policing test
    proves each deferred skill is *still* genuinely failing the rule.
    """
    manifest_path, fixture = case
    manifest = RSkillManifest.from_yaml(str(manifest_path))
    if manifest.name in _KNOWN_DEFERRED:
        pytest.skip(f"{manifest.name}: deferred — {_KNOWN_DEFERRED[manifest.name]}")
    robot = RobotDescription.from_yaml(str(_ROBOTS_ROOT / fixture / "robot.yaml"))
    _check_action_contract_executable(manifest, robot)


@pytest.mark.parametrize("deferred_name", sorted(_KNOWN_DEFERRED), ids=sorted(_KNOWN_DEFERRED))
def test_deferred_skill_still_genuinely_fails(deferred_name: str) -> None:
    """Each ``_KNOWN_DEFERRED`` entry must still fail the rule (self-policing).

    The moment a deferred skill's contract is fixed (so the rule passes), this
    test fails — telling the dev to remove it from ``_KNOWN_DEFERRED`` rather
    than leave a dead exception masking a real future regression.
    """
    fixtures = _robot_fixture_names()
    # Locate the deferred manifest + its first resolvable robot fixture.
    for manifest_path in sorted(_RSKILLS_ROOT.glob("*/rskill.yaml")):
        manifest = RSkillManifest.from_yaml(str(manifest_path))
        if manifest.name != deferred_name:
            continue
        for tag in manifest.embodiment_tags or []:
            fixture = _resolve_tag_to_fixture(tag, fixtures)
            if fixture is None:
                continue
            robot = RobotDescription.from_yaml(str(_ROBOTS_ROOT / fixture / "robot.yaml"))
            try:
                _check_action_contract_executable(manifest, robot)
            except AssertionError:
                return  # still genuinely failing — exception is justified
            pytest.fail(
                f"{deferred_name!r} now PASSES the action-contract executability rule on "
                f"{fixture!r}; remove it from _KNOWN_DEFERRED — its deferral no longer applies."
            )
        pytest.fail(
            f"{deferred_name!r} resolves to no robot fixture; cannot prove it still fails. "
            f"Re-check _KNOWN_DEFERRED / the embodiment→fixture mapping."
        )
    pytest.fail(
        f"_KNOWN_DEFERRED name {deferred_name!r} matches no rskills/*/rskill.yaml manifest."
    )
