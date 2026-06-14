"""Lockstep proof: ``SIM_EXECUTABLE_CONTROL_MODES`` == the sim HAL packers.

ADR-0036 (amended 2026-06-04). The reasoner's ``hal_mode="sim"`` palette
gate admits a VLA rSkill only when every :class:`ControlMode` its action
contract demands is in :data:`openral_core.SIM_EXECUTABLE_CONTROL_MODES`.
That constant is only safe if it is *exactly* the set of modes the default
sim HAL action-packers can execute — admit a mode no packer implements and
the skill boots fine, then E-stops mid-run when the first chunk of that
mode hits a packer ``else`` branch.

This test pins the two sides together, **both directions**, against the
REAL packers in :mod:`openral_hal.sim_attached` (CLAUDE.md §1.11 — no
mocks; pure-numpy packers driven with real :class:`Action` chunks and real
:class:`RobotDescription` manifests loaded from ``robots/``):

* :func:`pack_action_for_env` — the default free-function packer.
* :meth:`SimAttachedHAL._pack_with_composite_split` — the robosuite
  composite-slot packer.
* The ``BODY_TWIST`` direct-qpos path intercepted in
  :meth:`SimAttachedHAL.send_action` (NOT a packer ``else`` branch — see
  the BODY_TWIST note below).

A mode is classified **handled** by a packer when driving a representative
chunk of that mode does *not* raise the packer's unsupported-mode
``ROSConfigError`` (the one carrying ``"unsupported control_mode"``). A
*different* ``ROSConfigError`` (e.g. "composite has no 'right' part" when
no live env is bound) still means the mode *reached its own branch* — i.e.
the packer knows how to execute it — so it counts as handled. Only the
closing ``else`` (``"unsupported control_mode"``) marks a mode as
**rejected**.

To make the discrimination clean, every representative chunk also carries a
``joint_targets`` row: ``pack_action_for_env`` guards ``not joint_targets``
*before* the closing ``else``, so without a row an unsupported mode would
raise the empty-payload guard rather than the unsupported-mode ``else``.
Populating both fields lets every unsupported mode fall through to the real
``else``.

The union of the two packers' handled sets must equal
``SIM_EXECUTABLE_CONTROL_MODES`` exactly. The four latent false-admits
removed by this amendment (JOINT_TORQUE, JOINT_TRAJECTORY, CARTESIAN_POSE,
GRIPPER_BINARY) and the three never-admitted modes (CARTESIAN_TWIST,
FOOT_PLACEMENT, DEX_HAND_JOINT) must be rejected by BOTH packers AND absent
from the constant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The composite-split packer imports robosuite at call time; the default
# packer is pure numpy. Guard so a host without the sim extras skips
# rather than errors. (This test does NOT need rclpy/openral_msgs — it
# drives the HAL packers directly, not the reasoner node.)
pytest.importorskip("numpy")

from openral_core import SIM_EXECUTABLE_CONTROL_MODES, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import Action, ControlMode, Pose6D
from openral_hal.sim_attached import SimAttachedHAL, pack_action_for_env

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = REPO_ROOT / "robots"

# The else-branch message both packers raise for a mode they do not
# implement. Matching on this substring is what distinguishes a genuine
# "this packer cannot execute this mode" rejection from an in-branch
# guard (empty payload, missing composite part) that still proves the
# mode IS implemented.
_UNSUPPORTED_MARKER = "unsupported control_mode"

# The amendment's deletions and the perennial exclusions — neither group
# may appear in the canonical constant, and both packers must reject them.
_REMOVED_FALSE_ADMITS = frozenset(
    {
        ControlMode.JOINT_TORQUE,
        ControlMode.JOINT_TRAJECTORY,
        ControlMode.CARTESIAN_POSE,
        ControlMode.GRIPPER_BINARY,
    }
)
_NEVER_ADMITTED = frozenset(
    {
        ControlMode.CARTESIAN_TWIST,
        ControlMode.FOOT_PLACEMENT,
        ControlMode.DEX_HAND_JOINT,
    }
)


def _load_robot(name: str) -> RobotDescription:
    path = ROBOTS_DIR / name / "robot.yaml"
    if not path.exists():
        pytest.skip(f"robot fixture missing: {path}")
    return RobotDescription.from_yaml(str(path))


def _franka() -> RobotDescription:
    """Joint-only 7-DoF arm (base_dim=0, arm_dim=7) — cartesian/joint modes."""
    return _load_robot("franka_panda")


def _panda_mobile() -> RobotDescription:
    """Robosuite-composite mobile manipulator (base_dim=3, arm_dim=7) — base/composite modes."""
    return _load_robot("panda_mobile")


def _representative_action(mode: ControlMode) -> Action:
    """A real :class:`Action` of ``mode`` whose payload reaches the mode's branch.

    Widths are chosen so a handled mode passes its in-branch shape guard
    (so it never raises for the wrong reason); for the unhandled modes the
    payload is irrelevant because the packers fall straight through to the
    unsupported-mode ``else`` without inspecting the field.
    """
    pose = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="world")
    payloads: dict[ControlMode, dict[str, object]] = {
        # arm_dim=7 for both franka and panda_mobile.
        ControlMode.JOINT_POSITION: {"joint_targets": [[0.0] * 7]},
        # JOINT_VELOCITY chunk is padded to full n_dof (panda_mobile=11).
        ControlMode.JOINT_VELOCITY: {"joint_velocities": [[0.0] * 11]},
        ControlMode.JOINT_TORQUE: {"joint_torques": [[0.0] * 7]},
        ControlMode.JOINT_TRAJECTORY: {"joint_targets": [[0.0] * 7]},
        ControlMode.CARTESIAN_POSE: {"cartesian_pose": [pose]},
        ControlMode.CARTESIAN_DELTA: {"cartesian_delta": [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]},
        ControlMode.CARTESIAN_TWIST: {"cartesian_twist": [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]},
        ControlMode.BODY_TWIST: {"body_twist": [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]},
        ControlMode.FOOT_PLACEMENT: {"foot_placements": [{"x": 0.0}]},
        ControlMode.GRIPPER_BINARY: {"gripper": [0.0]},
        ControlMode.GRIPPER_POSITION: {"gripper": [0.0]},
        ControlMode.DEX_HAND_JOINT: {"dex_hand_joints": [[0.0] * 12]},
        ControlMode.COMPOSITE_MODE: {"composite_mode": [-1.0]},
    }
    # Always carry a joint_targets row so an unsupported mode reaches the
    # real "unsupported control_mode" else in pack_action_for_env instead
    # of tripping its earlier ``not joint_targets`` empty-payload guard.
    fields: dict[str, object] = {"joint_targets": [[0.0] * 7]}
    fields.update(payloads[mode])
    return Action(control_mode=mode, **fields)  # type: ignore[arg-type]  # reason: per-mode payload kwargs


def _bare_sim_hal(description: RobotDescription) -> SimAttachedHAL:
    """A SimAttachedHAL with no live env (composite split absent → 'no X part').

    ``_pack_with_composite_split`` resolves the composite controller from
    ``env.robots[0]``; an env object without ``robots`` yields ``None``, so
    a *supported* slot mode raises a "composite has no 'X' part"
    ``ROSConfigError`` (still in-branch → handled) while an unsupported
    mode raises the closing ``else`` ("unsupported control_mode" →
    rejected). That is exactly the discrimination this test needs, and it
    avoids standing up a full robosuite env for a packer-dispatch check.
    """

    class _NoCompositeEnv:
        """Minimal SimRollout stand-in (process boundary): no robots, no step."""

        action_dim = 12

    hal = SimAttachedHAL(_NoCompositeEnv(), description, env_action_dim=12)  # type: ignore[arg-type]  # reason: bare env is a deliberate no-composite SimRollout stand-in
    return hal


def _is_unsupported(exc: ROSConfigError) -> bool:
    return _UNSUPPORTED_MARKER in str(exc)


def _free_packer_handles(mode: ControlMode, description: RobotDescription) -> bool:
    """True iff ``pack_action_for_env`` does NOT reject ``mode`` as unsupported."""
    try:
        pack_action_for_env(_representative_action(mode), description, 12)
    except ROSConfigError as exc:
        if _is_unsupported(exc):
            return False
    return True


def _composite_packer_handles(mode: ControlMode, hal: SimAttachedHAL) -> bool:
    """True iff ``_pack_with_composite_split`` does NOT reject ``mode`` as unsupported."""
    # The composite packer imports robosuite.controllers.composite at call
    # time; that subpackage exists only in robosuite>=1.5 (the RoboCasa pin).
    # The base venv may carry robosuite==1.4 (the LIBERO pin, ADR-0011, which
    # is mutually exclusive with >=1.5) — skip cleanly there rather than
    # raising ModuleNotFoundError from inside the packer.
    pytest.importorskip("robosuite.controllers.composite.composite_controller")
    try:
        hal._pack_with_composite_split(_representative_action(mode))
    except ROSConfigError as exc:
        if _is_unsupported(exc):
            return False
    return True


def test_packer_union_equals_canonical_constant() -> None:
    """The union of both sim packers' handled modes == SIM_EXECUTABLE_CONTROL_MODES.

    Asserted both directions (set equality), the drift-catcher: a mode
    admitted by the gate but unimplemented by every packer, or a mode a
    packer gained without updating the gate, fails here.
    """
    franka = _franka()
    panda_mobile = _panda_mobile()
    hal = _bare_sim_hal(panda_mobile)

    handled: set[ControlMode] = set()
    for mode in ControlMode:
        # Joint/cartesian/gripper modes exercise on the joint-only franka;
        # base/composite modes need panda_mobile's declared base_joints.
        free_robot = panda_mobile if mode in {ControlMode.JOINT_VELOCITY} else franka
        if _free_packer_handles(mode, free_robot):
            handled.add(mode)
        if _composite_packer_handles(mode, hal):
            handled.add(mode)

    canonical = set(SIM_EXECUTABLE_CONTROL_MODES)
    assert handled == canonical, (
        "Sim packer handled-mode union drifted from SIM_EXECUTABLE_CONTROL_MODES.\n"
        f"  only in packers: {sorted(m.value for m in handled - canonical)}\n"
        f"  only in constant: {sorted(m.value for m in canonical - handled)}"
    )


def test_free_packer_handled_set_is_exact() -> None:
    """``pack_action_for_env`` handles exactly its four documented modes."""
    franka = _franka()
    panda_mobile = _panda_mobile()
    handled = {
        m
        for m in ControlMode
        if _free_packer_handles(m, panda_mobile if m is ControlMode.JOINT_VELOCITY else franka)
    }
    assert handled == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
        ControlMode.BODY_TWIST,
        ControlMode.JOINT_POSITION,
    }


def test_composite_packer_handled_set_is_exact() -> None:
    """``_pack_with_composite_split`` handles exactly its five documented modes.

    BODY_TWIST is intentionally absent here: the composite-split packer's
    ``else`` rejects it because BODY_TWIST has its own direct-qpos path in
    ``SimAttachedHAL.send_action`` (it never reaches this packer). It is
    still sim-executable, picked up via ``pack_action_for_env`` in the
    union test above.
    """
    pytest.importorskip("robosuite.controllers.composite.composite_controller")
    hal = _bare_sim_hal(_panda_mobile())
    handled = {m for m in ControlMode if _composite_packer_handles(m, hal)}
    assert handled == {
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
        ControlMode.JOINT_VELOCITY,
        ControlMode.COMPOSITE_MODE,
        ControlMode.JOINT_POSITION,
    }


def test_body_twist_uses_direct_qpos_path_not_a_packer() -> None:
    """BODY_TWIST is sim-executable via the send_action direct-qpos interception.

    It is in the canonical constant, and ``pack_action_for_env`` carries a
    BODY_TWIST branch (so the union closes there), but the *real* execution
    route is ``SimAttachedHAL.send_action`` → ``_apply_body_twist_to_qpos``
    (direct base-qpos integration, skipping ``env.step``). The composite
    packer therefore deliberately rejects BODY_TWIST. We assert both facts
    so the special-case documented in the constant cannot rot.
    """
    franka = _franka()
    panda_mobile = _panda_mobile()
    hal = _bare_sim_hal(panda_mobile)
    assert ControlMode.BODY_TWIST in SIM_EXECUTABLE_CONTROL_MODES
    assert _free_packer_handles(ControlMode.BODY_TWIST, franka) is True
    assert _composite_packer_handles(ControlMode.BODY_TWIST, hal) is False


def test_removed_and_excluded_modes_are_rejected_everywhere() -> None:
    """The 4 removed false-admits + 3 perennial exclusions: absent + rejected.

    Negative guard for the trim. Each of these modes must be (a) absent
    from ``SIM_EXECUTABLE_CONTROL_MODES`` and (b) rejected as unsupported
    by BOTH packers — proving the gate can no longer boot-pass a skill that
    would E-stop mid-run.
    """
    franka = _franka()
    hal = _bare_sim_hal(_panda_mobile())
    for mode in _REMOVED_FALSE_ADMITS | _NEVER_ADMITTED:
        assert mode not in SIM_EXECUTABLE_CONTROL_MODES, f"{mode} must not be admitted"
        assert _free_packer_handles(mode, franka) is False, (
            f"{mode} must be rejected by free packer"
        )
        assert _composite_packer_handles(mode, hal) is False, (
            f"{mode} must be rejected by composite packer"
        )
