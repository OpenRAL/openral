"""End-to-end sim test for the SO-100 × robosuite integration.

Exercises the full pipeline:

* :class:`openral_sim.backends.so100_robosuite.SO100` registers as
  a robosuite robot model + gripper that the gripper/robot factories
  can load by name;
* :func:`make_so100_lift_env` builds a :class:`Lift`-derived env that
  resets and steps against real MuJoCo physics, with robosuite's
  stock ``OSC_POSITION`` (3-DOF Cartesian) controller wiring for the
  arm and a ``SimpleGripController`` for the jaw;
* :class:`ScriptedPickPolicy` walks the env through approach →
  descend → close → lift, exercising joint-position commands +
  gripper torque, and at least **attempts to grasp** the block on
  the table (gets the gripper site to within a few cm of the cube,
  closes the jaw on contact).

No mocks. The test loads the upstream DeepMind ``mujoco_menagerie``
SO-100 MJCF via :mod:`robot_descriptions`, rewrites it into the
robosuite-compatible body/gripper format at import time, and runs a
real ``mj_step`` loop. When robosuite or its assets are unavailable
on the runner, the module skips with a typed reason rather than
faking the components (CLAUDE.md §1.11).
"""

from __future__ import annotations

import pytest

# Use try/except → boolean + `pytestmark.skipif` rather than module-level
# `pytest.skip(allow_module_level=True)`: with `tests/sim/__init__.py`
# making this directory a Package, a Skipped raised at module-import time
# poisons the whole `tests/sim` Package collection ("found no collectors
# for ..." on every sibling). Deferring the decision to `pytestmark` keeps
# this module importable when optional deps fail (robosuite missing, the
# OSMesa renderer probe crashes on a headless host, etc.) so sibling files
# remain reachable.
try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    import robosuite  # noqa: F401
except ImportError as exc:
    _ROBOSUITE_ERROR: str | None = str(exc)
else:
    _ROBOSUITE_ERROR = None

# ``robot_descriptions`` lazily clones mujoco_menagerie (~650 MB) from
# GitHub the first time a model is loaded; on CI runners with restricted
# network or first-run cold caches the clone may fail or time out.
try:
    from robot_descriptions import so_arm100_mj_description as _so100_desc

    _ = _so100_desc.MJCF_PATH  # triggers lazy clone / cache lookup
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _so100_desc = None  # type: ignore[assignment]
    _MJCF_ERROR = str(exc)

# `openral_sim.backends.so100_robosuite.model` imports `register_gripper`
# from robosuite — only present in robosuite>=1.5. The libero extras pin
# robosuite==1.4 (mutually exclusive with the robocasa group per
# pyproject.toml), so importing this module on a libero-flavoured venv
# raises ImportError. Gate it the same way as the other optional deps.
try:
    if _ROBOSUITE_ERROR is None:
        from openral_sim.backends.so100_robosuite import (
            SO100,
            SO100Gripper,
            make_so100_lift_env,
        )
        from openral_sim.backends.so100_robosuite.policy import (
            PolicyTelemetry,
            ScriptedPickPolicy,
        )

        _SO100_BACKEND_ERROR: str | None = None
    else:
        _SO100_BACKEND_ERROR = "robosuite unavailable"
except ImportError as exc:
    _SO100_BACKEND_ERROR = str(exc)

import numpy as np

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _ROBOSUITE_ERROR is not None,
        reason=(
            "robosuite unavailable (install via "
            f"`just sync --all-packages --group robocasa`): {_ROBOSUITE_ERROR}"
        ),
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"SO-100 MJCF unavailable: {_MJCF_ERROR}",
    ),
    pytest.mark.skipif(
        _SO100_BACKEND_ERROR is not None,
        reason=(
            f"openral_sim.backends.so100_robosuite unavailable (needs robosuite>=1.5): "
            f"{_SO100_BACKEND_ERROR}"
        ),
    ),
]


# ── Robot / gripper model registration ────────────────────────────────────────


class TestRobotModelRegistration:
    """:class:`SO100` and :class:`SO100Gripper` must surface in the robosuite
    factories so downstream env constructors can resolve them by name."""

    def test_so100_registered_in_robosuite(self) -> None:
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

        assert "SO100" in REGISTERED_ROBOTS
        assert REGISTERED_ROBOTS["SO100"] is SO100

    def test_so100_in_robot_class_mapping(self) -> None:
        from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

        assert "SO100" in ROBOT_CLASS_MAPPING
        assert issubclass(ROBOT_CLASS_MAPPING["SO100"], FixedBaseRobot)

    def test_so100_gripper_registered(self) -> None:
        from robosuite.models.grippers import GRIPPER_MAPPING

        assert "SO100Gripper" in GRIPPER_MAPPING
        assert GRIPPER_MAPPING["SO100Gripper"] is SO100Gripper

    def test_arm_model_loads_with_five_joints(self) -> None:
        robot = SO100()
        assert robot.arm_type == "single"
        # Five arm joints; the Jaw moved to the gripper.
        assert robot.dof == 5
        for joint_name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"):
            assert any(joint_name in j for j in robot.joints), (
                f"missing arm joint {joint_name!r}; got {robot.joints}"
            )

    def test_gripper_has_single_jaw_dof(self) -> None:
        gripper = SO100Gripper()
        assert gripper.dof == 1
        assert any("Jaw" in j for j in gripper.joints)
        # Both finger groups present so ``_check_grasp`` can resolve
        # left_fingerpad / right_fingerpad geoms.
        groups = gripper._important_geoms
        assert "left_fingerpad" in groups and "right_fingerpad" in groups
        assert len(groups["left_fingerpad"]) > 0 and len(groups["right_fingerpad"]) > 0


# ── End-to-end env ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env():
    """Single env reused across tests in the module — robosuite + MuJoCo
    setup is slow (~2 s) so we amortize it."""
    e = make_so100_lift_env(
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        seed=42,
    )
    yield e
    e.close()


class TestEnvBasics:
    """Confirm the env round-trips reset/step and exposes the right shapes."""

    def test_env_reset_returns_full_obs(self, env) -> None:
        obs = env.reset()
        # Cube + robot state must be present (no camera obs in this fixture).
        for key in ("cube_pos", "cube_quat", "robot0_eef_pos", "robot0_joint_pos"):
            assert key in obs, f"missing observation key {key!r}; got {sorted(obs)[:10]}…"
        # OSC_POSITION (3 Cartesian deltas) + GRIP (1 jaw torque) = 4 dims.
        assert env.action_dim == 4

    def test_zero_step_does_not_raise(self, env) -> None:
        env.reset()
        action = np.zeros(env.action_dim, dtype=np.float32)
        _obs, reward, done, _info = env.step(action)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, (bool, np.bool_))

    def test_cube_initially_above_table(self, env) -> None:
        obs = env.reset()
        table_top_z = float(env.model.mujoco_arena.table_offset[2])
        assert obs["cube_pos"][2] > table_top_z, (
            f"cube spawned at z={obs['cube_pos'][2]:.3f} but table top is at {table_top_z:.3f}"
        )


# ── Scripted policy ──────────────────────────────────────────────────────────


def _run_policy(
    env, *, max_steps: int = 280, settle_steps: int = 20
) -> tuple[list[PolicyTelemetry], dict[str, float]]:
    """Run the scripted policy and collect per-step telemetry + summary.

    Returns:
        ``(telemetry_per_step, summary)`` where ``summary`` carries the
        derived scalars the test asserts on.
    """
    obs = env.reset()
    # Hold a zero Cartesian command + open jaw so the cube settles and
    # the OSC controller doesn't accelerate the arm from a cold start.
    # The action layout is ``[dx, dy, dz, gripper]`` per OSC_POSITION
    # + GRIP; ``+1`` opens the SO-100 jaw.
    home = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(settle_steps):
        obs, _r, _done, _info = env.step(home)

    initial_cube_z = float(obs["cube_pos"][2])
    policy = ScriptedPickPolicy()
    policy.reset()

    import mujoco as _mj

    m = env.sim.model._model
    jaw_jnt_id = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_JOINT, "gripper0_right_Jaw")
    jaw_addr = int(m.jnt_qposadr[jaw_jnt_id])

    telemetry: list[PolicyTelemetry] = []
    min_eef_to_cube = float("inf")
    max_cube_z = initial_cube_z
    min_jaw_qpos = float(env.sim.data._data.qpos[jaw_addr])
    max_jaw_qpos = min_jaw_qpos

    for _step in range(max_steps):
        action, tele = policy.step(env, obs)
        obs, _r, _done, _info = env.step(action)
        telemetry.append(tele)
        min_eef_to_cube = min(min_eef_to_cube, tele.eef_to_cube_distance_m)
        max_cube_z = max(max_cube_z, float(obs["cube_pos"][2]))
        jaw_q = float(env.sim.data._data.qpos[jaw_addr])
        min_jaw_qpos = min(min_jaw_qpos, jaw_q)
        max_jaw_qpos = max(max_jaw_qpos, jaw_q)

    summary = {
        "initial_cube_z": initial_cube_z,
        "max_cube_z": max_cube_z,
        "cube_z_delta": max_cube_z - initial_cube_z,
        "min_eef_to_cube_m": min_eef_to_cube,
        "min_jaw_qpos": min_jaw_qpos,
        "max_jaw_qpos": max_jaw_qpos,
        "jaw_qpos_range": max_jaw_qpos - min_jaw_qpos,
    }
    return telemetry, summary


class TestScriptedPolicy:
    """The scripted policy must run end-to-end and at least *attempt*
    to grasp — i.e. drive the gripper near the cube, close the jaw,
    and try to lift. Real grasp success on a 5-DOF arm with a
    plate-thin parallel jaw is not guaranteed across all seeds (OSC
    can track the eef to within ~3 cm of the cube but the 5-DOF arm
    can't always orient the jaws to wrap the block), so we assert on
    the *attempt* signal rather than the lift outcome alone.
    """

    @pytest.fixture(scope="class")
    def rollout(self) -> tuple[list[PolicyTelemetry], dict[str, float]]:
        # Build a dedicated env per class so the rollout doesn't share
        # state with the module-scope ``env`` fixture.
        e = make_so100_lift_env(
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=400,
            seed=42,
        )
        try:
            telemetry, summary = _run_policy(e)
        finally:
            e.close()
        return telemetry, summary

    def test_policy_runs_all_phases(self, rollout) -> None:
        telemetry, _summary = rollout
        phases = {t.phase for t in telemetry}
        assert phases == {"approach", "descend", "close", "lift"}, (
            f"policy did not visit every phase; got {sorted(phases)}"
        )

    def test_policy_completes_without_runtime_error(self, rollout) -> None:
        telemetry, _summary = rollout
        # _run_policy raises if env.step ever does; reaching here implies
        # the full rollout completed. We additionally assert we collected
        # the expected number of steps.
        assert len(telemetry) >= 200

    def test_eef_reaches_near_the_cube(self, rollout) -> None:
        _telemetry, summary = rollout
        # robosuite's stock OSC_POSITION drives the eef to within ~3
        # cm of the cube on the SO-100; tighter tracking would need
        # OSC_POSE (over-determined on a 5-DOF arm) or a mink-style
        # SE(3) solver, neither of which is required for the scripted
        # attempt policy.
        assert summary["min_eef_to_cube_m"] < 0.05, (
            f"gripper never got close to the cube: min dist {summary['min_eef_to_cube_m']:.3f} m"
        )

    def test_gripper_closes_during_close_phase(self, rollout) -> None:
        _telemetry, summary = rollout
        # The Jaw joint range is [-0.174, 0.5] (see :func:`_assets`). Under the
        # scripted open(+1) command the jaw rests near ~0.10 (not the -0.174
        # lower limit), then closes fully to the 0.5 upper limit — an achievable
        # sweep of ~0.40 rad, not the full 0.674. Assert it both (a) reaches the
        # closed limit and (b) sweeps a healthy range; a run where the gripper
        # never actuated would show a near-zero range and a max far below 0.5.
        assert summary["max_jaw_qpos"] > 0.45, (
            f"jaw never reached its closed limit (0.5): max={summary['max_jaw_qpos']:.3f} rad"
        )
        assert summary["jaw_qpos_range"] > 0.3, (
            f"jaw did not actuate end-to-end: range={summary['jaw_qpos_range']:.3f} rad "
            f"(min={summary['min_jaw_qpos']:.3f}, max={summary['max_jaw_qpos']:.3f})"
        )

    def test_close_phase_commands_negative_gripper(self, rollout) -> None:
        telemetry, _summary = rollout
        # During the close phase the policy must drive the gripper
        # command negative (the convention that drives the SO-100 jaw
        # toward its closed limit). This guards against accidental
        # sign flips like the one we hit during development.
        close_cmds = [t.gripper_command for t in telemetry if t.phase == "close"]
        assert close_cmds and min(close_cmds) < -0.5, (
            f"close phase never commanded a negative gripper torque; "
            f"got min cmd {min(close_cmds) if close_cmds else 'EMPTY'}"
        )

    def test_lift_phase_commands_negative_gripper(self, rollout) -> None:
        telemetry, _summary = rollout
        # Lift must hold the gripper closed; a positive command would
        # release whatever was grasped.
        lift_cmds = [t.gripper_command for t in telemetry if t.phase == "lift"]
        assert lift_cmds and max(lift_cmds) <= -0.5, (
            f"lift phase released the gripper; max cmd was {max(lift_cmds)}"
        )
