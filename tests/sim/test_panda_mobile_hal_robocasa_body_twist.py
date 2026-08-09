"""A BODY_TWIST command must actually move the RoboCasa base, and only once.

``SimAttachedHAL._apply_body_twist_to_qpos`` drives the mobile base by writing
the three base qpos slots directly, then calls ``refresh_obs`` so cameras and
state slots track the new pose. ``refresh_obs`` used to do that by driving a
zero-action ``env.step``, on the reasoning that a zero action means "no
controller effort". It does not: robosuite's mobile-base controller reads it as
"hold your setpoint" and regulated the base back toward its pre-write pose.

Measured on this scene before the fix, 40 commands at 0.5 m/s wrote 1.0000 m of
displacement and kept 0.0396 m — 4.0% — with per-command survival decaying
1.000 -> 0.458 -> 0.085 -> 0.015 -> 0.002 -> 0.000. End to end that starved
Nav2's ``SimpleProgressChecker``: an identical 1.2 m ``navigate_to_pose`` goal
returned no terminal status in 240 s and left the base 1.984 m from the goal,
versus SUCCEEDED at 0.229 m with the fix.

That step also advanced MuJoCo's clock on top of the method's own ``data.time``
bookkeeping, so sim time ran at 2x the rate of the motion it was timing.

No mocks (CLAUDE.md §1.11): real ``panda_mobile`` manifest, real RoboCasa
``PickPlaceCounterToCabinet`` env via the deploy-sim HAL path, real MuJoCo.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

from openral_core import RobotDescription, extract_base_sim_joint_names
from openral_core.schemas import Action, ControlMode
from openral_hal import build_hal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_BAGUETTE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"

_VX_M_S = 0.5
_N_COMMANDS = 20
# The regulator erased ~96% of commanded motion; exact tracking is ~100%. 0.9
# separates the two by a wide margin while tolerating contact with scene
# geometry on a longer run.
_MIN_TRACKING_RATIO = 0.9


@pytest.fixture
def hal() -> Iterator[Any]:
    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    built = build_hal(desc, mode="sim", sim_env_yaml=str(_BAGUETTE))
    built.connect()
    try:
        yield built
    finally:
        built.disconnect()


def _base_xy_yaw(model: Any, data: Any, desc: RobotDescription) -> tuple[float, float, float]:
    import mujoco

    vals = []
    for jname in extract_base_sim_joint_names(desc):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"base joint {jname!r} missing from the composed MJCF"
        vals.append(float(data.qpos[int(model.jnt_qposadr[jid])]))
    return vals[0], vals[1], vals[2]


def test_body_twist_displacement_survives_the_obs_refresh(hal: Any) -> None:
    """The base tracks a commanded forward twist instead of being regulated back."""
    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    handles = hal.mujoco_handles()
    assert handles is not None, "expected live MuJoCo handles for a RoboCasa scene"
    model, data = handles
    dt = hal._body_twist_dt_s  # reason: the integration stride under test

    x0, y0, _ = _base_xy_yaw(model, data, desc)
    for _ in range(_N_COMMANDS):
        hal.send_action(
            Action(
                control_mode=ControlMode.BODY_TWIST,
                horizon=1,
                body_twist=[(_VX_M_S, 0.0, 0.0, 0.0, 0.0, 0.0)],
            )
        )
    x1, y1, _ = _base_xy_yaw(model, data, desc)

    moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    commanded = _VX_M_S * dt * _N_COMMANDS
    ratio = moved / commanded
    assert ratio >= _MIN_TRACKING_RATIO, (
        f"base kept only {ratio:.1%} of {commanded:.4f} m commanded over "
        f"{_N_COMMANDS} twists (moved {moved:.4f} m) — the observation refresh "
        "is stepping physics again and the base controller is regulating the "
        "qpos write away"
    )


def test_body_twist_advances_sim_clock_exactly_once_per_command(hal: Any) -> None:
    """Sim time gains exactly ``body_twist_dt_s`` per twist — not double."""
    handles = hal.mujoco_handles()
    assert handles is not None
    _model, data = handles
    dt = hal._body_twist_dt_s  # reason: the stride under test

    t0 = float(data.time)
    for _ in range(_N_COMMANDS):
        hal.send_action(
            Action(
                control_mode=ControlMode.BODY_TWIST,
                horizon=1,
                body_twist=[(_VX_M_S, 0.0, 0.0, 0.0, 0.0, 0.0)],
            )
        )
    per_command = (float(data.time) - t0) / _N_COMMANDS
    assert per_command == pytest.approx(dt, rel=1e-6), (
        f"sim clock advanced {per_command:.4f} s per twist, expected {dt:.4f} s. "
        "A double advance makes /clock outrun the motion it is timing, and "
        "Nav2 judges progress against that clock."
    )
