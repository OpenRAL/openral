"""Unit tests for :func:`openral_hal.build_hal` — the single sim/real HAL
construction seam (ADR-0031).

The resolver is what makes ``deploy sim`` deterministically build a
**simulation** HAL and ``deploy run`` a **real** HAL, with the choice driven
only by the manifest's ``hal:`` block (never env config). These tests pin:

* sim mode → a simulation HAL (explicit subclass *or* derived ``MujocoArmHAL``);
* real mode → the real-hardware HAL, with ``transport`` kwargs threaded through;
* a robot lacking the requested mode → typed ``ROSCapabilityMismatch``;
* a malformed / unresolvable entrypoint → typed ``ROSConfigError``.

Fixtures are the real ``robots/<id>/robot.yaml`` manifests (CLAUDE.md §1.11 —
no ``"foo"`` placeholders).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import HalEntrypoints, RobotDescription
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_hal import build_hal

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(robot_id: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(REPO_ROOT / "robots" / robot_id / "robot.yaml"))


class TestSimMode:
    """``mode="sim"`` always yields a simulation HAL."""

    def test_explicit_sim_subclass(self) -> None:
        """franka names ``FrankaPandaHAL`` explicitly → that class is built."""
        from openral_hal.franka_panda import FrankaPandaHAL

        hal = build_hal(_load("franka_panda"), mode="sim")
        assert isinstance(hal, FrankaPandaHAL)

    def test_derived_mujoco_arm_when_sim_entry_null(self) -> None:
        """so100 leaves ``hal.sim`` null + has a ``sim:`` block → MujocoArmHAL."""
        from openral_hal._mujoco_arm import MujocoArmHAL

        hal = build_hal(_load("so100_follower"), mode="sim")
        assert isinstance(hal, MujocoArmHAL)

    def test_panda_mobile_explicit_sim_without_sim_block(self) -> None:
        """panda_mobile has no ``sim:`` block; the explicit entry is honoured."""
        from openral_hal.panda_mobile import PandaMobileHAL

        hal = build_hal(_load("panda_mobile"), mode="sim")
        assert isinstance(hal, PandaMobileHAL)


class TestRealMode:
    """``mode="real"`` builds the real-hardware HAL and threads ``transport``."""

    def test_real_hal_with_description_arg(self) -> None:
        """ur5e's real HAL takes ``description`` positionally + ``robot_ip``."""
        from openral_hal.ur_real import UR5eRealHAL

        hal = build_hal(_load("ur5e"), mode="real", transport={"robot_ip": "192.168.1.10"})
        assert isinstance(hal, UR5eRealHAL)

    def test_real_hal_with_transport_kwargs(self) -> None:
        """so100's real HAL takes a serial ``port`` via transport."""
        from openral_hal.so100_follower import SO100FollowerHAL

        hal = build_hal(_load("so100_follower"), mode="real", transport={"port": "/dev/ttyUSB0"})
        assert isinstance(hal, SO100FollowerHAL)


class TestMissingMode:
    """A robot that lacks the requested mode raises ``ROSCapabilityMismatch``."""

    def test_sim_only_robot_has_no_real_hal(self) -> None:
        with pytest.raises(ROSCapabilityMismatch, match="rizon4"):
            build_hal(_load("rizon4"), mode="real")

    def test_real_only_robot_has_no_sim_hal(self) -> None:
        with pytest.raises(ROSCapabilityMismatch, match="sawyer"):
            build_hal(_load("sawyer"), mode="sim")


class TestBadEntrypoint:
    """A malformed or unresolvable entrypoint raises ``ROSConfigError``."""

    def test_unresolvable_module(self) -> None:
        desc = _load("ur5e").model_copy(
            update={"hal": HalEntrypoints(real="openral_hal.does_not_exist:Nope")}
        )
        with pytest.raises(ROSConfigError, match="does_not_exist"):
            build_hal(desc, mode="real")

    def test_malformed_entry_no_colon(self) -> None:
        desc = _load("ur5e").model_copy(
            update={"hal": HalEntrypoints(real="openral_hal.ur_real.UR5eRealHAL")}
        )
        with pytest.raises(ROSConfigError, match="malformed"):
            build_hal(desc, mode="real")
