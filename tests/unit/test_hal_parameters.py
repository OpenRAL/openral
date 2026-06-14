"""Unit tests for the manifest ``hal.parameters`` block (ADR-0029, issue #191).

``HalParameters.defaults`` lets a robot's HAL construction kwargs (serial
``port``, ``robot_ip``, …) live in ``robots/<id>/robot.yaml`` instead of a
per-robot lifecycle-node subclass, so the unified
:class:`openral_hal.lifecycle.ManifestHALLifecycleNode` can serve a
parameterised HAL with no bespoke ``_create_hal``. These tests pin:

* the schema (default-empty, round-trip, ``extra="forbid"``);
* the new field is backward-compatible — every real manifest still loads and
  defaults to an empty block;
* :func:`openral_hal.build_hal` threads ``hal.parameters.defaults`` into the
  HAL constructor, with an explicit ``transport`` override winning and
  unaccepted keys dropped.

Fixtures are the real ``robots/<id>/robot.yaml`` manifests (CLAUDE.md §1.11 —
no ``"foo"`` placeholders, no mocks).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import HalEntrypoints, HalParameters, RobotDescription
from openral_hal import build_hal

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = REPO_ROOT / "robots"

_MANIFESTS = sorted(ROBOTS_DIR.glob("*/robot.yaml"))


def _load(robot_id: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(ROBOTS_DIR / robot_id / "robot.yaml"))


class TestHalParametersSchema:
    """The ``HalParameters`` model and its place on ``HalEntrypoints``."""

    def test_defaults_empty(self) -> None:
        assert HalParameters().defaults == {}

    def test_entrypoints_defaults_to_empty_parameters(self) -> None:
        """A manifest that omits the block still gets an empty ``HalParameters``."""
        entry = HalEntrypoints(real="openral_hal.ur_real:UR5eRealHAL")
        assert entry.parameters == HalParameters()
        assert entry.parameters.defaults == {}

    def test_round_trip(self) -> None:
        entry = HalEntrypoints(
            real="openral_hal.so100_follower:SO100FollowerHAL",
            parameters=HalParameters(defaults={"port": "/dev/ttyACM0", "baud": 1_000_000}),
        )
        reloaded = HalEntrypoints.model_validate(entry.model_dump())
        assert reloaded == entry
        assert reloaded.parameters.defaults["port"] == "/dev/ttyACM0"

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            HalParameters.model_validate({"defaults": {}, "bogus": 1})


class TestManifestBackwardCompatibility:
    """Adding the optional block must not break any shipped manifest."""

    @pytest.mark.parametrize("manifest", _MANIFESTS, ids=lambda p: p.parent.name)
    def test_real_manifest_loads_with_parameters_block(self, manifest: Path) -> None:
        desc = RobotDescription.from_yaml(str(manifest))
        # Every manifest parses; the block is always a HalParameters (empty for
        # most, populated for the manifest-driven serial arms — see below).
        assert isinstance(desc.hal.parameters, HalParameters)

    def test_so100_so101_declare_serial_port_defaults(self) -> None:
        """SO-100/SO-101 carry their real-HAL `port` in the manifest (issue #191)."""
        for robot_id in ("so100_follower", "so101_follower"):
            desc = RobotDescription.from_yaml(str(ROBOTS_DIR / robot_id / "robot.yaml"))
            assert desc.hal.parameters.defaults["port"] == "/dev/ttyUSB0"
            assert desc.hal.parameters.defaults["calibrate_on_connect"] is False


class TestBuildHalThreadsManifestDefaults:
    """``build_hal`` merges ``hal.parameters.defaults`` into the constructor."""

    def test_manifest_port_threaded_without_transport(self) -> None:
        """SO-100's serial ``port`` comes from the manifest when no transport is passed."""
        from openral_hal.so100_follower import SO100FollowerHAL

        desc = _load("so100_follower").model_copy(
            update={
                "hal": HalEntrypoints(
                    real="openral_hal.so100_follower:SO100FollowerHAL",
                    parameters=HalParameters(defaults={"port": "/dev/ttyMANIFEST0"}),
                )
            }
        )
        hal = build_hal(desc, mode="real")
        assert isinstance(hal, SO100FollowerHAL)
        assert hal._port == "/dev/ttyMANIFEST0"

    def test_explicit_transport_overrides_manifest_default(self) -> None:
        """An explicit ``deploy run`` transport kwarg wins over the manifest default."""
        from openral_hal.so100_follower import SO100FollowerHAL

        desc = _load("so100_follower").model_copy(
            update={
                "hal": HalEntrypoints(
                    real="openral_hal.so100_follower:SO100FollowerHAL",
                    parameters=HalParameters(defaults={"port": "/dev/ttyMANIFEST0"}),
                )
            }
        )
        hal = build_hal(desc, mode="real", transport={"port": "/dev/ttyOVERRIDE0"})
        assert isinstance(hal, SO100FollowerHAL)
        assert hal._port == "/dev/ttyOVERRIDE0"

    def test_unaccepted_manifest_keys_dropped(self) -> None:
        """A default key the constructor doesn't accept is filtered, not an error."""
        from openral_hal.so100_follower import SO100FollowerHAL

        desc = _load("so100_follower").model_copy(
            update={
                "hal": HalEntrypoints(
                    real="openral_hal.so100_follower:SO100FollowerHAL",
                    parameters=HalParameters(
                        defaults={"port": "/dev/ttyMANIFEST0", "not_a_ctor_arg": 42}
                    ),
                )
            }
        )
        hal = build_hal(desc, mode="real")
        assert isinstance(hal, SO100FollowerHAL)
        assert hal._port == "/dev/ttyMANIFEST0"
