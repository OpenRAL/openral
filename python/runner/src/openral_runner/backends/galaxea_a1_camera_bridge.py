"""Galaxea A1 Runtime Camera Bridge reader.

The A1 Runtime remains the sole owner of both RealSense devices.  This
backend consumes its public, paired raw-frame bridge and presents one view as
an ordinary OpenRAL :class:`SensorReader`.  It never opens a camera device
itself and therefore cannot race the A1 Runtime camera monitor.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from openral_core import FrameEncoding, SensorFrame
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale

__all__ = ["GalaxeaA1CameraBridgeReader"]

_COLOR_NDIM = 3
_COLOR_CHANNELS = 3


@dataclass(frozen=True)
class _CachedCameraFrame:
    """One view from an atomically cached paired-camera observation."""

    rgb: NDArray[np.uint8]
    stamp_monotonic_ns: int
    stamp_wall_ns: int


class _PairedObservationCache:
    """Serve both camera views from one Runtime bridge observation."""

    def __init__(
        self,
        read_observation: Callable[[], dict[str, NDArray[np.uint8]] | None],
    ) -> None:
        self._read_observation = read_observation
        self._lock = RLock()
        self._observation: dict[str, NDArray[np.uint8]] | None = None
        self._consumed: set[str] = set()
        self._stamp_monotonic_ns = 0
        self._stamp_wall_ns = 0

    def read(
        self,
        camera: Literal["front", "wrist"],
        *,
        max_age_ms: int,
    ) -> _CachedCameraFrame:
        """Return ``camera`` while pairing one front and one wrist read."""
        if max_age_ms <= 0:
            raise ROSConfigError("max_age_ms must be positive")
        with self._lock:
            if self._observation is None or camera in self._consumed:
                started_ns = time.monotonic_ns()
                observation = self._read_observation()
                completed_ns = time.monotonic_ns()
                if observation is None:
                    raise ROSPerceptionStale("no fresh synchronized Galaxea A1 camera pair")
                elapsed_ms = (completed_ns - started_ns) / 1e6
                if elapsed_ms > max_age_ms:
                    raise ROSPerceptionStale(
                        "Galaxea A1 paired camera bridge read took "
                        f"{elapsed_ms:.1f} ms (budget {max_age_ms} ms)"
                    )
                self._observation = {
                    name: self._validated_rgb(observation, name) for name in ("front", "wrist")
                }
                self._consumed.clear()
                self._stamp_monotonic_ns = completed_ns
                self._stamp_wall_ns = time.time_ns()

            rgb = self._observation[camera]
            self._consumed.add(camera)
            return _CachedCameraFrame(
                rgb=rgb,
                stamp_monotonic_ns=self._stamp_monotonic_ns,
                stamp_wall_ns=self._stamp_wall_ns,
            )

    @staticmethod
    def _validated_rgb(
        observation: dict[str, NDArray[np.uint8]],
        camera: str,
    ) -> NDArray[np.uint8]:
        try:
            rgb = np.asarray(observation[camera])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Galaxea A1 paired observation is missing {camera!r}") from exc
        if rgb.ndim != _COLOR_NDIM or rgb.shape[2] != _COLOR_CHANNELS or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"Galaxea A1 {camera} camera returned shape={rgb.shape} "
                f"dtype={rgb.dtype}, expected HWC uint8 RGB"
            )
        return rgb


class _GalaxeaA1CameraBridgeSession:
    """Reference-counted owner of one paired A1 Runtime camera client."""

    def __init__(
        self,
        *,
        runtime_root_env: str,
        system_config: str,
    ) -> None:
        """Store the Runtime root selector and relative System config."""
        if not runtime_root_env:
            raise ROSConfigError("runtime_root_env must be a non-empty environment variable name")
        if Path(system_config).is_absolute():
            raise ROSConfigError("system_config must be relative to the A1 Runtime root")
        self.runtime_root_env = runtime_root_env
        self.system_config = system_config
        self._lock = RLock()
        self._owners = 0
        self._policy_session: Any | None = None
        self._cache: _PairedObservationCache | None = None

    def open(self) -> None:
        """Acquire this shared session, opening the Runtime client once."""
        with self._lock:
            if self._owners:
                self._owners += 1
                return
            root, config_path = self._runtime_paths()
            embodied_ops = root / "external" / "embodied-ops" / "src"
            for import_root in (embodied_ops, root):
                value = str(import_root)
                if value not in sys.path:
                    sys.path.insert(0, value)

            config_module = import_module("galaxea_a1_runtime.configuration.system")
            camera_module = import_module("galaxea_a1_runtime.apps.policy_camera")
            system = config_module.load_system_config(config_path, repo_root=root)
            policy_session = camera_module.PolicyCameraSession(
                system,
                front_key="front",
                wrist_key="wrist",
            )
            self._policy_session = policy_session
            self._cache = _PairedObservationCache(policy_session.read_observation)
            self._owners = 1

    def close(self) -> None:
        """Release one reader and close the Runtime client after the last."""
        with self._lock:
            if not self._owners:
                return
            self._owners -= 1
            if self._owners:
                return
            policy_session, self._policy_session = self._policy_session, None
            self._cache = None
            if policy_session is not None:
                policy_session.close()

    def read(
        self,
        camera: Literal["front", "wrist"],
        *,
        max_age_ms: int,
    ) -> _CachedCameraFrame:
        """Read one view from the current shared observation pair."""
        with self._lock:
            if self._cache is None:
                raise RuntimeError("Galaxea A1 paired camera session is closed")
            return self._cache.read(camera, max_age_ms=max_age_ms)

    def _runtime_paths(self) -> tuple[Path, Path]:
        raw_root = os.environ.get(self.runtime_root_env)
        if raw_root is None:
            raise ROSConfigError(
                f"{self.runtime_root_env} is required for the Galaxea A1 camera bridge"
            )
        root = Path(raw_root).expanduser().resolve()
        config_path = (root / self.system_config).resolve()
        if not config_path.is_relative_to(root):
            raise ROSConfigError("system_config escapes the A1 Runtime root")
        if not config_path.is_file():
            raise ROSConfigError(f"A1 Runtime system config not found: {config_path}")
        return root, config_path


class GalaxeaA1CameraBridgeReader:
    """Expose one synchronized A1 Camera Bridge view as an RGB8 sensor."""

    def __init__(
        self,
        *,
        sensor_id: str,
        camera: Literal["front", "wrist"],
        session: _GalaxeaA1CameraBridgeSession,
        default_max_age_ms: int,
    ) -> None:
        """Store the camera view and explicitly shared paired session."""
        if default_max_age_ms <= 0:
            raise ROSConfigError("default_max_age_ms must be positive")
        self.sensor_id = sensor_id
        self.is_open = False
        self._camera = camera
        self._session = session
        self._default_max_age_ms = default_max_age_ms

    def open(self) -> None:
        """Connect to the persistent A1 Camera Bridge without opening hardware."""
        if self.is_open:
            return
        self._session.open()
        self.is_open = True

    def close(self) -> None:
        """Close only this bridge client; the persistent camera owner remains alive."""
        if not self.is_open:
            return
        try:
            self._session.close()
        finally:
            self.is_open = False

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the latest synchronized view as an inline RGB8 frame."""
        if not self.is_open:
            raise RuntimeError(f"GalaxeaA1CameraBridgeReader({self.sensor_id!r}) is closed")
        budget_ms = self._default_max_age_ms if max_age_ms is None else max_age_ms
        frame = self._session.read(self._camera, max_age_ms=budget_ms)
        height, width, channels = frame.rgb.shape
        return SensorFrame(
            sensor_id=self.sensor_id,
            stamp_monotonic_ns=frame.stamp_monotonic_ns,
            stamp_wall_ns=frame.stamp_wall_ns,
            encoding=FrameEncoding.RGB8,
            width=int(width),
            height=int(height),
            channels=int(channels),
            data=frame.rgb.tobytes(),
        )
