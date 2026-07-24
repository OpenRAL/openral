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
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np
from openral_core import FrameEncoding, SensorFrame
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale

__all__ = ["GalaxeaA1CameraBridgeReader"]

_COLOR_NDIM = 3
_COLOR_CHANNELS = 3


class GalaxeaA1CameraBridgeReader:
    """Expose one synchronized A1 Camera Bridge view as an RGB8 sensor."""

    def __init__(
        self,
        *,
        sensor_id: str,
        camera: Literal["front", "wrist"],
        runtime_root_env: str,
        deployment_config: str,
        default_max_age_ms: int,
    ) -> None:
        """Store the explicit bridge and A1 Runtime configuration."""
        if not runtime_root_env:
            raise ROSConfigError("runtime_root_env must be a non-empty environment variable name")
        if Path(deployment_config).is_absolute():
            raise ROSConfigError("deployment_config must be relative to the A1 Runtime root")
        if default_max_age_ms <= 0:
            raise ROSConfigError("default_max_age_ms must be positive")
        self.sensor_id = sensor_id
        self.is_open = False
        self._camera = camera
        self._runtime_root_env = runtime_root_env
        self._deployment_config = deployment_config
        self._default_max_age_ms = default_max_age_ms
        self._session: Any | None = None

    def open(self) -> None:
        """Connect to the persistent A1 Camera Bridge without opening hardware."""
        if self.is_open:
            return
        raw_root = os.environ.get(self._runtime_root_env)
        if raw_root is None:
            raise ROSConfigError(
                f"{self._runtime_root_env} is required for the Galaxea A1 camera bridge"
            )
        root = Path(raw_root).expanduser().resolve()
        config_path = (root / self._deployment_config).resolve()
        if not config_path.is_relative_to(root):
            raise ROSConfigError("deployment_config escapes the A1 Runtime root")
        if not config_path.is_file():
            raise ROSConfigError(f"A1 Runtime deployment config not found: {config_path}")
        embodied_ops = root / "external" / "embodied-ops" / "src"
        for import_root in (embodied_ops, root):
            value = str(import_root)
            if value not in sys.path:
                sys.path.insert(0, value)

        config_module = import_module("galaxea_a1_runtime.apps.lingbot.config")
        camera_module = import_module("galaxea_a1_runtime.apps.policy_camera")
        config = config_module.load_lingbot_config(config_path, repo_root=root)
        self._session = camera_module.PolicyCameraSession(
            config.system,
            front_key="front",
            wrist_key="wrist",
        )
        self.is_open = True

    def close(self) -> None:
        """Close only this bridge client; the persistent camera owner remains alive."""
        session, self._session = self._session, None
        try:
            if session is not None:
                session.close()
        finally:
            self.is_open = False

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the latest synchronized view as an inline RGB8 frame."""
        if not self.is_open or self._session is None:
            raise RuntimeError(f"GalaxeaA1CameraBridgeReader({self.sensor_id!r}) is closed")
        budget_ms = self._default_max_age_ms if max_age_ms is None else max_age_ms
        started_ns = time.monotonic_ns()
        observation = self._session.read_observation()
        if observation is None:
            raise ROSPerceptionStale(
                f"GalaxeaA1CameraBridgeReader({self.sensor_id!r}): "
                "no fresh synchronized camera pair"
            )
        rgb = np.asarray(observation[self._camera], dtype=np.uint8)
        if rgb.ndim != _COLOR_NDIM or rgb.shape[2] != _COLOR_CHANNELS:
            raise RuntimeError(
                f"Galaxea A1 {self._camera} camera returned shape {rgb.shape}, expected HWC RGB"
            )
        age_ms = (time.monotonic_ns() - started_ns) / 1e6
        if age_ms > budget_ms:
            raise ROSPerceptionStale(
                f"GalaxeaA1CameraBridgeReader({self.sensor_id!r}): bridge read "
                f"took {age_ms:.1f} ms (budget {budget_ms} ms)"
            )
        stamp_monotonic_ns = time.monotonic_ns()
        height, width, channels = rgb.shape
        return SensorFrame(
            sensor_id=self.sensor_id,
            stamp_monotonic_ns=stamp_monotonic_ns,
            stamp_wall_ns=time.time_ns(),
            encoding=FrameEncoding.RGB8,
            width=int(width),
            height=int(height),
            channels=int(channels),
            data=rgb.tobytes(),
        )
