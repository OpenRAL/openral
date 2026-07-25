"""Native SensorReader for the A1 Runtime paired Camera Bridge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from openral_core import FrameEncoding, SensorFrame
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale, ROSRuntimeError

from openral_runner.backends.galaxea_a1_ipc import (
    RuntimeLocalClient,
    decode_array,
)

__all__ = ["GalaxeaA1CameraBridgeReader"]

_PROTOCOL_VERSION = 2
_SOCKET_NAME = "a1-camera-bridge.sock"
_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
_SHA256_HEX_LENGTH = 64
_COLOR_NDIM = 3
_COLOR_CHANNELS = 3


@dataclass(frozen=True)
class _CachedCameraFrame:
    rgb: NDArray[np.uint8]
    stamp_monotonic_ns: int
    stamp_wall_ns: int


class _GalaxeaA1CameraBridgeSession:
    """Reference-counted native client for one atomic paired-frame stream."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._owners = 0
        self._client: RuntimeLocalClient | None = None
        self._contract_digest: str | None = None
        self._shapes: dict[str, tuple[int, int, int]] = {}
        self._after = {"front": -1, "wrist": -1}
        self._pair: dict[str, _CachedCameraFrame] | None = None
        self._consumed: set[str] = set()

    def open(self) -> None:
        with self._lock:
            if self._owners:
                self._owners += 1
                return
            client = RuntimeLocalClient(
                _SOCKET_NAME,
                label="Galaxea A1 Camera Bridge",
                max_response_bytes=_MAX_RESPONSE_BYTES,
            )
            client.connect(timeout_s=5.0)
            try:
                response = client.call(
                    {"version": _PROTOCOL_VERSION, "op": "describe"},
                    timeout_s=5.0,
                )
                self._install_description(response.get("metadata"))
            except BaseException:
                client.close()
                raise
            self._client = client
            self._owners = 1

    def close(self) -> None:
        with self._lock:
            if not self._owners:
                return
            self._owners -= 1
            if self._owners:
                return
            client, self._client = self._client, None
            self._pair = None
            self._consumed.clear()
            if client is not None:
                client.close()

    def read(
        self,
        camera: Literal["front", "wrist"],
        *,
        max_age_ms: int,
    ) -> _CachedCameraFrame:
        if max_age_ms <= 0:
            raise ROSConfigError("max_age_ms must be positive")
        with self._lock:
            if self._client is None:
                raise ROSRuntimeError("Galaxea A1 Camera Bridge session is closed")
            if self._pair is None or camera in self._consumed:
                self._pair = self._read_pair(timeout_s=max_age_ms / 1000.0)
                self._consumed.clear()
            frame = self._pair[camera]
            age_ms = (time.monotonic_ns() - frame.stamp_monotonic_ns) / 1e6
            if age_ms > max_age_ms:
                raise ROSPerceptionStale(
                    f"Galaxea A1 {camera} frame is {age_ms:.1f} ms old (budget {max_age_ms} ms)"
                )
            self._consumed.add(camera)
            return frame

    def _install_description(self, value: object) -> None:
        if not isinstance(value, dict):
            raise ROSRuntimeError("A1 Camera Bridge metadata is missing")
        metadata = cast(dict[str, object], value)
        digest = metadata.get("contract_digest")
        if not isinstance(digest, str) or len(digest) != _SHA256_HEX_LENGTH:
            raise ROSRuntimeError("A1 Camera Bridge contract digest is invalid")
        self._contract_digest = digest
        self._shapes = {
            camera: _shape3(metadata.get(f"{camera}_color_shape"), camera=camera)
            for camera in ("front", "wrist")
        }

    def _read_pair(self, *, timeout_s: float) -> dict[str, _CachedCameraFrame]:
        assert self._client is not None
        assert self._contract_digest is not None
        response = self._client.call(
            {
                "version": _PROTOCOL_VERSION,
                "op": "next_pair",
                "contract_digest": self._contract_digest,
                "after": dict(self._after),
            },
            timeout_s=max(1.0, timeout_s),
        )
        if response.get("changed") is not True:
            raise ROSPerceptionStale("no fresh synchronized Galaxea A1 camera pair")
        metadata = response.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("contract_digest") != self._contract_digest
        ):
            raise ROSRuntimeError("A1 Camera Bridge contract changed while connected")
        wall_ns = time.time_ns()
        pair = {
            camera: self._decode_frame(
                response.get(camera),
                camera=camera,
                wall_ns=wall_ns,
            )
            for camera in ("front", "wrist")
        }
        self._after = {
            camera: _plain_int(response[camera].get("seq"), label=f"{camera}.seq")
            for camera in ("front", "wrist")
        }
        return pair

    def _decode_frame(
        self,
        value: object,
        *,
        camera: str,
        wall_ns: int,
    ) -> _CachedCameraFrame:
        if not isinstance(value, dict):
            raise ROSRuntimeError(f"A1 Camera Bridge response is missing {camera}")
        payload = cast(dict[str, object], value)
        monotonic_s = payload.get("monotonic_s")
        if (
            isinstance(monotonic_s, bool)
            or not isinstance(monotonic_s, (int, float))
            or not np.isfinite(monotonic_s)
        ):
            raise ROSRuntimeError(f"A1 Camera Bridge {camera} timestamp is invalid")
        bgr = decode_array(
            payload.get("color_bgr"),
            shape=self._shapes[camera],
            dtype=np.dtype(np.uint8),
            label=f"A1 Camera Bridge {camera} color",
        )
        return _CachedCameraFrame(
            rgb=np.ascontiguousarray(bgr[..., ::-1]),
            stamp_monotonic_ns=int(float(monotonic_s) * 1e9),
            stamp_wall_ns=wall_ns,
        )


class GalaxeaA1CameraBridgeReader:
    """Expose one synchronized A1 Runtime camera view as RGB8."""

    def __init__(
        self,
        *,
        sensor_id: str,
        camera: Literal["front", "wrist"],
        session: _GalaxeaA1CameraBridgeSession,
        default_max_age_ms: int,
    ) -> None:
        """Store one camera view over an explicitly shared native session."""
        if default_max_age_ms <= 0:
            raise ROSConfigError("default_max_age_ms must be positive")
        self.sensor_id = sensor_id
        self.is_open = False
        self._camera = camera
        self._session = session
        self._default_max_age_ms = default_max_age_ms

    def open(self) -> None:
        """Acquire the shared Runtime camera connection."""
        if self.is_open:
            return
        self._session.open()
        self.is_open = True

    def close(self) -> None:
        """Release this view and close the connection after the last owner."""
        if not self.is_open:
            return
        try:
            self._session.close()
        finally:
            self.is_open = False

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        """Return the latest source-timestamped RGB frame."""
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


def _shape3(value: object, *, camera: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != _COLOR_NDIM
        or value[2] != _COLOR_CHANNELS
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ROSRuntimeError(f"A1 Camera Bridge {camera} shape is invalid")
    return cast(tuple[int, int, int], tuple(value))


def _plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ROSRuntimeError(f"A1 Camera Bridge {label} must be an integer")
    return value
