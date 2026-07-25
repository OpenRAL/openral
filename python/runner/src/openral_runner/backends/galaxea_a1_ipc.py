"""Native client for Galaxea A1 Runtime local MessagePack services."""

from __future__ import annotations

import os
import socket
import struct
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

_PACKET_LENGTH = struct.Struct("!I")


def runtime_socket_path(name: str) -> Path:
    """Resolve one public Runtime endpoint without locating its source checkout."""
    if not name or Path(name).name != name:
        raise ROSConfigError("A1 Runtime socket name must be one path component")
    configured = os.environ.get("A1_PROCESS_STATE_ROOT", "").strip()
    if configured:
        state_root = Path(configured)
    else:
        runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
        state_root = runtime_root / f"galaxea-a1-runtime-{os.getuid()}"
    return state_root.expanduser().resolve() / name


class RuntimeLocalClient:
    """One synchronous connection to a private A1 Runtime service."""

    def __init__(
        self,
        socket_name: str,
        *,
        label: str,
        max_response_bytes: int,
    ) -> None:
        """Store one endpoint contract without opening the socket."""
        if max_response_bytes <= 0:
            raise ROSConfigError("A1 Runtime max_response_bytes must be positive")
        self.path = runtime_socket_path(socket_name)
        self.label = label
        self.max_response_bytes = max_response_bytes
        self._socket: socket.socket | None = None

    def connect(self, *, timeout_s: float) -> None:
        """Connect to the Runtime-owned Unix socket."""
        if timeout_s <= 0:
            raise ROSConfigError(f"{self.label} connect timeout must be positive")
        if self._socket is not None:
            return
        active_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        active_socket.settimeout(timeout_s)
        try:
            active_socket.connect(str(self.path))
        except OSError as exc:
            active_socket.close()
            raise ROSConfigError(
                f"{self.label} is unavailable at {self.path}; start the "
                "corresponding A1 Runtime service"
            ) from exc
        self._socket = active_socket

    def call(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Perform one bounded request and require an explicit success reply."""
        active_socket = self._socket
        if active_socket is None:
            raise ROSRuntimeError(f"{self.label} client is closed")
        if timeout_s <= 0:
            raise ROSConfigError(f"{self.label} request timeout must be positive")
        active_socket.settimeout(timeout_s)
        try:
            _send_packet(active_socket, request)
            response = _receive_packet(
                active_socket,
                max_bytes=self.max_response_bytes,
            )
        except (OSError, ValueError) as exc:
            self.close()
            raise ROSRuntimeError(f"{self.label} transport failed: {exc}") from exc
        if not isinstance(response, dict):
            self.close()
            raise ROSRuntimeError(f"{self.label} returned a non-map response")
        if response.get("ok") is not True:
            raise ROSRuntimeError(
                f"{self.label} rejected the request: {response.get('error', 'unknown error')}"
            )
        return response

    def close(self) -> None:
        """Close the local connection idempotently."""
        active_socket, self._socket = self._socket, None
        if active_socket is not None:
            active_socket.close()


def encode_array(value: NDArray[Any]) -> dict[str, Any]:
    """Encode one contiguous ndarray with an explicit shape and dtype."""
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "data": array.tobytes(),
    }


def decode_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> NDArray[Any]:
    """Decode an exact ndarray payload and reject protocol drift."""
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "data"}:
        raise ROSRuntimeError(f"{label} array payload is invalid")
    if value["shape"] != list(shape) or value["dtype"] != dtype.str:
        raise ROSRuntimeError(f"{label} array contract mismatch: expected {shape} {dtype}")
    data = value["data"]
    size = int(np.prod(shape)) * dtype.itemsize
    if not isinstance(data, bytes) or len(data) != size:
        raise ROSRuntimeError(f"{label} array byte length mismatch")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _send_packet(active_socket: socket.socket, value: object) -> None:
    payload = msgpack.packb(value, use_bin_type=True)
    active_socket.sendall(_PACKET_LENGTH.pack(len(payload)) + payload)


def _receive_packet(
    active_socket: socket.socket,
    *,
    max_bytes: int,
) -> object | None:
    header = _receive_exact(active_socket, _PACKET_LENGTH.size)
    if header is None:
        return None
    (size,) = _PACKET_LENGTH.unpack(header)
    if size <= 0 or size > max_bytes:
        raise ValueError(f"invalid A1 Runtime packet size: {size}")
    payload = _receive_exact(active_socket, size)
    if payload is None:
        raise ConnectionError("A1 Runtime packet ended early")
    decoded: object = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    return decoded


def _receive_exact(active_socket: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = active_socket.recv(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise ConnectionError("A1 Runtime connection ended mid-packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
