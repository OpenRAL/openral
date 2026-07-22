"""Pinned MuJoCo Playground walking assets and controller for the G1 sim HAL."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    import mujoco

_PINNED_SHA = "43d180a226da3aae091d918b63c06c3a343519ad"
_RAW_ROOT = f"https://raw.githubusercontent.com/google-deepmind/mujoco_playground/{_PINNED_SHA}"
_ASSETS = {
    "mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml": (
        "a05233736c914984859fc0c94161036363a8972760699a59471fdd2aecb0bf2e"
    ),
    "mujoco_playground/_src/locomotion/g1/xmls/g1_mjx_feetonly.xml": (
        "fae73d06ae6957ab38e59c67cac5b3665db6de27cd741602dd93951e3f82b72c"
    ),
    "mujoco_playground/_src/locomotion/g1/xmls/sensor.xml": (
        "9f290a8614700c4fe51323208bf91594d960bcf5622806aeda6f9722d754e56a"
    ),
    "mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx": (
        "db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b"
    ),
}


class _InferenceSession(Protocol):
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(relative_path: str, expected_sha256: str, destination: Path) -> None:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    request = urllib.request.Request(
        f"{_RAW_ROOT}/{relative_path}",
        headers={"User-Agent": "OpenRAL-G1-walking-assets"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise ROSConfigError(
            f"Could not download pinned G1 walking asset {relative_path!r}: {exc}"
        ) from exc
    actual_sha256 = _sha256(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ROSConfigError(
            f"G1 walking asset {relative_path!r} failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    os.replace(temporary, destination)


def ensure_g1_walking_assets() -> tuple[str, str]:
    """Return the flattened MJCF and ONNX paths for the pinned walking controller."""
    root = (
        Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
        / "g1_walking"
        / _PINNED_SHA
    )
    for relative_path, expected_sha256 in _ASSETS.items():
        _download(relative_path, expected_sha256, root / relative_path)

    from robot_descriptions import (  # noqa: PLC0415  # reason: optional sim asset
        g1_mj_description,
    )

    menagerie = Path(g1_mj_description.MJCF_PATH).parents[1]
    menagerie_link = root / "mujoco_menagerie"
    if menagerie_link.is_symlink():
        if menagerie_link.resolve() != menagerie.resolve():
            menagerie_link.unlink()
    elif menagerie_link.exists():
        raise ROSConfigError(
            f"G1 walking cache path {menagerie_link} exists but is not the expected symlink."
        )
    if not menagerie_link.exists():
        menagerie_link.symlink_to(menagerie, target_is_directory=True)

    xml_dir = root / "mujoco_playground" / "_src" / "locomotion" / "g1" / "xmls"
    source_mjcf = xml_dir / "scene_mjx_feetonly_flat_terrain.xml"
    flat_mjcf = xml_dir / "g1_walking_flat.xml"
    if not flat_mjcf.is_file():
        try:
            import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dependency

            model = mj.MjModel.from_xml_path(str(source_mjcf))
            temporary = flat_mjcf.with_name(f".{flat_mjcf.name}.{os.getpid()}.tmp")
            mj.mj_saveLastXML(str(temporary), model)
            os.replace(temporary, flat_mjcf)
        except (OSError, ValueError) as exc:
            raise ROSConfigError(f"Could not flatten pinned G1 walking MJCF: {exc}") from exc

    policy = root / "mujoco_playground" / "experimental" / "sim2sim" / "onnx" / "g1_policy.onnx"
    return str(flat_mjcf), str(policy)


class G1WalkingController:
    """50 Hz ONNX joystick policy over the G1's 500 Hz MuJoCo physics."""

    _CONTROL_DECIMATION = 10
    _ACTION_SCALE = 0.5
    _PHASE_DT = 2.0 * np.pi * 1.5 * 0.02

    def __init__(self, policy_path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415  # reason: optional backend has no py.typed
        except ImportError as exc:
            raise ROSConfigError(
                "G1 walking needs onnxruntime. Install the HAL sim extras with: "
                "just sync --group sim"
            ) from exc
        try:
            self._session: _InferenceSession = ort.InferenceSession(
                policy_path,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ROSConfigError(
                f"Could not load G1 walking policy {policy_path!r}: {exc}"
            ) from exc
        self._default_angles: NDArray[np.float32] = np.zeros(29, dtype=np.float32)
        self._last_action: NDArray[np.float32] = np.zeros(29, dtype=np.float32)
        self._phase: NDArray[np.float32] = np.array([0.0, np.pi], dtype=np.float32)
        self._counter = 0

    def initialize(self, data: mujoco.MjData) -> None:
        self._default_angles = np.asarray(data.qpos[7:], dtype=np.float32).copy()
        self.reset_state()

    def reset_state(self) -> None:
        self._last_action.fill(0.0)
        self._phase[:] = (0.0, np.pi)
        self._counter = 0

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: tuple[float, float, float],
    ) -> None:
        if self._counter % self._CONTROL_DECIMATION == 0:
            imu_xmat = np.asarray(
                data.site_xmat[model.site("imu_in_pelvis").id],
                dtype=np.float32,
            ).reshape(3, 3)
            observation = np.hstack(
                (
                    np.asarray(data.sensor("local_linvel_pelvis").data, dtype=np.float32),
                    np.asarray(data.sensor("gyro_pelvis").data, dtype=np.float32),
                    imu_xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32),
                    np.asarray(command, dtype=np.float32),
                    np.asarray(data.qpos[7:], dtype=np.float32) - self._default_angles,
                    np.asarray(data.qvel[6:], dtype=np.float32),
                    self._last_action,
                    np.cos(self._phase),
                    np.sin(self._phase),
                )
            ).astype(np.float32)
            raw = self._session.run(
                ["continuous_actions"],
                {"obs": observation.reshape(1, -1)},
            )[0]
            action = np.asarray(raw, dtype=np.float32).reshape(-1)
            if action.shape != (29,) or not np.isfinite(action).all():
                raise ROSConfigError(
                    "G1 walking policy must emit 29 finite joint targets; "
                    f"got shape={action.shape}, finite={bool(np.isfinite(action).all())}."
                )
            self._last_action = action
            targets = action * self._ACTION_SCALE + self._default_angles
            data.ctrl[:] = np.clip(
                targets,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            self._phase[:] = np.fmod(self._phase + self._PHASE_DT + np.pi, 2.0 * np.pi) - np.pi
        self._counter += 1
