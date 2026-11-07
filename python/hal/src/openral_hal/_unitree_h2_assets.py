"""Pinned Unitree H2 description assets (URDF + MJCF + meshes), fetched on demand.

The Unitree H2 is in neither ``mujoco_menagerie`` nor ``robot_descriptions``, so
``robots/h2/robot.yaml`` names its assets with the ``unitree:h2`` scheme
(``openral_core.assets``) and this module fetches the official upstream files:

* URDF — ``unitreerobotics/unitree_ros`` @ ``41dec58c`` (BSD-3-Clause),
  ``robots/h2_description/H2.urdf`` + the 34 STL meshes it references.
* MJCF — ``unitreerobotics/unitree_mujoco`` @ ``1eb6642e`` (BSD-3-Clause),
  ``unitree_robots/h2/{h2_mujoco.xml,scene.xml}`` + the 32 STL meshes the MJCF
  references (``motor`` torque actuators, SDK motor order).

Fetched, not vendored: each mesh set is ~13 MB, the two sets are NOT
byte-identical (head + wrist meshes differ, the URDF adds hand meshes), and both
description files reference their meshes by a path relative to the file
(``meshes/*.stl``). Keeping each file beside its own meshes in the cache lets the
upstream relative refs resolve unchanged — MuJoCo's ``meshdir`` and yourdfpy's
URDF mesh lookup (collision lowering) both work with zero patching.

Every file is pinned by commit SHA **and** its SHA-256; a mismatch raises
``ROSConfigError`` and installs nothing (fail-closed, ``_pinned_files``). Cache:
``$OPENRAL_CACHE_DIR/unitree_h2/<repo>/<sha>/<path>`` (default
``~/.cache/openral``).
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_hal._pinned_files import fetch_pinned_file

__all__ = [
    "UNITREE_MUJOCO_SHA",
    "UNITREE_ROS_SHA",
    "ensure_h2_mjcf",
    "ensure_h2_urdf",
]

#: ``unitreerobotics/unitree_ros`` HEAD on 2026-11-07 (BSD-3-Clause).
UNITREE_ROS_SHA: str = "41dec58c7a3ccb3050243ef16ac2ca74b2b0f9ca"
#: ``unitreerobotics/unitree_mujoco`` HEAD on 2026-11-07 (BSD-3-Clause).
UNITREE_MUJOCO_SHA: str = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"

_RAW = "https://raw.githubusercontent.com/unitreerobotics"
_ROS_DIR = "robots/h2_description"
_MUJOCO_DIR = "unitree_robots/h2"

# SHA-256 of every file, relative to ``_ROS_DIR`` / ``_MUJOCO_DIR``, computed from
# the pinned commits; a re-pin recomputes the whole table.
_ROS_FILES: dict[str, str] = {
    "H2.urdf": "ecd8423b947d7b01e3d362876e3e0c66ee9d78e03bc709c504900eee422d2571",
    "meshes/head_pitch_link.stl": (
        "209fd2e1d91b088a75571513628688c5423e3e5ea7c92c750ddae4225c8aac53"
    ),
    "meshes/head_yaw_link.stl": "02482147dd11cf6f4f36cd599e4beff78b6d3be65aa799cda410d85a646189ac",
    "meshes/left_ankle_pitch_link.stl": (
        "0b15f8fcbfaece06fbbf16684a96d8a31914ab870c55b292ac18203d7c6c7a62"
    ),
    "meshes/left_ankle_roll_link.stl": (
        "65e35f53cf85e55c0c61a6b10e7c2e6e4c3a46b18bf9ce6dffa520ec1e4e5b66"
    ),
    "meshes/left_elbow_link.stl": (
        "3da8b8c0a387fd91911a3fb3dda3ce6a0c56903490d9e08b4536216066211b5f"
    ),
    "meshes/left_hand_link.stl": "b326d6cbdeb2ca202feb193b3103f8194fc594e6177e7d1a73791bcea39d4c47",
    "meshes/left_hip_pitch_link.stl": (
        "d1bccf3c881baf960ff75c17d570f9a5871ebbf1f8935f4d8873677533d6d3ec"
    ),
    "meshes/left_hip_roll_link.stl": (
        "32c8554cee739281bee25da955f61dcf4fb828c4deeee94b963403c512875265"
    ),
    "meshes/left_hip_yaw_link.stl": (
        "1222ef2aed7b276e05a86b588d92df934c7feac537ae2491243db2d32fd9aac6"
    ),
    "meshes/left_knee_link.stl": "f24801b1dde9fd004b2f2178ef65318364a48daede96ab95c8ea2d979a642219",
    "meshes/left_shoulder_pitch_link.stl": (
        "7d10e524f5124b8dd37d8b9b61f2ae9342f0605bdc87ba49c6953c2863fa32c9"
    ),
    "meshes/left_shoulder_roll_link.stl": (
        "cc86323ffe6477cacc6f6ee2a7e000940432e47ba53d3514716f3539f8c83208"
    ),
    "meshes/left_shoulder_yaw_link.stl": (
        "253bde468b2156e13a0172436ab0327001d08b2a7872b729426e2f4a6f70183f"
    ),
    "meshes/left_wrist_pitch_link.stl": (
        "6ae7cbf9941e3507546c2d568d0bac32b93eaaafaa83117371f5c7363172bcb3"
    ),
    "meshes/left_wrist_roll_link.stl": (
        "7fed0dde408dd63bec1a5a13721173e66e4a063b68850a2ab7701719da6bec62"
    ),
    "meshes/left_wrist_yaw_link.stl": (
        "c8fc77bf0b8617f9e8405c4a1006280b1605a3b3a130b45428d4be8531d75005"
    ),
    "meshes/pelvis.stl": "1f2a2e99ce060eb1a25efcbadfdd4424347f58e6d954e023dcfcb05d87c55a4e",
    "meshes/right_ankle_pitch_link.stl": (
        "6270e30af526c306e4a1006f7ad9a70fe010ffa8451c9630ff98527a5bdce2d1"
    ),
    "meshes/right_ankle_roll_link.stl": (
        "27cbf682efac52262a35867ce39c969a56e87e00395a7f29ca4840149cf0a96a"
    ),
    "meshes/right_elbow_link.stl": (
        "421fdb043f2775038c72e692a598f96840f1bd2b898ce19b35e240fe5617abc4"
    ),
    "meshes/right_hand_link.stl": (
        "8e5d75d26f970b05148436d21973cfe29cc1bc335783cd6e5104d6170bd8a139"
    ),
    "meshes/right_hip_pitch_link.stl": (
        "33de10c25bee624038bf31bccb41aa84bfea0cd6bca5fcd5dfee8fa84eb0eab1"
    ),
    "meshes/right_hip_roll_link.stl": (
        "a1bc4cf4a5c2ad261af93c29972dce9bd8a1818c78fea513dec46d25b729b9fc"
    ),
    "meshes/right_hip_yaw_link.stl": (
        "9095a160b6c5ba60a8460439f748764baee08bf85ef9d05c0bdc14b7a1c66437"
    ),
    "meshes/right_knee_link.stl": (
        "4db181250da6aaec995b4eda88a3729c5144128704c3d949fa9cb5a56f437b1b"
    ),
    "meshes/right_shoulder_pitch_link.stl": (
        "e2b725c2592250811135c05004bb5ec6356b357fdf45781cb2affc7e61499cfb"
    ),
    "meshes/right_shoulder_roll_link.stl": (
        "be48510290d600d0656cd6055d2a6c58d94665e2f8210fb39ebed8cfdcf0abed"
    ),
    "meshes/right_shoulder_yaw_link.stl": (
        "78b6403fc846c6f64cd6740ec3432f1933635724f988313fdf848063e261c73d"
    ),
    "meshes/right_wrist_pitch_link.stl": (
        "216abd2755c2288dd408afb1b59867cdac3e22ef7a5ee111b54f456b38b25074"
    ),
    "meshes/right_wrist_roll_link.stl": (
        "545475c0f7377661a9c67a983f4017985578e805045d2d7b42df8d37c5641d7d"
    ),
    "meshes/right_wrist_yaw_link.stl": (
        "c13910f88609dd2b801c1398150ceb6c82f65ca308bc8e2878f19ea9a3dabda5"
    ),
    "meshes/torso_link.stl": "0f0c12bddbe2e29df43f38fbaeb3a65c08d43123fbb62379a1cc58e9c6df47b5",
    "meshes/waist_roll_link.stl": (
        "e229c2617f2c1ed26d29b44bc45cfe9622cfd0dcb6974f1896e4a21941706c7c"
    ),
    "meshes/waist_yaw_link.stl": "385f8caecd05d046aabd3cd4721e2d23a27f190646a247f71c8d0c19114456a2",
}
_MUJOCO_FILES: dict[str, str] = {
    "h2_mujoco.xml": "2ac63ffc436fd4a368a448acacfd4f0915e7f986596b254fa681b6a95b69c47f",
    "scene.xml": "06f6b79b672a2a13031c63c34b0bf948b9aed300eb38cd45c2366dedd55c345b",
    "meshes/pelvis.stl": "1f2a2e99ce060eb1a25efcbadfdd4424347f58e6d954e023dcfcb05d87c55a4e",
    "meshes/left_hip_pitch_link.stl": (
        "d1bccf3c881baf960ff75c17d570f9a5871ebbf1f8935f4d8873677533d6d3ec"
    ),
    "meshes/left_hip_roll_link.stl": (
        "32c8554cee739281bee25da955f61dcf4fb828c4deeee94b963403c512875265"
    ),
    "meshes/left_hip_yaw_link.stl": (
        "1222ef2aed7b276e05a86b588d92df934c7feac537ae2491243db2d32fd9aac6"
    ),
    "meshes/left_knee_link.stl": "f24801b1dde9fd004b2f2178ef65318364a48daede96ab95c8ea2d979a642219",
    "meshes/left_ankle_roll_link.stl": (
        "65e35f53cf85e55c0c61a6b10e7c2e6e4c3a46b18bf9ce6dffa520ec1e4e5b66"
    ),
    "meshes/left_ankle_pitch_link.stl": (
        "0b15f8fcbfaece06fbbf16684a96d8a31914ab870c55b292ac18203d7c6c7a62"
    ),
    "meshes/right_hip_pitch_link.stl": (
        "33de10c25bee624038bf31bccb41aa84bfea0cd6bca5fcd5dfee8fa84eb0eab1"
    ),
    "meshes/right_hip_roll_link.stl": (
        "a1bc4cf4a5c2ad261af93c29972dce9bd8a1818c78fea513dec46d25b729b9fc"
    ),
    "meshes/right_hip_yaw_link.stl": (
        "9095a160b6c5ba60a8460439f748764baee08bf85ef9d05c0bdc14b7a1c66437"
    ),
    "meshes/right_knee_link.stl": (
        "4db181250da6aaec995b4eda88a3729c5144128704c3d949fa9cb5a56f437b1b"
    ),
    "meshes/right_ankle_roll_link.stl": (
        "27cbf682efac52262a35867ce39c969a56e87e00395a7f29ca4840149cf0a96a"
    ),
    "meshes/right_ankle_pitch_link.stl": (
        "6270e30af526c306e4a1006f7ad9a70fe010ffa8451c9630ff98527a5bdce2d1"
    ),
    "meshes/waist_yaw_link.stl": "385f8caecd05d046aabd3cd4721e2d23a27f190646a247f71c8d0c19114456a2",
    "meshes/waist_roll_link.stl": (
        "e229c2617f2c1ed26d29b44bc45cfe9622cfd0dcb6974f1896e4a21941706c7c"
    ),
    "meshes/torso_link.stl": "0f0c12bddbe2e29df43f38fbaeb3a65c08d43123fbb62379a1cc58e9c6df47b5",
    "meshes/head_pitch_link.stl": (
        "1800daa300a7319bb14dda0b9bddc5c76fa8ca2b7943efef3ecc298886e89e33"
    ),
    "meshes/head_yaw_link.stl": "7f9ad04b2061e7ec34881a6c38b35c41844ea5733b5f9097395c0fe4611a2104",
    "meshes/left_shoulder_pitch_link.stl": (
        "7d10e524f5124b8dd37d8b9b61f2ae9342f0605bdc87ba49c6953c2863fa32c9"
    ),
    "meshes/left_shoulder_roll_link.stl": (
        "cc86323ffe6477cacc6f6ee2a7e000940432e47ba53d3514716f3539f8c83208"
    ),
    "meshes/left_shoulder_yaw_link.stl": (
        "253bde468b2156e13a0172436ab0327001d08b2a7872b729426e2f4a6f70183f"
    ),
    "meshes/left_elbow_link.stl": (
        "3da8b8c0a387fd91911a3fb3dda3ce6a0c56903490d9e08b4536216066211b5f"
    ),
    "meshes/left_wrist_roll_link.stl": (
        "7fed0dde408dd63bec1a5a13721173e66e4a063b68850a2ab7701719da6bec62"
    ),
    "meshes/left_wrist_pitch_link.stl": (
        "b8bfc833fc1ae5ebce128d760f72337e2d2a75a51994e3ec7b3d9e0ebea4eef6"
    ),
    "meshes/left_wrist_yaw_link.stl": (
        "51fb8b27cc66f5fbc8fe43a7506a51553a57aa79479e6f61f639254c3b7e112a"
    ),
    "meshes/right_shoulder_pitch_link.stl": (
        "e2b725c2592250811135c05004bb5ec6356b357fdf45781cb2affc7e61499cfb"
    ),
    "meshes/right_shoulder_roll_link.stl": (
        "be48510290d600d0656cd6055d2a6c58d94665e2f8210fb39ebed8cfdcf0abed"
    ),
    "meshes/right_shoulder_yaw_link.stl": (
        "78b6403fc846c6f64cd6740ec3432f1933635724f988313fdf848063e261c73d"
    ),
    "meshes/right_elbow_link.stl": (
        "421fdb043f2775038c72e692a598f96840f1bd2b898ce19b35e240fe5617abc4"
    ),
    "meshes/right_wrist_roll_link.stl": (
        "545475c0f7377661a9c67a983f4017985578e805045d2d7b42df8d37c5641d7d"
    ),
    "meshes/right_wrist_pitch_link.stl": (
        "31a1f13952056b9f0cf2a3b5095678675940550e2f16bca7a0bc7fa8984c8e76"
    ),
    "meshes/right_wrist_yaw_link.stl": (
        "3fb5391ed692447d273bbc0c766a2d8a816753a4912e277d5bfe692647fc2cfb"
    ),
}


def _cache_root() -> Path:
    return Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")


def _fetch_tree(repo: str, sha: str, subdir: str, files: dict[str, str]) -> Path:
    root = _cache_root() / "unitree_h2" / repo / sha / subdir
    for rel, digest in files.items():
        fetch_pinned_file(
            f"{_RAW}/{repo}/{sha}/{subdir}/{rel}",
            digest,
            root / rel,
            what=f"Unitree H2 asset {repo}/{subdir}/{rel} (pin {sha[:10]})",
            user_agent="OpenRAL-unitree-h2-assets",
        )
    return root


def ensure_h2_urdf() -> Path:
    """Return the pinned upstream ``H2.urdf`` path, its meshes fetched beside it.

    Idempotent: verified cached files are reused without network access.

    Raises:
        ROSConfigError: A download fails or a file fails SHA-256 verification.
    """
    return _fetch_tree("unitree_ros", UNITREE_ROS_SHA, _ROS_DIR, _ROS_FILES) / "H2.urdf"


def ensure_h2_mjcf() -> Path:
    """Return the pinned upstream ``h2_mujoco.xml`` path (robot only, no floor).

    ``scene.xml`` (floor + lights, includes ``h2_mujoco.xml``) is fetched
    alongside for viewers; the HAL loads the bare robot like the menagerie
    ``g1.xml`` / ``h1.xml`` twins.

    Raises:
        ROSConfigError: A download fails or a file fails SHA-256 verification.
    """
    root = _fetch_tree("unitree_mujoco", UNITREE_MUJOCO_SHA, _MUJOCO_DIR, _MUJOCO_FILES)
    return root / "h2_mujoco.xml"
