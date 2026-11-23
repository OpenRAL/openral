"""The RoboTwin 2.0 aloha-agilex URDF and meshes, fetched on demand at a pinned revision.

``robots/aloha_agilex/robot.yaml`` describes the dual-arm AgileX platform that
RoboTwin 2.0 simulates. RoboTwin ships its embodiments in one archive,
``embodiments.zip`` in the ``TianxingChen/RoboTwin2.0`` Hugging Face dataset
(MIT). This module reads only the aloha-agilex URDF
(``embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf``) and the meshes
it references out of that archive at ``ROBOTWIN_ASSETS_REVISION`` (HTTP range
reads, ~98 MB of the 220 MB archive), into
``$OPENRAL_CACHE_DIR/robotwin_aloha_agilex/<rev12>/`` in the archive's own layout,
so the URDF's relative mesh refs resolve unchanged. Every file is checked
against its SHA-256; nothing is vendored.

Two steps, so a caller can tell an unreachable host from bad bytes:
``fetch_aloha_agilex_files`` downloads what is missing, and
``ensure_aloha_agilex_urdf`` verifies every digest and returns the URDF.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeVar

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_files import sha256_file

_T = TypeVar("_T")

__all__ = [
    "ARM_JOINT_RANGES",
    "EFFORT_LIMIT",
    "GRIPPER_SCALE",
    "HOME_ARM_QPOS",
    "ROBOTWIN_ASSETS_REVISION",
    "ensure_aloha_agilex_mjcf",
    "ensure_aloha_agilex_urdf",
    "fetch_aloha_agilex_files",
]

#: ``TianxingChen/RoboTwin2.0`` dataset commit (MIT).
ROBOTWIN_ASSETS_REVISION: str = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
_ARCHIVE = f"datasets/TianxingChen/RoboTwin2.0@{ROBOTWIN_ASSETS_REVISION}/embodiments.zip"
_MEMBER_DIR = "embodiments/aloha-agilex/urdf/"
_URDF = "arx5_description_isaac.urdf"
# SHA-256 of every file, relative to ``_MEMBER_DIR``, at the pinned revision.
_FILES: dict[str, str] = {
    "arx5_description_isaac.urdf": (
        "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
    ),
    "aloha_maniskill_sim/meshes/back_link7.STL": (
        "baea38e3edf22c80d6cbe813bfdc01439dc51406fa176c1f62732fba380267c5"
    ),
    "aloha_maniskill_sim/meshes/back_link7.dae": (
        "5ef2f06e003b0463868460504951fc864130779cc35e87bb54a8ad7c524bc513"
    ),
    "aloha_maniskill_sim/meshes/base_arm.STL": (
        "3d6524754df3c5b50ea4196c473b447cd7dc0c00afb9a31e7db3bb6d1cd52c60"
    ),
    "aloha_maniskill_sim/meshes/base_arm.dae": (
        "cfe3f44bf9860d72acac2abd21a50422d72efd4de6fac72af44e02b2ea6a52df"
    ),
    "aloha_maniskill_sim/meshes/box1_Link.STL": (
        "e48fbe760dc7c1d98c3e655ca5788dd09a1c1c6544b1ad3f0aeaaaa2c8121917"
    ),
    "aloha_maniskill_sim/meshes/box2_Link.dae": (
        "511af26229e2ec9c9c014c12b04bfda50253b800dbaa129aca69c2c2b81ecc74"
    ),
    "aloha_maniskill_sim/meshes/camera_base_link.STL": (
        "d4cf9f1b01727e55283b7ac1c311eca1e807908e2d2c8e9e43bf7f6936b6e847"
    ),
    "aloha_maniskill_sim/meshes/camera_link1.STL": (
        "e37b71296958e665006034f3432ac4f74b25aea8268dfaa9bfb2164d452ac015"
    ),
    "aloha_maniskill_sim/meshes/camera_link1.dae": (
        "0692aa9f29f5d3aa9c21b81bff671e22128ae9110f3af809196a0d1f9a1b2e02"
    ),
    "aloha_maniskill_sim/meshes/camera_link2.STL": (
        "e37b71296958e665006034f3432ac4f74b25aea8268dfaa9bfb2164d452ac015"
    ),
    "aloha_maniskill_sim/meshes/camera_link2.dae": (
        "0692aa9f29f5d3aa9c21b81bff671e22128ae9110f3af809196a0d1f9a1b2e02"
    ),
    "aloha_maniskill_sim/meshes/castor.dae": (
        "0a61bca6839f37ebebd89021474821abfd908d2b50867f221e3504db4d84a06a"
    ),
    "aloha_maniskill_sim/meshes/castor_joint.dae": (
        "901cad2da533c8fd0f3e3ce25a4d5b5d1c689d5f852640cd7dab041dc647d629"
    ),
    "aloha_maniskill_sim/meshes/d435.dae": (
        "552a84a19ffbce76329911860095f1b16e5f837cc4228d35d6481b178a721f8e"
    ),
    "aloha_maniskill_sim/meshes/link1.STL": (
        "079a59fd0a94237c5042d6c7542b79bdc4c8db396758fdb7860f7b464e3c25ed"
    ),
    "aloha_maniskill_sim/meshes/link1.dae": (
        "49480bb69e5daffdd9092b33d9af6c6b3e6b9a069419e3e76fb9947e3e1753f9"
    ),
    "aloha_maniskill_sim/meshes/link2.STL": (
        "87e1386ed2533e68743f50f448e8cbdc366c2b37f0df3ede1fa2de340666786c"
    ),
    "aloha_maniskill_sim/meshes/link2.dae": (
        "9bb90c7a6c6d9bfab562349bf647f833433d71f2804a7981eba775f31e78723d"
    ),
    "aloha_maniskill_sim/meshes/link3.STL": (
        "76ded14e5f919e7a77ae7501883d55fb349be5c3c16499f59c1de5b552cad842"
    ),
    "aloha_maniskill_sim/meshes/link3.dae": (
        "1eabf99eb30c6cc9bf7c61bf650102fd22d35ac2ce67f36e42d700e3e2442318"
    ),
    "aloha_maniskill_sim/meshes/link4.STL": (
        "d234f4b96822ecf8a2f62219031053f7950d685faa15b4d23f8115c1b434cbbd"
    ),
    "aloha_maniskill_sim/meshes/link4.dae": (
        "3b50cf11a5271fe34aca0c186e623137875fd02c764cb7238c20b898f175f150"
    ),
    "aloha_maniskill_sim/meshes/link5.STL": (
        "dc58ea92599295aaa980e090a034c68397e505b79b681a223cd98ba7f10bab03"
    ),
    "aloha_maniskill_sim/meshes/link5.dae": (
        "c8a96bc00fa9c67723d0a2626bfd015b00a872d7be1588b61f0ae002dd189b13"
    ),
    "aloha_maniskill_sim/meshes/link6.STL": (
        "ffb2de0357066957db00fd7599dd4a7fbc147c8ef5916a492aa9070fdbdf5740"
    ),
    "aloha_maniskill_sim/meshes/link6.dae": (
        "e0c0f316065921fdab8704e5301baaa2b8c50867846a19c489393bc4ab83cbfa"
    ),
    "aloha_maniskill_sim/meshes/link7.STL": (
        "ec3d40b39cccde0075ca3d16603f14f180fce41a46b446473d384def7a1b0be0"
    ),
    "aloha_maniskill_sim/meshes/link7.dae": (
        "cf87ef550d6ab5d71058e7da63ba6ac4c492a9385b3ca0eed37eee0fd0cab97e"
    ),
    "aloha_maniskill_sim/meshes/link8.STL": (
        "29f7a8379cad81797e82e4ba099455cc275ac6ac5bb9e264ce46719fd7d2bb60"
    ),
    "aloha_maniskill_sim/meshes/link8.dae": (
        "e58e8cbc674bb7465ec81a2db6c4b1d08ccf1d63352e245739aa6418d62b0fd7"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link.STL": (
        "bf816a05cd083357ebd4ccc554b0bedf4d1459c60359db425b813a797bfec216"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link.dae": (
        "53c27a6f52d019e02f93748e463448e0c5173d256a9a5772a1b7267047d15c08"
    ),
    "aloha_maniskill_sim/meshes/tracer_base_link_no_wheel.dae": (
        "d3a1ee3b34dbc418e888d68148d04fccda589e00d5100fbd3cbc2f17154e0a08"
    ),
    "aloha_maniskill_sim/meshes/tracer_wheel.dae": (
        "42784402bca05e9f406f4e08d8ea1cc7ff57db9fee0e0b0b557670d0ee1d10c1"
    ),
}


def _root() -> Path:
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "robotwin_aloha_agilex" / ROBOTWIN_ASSETS_REVISION[:12]


def _remote(read: Callable[[], _T]) -> _T:
    """Run one read of the remote archive, every failure a ``ROSConfigError`` (a fetch)."""
    try:
        return read()
    except Exception as exc:  # every hub / network / archive read failure is a fetch failure
        raise ROSConfigError(
            f"Failed to read {_MEMBER_DIR} from {_ARCHIVE}: {type(exc).__name__}: {exc}"
        ) from exc


def fetch_aloha_agilex_files() -> Path:
    """Download every pinned file not already cached; return the cache root.

    Only presence is checked here (``ensure_aloha_agilex_urdf`` verifies), so a
    warm cache needs no network.

    Raises:
        ROSConfigError: ``huggingface_hub`` is missing, or the archive could not
            be read (``.__cause__`` carries the hub / network error). Only remote
            reads are wrapped, so a caller can tell them from a local cache error.
        OSError: Writing into the local cache failed (a full or read-only disk).
    """
    root = _root()
    missing = [rel for rel in _FILES if not (root / rel).is_file()]
    if not missing:
        return root
    try:
        from huggingface_hub import HfFileSystem  # noqa: PLC0415  # reason: optional dep
    except ImportError as exc:
        raise ROSConfigError(f"robotwin:aloha_agilex needs huggingface_hub: {exc}") from exc
    # The cache root first: a full or read-only disk fails here, before any network.
    root.mkdir(parents=True, exist_ok=True)
    (root / ".verified").unlink(missing_ok=True)
    remote = _remote(lambda: HfFileSystem().open(_ARCHIVE, "rb", block_size=8 << 20))
    with remote, _remote(lambda: zipfile.ZipFile(remote)) as archive:
        for rel in missing:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
            try:
                with (
                    _remote(partial(archive.open, _MEMBER_DIR + rel)) as src,
                    tmp.open("wb") as out,
                ):
                    while chunk := _remote(partial(src.read, 1 << 20)):
                        out.write(chunk)
                os.replace(tmp, dest)
            finally:  # a failed read or write leaves no partial mesh behind
                tmp.unlink(missing_ok=True)
    return root


def ensure_aloha_agilex_urdf() -> Path:
    """Return the aloha-agilex URDF, every file fetched and SHA-256-verified.

    A file that fails verification is deleted, so the next call fetches it again.
    A passed check leaves a ``.verified`` stamp, so a warm cache is not re-hashed
    (~430 MB) on every resolve.

    Raises:
        ROSConfigError: A fetch failed or a file failed SHA-256 verification.
    """
    root = fetch_aloha_agilex_files()
    stamp = root / ".verified"
    if stamp.is_file():
        return root / _URDF
    for rel, digest in _FILES.items():
        path = root / rel
        if sha256_file(path) != digest:
            path.unlink()
            raise ROSConfigError(
                f"robotwin:aloha_agilex {rel} failed SHA-256 verification "
                f"(pin {ROBOTWIN_ASSETS_REVISION[:10]}); removed it from the cache."
            )
    stamp.touch()
    return root / _URDF


# ── MuJoCo twin ─────────────────────────────────────────────────────────────────
#
# The derived cache is keyed by this module's own source (every derive constant
# and step lives here), so an edit invalidates a stale MJCF without a hand-bumped
# version; the pinned revision in the cache root covers the upstream files.
_DERIVE_KEY: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

#: Arm joint ranges (rad), joint1..joint6, from ``robots/aloha_agilex/robot.yaml``
#: (the URDF declares ±10 rad placeholders). Both front arms share them;
#: ``python/core/tests/test_assets.py`` holds the two in lock-step.
ARM_JOINT_RANGES: tuple[tuple[float, float], ...] = (
    (-2.618, 2.618),
    (0.0, 3.14),
    (0.0, 2.967),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-3.14159, 3.14159),
)
#: The twin's start pose, joint1..joint6 per arm (rad): RoboTwin's config.yml home pose
#: (all zeros) with joint2 / joint3 lifted 0.05 rad off their lower limit of 0.
#: At exactly 0, gravity settles them ~0.0005 rad below the manifest limit, and
#: the safety kernel refuses any target that repeats that measured pose.
HOME_ARM_QPOS: tuple[float, ...] = (0.0, 0.05, 0.05, 0.0, 0.0, 0.0)
#: Gripper command span (m of joint7) for the normalised [0, 1] gripper: RoboTwin's
#: ``config.yml`` ``gripper_scale``. 0 targets 1 cm past the closed stop (the
#: grip squeeze); 1 is 0.045 m, inside the 0.04765 m slide.
GRIPPER_SCALE: tuple[float, float] = (-0.01, 0.045)
#: Actuator force clamp (N·m / N), the manifest's ``effort_limit`` for every joint.
EFFORT_LIMIT: float = 30.0
_ARMS: tuple[str, ...] = ("fl_", "fr_")
# RoboTwin's ``config.yml`` couples each gripper as ``joint8 = 1.0 * joint7``
# (both 0..0.04765 m, 0 = closed; joint8's axis is mirrored in the URDF).
_DRIVEN_FINGER: str = "joint7"
_MIMIC_FINGER: str = "joint8"
# Sim-only position-servo gains (kp N·m/rad or N/m, kv N·m·s/rad or N·s/m).
# RoboTwin's SAPIEN drive (stiffness 1000, damping 200) under the 30 N·m clamp
# would cap a joint at 30 / 200 = 0.15 rad/s; kv = 10 puts that cap at the
# manifest's 3 rad/s velocity limit instead.
_KP_KV: dict[str, tuple[float, float]] = {
    "joint1": (300.0, 10.0),
    "joint2": (400.0, 10.0),
    "joint3": (300.0, 10.0),
    "joint4": (100.0, 3.0),
    "joint5": (100.0, 3.0),
    "joint6": (50.0, 1.5),
    _DRIVEN_FINGER: (400.0, 10.0),
}
# Mesh-contact pairs to exclude per arm. ``base_link`` is welded to the world,
# so MuJoCo's parent/child filter does not cover it, and its STL overlaps
# ``link1``'s by ~2 mm at every pose; the two fingers touch when closed.
_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = (("base_link", "link1"), ("link7", "link8"))
_JOINT_DAMPING: float = 0.5
_JOINT_ARMATURE: float = 0.02


def _is_front_arm_joint(name: str) -> bool:
    """``fl_joint1``..``fr_joint8`` (``fl_castor_wheel`` shares the prefix, not the stem)."""
    return name.startswith(tuple(p + "joint" for p in _ARMS))


def _frozen_urdf_xml(urdf: Path) -> str:
    """The URDF with every joint but the two front arms' turned ``fixed``, minus COLLADA.

    The wheels, castors and the two rear (``lr_`` / ``rr_``) arms are welded at
    their zero pose: RoboTwin drives only the front arms, and the manifest's
    14 joints name only those. MuJoCo cannot load the ``.dae`` visuals, so they
    are dropped; those links render their STL collision mesh (the same shape),
    and the visual-only STL frame (``box1_Link``, the camera mast) is kept.
    """
    try:
        root = ET.parse(urdf).getroot()
    except ET.ParseError as exc:
        raise ROSConfigError(f"aloha_agilex URDF {urdf!s} is not valid XML: {exc}") from exc
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is not None and str(mesh.get("filename")).lower().endswith(".dae"):
                link.remove(visual)
    for joint in root.findall("joint"):
        name = str(joint.get("name"))
        if joint.get("type") != "fixed" and not _is_front_arm_joint(name):
            joint.set("type", "fixed")
    return ET.tostring(root, encoding="unicode")


def _aloha_agilex_mjcf_xml(urdf: Path) -> str:
    """Compile the frozen URDF with MuJoCo and add actuators + finger equalities.

    Raises:
        ROSConfigError: ``mujoco`` is missing or the URDF does not compile, or
            the compiled model lacks a front-arm joint.
    """
    try:
        import mujoco  # noqa: PLC0415 — sim-only optional dep, lazy
    except ImportError as exc:
        raise ROSConfigError(
            f"Deriving the aloha_agilex MJCF needs the `mujoco` package "
            f"({type(exc).__name__}: {exc}); install the sim extras."
        ) from exc

    root = ET.fromstring(_frozen_urdf_xml(urdf))
    ext = ET.SubElement(root, "mujoco")
    # fusestatic=false keeps every URDF link a body (the camera mounts and
    # link names the manifest refers to); the remaining STL visuals are kept.
    ET.SubElement(
        ext,
        "compiler",
        {
            "meshdir": str(urdf.parent),
            "fusestatic": "false",
            "discardvisual": "false",
            "balanceinertia": "true",
            "angle": "radian",
        },
    )
    try:
        # MjSpec, not ``mj_saveLastXML`` (process-global, racy across threads).
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        spec.compile()
    except ValueError as exc:
        raise ROSConfigError(f"Deriving the aloha_agilex MJCF failed: {exc}") from exc
    mjcf = ET.fromstring(spec.to_xml())

    mjcf.set("model", "aloha_agilex")
    compiler = mjcf.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        mjcf.insert(0, compiler)
    compiler.set("meshdir", str(urdf.parent))
    compiler.set("angle", "radian")
    option = mjcf.find("option")
    if option is None:
        option = ET.Element("option")
        mjcf.insert(1, option)
    option.set("integrator", "implicitfast")

    joints = {str(j.get("name")): j for j in mjcf.iter("joint")}
    for joint in joints.values():
        joint.set("damping", f"{_JOINT_DAMPING}")
        joint.set("armature", f"{_JOINT_ARMATURE}")

    for old in mjcf.findall("actuator") + mjcf.findall("equality") + mjcf.findall("contact"):
        mjcf.remove(old)
    contact = ET.SubElement(mjcf, "contact")
    actuator = ET.SubElement(mjcf, "actuator")
    equality = ET.SubElement(mjcf, "equality")
    # Actuators in manifest order: left joint1..6, left gripper, right ...
    for prefix in _ARMS:
        names = [f"joint{i}" for i in range(1, 7)] + [_DRIVEN_FINGER]
        for name in names:
            mj_joint = joints.get(prefix + name)
            if mj_joint is None:
                raise ROSConfigError(
                    f"aloha_agilex: compiled model has no joint {prefix + name!r} — "
                    f"the pinned RoboTwin URDF changed its joint names."
                )
            if name == _DRIVEN_FINGER:
                lo, hi = GRIPPER_SCALE
            else:
                lo, hi = ARM_JOINT_RANGES[int(name[-1]) - 1]
                mj_joint.set("range", f"{lo} {hi}")
                mj_joint.set("limited", "true")
            kp, kv = _KP_KV[name]
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": prefix + name,
                    "joint": prefix + name,
                    "kp": f"{kp}",
                    "kv": f"{kv}",
                    "ctrlrange": f"{lo} {hi}",
                    "forcerange": f"{-EFFORT_LIMIT} {EFFORT_LIMIT}",
                },
            )
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"{prefix}finger_mimic",
                "joint1": prefix + _MIMIC_FINGER,
                "joint2": prefix + _DRIVEN_FINGER,
                "polycoef": "0 1 0 0 0",
            },
        )
        for body1, body2 in _CONTACT_EXCLUDES:
            ET.SubElement(contact, "exclude", {"body1": prefix + body1, "body2": prefix + body2})
    _add_home_keyframe(mjcf)
    ET.indent(mjcf)
    return ET.tostring(mjcf, encoding="unicode")


def _add_home_keyframe(mjcf: ET.Element) -> None:
    """Replace any keyframes with ``home``: ``HOME_ARM_QPOS`` per arm, grippers closed."""
    # qpos per arm: joint1..6, joint7, joint8; ctrl per arm: joint1..6, joint7.
    arm_qpos = " ".join(f"{q}" for q in (*HOME_ARM_QPOS, 0.0, 0.0))
    arm_ctrl = " ".join(f"{q}" for q in (*HOME_ARM_QPOS, 0.0))
    for old in mjcf.findall("keyframe"):
        mjcf.remove(old)
    keyframe = ET.SubElement(mjcf, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": f"{arm_qpos} {arm_qpos}", "ctrl": f"{arm_ctrl} {arm_ctrl}"},
    )


def ensure_aloha_agilex_mjcf() -> Path:
    """Return the aloha-agilex MuJoCo twin MJCF, deriving it from the URDF if needed.

    The front two arms are actuated (one ``<position>`` actuator per manifest
    joint, in manifest order, ranges from the manifest) with ``joint8``
    mirroring ``joint7`` per gripper; the base, wheels and rear arms are welded.
    The gains are OpenRAL's sim-only choice, not a model of the real servos.

    Raises:
        ROSConfigError: The URDF fetch / verification fails or ``mujoco`` is
            not installed.
    """
    urdf = ensure_aloha_agilex_urdf()
    path = _root() / f"derived-{_DERIVE_KEY}" / "aloha_agilex.xml"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(_aloha_agilex_mjcf_xml(urdf))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return path
