"""Fetch the MakerMods Metal arm description and derive its URDF / MJCF.

Upstream ``makermods-robotics/metal-python-ros`` ships the ROS 2 description
package ``metal_description`` (``metal_ros2/src/metal_description``): a
6-revolute + parallel-gripper URDF (``metal_with_gripper.urdf``) and its STL
meshes. Nothing is vendored into this repository — the upstream repository
carries no LICENSE file at the pinned SHA, so OpenRAL fetches it as a pinned
clone into ``$OPENRAL_CACHE_DIR`` (the same pattern as the OpenArm fetchers)
and derives every file it needs from that clone at runtime:

* ``ensure_metal_description()`` — the ``metal_description`` package root
  (``package.xml`` + ``meshes/``), registered in
  ``openral_hal.ros_package_overlay.PUBLIC_ROS_PACKAGES`` so a ``package://``
  mesh ref resolves on a host without a sourced MakerMods workspace.
* ``ensure_metal_urdf()`` / ``ensure_metal_station_urdf()`` — the single-arm
  URDF rooted at ``base_link`` (the upstream ``world`` link and its fixed
  joint dropped so the root equals the manifest's ``base_frame``), and the
  two-arm station URDF (``left_`` / ``right_`` prefixed copies on one
  ``station_base`` link). Mesh refs stay ``package://metal_description/…``.
* ``ensure_metal_mjcf()`` / ``ensure_metal_station_mjcf()`` — MuJoCo models
  compiled from those URDFs, with ``<position>`` actuators (one per arm joint
  plus one gripper finger), joint damping/armature, and an ``<equality>``
  coupling the second finger as the upstream ``<mimic multiplier="-1">``
  declares. The URDF carries no actuator model, so the gains are OpenRAL's
  sim-only choice and say nothing about the real Damiao CAN loop.

Station geometry: MakerMods publishes no base-to-base spacing for the two-arm
Metal Arm Station, so ``STATION_ARM_SPACING_M`` is a provisional value. The
station manifest marks every frame derived from it as not calibrated; measure
the real station and update the constant (and the manifest) together.
"""

from __future__ import annotations

import copy
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_clone import fetch_pinned_clone

__all__ = [
    "METAL_PINNED_SHA",
    "STATION_ARM_SPACING_M",
    "ensure_metal_description",
    "ensure_metal_mjcf",
    "ensure_metal_station_mjcf",
    "ensure_metal_station_urdf",
    "ensure_metal_urdf",
]

#: ``makermods-robotics/metal-python-ros`` master at 2026-09-16. Keep pinned
#: (never a branch) so the sim contract is reproducible; bump deliberately.
METAL_PINNED_SHA: str = "bccf3c82a21453a7b1784a94966456e16ff0f008"
_REPO_URL: str = "https://github.com/makermods-robotics/metal-python-ros.git"
_PKG_REL: str = "metal_ros2/src/metal_description"
_URDF_REL: str = "urdf/metal_with_gripper.urdf"

#: Provisional lateral distance (metres) between the two arm bases on the
#: Metal Arm Station. NOT a MakerMods figure — none is published. Update it
#: together with ``robots/makermods_metal_station/robot.yaml`` once measured.
STATION_ARM_SPACING_M: float = 0.60

# Bump when the derived URDF/MJCF content changes, so a stale cache is not
# reused (the pinned SHA alone does not cover OpenRAL's post-processing).
_DERIVE_VERSION: str = "2"

_ARM_JOINTS: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_DRIVEN_FINGER: str = "joint8"  # upstream range [0, 0.05] m; 0 = fingers closed
_MIMIC_FINGER: str = "joint7"  # upstream <mimic joint="joint7" multiplier="-1"> on joint8

# Sim-only position-servo gains (kp N·m/rad, kv N·m·s/rad). Shoulder/elbow
# carry the heavy links (joint2 effort limit 100 N·m in the URDF); the wrist
# joints are 10 N·m motors.
_KP_KV: dict[str, tuple[float, float]] = {
    "joint1": (150.0, 6.0),
    "joint2": (400.0, 15.0),
    "joint3": (200.0, 8.0),
    "joint4": (80.0, 3.0),
    "joint5": (60.0, 2.0),
    "joint6": (30.0, 1.0),
    _DRIVEN_FINGER: (400.0, 10.0),
}
# Mesh-contact pairs to exclude per arm. ``base_link`` is welded to the world,
# so MuJoCo's parent/child filter does not cover it and its STL overlaps
# ``link1``'s by ~6 mm at every pose (jamming joint1); the two fingers touch at
# the closed position by construction.
_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = (("base_link", "link1"), ("link7", "link8"))
_JOINT_DAMPING: float = 0.5
_JOINT_ARMATURE: float = 0.02
_FINGER_FORCE_N: float = 20.0


def _cache_root() -> Path:
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "makermods_metal"


def _derived_dir() -> Path:
    return _cache_root() / f"derived-{METAL_PINNED_SHA[:12]}-v{_DERIVE_VERSION}"


def ensure_metal_description() -> Path:
    """Return the ``metal_description`` package root, fetching it if needed.

    Returns:
        The directory holding ``package.xml``, ``urdf/`` and ``meshes/``.

    Raises:
        ROSConfigError: ``git`` is missing, the clone fails, or the pinned tree
            has no ``metal_description`` package.
    """
    repo_dir = _cache_root() / METAL_PINNED_SHA
    pkg = repo_dir / _PKG_REL
    if not (pkg / "package.xml").is_file():
        fetch_pinned_clone(_REPO_URL, METAL_PINNED_SHA, repo_dir, what="MakerMods Metal")
    if not (pkg / "package.xml").is_file() or not (pkg / _URDF_REL).is_file():
        raise ROSConfigError(
            f"MakerMods Metal clone at {repo_dir!s} has no {_PKG_REL}/package.xml or "
            f"{_URDF_REL} — upstream may have moved the description package."
        )
    return pkg


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a same-directory temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _upstream_arm_tree() -> ET.Element:
    """The upstream URDF root with the ``world`` link and its fixed joint removed."""
    root = ET.parse(ensure_metal_description() / _URDF_REL).getroot()
    for joint in list(root.findall("joint")):
        parent = joint.find("parent")
        if parent is not None and parent.get("link") == "world":
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") == "world":
            root.remove(link)
    return root


def _arm_urdf_xml() -> str:
    root = _upstream_arm_tree()
    root.set("name", "makermods_metal")
    return ET.tostring(root, encoding="unicode")


def _prefixed_arm(prefix: str) -> list[ET.Element]:
    """Deep copies of every arm ``<link>``/``<joint>`` with names prefixed."""
    out: list[ET.Element] = []
    for src in _upstream_arm_tree():
        if src.tag not in ("link", "joint"):
            continue
        el = copy.deepcopy(src)
        el.set("name", prefix + str(el.get("name")))
        for tag in ("parent", "child"):
            ref = el.find(tag)
            if ref is not None:
                ref.set("link", prefix + str(ref.get("link")))
        mimic = el.find("mimic")
        if mimic is not None:
            mimic.set("joint", prefix + str(mimic.get("joint")))
        out.append(el)
    return out


def _station_urdf_xml() -> str:
    """Two prefixed arms mounted ±spacing/2 along y on one ``station_base`` link."""
    root = ET.Element("robot", {"name": "makermods_metal_station"})
    ET.SubElement(root, "link", {"name": "station_base"})
    half = STATION_ARM_SPACING_M / 2.0
    for prefix, y in (("left_", half), ("right_", -half)):
        mount = ET.SubElement(root, "joint", {"name": f"{prefix}mount", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"0 {y:.6f} 0", "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "station_base"})
        ET.SubElement(mount, "child", {"link": f"{prefix}base_link"})
        root.extend(_prefixed_arm(prefix))
    return ET.tostring(root, encoding="unicode")


def ensure_metal_urdf() -> Path:
    """Return the single-arm Metal URDF (root ``base_link``), deriving it if needed.

    Raises:
        ROSConfigError: Propagated from ``ensure_metal_description``.
    """
    path = _derived_dir() / "makermods_metal.urdf"
    if not path.is_file():
        _atomic_write(path, _arm_urdf_xml())
    return path


def ensure_metal_station_urdf() -> Path:
    """Return the two-arm Metal Arm Station URDF (root ``station_base``).

    Raises:
        ROSConfigError: Propagated from ``ensure_metal_description``.
    """
    path = _derived_dir() / "makermods_metal_station.urdf"
    if not path.is_file():
        _atomic_write(path, _station_urdf_xml())
    return path


def _compile_urdf_to_mjcf(urdf_xml: str, model_name: str, prefixes: tuple[str, ...]) -> str:
    """Compile ``urdf_xml`` with MuJoCo and add actuators + finger equality."""
    try:
        import mujoco  # noqa: PLC0415 — sim-only optional dep, lazy
    except ImportError as exc:
        raise ROSConfigError(
            f"Deriving the {model_name} MJCF needs the `mujoco` package "
            f"({type(exc).__name__}: {exc}); install the sim extras."
        ) from exc

    meshdir = ensure_metal_description() / "meshes"
    root = ET.fromstring(urdf_xml)
    ext = ET.SubElement(root, "mujoco")
    # strippath: MuJoCo keeps only each mesh's basename, so the package:// refs
    # resolve against ``meshdir``. fusestatic=false keeps every URDF link a body
    # (link names are the frames the manifest's sensors and safety refer to).
    ET.SubElement(
        ext,
        "compiler",
        {
            "meshdir": str(meshdir),
            "strippath": "true",
            "fusestatic": "false",
            "discardvisual": "true",
            "balanceinertia": "true",
            "angle": "radian",
        },
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / f"{model_name}.xml"
        mujoco.mj_saveLastXML(str(saved), model)
        mjcf = ET.parse(saved).getroot()

    mjcf.set("model", model_name)
    compiler = mjcf.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        mjcf.insert(0, compiler)
    compiler.set("meshdir", str(meshdir))
    compiler.set("angle", "radian")

    joints = {str(j.get("name")): j for j in mjcf.iter("joint")}
    for joint in joints.values():
        joint.set("damping", f"{_JOINT_DAMPING}")
        joint.set("armature", f"{_JOINT_ARMATURE}")

    for old in mjcf.findall("actuator") + mjcf.findall("equality") + mjcf.findall("contact"):
        mjcf.remove(old)
    contact = ET.SubElement(mjcf, "contact")
    for prefix in prefixes:
        for body1, body2 in _CONTACT_EXCLUDES:
            ET.SubElement(contact, "exclude", {"body1": prefix + body1, "body2": prefix + body2})
    actuator = ET.SubElement(mjcf, "actuator")
    equality = ET.SubElement(mjcf, "equality")
    for prefix in prefixes:
        for name in (*_ARM_JOINTS, _DRIVEN_FINGER):
            mj_joint = joints.get(prefix + name)
            if mj_joint is None:
                raise ROSConfigError(
                    f"{model_name}: compiled model has no joint {prefix + name!r} — "
                    f"the pinned upstream URDF changed its joint names."
                )
            kp, kv = _KP_KV[name]
            lo, hi = (float(v) for v in str(mj_joint.get("range")).split())
            force = (
                _FINGER_FORCE_N
                if name == _DRIVEN_FINGER
                else _effort_limit(urdf_xml, prefix + name)
            )
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": prefix + name,
                    "joint": prefix + name,
                    "kp": f"{kp}",
                    "kv": f"{kv}",
                    "ctrlrange": f"{lo} {hi}",
                    "forcerange": f"{-force} {force}",
                },
            )
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"{prefix}finger_mimic",
                "joint1": prefix + _MIMIC_FINGER,
                "joint2": prefix + _DRIVEN_FINGER,
                "polycoef": "0 -1 0 0 0",
            },
        )
    ET.indent(mjcf)
    return ET.tostring(mjcf, encoding="unicode")


def _effort_limit(urdf_xml: str, joint_name: str) -> float:
    """The URDF ``<limit effort>`` of ``joint_name`` (the actuator force clamp)."""
    for joint in ET.fromstring(urdf_xml).iter("joint"):
        if joint.get("name") == joint_name:
            limit = joint.find("limit")
            if limit is not None and limit.get("effort") is not None:
                return float(str(limit.get("effort")))
    raise ROSConfigError(f"URDF joint {joint_name!r} declares no <limit effort>.")


def ensure_metal_mjcf() -> Path:
    """Return the single-arm Metal MJCF, deriving it from the pinned URDF if needed.

    Raises:
        ROSConfigError: The fetch fails or ``mujoco`` is not installed.
    """
    path = _derived_dir() / "makermods_metal.xml"
    if not path.is_file():
        _atomic_write(path, _compile_urdf_to_mjcf(_arm_urdf_xml(), "makermods_metal", ("",)))
    return path


def ensure_metal_station_mjcf() -> Path:
    """Return the two-arm Metal Arm Station MJCF, deriving it if needed.

    Raises:
        ROSConfigError: The fetch fails or ``mujoco`` is not installed.
    """
    path = _derived_dir() / "makermods_metal_station.xml"
    if not path.is_file():
        _atomic_write(
            path,
            _compile_urdf_to_mjcf(
                _station_urdf_xml(), "makermods_metal_station", ("left_", "right_")
            ),
        )
    return path
