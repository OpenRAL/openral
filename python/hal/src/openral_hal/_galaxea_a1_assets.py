"""The Galaxea A1 URDF and meshes, fetched on demand at a pinned commit.

``robots/galaxea_a1/robot.yaml`` was transcribed from Galaxea's
``A1_URDF_0607_0028.urdf``, which Galaxea publishes in
``userguide-galaxea/robot_descriptions`` (``a1_description/``). That repository
carries no license, so nothing is vendored: this module downloads the URDF and
its nine STL meshes at ``GALAXEA_DESCRIPTIONS_SHA``, each checked against its
SHA-256 (``_pinned_files``), into ``$OPENRAL_CACHE_DIR/galaxea_a1/<sha>/``.

Upstream names its meshes ``package://mobiman/urdf/A1/meshes/<file>`` (a
package of Galaxea's own workspace that is not published), so the URDF
returned is a derived copy beside the original with those refs rewritten to
``../meshes/<file>``, relative to itself.

``ensure_a1_mjcf()`` derives the MuJoCo twin's model from that URDF (issue #740).
Upstream models the two fingers as ``fixed`` joints ``gripper1_axis`` /
``gripper2_axis``, yet each still declares ``<axis xyz="0 0 1">`` and
``<limit lower="0" upper="0.03">``: the vendor's own finger slide, 0-30 mm per
finger (+q opens). The derive step turns both into those prismatic joints,
drives ``gripper1_axis`` with a ``<position>`` actuator and couples
``gripper2_axis`` to it with an ``<equality>``, and adds one ``<position>``
actuator per arm joint clamped at the URDF effort. The URDF has no actuator
model, so the gains are OpenRAL's sim-only choice and say nothing about the
real A1 joint loop. The real HAL's 0-104 mm ``gripper_stroke_*_mm`` defaults are
a different figure (the sidecar's normalised stroke): the twin only ever
exchanges the normalised ``[0, 1]`` value, as the real HAL does.
"""

from __future__ import annotations

import hashlib
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_files import fetch_pinned_file

__all__ = ["GALAXEA_DESCRIPTIONS_SHA", "ensure_a1_mjcf", "ensure_a1_urdf"]

#: ``userguide-galaxea/robot_descriptions`` ``galaxea/main`` on 2026-11-12.
GALAXEA_DESCRIPTIONS_SHA: str = "72008ea950a4c9258387d30821a7d8d8a2b588ca"
_RAW = "https://raw.githubusercontent.com/userguide-galaxea/robot_descriptions"
_SUBDIR = "a1_description"
_URDF = "urdf/A1_URDF_0607_0028.urdf"
_UPSTREAM_MESH_PREFIX = "package://mobiman/urdf/A1/meshes/"
# SHA-256 of every file, relative to ``_SUBDIR``, at the pinned commit.
_FILES: dict[str, str] = {
    _URDF: "9c7f3f391d3bd283c406a921565b4d504765cbc4c339c6c8075a5c1803a49c2c",
    "meshes/arm_seg1.STL": "b80652f37e0d2d4b59d5127cbd97b404ddf661cd421d2ea47c100cc991bedcae",
    "meshes/arm_seg2.STL": "98d224d819f1ecbbd19c979e4f18409bd16e277bd752c4bb3fd913182bee41d8",
    "meshes/arm_seg3.STL": "dbfd8fea24de4480195d62096b63c206c002ba122021dbdab71ccfe4ea1a56f0",
    "meshes/arm_seg4.STL": "df203b499f3c2eff565b2078a9b8b0888552c9453b97d7734f82b0689208a6bf",
    "meshes/arm_seg5.STL": "cf71a6682a3a702d92ff6278742549a9fe2620f6edc338c4d5b45621743d2ff6",
    "meshes/arm_seg6.STL": "c0073262d2cc5f950eb66ef4bec90662f30791c4be8ba6b88e0a6b5584f4153f",
    "meshes/base_link.STL": "be3d34b3244d745f527273f2005ca4ae9b08118c7778c4f3fae88a6e361203a0",
    "meshes/gripper1.STL": "e888cc5f0cd77a8e1abef7b436378a38a2d0ee7268cc739fb51f9913d4ce5a1d",
    "meshes/gripper2.STL": "393dfe556f193ff31694e765e01161c00e2a2bb563ed1f5ea2f28d4c00048a47",
}


def ensure_a1_urdf() -> Path:
    """Return the A1 URDF with relative mesh refs, fetching the pinned files if needed.

    Idempotent: verified cached files are reused without network access.

    Raises:
        ROSConfigError: A download fails or a file fails SHA-256 verification.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    root = base / "galaxea_a1" / GALAXEA_DESCRIPTIONS_SHA
    for rel, digest in _FILES.items():
        fetch_pinned_file(
            f"{_RAW}/{GALAXEA_DESCRIPTIONS_SHA}/{_SUBDIR}/{rel}",
            digest,
            root / rel,
            what=f"Galaxea A1 asset {rel} (pin {GALAXEA_DESCRIPTIONS_SHA[:10]})",
            user_agent="OpenRAL-galaxea-a1-assets",
        )
    upstream = root / _URDF
    derived = upstream.with_name("a1.urdf")
    xml = upstream.read_text(encoding="utf-8").replace(_UPSTREAM_MESH_PREFIX, "../meshes/")
    if not derived.is_file() or derived.read_text(encoding="utf-8") != xml:
        tmp = derived.with_name(f".{derived.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(xml, encoding="utf-8")
        os.replace(tmp, derived)
    return derived


# The derived MJCF is keyed by this module's source (every derive constant
# lives here), so an edit invalidates a stale cache without a hand-bumped version.
_DERIVE_KEY: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
_ARM_JOINTS: tuple[str, ...] = tuple(f"arm_joint{i}" for i in range(1, 7))
_DRIVEN_FINGER = "gripper1_axis"
_MIMIC_FINGER = "gripper2_axis"
# Sim-only position-servo gains (kp, kv). Joints 1-3 carry the 40/40/27 N·m
# motors of the URDF, the wrist the 7 N·m ones; the finger is a slide in metres.
_KP_KV: dict[str, tuple[float, float]] = {
    "arm_joint1": (150.0, 6.0),
    "arm_joint2": (800.0, 30.0),
    "arm_joint3": (600.0, 20.0),
    "arm_joint4": (100.0, 3.0),
    "arm_joint5": (80.0, 2.5),
    "arm_joint6": (20.0, 0.8),
    _DRIVEN_FINGER: (1000.0, 20.0),
}
# The finger meshes' convex hulls overlap at the closed position (0 m) by
# construction, and ``base_link`` is welded to the world (MuJoCo's parent/child
# filter does not cover it).
_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("gripper1", "gripper2"),
    ("base_link", "arm_seg1"),
)
_JOINT_DAMPING = 0.5
_JOINT_ARMATURE = 0.02


def _derive_mjcf(urdf: Path) -> str:
    """Compile the A1 URDF with sliding fingers and add actuators + finger equality.

    Raises:
        ROSConfigError: ``mujoco`` is missing, or the pinned URDF lost a joint,
            range or effort limit this step needs.
    """
    try:
        import mujoco  # noqa: PLC0415 — sim-only optional dep, lazy
    except ImportError as exc:
        raise ROSConfigError(
            f"Deriving the Galaxea A1 MJCF needs the `mujoco` package "
            f"({type(exc).__name__}: {exc}); install the sim extras."
        ) from exc
    root = ET.parse(urdf).getroot()
    urdf_joints = {str(j.get("name")): j for j in root.iter("joint")}
    efforts: dict[str, float] = {}
    for name in (*_ARM_JOINTS, _DRIVEN_FINGER, _MIMIC_FINGER):
        joint = urdf_joints.get(name)
        limit = None if joint is None else joint.find("limit")
        if joint is None or limit is None or limit.get("effort") is None:
            raise ROSConfigError(f"Galaxea A1 URDF: joint {name!r} or its <limit> is missing.")
        efforts[name] = float(str(limit.get("effort")))
        if name in (_DRIVEN_FINGER, _MIMIC_FINGER):
            joint.set("type", "prismatic")  # the vendor's declared axis + limit
    # Absolute: MuJoCo resolves a relative meshdir against the generated file's own
    # directory, so a relative OPENRAL_CACHE_DIR would point the cached MJCF nowhere.
    meshdir = (urdf.parent.parent / "meshes").resolve()
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(
        ext,
        "compiler",
        {
            "meshdir": str(meshdir),
            "strippath": "true",
            "fusestatic": "false",
            "discardvisual": "true",
            "balanceinertia": "true",
        },
    )
    try:
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        spec.compile()
        mjcf = ET.fromstring(spec.to_xml())
    except (ValueError, ET.ParseError) as exc:
        raise ROSConfigError(f"Deriving the Galaxea A1 MJCF failed: {exc}") from exc
    mjcf.set("model", "galaxea_a1")
    compiler = mjcf.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        mjcf.insert(0, compiler)
    compiler.set("meshdir", str(meshdir))
    compiler.set("angle", "radian")
    mj_joints = {str(j.get("name")): j for j in mjcf.iter("joint")}
    for joint in mj_joints.values():
        joint.set("damping", f"{_JOINT_DAMPING}")
        joint.set("armature", f"{_JOINT_ARMATURE}")
    for old in mjcf.findall("actuator") + mjcf.findall("equality") + mjcf.findall("contact"):
        mjcf.remove(old)
    contact = ET.SubElement(mjcf, "contact")
    for body1, body2 in _CONTACT_EXCLUDES:
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
    actuator = ET.SubElement(mjcf, "actuator")
    for name in (*_ARM_JOINTS, _DRIVEN_FINGER):
        kp, kv = _KP_KV[name]
        lo, hi = str(mj_joints[name].get("range")).split()
        force = efforts[name]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": "gripper" if name == _DRIVEN_FINGER else name,
                "joint": name,
                "kp": f"{kp}",
                "kv": f"{kv}",
                "ctrlrange": f"{lo} {hi}",
                "forcerange": f"{-force} {force}",
            },
        )
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "finger_mimic",
            "joint1": _MIMIC_FINGER,
            "joint2": _DRIVEN_FINGER,
            "polycoef": "0 1 0 0 0",
        },
    )
    ET.indent(mjcf)
    return ET.tostring(mjcf, encoding="unicode")


def ensure_a1_mjcf() -> Path:
    """Return the A1 twin MJCF (6 arm + 2 finger joints, 7 actuators), deriving it if needed.

    Raises:
        ROSConfigError: The pinned fetch fails or ``mujoco`` is not installed.
    """
    urdf = ensure_a1_urdf()
    path = urdf.parent.parent / f"derived-{_DERIVE_KEY}" / "a1.xml"
    # The derived XML pins an absolute meshdir: a cache tree moved since (copied from a
    # build host, say) would still point at the old meshes, so re-derive when it differs.
    meshdir = f'meshdir="{(urdf.parent.parent / "meshes").resolve()}"'
    if not path.is_file() or meshdir not in path.read_text(encoding="utf-8"):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(_derive_mjcf(urdf), encoding="utf-8")
        os.replace(tmp, path)
    return path
