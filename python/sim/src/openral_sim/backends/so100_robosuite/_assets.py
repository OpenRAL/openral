"""Asset preparation for the robosuite-flavored SO-100 model.

The DeepMind ``mujoco_menagerie`` MJCF for the SO-100 (``trs_so_arm100``)
ships as a single monolithic robot description: arm + jaw in one XML, with
``position`` actuators and a relative ``meshdir``. robosuite expects:

* the arm and the gripper as **two** XMLs (``ManipulatorModel`` +
  ``GripperModel``);
* an outer ``base`` body wrapping the arm with a ``robotview`` camera and a
  ``right_hand`` body where the gripper attaches;
* ``motor`` actuators (torque control) so robosuite's stock
  OSC_POSITION composite controller can feed the computed torques
  into ``sim.data.ctrl``;
* an ``eef`` body inside the gripper exposing the conventional
  ``grip_site`` / ``ee_x`` / ``ee_y`` / ``ee_z`` sites.

We adapt the menagerie XML at import time and cache the rewritten files
under ``~/.cache/openral/so100_robosuite/`` next to absolute mesh
references — that avoids both shipping the STLs in-tree and copying them
around. The cache key embeds the menagerie revision so a menagerie upgrade
invalidates the cache automatically.

The transformation is done with ``xml.etree.ElementTree`` against the
upstream tree; no string templating, no regex hacks. Tests exercise it
end-to-end against a real ``robosuite.make`` + ``MjSim`` run, so any drift
in the menagerie schema surfaces immediately.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from openral_core.exceptions import ROSConfigError

__all__ = ["SO100Assets", "ensure_so100_assets"]


_GRIPPER_JOINT: str = "Jaw"
"""Single DOF parallel jaw — the only joint that moves to the gripper XML."""


@dataclass(frozen=True)
class SO100Assets:
    """Resolved on-disk paths for the robosuite-flavored SO-100 model.

    Attributes:
        robot_xml: Path to the arm-only robosuite XML (5 motor actuators).
        gripper_xml: Path to the jaw gripper XML (1 motor actuator).
        menagerie_dir: Path to the upstream menagerie SO-100 directory,
            kept around so debugging can compare against the original.
    """

    robot_xml: Path
    gripper_xml: Path
    menagerie_dir: Path


def _menagerie_paths() -> tuple[Path, Path]:
    """Resolve the SO-100 menagerie MJCF path, downloading it if needed.

    Returns:
        Tuple of ``(mjcf_path, menagerie_dir)``. ``mjcf_path`` points at
        ``so_arm100.xml``; ``menagerie_dir`` is its parent dir (holds the
        ``assets/`` STL collection).
    """
    try:
        from robot_descriptions import so_arm100_mj_description as desc
    except ImportError as exc:
        raise ROSConfigError(
            "SO-100 robosuite backend requires the `robot_descriptions` "
            "package (pulls the mujoco_menagerie SO-100 MJCF). Install it "
            "with `just sync --all-packages --group sim` or "
            "`just sync --all-packages --group robocasa`."
        ) from exc
    mjcf = Path(desc.MJCF_PATH)
    if not mjcf.is_file():
        raise ROSConfigError(
            f"robot_descriptions reports the SO-100 MJCF at {mjcf!s} but the "
            "file is missing — clear the cache at ~/.cache/robot_descriptions/ "
            "and retry."
        )
    return mjcf, mjcf.parent


def _cache_dir() -> Path:
    """OpenRAL cache root for generated robosuite-flavored SO-100 assets."""
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "so100_robosuite"


def _menagerie_fingerprint(mjcf: Path, mesh_dir: Path) -> str:
    """Hash inputs so a menagerie upgrade invalidates the rewritten XML cache.

    Hashes the MJCF bytes plus the names+sizes of the asset STLs (we don't
    hash STL contents — they are large and the menagerie pins each release
    to a fixed commit, so name+size is enough to detect a real change).
    """
    h = hashlib.sha256()
    h.update(mjcf.read_bytes())
    if mesh_dir.is_dir():
        for stl in sorted(mesh_dir.iterdir()):
            if stl.is_file():
                h.update(stl.name.encode())
                h.update(str(stl.stat().st_size).encode())
    return h.hexdigest()[:16]


def _flatten_defaults(menagerie_default: ET.Element) -> ET.Element:
    """Flatten nested ``<default class=...>`` blocks into a single-level list.

    The menagerie XML organises defaults hierarchically (e.g. ``Rotation``
    nested inside ``so_arm100``). MuJoCo's compiler resolves this fine,
    but robosuite's :meth:`MujocoXML._replace_defaults_inline` only
    looks at the **first** layer of children inside ``<default>`` — a
    body with ``class="Rotation"`` triggers a ``KeyError`` because
    ``Rotation`` is two levels deep.

    We walk the menagerie tree once, accumulating attribute defaults
    per tag (``joint``, ``geom``, …) along the way, and emit a flat
    ``<default>`` block where every nested class is hoisted to the top
    level with its inherited attributes baked in. ``<position>``
    prototypes are dropped — we converted the actuators to ``motor``.
    """
    new_root = ET.Element("default")

    def _walk(node: ET.Element, parent_tag_attrs: dict[str, dict[str, str]]) -> None:
        # Aggregate this class's own per-tag attributes on top of the parent's.
        own_tag_attrs: dict[str, dict[str, str]] = {
            tag: dict(attrs) for tag, attrs in parent_tag_attrs.items()
        }
        for child in node:
            if child.tag == "default":
                continue
            if child.tag == "position":
                # Drop — we use motor actuators.
                continue
            merged = dict(own_tag_attrs.get(child.tag, {}))
            merged.update(child.attrib)
            own_tag_attrs[child.tag] = merged

        cls_name = node.get("class")
        if cls_name is not None:
            flat = ET.SubElement(new_root, "default")
            flat.set("class", cls_name)
            for tag, attrs in own_tag_attrs.items():
                el = ET.SubElement(flat, tag)
                for k, v in attrs.items():
                    el.set(k, v)

        for child in node:
            if child.tag == "default":
                _walk(child, own_tag_attrs)

    _walk(menagerie_default, {})

    # robosuite's MujocoXMLModel filter only knows group "0" (collision)
    # and "1" (visual); the menagerie uses "2" for visual and "3" for
    # collision. Remap inside the flattened defaults so the inlined
    # attributes drop in clean — without this the unnamed visual geoms
    # in the arm chain trip a KeyError during ``sort_elements``.
    group_remap = {"2": "1", "3": "0"}
    for cls in new_root.iter("default"):
        for tag in cls.iter():
            grp = tag.get("group")
            if grp in group_remap:
                tag.set("group", group_remap[grp])
    return new_root


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """Locate a ``<body name=...>`` element, raising on miss."""
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ROSConfigError(f"SO-100 MJCF: missing <body name={name!r}> — menagerie schema drift?")


def _write_robot_xml(menagerie_xml: ET.Element, mesh_dir: Path, out: Path) -> None:
    """Rewrite the menagerie XML into a robosuite-compatible arm-only XML.

    The transformation:

    1. Set ``meshdir`` on the compiler element to the absolute menagerie
       assets dir so mesh references resolve regardless of cwd.
    2. Drop the ``Jaw`` actuator (it moves to the gripper) and convert the
       remaining ``position`` actuators to ``motor`` actuators with
       symmetric torque limits matching the menagerie ``forcerange``
       generous (``±200 N·m``). robosuite's OSC controllers compute
       ``mass_matrix @ desired_acc + gravity_compensation`` and write
       the result straight to ``sim.data.ctrl``; for the SO-100's
       tiny mass-matrix entries (≈ 1e-3 to 1e-1) those torques get
       amplified into the tens of newton-metres even though the
       resulting joint motion is small, so the ctrl range must be
       wide enough to not clip.
    3. Drop the ``Moving_Jaw`` subtree from ``Fixed_Jaw`` so the gripper
       can attach its own moving jaw there.
    4. Wrap the existing ``Base`` body in a robosuite-style ``<body
       name="base">`` that exposes a ``robotview`` camera and a
       ``right_center`` site on the inner body (robosuite uses both).
    5. Add a ``right_hand`` body inside ``Fixed_Jaw`` where the gripper
       merges, with an ``eye_in_hand`` camera pointing along the jaw.
    6. Drop the ``<keyframe>`` element (it references the Jaw joint we
       removed).
    """
    # Work on a deep copy via tostring/fromstring so we don't mutate the
    # caller's tree.
    root = ET.fromstring(ET.tostring(menagerie_xml))

    # 1. absolute mesh file paths. We can't rely on ``meshdir`` because
    # robosuite's :class:`MujocoXML.resolve_asset_dependency` rewrites
    # every ``<mesh file="...">`` to ``os.path.join(self.folder, file)``
    # without honouring ``meshdir`` — so a relative file inside the
    # cache dir gets a wrong absolute path. Setting ``file`` to an
    # absolute path makes ``os.path.join`` a no-op (its second absolute
    # argument wins).
    compiler = root.find("compiler")
    if compiler is None:
        raise ROSConfigError("SO-100 MJCF missing <compiler> element")
    compiler.set("meshdir", str(mesh_dir.resolve()))
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            f = mesh.get("file")
            if f is not None and not os.path.isabs(f):
                mesh.set("file", str((mesh_dir / f).resolve()))

    # 2. actuators: remove Jaw, motorize the rest
    actuator = root.find("actuator")
    if actuator is None:
        raise ROSConfigError("SO-100 MJCF missing <actuator> element")
    for act in list(actuator):
        if act.get("joint") == _GRIPPER_JOINT or act.get("name") == _GRIPPER_JOINT:
            actuator.remove(act)
            continue
        joint_name = act.get("joint")
        motor = ET.SubElement(actuator, "motor")
        motor.set("name", joint_name or "")
        motor.set("joint", joint_name or "")
        motor.set("ctrllimited", "true")
        # Wide torque headroom (200 N·m). The menagerie's
        # ``forcerange="-3.5 3.5"`` is the internal torque the *physical*
        # position-controlled servo produces; here we drive the joint
        # via ``motor`` actuators + robosuite's stock OSC composite
        # controller, which computes ``mass_matrix @ desired_acc +
        # gravity_compensation``. The mass_matrix multiplication
        # amplifies the OSC output by 30-50× for the SO-100's chain
        # (vs the Panda's 1-5×), so peak torque during a settle step
        # routinely exceeds 30 N·m even though the resulting joint
        # motion is small. Clipping below 200 N·m leaves the arm
        # unable to track its commanded pose against gravity.
        motor.set("ctrlrange", "-200 200")
        actuator.remove(act)

    # Flatten the menagerie's nested defaults so robosuite's parser can
    # resolve `class="Rotation"` etc. (it only looks one level deep).
    default = root.find("default")
    if default is not None:
        flat = _flatten_defaults(default)
        # Drop the Jaw default class — its joint moved to the gripper.
        for inner in list(flat):
            if inner.get("class") == "Jaw":
                flat.remove(inner)
        root.remove(default)
        # Insert in the same slot the menagerie used (after compiler / option).
        root.insert(2, flat)

    # 3. Strip the entire jaw geometry from the arm. The fixed jaw +
    # moving jaw both live in the gripper XML; leaving them in the arm
    # too creates duplicate geoms at the same world position (phantom
    # contacts that push the cube away during grasp). What stays on
    # the arm is only:
    #   * the Wrist_Roll joint (necessary for wrist rotation);
    #   * the Fixed_Jaw inertial (so the body has mass for the
    #     dynamics — without inertial MuJoCo errors on bodies with
    #     joints);
    #   * the new ``right_hand`` mount body added below.
    # Visuals and collisions all move to the gripper.
    fixed_jaw = _find_body(root, "Fixed_Jaw")
    for child in list(fixed_jaw):
        if child.tag == "geom" or (child.tag == "body" and child.get("name") == "Moving_Jaw"):
            fixed_jaw.remove(child)

    # Strip ``childclass`` attributes. robosuite's MujocoXML inlines every
    # ``class=`` reference and then **removes** the entire ``<default>``
    # block before passing the XML to MuJoCo's compiler. Any remaining
    # ``childclass`` reference becomes a dangling pointer at that point
    # ("unknown default childclass"). The inlining already covered every
    # element with an explicit class, so dropping ``childclass`` outright
    # is safe.
    for body in root.iter("body"):
        body.attrib.pop("childclass", None)

    # 4. wrap the existing Base body in robosuite's outer <body name="base">
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ROSConfigError("SO-100 MJCF missing <worldbody> element")
    arm_root = _find_body(root, "Base")
    worldbody.remove(arm_root)
    outer_base = ET.SubElement(worldbody, "body")
    outer_base.set("name", "base")
    outer_base.set("pos", "0 0 0")
    # Third-person view, ~80 cm in front of the arm, height ≈ 60 cm,
    # looking back at the workspace. Quaternion derived from a
    # 60° pitch toward -x at 60° yaw above the table.
    cam = ET.SubElement(outer_base, "camera")
    cam.set("mode", "fixed")
    cam.set("name", "robotview")
    cam.set("pos", "0.8 0.0 0.7")
    cam.set("quat", "0.653 0.271 0.271 0.653")
    inertial = ET.SubElement(outer_base, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", "0")
    inertial.set("diaginertia", "0 0 0")
    outer_base.append(arm_root)

    # robosuite uses a `right_center` site on the *inner* root body as the
    # mount-attachment marker. The original SO-100 Base body is empty of
    # sites; add it explicitly.
    site = ET.Element("site")
    site.set("name", "right_center")
    site.set("pos", "0 0 0")
    site.set("size", "0.01")
    site.set("rgba", "1 0.3 0.3 1")
    site.set("group", "2")
    arm_root.insert(0, site)

    # 5. right_hand body inside Fixed_Jaw (0, 0, 0 offset — gripper geometry
    # in the gripper XML carries its own offsets). The eye_in_hand camera
    # points outward along the local -y axis (toward the fingers).
    right_hand = ET.SubElement(fixed_jaw, "body")
    right_hand.set("name", "right_hand")
    right_hand.set("pos", "0 0 0")
    rh_inertial = ET.SubElement(right_hand, "inertial")
    rh_inertial.set("pos", "0 0 0")
    rh_inertial.set("mass", "0.01")
    rh_inertial.set("diaginertia", "1e-5 1e-5 1e-5")
    eye = ET.SubElement(right_hand, "camera")
    eye.set("mode", "fixed")
    eye.set("name", "eye_in_hand")
    eye.set("pos", "0 -0.04 0.04")
    eye.set("quat", "0.707 0.707 0 0")
    eye.set("fovy", "75")

    # 6. drop keyframe (refs the Jaw joint we removed)
    kf = root.find("keyframe")
    if kf is not None:
        root.remove(kf)

    _write_xml(root, out)


def _write_gripper_xml(menagerie_xml: ET.Element, mesh_dir: Path, out: Path) -> None:
    """Build a robosuite gripper XML from the SO-100 jaw + moving finger.

    The gripper XML must expose:

    * a single root body (becomes a child of ``right_hand`` after
      ``add_gripper``);
    * an ``eef`` child body with ``grip_site`` / ``ee_x|y|z`` /
      ``grip_site_cylinder`` sites for visualization and contact queries;
    * an ``ft_frame`` site + ``force_ee`` / ``torque_ee`` sensors;
    * a ``motor`` actuator on the ``Jaw`` joint with a torque range
      matching the menagerie value.

    We copy the menagerie's ``Fixed_Jaw`` geom set (the static jaw + its
    finger pads) and the ``Moving_Jaw`` subtree (with its ``Jaw`` joint
    and pads) verbatim — the only edits are wrapping them inside the
    robosuite-expected body hierarchy and shifting the eef site to sit
    between the closed pads.
    """
    src = ET.fromstring(ET.tostring(menagerie_xml))
    fixed_jaw_src = _find_body(src, "Fixed_Jaw")
    moving_jaw_src = next(
        (c for c in fixed_jaw_src.findall("body") if c.get("name") == "Moving_Jaw"),
        None,
    )
    if moving_jaw_src is None:
        raise ROSConfigError("SO-100 MJCF: missing Moving_Jaw body")

    root = ET.Element("mujoco")
    root.set("model", "so100_gripper")
    compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(mesh_dir.resolve()))

    # Reuse the menagerie's mesh asset list verbatim — robosuite's
    # `merge_assets` dedups so redeclaring the same names twice (also in
    # the arm XML) is fine. Absolute file paths for the same reason as
    # in the arm XML (see :func:`_write_robot_xml`).
    asset_src = src.find("asset")
    if asset_src is not None:
        cloned = ET.fromstring(ET.tostring(asset_src))
        for mesh in cloned.findall("mesh"):
            f = mesh.get("file")
            if f is not None and not os.path.isabs(f):
                mesh.set("file", str((mesh_dir / f).resolve()))
        root.append(cloned)

    # Defaults: flatten so robosuite's parser is happy. We keep only the
    # classes referenced in the gripper subtree (Jaw + visual / collision
    # variants) — the arm joint classes are irrelevant here.
    default_src = src.find("default")
    if default_src is not None:
        flat = _flatten_defaults(default_src)
        keep = {"so_arm100", "Jaw", "visual", "motor_visual", "collision", "finger_collision"}
        for inner in list(flat):
            if inner.get("class") not in keep:
                flat.remove(inner)
        root.append(flat)

    # Strip childclass on the gripper bodies for the same reason as the
    # arm (robosuite drops the defaults block before MuJoCo compiles).
    # We do this on cloned subtrees BELOW.

    # Tighten the Jaw range. The menagerie ships ``range="-0.174 1.75"``
    # which lets the moving jaw rotate ≈ 100° from closed. At full open
    # the jaw tip protrudes ~10 cm past the wrist — far enough that on
    # a low table (5 cm offset) it pierces the table surface during a
    # canonical top-down grasp, generating a contact force that pins
    # the shoulder joint. Capping the upper bound at 0.8 rad gives a
    # ~3 cm jaw aperture (plenty for the 2.4 cm cube) without the
    # moving jaw reaching the table.
    # Tighten the Jaw range. The menagerie ships ``range="-0.174 1.75"``
    # which lets the moving jaw rotate ≈ 100° from closed. At full open
    # the jaw tip protrudes ~10 cm past the wrist — far enough that on
    # a low table (5 cm offset) it can pierce the table surface during
    # a canonical top-down grasp, generating a contact force that pins
    # the shoulder joint. Capping the upper bound at 0.5 rad gives a
    # ~4 cm jaw aperture (plenty for the 2.4 cm cube) without the
    # moving jaw reaching the table.
    for cls in flat.iter("default"):
        if cls.get("class") == "Jaw":
            for child in cls:
                if child.tag == "joint" and child.get("range"):
                    child.set("range", "-0.174 0.5")

    # Single motor actuator for the Jaw. Range matches the menagerie's
    # forcerange="-3.5 3.5" position default — the jaw is small enough
    # that lower torque is more realistic, but we keep parity with the
    # arm actuators so the SimpleGripController scales action ∈ [-1, 1]
    # to ±3.5 N·m on close/open commands.
    actuator = ET.SubElement(root, "actuator")
    motor = ET.SubElement(actuator, "motor")
    motor.set("name", "Jaw")
    motor.set("joint", "Jaw")
    motor.set("ctrllimited", "true")
    motor.set("ctrlrange", "-1.5 1.5")

    worldbody = ET.SubElement(root, "worldbody")
    # The root gripper body — robosuite expects a single root in <worldbody>.
    # ``childclass="so_arm100"`` so the visual / collision defaults apply.
    gripper_root = ET.SubElement(worldbody, "body")
    gripper_root.set("name", "right_gripper")
    gripper_root.set("pos", "0 0 0")
    gripper_root.set("childclass", "so_arm100")
    # ft_frame site
    ft = ET.SubElement(gripper_root, "site")
    ft.set("name", "ft_frame")
    ft.set("pos", "0 0 0")
    ft.set("size", "0.01 0.01 0.01")
    ft.set("rgba", "1 0 0 1")
    ft.set("type", "sphere")
    ft.set("group", "1")
    # Inertial — the original Fixed_Jaw inertial expressed in the same
    # local frame (the merged-in body sits where Fixed_Jaw used to be
    # because right_hand is at offset 0,0,0 inside Fixed_Jaw).
    inertial = ET.SubElement(gripper_root, "inertial")
    inertial.set("pos", "0.00552377 -0.0280167 0.000483583")
    inertial.set("quat", "0.41836 0.620891 -0.350644 0.562599")
    inertial.set("mass", "0.0929859")
    inertial.set("diaginertia", "5.03136e-05 4.64098e-05 2.72961e-05")

    # eef body — the conventional robosuite gripper sites. We position
    # `eef` between the closed pads (the fixed pads sit around y ≈ -0.075
    # in local coords; the moving pads converge from y ≈ -0.05).
    eef = ET.SubElement(gripper_root, "body")
    eef.set("name", "eef")
    eef.set("pos", "0 -0.085 0")
    eef.set("quat", "1 0 0 0")
    for site_name, extra in (
        ("grip_site", {"size": "0.005 0.005 0.005", "rgba": "1 0 0 0.5", "type": "sphere"}),
        (
            "ee_x",
            {
                "size": "0.003 0.05",
                "pos": "0.05 0 0",
                "quat": "0.707105 0 0.707108 0",
                "rgba": "1 0 0 0",
                "type": "cylinder",
            },
        ),
        (
            "ee_y",
            {
                "size": "0.003 0.05",
                "pos": "0 0.05 0",
                "quat": "0.707105 0.707108 0 0",
                "rgba": "0 1 0 0",
                "type": "cylinder",
            },
        ),
        (
            "ee_z",
            {
                "size": "0.003 0.05",
                "pos": "0 0 0.05",
                "quat": "1 0 0 0",
                "rgba": "0 0 1 0",
                "type": "cylinder",
            },
        ),
        (
            "grip_site_cylinder",
            {"size": "0.003 1.0", "pos": "0 0 0", "rgba": "0 1 0 0.3", "type": "cylinder"},
        ),
    ):
        s = ET.SubElement(eef, "site")
        s.set("name", site_name)
        s.set("group", "1")
        s.set("pos", extra.pop("pos", "0 0 0"))
        for k, v in extra.items():
            s.set(k, v)

    # Fixed jaw geometry — copy the geoms from the menagerie Fixed_Jaw
    # but DROP the bulky ``Fixed_Jaw_Collision_*`` mesh collisions.
    # Those collide with the table at typical grasp heights (the SO-100
    # has only ~5 cm of clearance between the wrist and the table
    # top), which pins the shoulder joint mid-descent. The
    # ``finger_collision``-class box pads are enough for collision +
    # grasping; the mesh-class geoms are kept ONLY as visuals
    # (``contype=0`` already from the class default's group=1).
    for child in list(fixed_jaw_src):
        if child.tag in {"inertial", "joint"}:
            continue
        if child.tag == "body" and child.get("name") == "Moving_Jaw":
            continue
        if child.tag == "geom" and child.get("class") == "collision":
            # Mesh collision geom — skip entirely. The fixed jaw pads
            # (class="finger_collision", separate small boxes) handle
            # the grasp.
            continue
        gripper_root.append(ET.fromstring(ET.tostring(child)))

    # Moving jaw — copy verbatim but apply the same collision-mesh
    # strip so it doesn't bump the table on the way down either. The
    # moving-jaw pads (finger_collision class) are kept.
    moving_clone = ET.fromstring(ET.tostring(moving_jaw_src))
    for child in list(moving_clone):
        if child.tag == "geom" and child.get("class") == "collision":
            moving_clone.remove(child)
    gripper_root.append(moving_clone)

    # Sensors. Use distinct names from the earlier `f = mesh.get("file")`
    # bindings (str | None) to keep mypy happy in this scope.
    sensor = ET.SubElement(root, "sensor")
    force_el = ET.SubElement(sensor, "force")
    force_el.set("name", "force_ee")
    force_el.set("site", "ft_frame")
    torque_el = ET.SubElement(sensor, "torque")
    torque_el.set("name", "torque_ee")
    torque_el.set("site", "ft_frame")

    # Strip childclass on all bodies for the same reason as the arm XML.
    for body in root.iter("body"):
        body.attrib.pop("childclass", None)

    _write_xml(root, out)


def _write_xml(root: ET.Element, out: Path) -> None:
    """Serialize ``root`` to ``out`` atomically."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(out)


def ensure_so100_assets() -> SO100Assets:
    """Generate (and cache) the robosuite-flavored SO-100 XMLs.

    Cached under ``$OPENRAL_CACHE_DIR/so100_robosuite/<fingerprint>/``
    so a menagerie upgrade invalidates the cache automatically.

    Returns:
        :class:`SO100Assets` carrying absolute paths to both XMLs.
    """
    menagerie_mjcf, menagerie_dir = _menagerie_paths()
    mesh_dir = menagerie_dir / "assets"
    if not mesh_dir.is_dir():
        raise ROSConfigError(
            f"SO-100 menagerie at {menagerie_dir!s} missing the `assets/` "
            "mesh directory — clear ~/.cache/robot_descriptions/ and retry."
        )

    fingerprint = _menagerie_fingerprint(menagerie_mjcf, mesh_dir)
    cache = _cache_dir() / fingerprint
    robot_xml = cache / "so100_robot.xml"
    gripper_xml = cache / "so100_gripper.xml"

    if not (robot_xml.is_file() and gripper_xml.is_file()):
        tree = ET.parse(menagerie_mjcf)
        menagerie_root = tree.getroot()
        try:
            _write_robot_xml(menagerie_root, mesh_dir, robot_xml)
            _write_gripper_xml(menagerie_root, mesh_dir, gripper_xml)
        except Exception:
            # Half-written caches mislead the next process; nuke and re-raise.
            shutil.rmtree(cache, ignore_errors=True)
            raise

    return SO100Assets(
        robot_xml=robot_xml,
        gripper_xml=gripper_xml,
        menagerie_dir=menagerie_dir,
    )
