"""Generic sim camera rig (ADR-0065).

Splice a robot's manifest-declared RGB cameras into a bare-arm MJCF that ships
no ``<camera>`` elements, so a ``deploy sim`` :class:`~openral_hal._mujoco_arm.MujocoArmHAL`
twin renders the cameras the manifest declares — without a per-robot scene
composer or any ``scene_defaults.composition`` hook on the robot manifest.

Each RGB :class:`~openral_core.SensorSpec` that carries a
:class:`~openral_core.CameraSimPlacement` is spliced as a ``<camera>`` either
into its ``parent_body`` (a wrist camera that tracks the gripper) or into
``<worldbody>`` (a world-fixed overhead / third-person camera). The rig is
**idempotent**: a camera already present in the MJCF (a scene-attached or
already-composed model) is left untouched, so this composes cleanly with the
scene-attach path (ADR-0034) and prop composers (openarm).

The camera's MuJoCo name is ``sim_camera_name or name`` — the same key
:meth:`MujocoArmHAL.read_images` renders — so the rig and the reader agree by
construction.
"""

from __future__ import annotations

import math
import re

from openral_core import SensorSpec
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import look_at_quat_wxyz

__all__ = ["rig_cameras_into_mjcf"]


def _fovy_deg_for(sensor: SensorSpec) -> float:
    """Vertical FoV (degrees) for a sensor's sim camera.

    Uses ``sim_placement.fovy_deg`` when set, else derives it from the pinhole
    ``intrinsics`` (``2·atan(height / (2·fy))``) so the rendered FoV matches the
    declared camera model. Falls back to 60° when neither is available.
    """
    placement = sensor.sim_placement
    assert placement is not None  # caller guards
    if placement.fovy_deg is not None:
        return placement.fovy_deg
    intr = sensor.intrinsics
    if intr is None or intr.fy <= 0.0:
        return 60.0
    return math.degrees(2.0 * math.atan(intr.height / (2.0 * intr.fy)))


def _camera_element(sensor: SensorSpec) -> str:
    """Build the ``<camera>`` XML for one RGB sensor with a sim placement."""
    placement = sensor.sim_placement
    assert placement is not None  # caller guards
    name = sensor.sim_camera_name or sensor.name
    px, py, pz = placement.pos
    # MuJoCo cameras look along local -Z; orient -Z from pos toward target.
    w, x, y, z = look_at_quat_wxyz(placement.pos, placement.target, view_axis="-z")
    fovy = _fovy_deg_for(sensor)
    return (
        f'<camera name="{name}" pos="{px} {py} {pz}" '
        f'quat="{w} {x} {y} {z}" fovy="{fovy}" mode="fixed"/>'
    )


# Visual-only ground plane: gives the cameras a surface to see without any
# physics interaction (``contype=0 conaffinity=0`` → no collisions, so the arm
# never contacts it and the safety kernel is unaffected). A bare arm MJCF has no
# floor, so a forward-looking wrist camera would otherwise render pure void.
_STAGING_FLOOR = (
    '<geom name="camrig_floor" type="plane" size="2 2 0.1" pos="0 0 0" '
    'rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>'
)


def _ensure_staging(xml: str) -> str:
    """Add minimal deploy-twin staging — a visual-only floor + fill light.

    A bare arm MJCF ships no floor and (often) no lighting, so a gripper-mounted
    camera looking into the workspace renders pure black/void. This adds:

    - a **visual-only** ground plane (no collisions) when the MJCF has no
      ``type="plane"`` geom — universal, parameterless deploy staging, not task
      props (CLAUDE.md / ADR-0065 keep scene props in scene files);
    - a moderate ambient ``<visual><headlight>`` when no ``<visual>`` block
      exists, lifting shadows without washing out materials.

    Both are no-ops when the MJCF already declares them (a composed/scene MJCF
    set its own), so the rig composes with those rather than clobbering them.
    """
    if 'type="plane"' not in xml:
        xml, n = re.subn(r"(</worldbody>)", f"        {_STAGING_FLOOR}\n      \\1", xml, count=1)
        if n != 1:  # no worldbody to stage into — leave the model as-is
            pass
    if "<visual" not in xml:
        headlight = (
            "\n  <visual>\n"
            '    <headlight ambient="0.4 0.4 0.4" diffuse="0.4 0.4 0.4" '
            'specular="0.1 0.1 0.1"/>\n'
            "  </visual>"
        )
        out, n = re.subn(r"(<mujoco\b[^>]*>)", r"\1" + headlight, xml, count=1)
        if n == 1:
            xml = out
    return xml


def rig_cameras_into_mjcf(xml: str, sensors: list[SensorSpec]) -> tuple[str, bool]:
    """Splice each RGB sensor's missing sim camera into ``xml``; return ``(xml, changed)``.

    For every RGB :class:`SensorSpec` with a :class:`~openral_core.CameraSimPlacement`
    whose camera name is absent from ``xml``, splice a ``<camera>`` into the
    named ``parent_body`` (or ``<worldbody>`` when ``parent_body`` is ``None``)
    and ensure a fill light. Cameras already present are skipped (idempotent), so
    a scene-attached or already-composed MJCF passes through unchanged
    (``changed=False``) and the caller can load the original file.

    Raises:
        ROSConfigError: A sensor's ``parent_body`` is not found in the MJCF, or
            ``</worldbody>`` is missing for a world-fixed camera — the splice
            anchors are wrong and must surface loudly, not render a blank frame.
    """
    existing = set(re.findall(r'<camera[^>]*\bname="([^"]+)"', xml))
    rigged = [
        s
        for s in sensors
        if s.modality == "rgb"
        and s.sim_placement is not None
        and (s.sim_camera_name or s.name) not in existing
    ]
    if not rigged:
        return xml, False

    world_cams: list[str] = []
    body_cams: dict[str, list[str]] = {}
    for s in rigged:
        placement = s.sim_placement
        assert placement is not None  # filtered above
        if placement.parent_body is None:
            world_cams.append(_camera_element(s))
        else:
            body_cams.setdefault(placement.parent_body, []).append(_camera_element(s))

    for body, cams in body_cams.items():
        snippet = "".join(f"\n        {c}" for c in cams)
        pattern = re.compile(rf'(<body[^>]*\bname="{re.escape(body)}"[^>]*>)')
        xml, n = pattern.subn(rf"\g<1>{snippet}", xml, count=1)
        if n != 1:
            raise ROSConfigError(
                f'camera rig: cannot find <body name="{body}"> to mount the wrist '
                f"camera(s) {[c.split(chr(34))[1] for c in cams]}. Check the sensor's "
                "sim_placement.parent_body matches an MJCF body name.",
            )

    if world_cams:
        snippet = "".join(f"\n        {c}" for c in world_cams)
        xml, n = re.subn(r"(</worldbody>)", f"{snippet}\n      \\1", xml, count=1)
        if n != 1:
            raise ROSConfigError(
                "camera rig: cannot find </worldbody> to mount the world-fixed "
                "camera(s) — the MJCF has no worldbody to splice into.",
            )

    return _ensure_staging(xml), True
