"""The hand's pinch volume ``J_h`` from joint-state FK (issue #418, "Grasp binding").

``J_h`` is the volume the pads would still close through: the oriented box, in the
gripper's rigid parent link (the attach link), around every pad collision primitive
posed at jaw positions swept from the **measured** jaw to the manifest's closed position.
What keeps the jaws apart lies in it. The fingers are the pads: the gripper's
``grasp_contact_links`` no other finger link hangs off. Its axes, fixed per hand, are the
closing direction at contact (``c``), the approach (``a``: palm towards fingertips) and
the pad width (``b = c x a``). Nothing of it lies behind the palm plane: the plane
through the gripper joint's origin, normal to ``a`` (the HAL's ``palm_plane_m``
convention), moved forward to the wrist's front face where the wrist reaches past it.

Pure numpy over the manifest; mimic followers are posed through ``MimicJoint.value_at``
(``openral_core.kinematics``), never a tf that reads the SO-101's jaw fraction as radians.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from openral_core import FixedAttachment, JointSpec, MimicJoint, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.kinematics import (
    Box,
    bounding_half_extents,
    link_transform,
    transform_from_xyz_rpy,
)

__all__ = ["HandModel", "PinchVolume"]

#: Jaw samples over the closing sweep. 100 deg of SO-101 jaw in 15 steps sags < 1 mm
#: between samples on a 10 cm finger, inside the one cell ``J_h`` is grown by.
_SWEEP_SAMPLES = 16
#: Below this a closing-motion covariance is degenerate (no finger moves).
_DEGENERATE = 1e-12
_UNIT_BOX = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


@dataclass(frozen=True)
class PinchVolume:
    """An oriented box in the attach link, plus the palm plane it never crosses.

    Attributes:
        centre: Box centre in the attach link (m).
        axes: Columns ``(c, a, b)``: closing, approach, pad width.
        half: Half extents along ``(c, a, b)`` (m).
        palm_m: The palm plane, ``{x : x . a >= palm_m}`` in the attach link.
    """

    centre: NDArray[np.float64]
    axes: NDArray[np.float64]
    half: NDArray[np.float64]
    palm_m: float

    def box(self) -> Box:
        """``(centre, rotation, half)`` for ``box_gap_lower_bound_m``."""
        return (self.centre, self.axes, self.half)

    def grown(self, margin_m: float) -> PinchVolume:
        """The same box grown by ``margin_m`` on every face (``J_h`` takes one cell)."""
        return PinchVolume(self.centre, self.axes, self.half + margin_m, self.palm_m)


class HandModel:
    """One gripper's fingers, posed by FK from the manifest (``role: gripper`` joint).

    Raises:
        ROSConfigError: ``gripper_joint`` is not a gripper-role joint, has no
            ``closure_calibration`` (no closed end of the stroke), or its fingers carry
            no collision geometry.

    Example:
        >>> d = RobotDescription.from_yaml("robots/droid_franka_robotiq/robot.yaml")
        >>> hand = HandModel(d, "finger_joint")
        >>> hand.attach_link
        'robotiq_arg2f_base_link'
        >>> j = hand.pinch_in_link(0.3)
        >>> [round(float(v), 2) for v in np.abs(j.axes[:, 0])]  # closing along the link's y
        [0.0, 1.0, 0.0]
    """

    def __init__(self, description: RobotDescription, gripper_joint: str) -> None:
        """Resolve the attach link, touch links, palm point and finger primitives."""
        joint: JointSpec | None = next(
            (j for j in description.joints if j.name == gripper_joint and j.role == "gripper"),
            None,
        )
        if joint is None:
            raise ROSConfigError(f"{gripper_joint!r} is not a gripper-role joint.")
        if joint.closure_calibration is None:
            raise ROSConfigError(f"{gripper_joint!r} has no closure_calibration (closed end).")
        self.description = description
        self.joint = joint
        self.attach_link = joint.parent_link
        self.touch_links = description.grasp_contact_links(gripper_joints=[gripper_joint])
        self.closed = float(joint.closure_calibration.closed_position)
        self.palm = np.asarray(joint.origin_xyz, dtype=np.float64)
        # The pads: finger links no other finger hangs off (the Robotiq's knuckles and
        # bars carry its pads and stay outside the gap; the SO-101's moving jaw is one).
        edges: list[JointSpec | MimicJoint | FixedAttachment] = [
            *description.joints,
            *description.mimic_joints,
            *description.fixed_attachments,
        ]
        parents = {e.parent_link for e in edges if e.child_link in self.touch_links}
        pads = set(self.touch_links) - parents
        self._pads = [
            (
                g.link_name,
                transform_from_xyz_rpy(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:]),
                bounding_half_extents(g.shape),
            )
            for g in description.collision_geometry
            if g.link_name in pads
        ]
        if not self._pads:
            raise ROSConfigError(f"{gripper_joint!r}: no pad carries collision geometry.")
        self.axes = self._closing_axes()
        # The hand never evidences an object: the fingers, the palm link and the wrist.
        mount = next((j for j in description.joints if j.child_link == self.attach_link), None)
        wrist = mount.parent_link if mount is not None else None
        hand = {*self.touch_links, self.attach_link, *([wrist] if wrist else [])}
        # The palm plane: the jaw joint's origin, or the wrist's front face where the wrist
        # reaches past it at any angle of the joint carrying the hand (the SO-101's does, by
        # 17 mm: g20's fallback cube hit it).
        self.palm_m = float(self.palm @ self.axes[:, 1])
        if mount is not None and wrist is not None:
            lo, hi = mount.position_limits or (-math.pi, math.pi)
            for q in np.linspace(lo, hi, _SWEEP_SAMPLES):
                t_aw = np.linalg.inv(
                    link_transform(description, wrist, self.attach_link, {mount.name: float(q)})
                )
                for g in description.collision_geometry:
                    if g.link_name == wrist:
                        t = t_aw @ transform_from_xyz_rpy(
                            g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:]
                        )
                        corners = (_UNIT_BOX * bounding_half_extents(g.shape)) @ t[:3, :3].T
                        front = float(((corners + t[:3, 3]) @ self.axes[:, 1]).max())
                        self.palm_m = max(self.palm_m, front)
        self._hand = [
            (
                g.link_name,
                transform_from_xyz_rpy(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:]),
                bounding_half_extents(g.shape),
            )
            for g in description.collision_geometry
            if g.link_name in hand
        ]

    def _pad_corners(self, q: float) -> NDArray[np.float64]:
        """``(8 * n_primitives, 3)`` pad box corners in the attach link at jaw ``q``."""
        positions = {self.joint.name: q}
        out = []
        for link, local, half in self._pads:
            t = link_transform(self.description, self.attach_link, link, positions) @ local
            out.append((_UNIT_BOX * half) @ t[:3, :3].T + t[:3, 3])
        return np.concatenate(out)

    def _closing_axes(self) -> NDArray[np.float64]:
        """Columns ``(c, a, b)``, fixed per hand at the closed end of the stroke.

        ``c`` is the principal direction of the pads' motion as they close onto each other
        (or onto the fixed jaw): the pinch axis at contact. ``a`` points from the palm to
        the pads' centroid, square to ``c``.
        """
        lo, hi = self.joint.position_limits or (self.closed - 1.0, self.closed + 1.0)
        step = 1e-3 * (hi - lo) * (1.0 if self.closed < hi else -1.0)
        d = self._pad_corners(self.closed + step) - self._pad_corners(self.closed)
        cov = d.T @ d
        if float(np.trace(cov)) < _DEGENERATE:
            raise ROSConfigError(f"{self.joint.name!r}: the pads do not move with the jaw.")
        c = np.linalg.eigh(cov)[1][:, -1]
        reach = self._pad_corners(self.closed).mean(axis=0) - self.palm
        a = reach - (reach @ c) * c
        a /= np.linalg.norm(a)
        return np.column_stack((c, a, np.cross(c, a)))

    def pinch_in_link(self, jaw: float) -> PinchVolume:
        """``J_h`` at the measured ``jaw`` (unpadded), in the attach link."""
        lo, hi = self.joint.position_limits or (min(jaw, self.closed), max(jaw, self.closed))
        jaw = float(np.clip(jaw, lo, hi))
        swept = np.concatenate(
            [self._pad_corners(q) for q in np.linspace(self.closed, jaw, _SWEEP_SAMPLES)]
        )
        local = swept @ self.axes
        lo3, hi3 = local.min(axis=0), local.max(axis=0)
        lo3[1] = max(lo3[1], self.palm_m)  # never behind the palm
        return PinchVolume(
            self.axes @ ((lo3 + hi3) / 2.0), self.axes, (hi3 - lo3) / 2.0, self.palm_m
        )

    @property
    def open_end(self) -> float:
        """The end of the jaw's ``position_limits`` farther from the closed end."""
        lo, hi = self.joint.position_limits or (self.closed, self.closed)
        return lo if abs(lo - self.closed) > abs(hi - self.closed) else hi

    def fallback_in_link(self, jaw: float, opening: float | None = None) -> PinchVolume:
        """The ``PINCH_FALLBACK`` payload (OT-10, bounded), in the attach link.

        Along the closing axis: ``J_h`` at ``opening``, the jaw measured just before
        closure (the held width there is at most that gap); the whole stroke when no
        opening was seen. Along the approach and the pad width: the gripper's full stroke
        (the pad gap at the manifest's open end), from ``J_h``'s centre at ``jaw``. Its
        back face stays on or in front of the palm plane.
        """
        j = self.pinch_in_link(jaw)
        gap = self.pinch_in_link(self.open_end if opening is None else opening)
        stroke_half = float(self.pinch_in_link(self.open_end).half[0])
        jc, gc = j.centre @ self.axes, gap.centre @ self.axes
        lo = np.array([gc[0] - gap.half[0], self.palm_m, jc[2] - max(j.half[2], stroke_half)])
        hi = np.array(
            [
                gc[0] + gap.half[0],
                max(jc[1] + j.half[1], jc[1] + stroke_half),
                jc[2] + max(j.half[2], stroke_half),
            ]
        )
        return PinchVolume(self.axes @ ((lo + hi) / 2.0), self.axes, (hi - lo) / 2.0, self.palm_m)

    def hand_boxes(self, frame: str, positions: Mapping[str, float]) -> list[Box]:
        """The hand's collision primitives as boxes in ``frame`` by FK at ``positions``."""
        out: list[Box] = []
        for link, local, half in self._hand:
            t = link_transform(self.description, frame, link, positions) @ local
            out.append((t[:3, 3], t[:3, :3], half))
        return out

    def t_frame_from_attach(
        self, frame: str, positions: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """``T_frame<-attach_link`` by FK at ``positions`` (``frame`` must be an ancestor)."""
        return link_transform(self.description, frame, self.attach_link, positions)
