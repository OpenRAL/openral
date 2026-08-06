"""``so101_eraser`` scene — SO-101 places an eraser on a strip of blue tape.

The sim twin of the real bench that produced
``rskills/rskill-smolvla-so101-eraser_place-bf16`` (dataset
``makermods/eraser_place_unblurry_real``): an SO-101 follower clamped to a
white desk, a dark rectangular block sitting in front of it, and a small square
of blue tape beside it marking the drop target. The single training task string
is ``"place the erase on the blue square"`` (upstream typo included).

Layout (top view, arm at the world origin facing -Y — the SO-ARM101 MJCF
convention every SO-101 scene in this repo shares):

      +X ←──────── image right is -X ────────→ -X
                        [SO-101]              ↑ +Y (behind the arm)
                       ┌────────┐
                       │        │
             ▪ tape    ▪ eraser              ← eraser 25.8 cm in front of the
             8.4 cm apart                      base, tape 8.4 cm to the arm's
                                               right (= image right)

Every dimension below is MEASURED against the dataset, not eyeballed:

* **Object placement** — forward kinematics on all 25 teleop episodes. The
  frame where the gripper closes is the eraser, the frame where it reopens is
  the tape. TCP means (N=25): grasp ``(+0.024, -0.258)`` std ``(0.014, 0.013)``,
  release ``(-0.060, -0.267)`` std ``(0.024, 0.011)``.
* **Camera** — fitted against the moving arm's silhouette over multiple
  frames (the object-only fit has a depth ambiguity), landing high behind the
  arm at z 0.64 m, fovy 28.8°; the default then slides it 15 cm to the arm's
  right so the follower arm does not occlude the gripper + eraser during the
  grasp (see ``front_camera_pos``). Cross-check on the pure fit: the 8.4 cm
  object spacing and the tape's column/size reproject within a few px.
* **Exposure** — the ``bench`` lighting preset reproduces the frames' own
  histogram (global mean luminance 126/255, desk centre ~135) and both object
  colours to within ~2 counts per channel.
* **Wrist camera** — a second PnP, this time against the wrist video: the tape
  is world-fixed, so with the gripper pose from FK its position in the gripper
  frame sweeps 16 x 17 x 11 cm over the episodes. 49 frames, 26.6 px mean
  reprojection (24.6 px on a held-out half) against 172 px for the manifest's
  modelled placement.

The honest gaps, all visible in a side-by-side:

* The real rig has a SECOND arm — the teleop leader — intruding into the
  bottom-right of every training frame. This scene has only the follower, so
  the sim frame's bottom-right is bare desk.
* The "eraser" is a white plastic eraser in a dark navy paper sleeve — the
  label is legible in the wrist view. The scene models both tones as two box
  geoms, matched on silhouette, size and each tone's mean colour, but NOT on
  texture: the real sleeve carries printed white lettering this does not.
* The home pose is CLAMPED to the SO-101 MJCF joint limits — the operator's
  real home exceeds them on shoulder_lift and elbow_flex (see the rSkill
  manifest's JOINT ENVELOPE note) — which lifts the arm ~1.5 cm higher in the
  frame than the training video shows. Widening the limits to fix this is a
  safety-WG decision, not a scene tweak (CLAUDE.md §3).
* The wrist camera is posed against the dataset's own wrist video at matched
  joint poses so the jaws frame the eraser at grasp exactly as the real rig's
  fingers do (an earlier tape-only PnP fit matched the world objects but left
  the jaws out of frame — see ``wrist_camera_pos_local``). Wrist exposure
  still runs ~15% brighter than the real frames (MuJoCo has no per-camera
  exposure; dimming the shared lights would break the front camera's
  calibrated luminance), and the real frames' strong arm shadow and desk
  specular pool remain softer here.

Cameras are named ``front`` and ``wrist`` — the checkpoint's own input-feature
suffixes — so the eraser rSkill's ``image_preprocessing.aliases`` (which are
keyed by the LIBERO-style ``camera1``/``camera2`` slots) pass through untouched
and the batch lands on ``observation.images.{front,wrist}``.

Everything the composer needs comes from :class:`EraserSceneOptions`, filled
from the YAML ``scene.backend_options`` block, so a different desk / camera /
object size is a pure YAML edit. The robot splice itself (base re-anchor, wrist
camera, fill light, manifest MJCF resolution) is reused verbatim from the
``so101_box`` composer — same so_arm101 anchors, so ``fixed_robot`` stays
``so101_follower``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import look_at_quat_wxyz, yaw_to_quat_wxyz

from openral_sim.backends.so101_box._assets import (
    _reanchor_robot_base,
    _resolve_robot_mjcf,
    _splice_wrist_camera,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import RobotDescription, SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

__all__ = [
    "EraserSceneOptions",
    "compose_so101_eraser_deploy_mjcf",
    "compose_so101_eraser_mjcf",
]

_SO101_ARM_DOF = 6
_DEFAULT_MAX_STEPS = 500
_DEFAULT_RENDER_HEIGHT = 480
_DEFAULT_RENDER_WIDTH = 640
_XYZ_LEN = 3
_INSTRUCTION = "place the erase on the blue square"


class _LightingPreset(NamedTuple):
    """One named lighting condition for the desk.

    Attributes:
        ambient: Headlight ambient level — the flat term that lifts every
            surface, including the ones the lamp does not reach.
        lamp_diffuse: Diffuse intensity of the desk lamp (the key light).
        lamp_specular: Specular intensity of the desk lamp; this is what
            paints the bright pool on the desk finish.
        lamp_pos: World position of the desk lamp, metres.
        fill_diffuse: Diffuse intensity of the soft overhead fill.
        tint: Per-channel multiplier applied to BOTH lights — ``(1, 1, 1)`` is
            neutral, warmer presets push red, cooler ones push blue.
    """

    ambient: float
    lamp_diffuse: float
    lamp_specular: float
    lamp_pos: tuple[float, float, float]
    fill_diffuse: float
    tint: tuple[float, float, float]


# ``bench`` is the calibrated one: it reproduces the dataset's own exposure.
# Measured on the real frames — global mean luminance 126/255 (std 30), desk
# centre 140, brightest at the bottom-left (161) and darkest at the top-right
# (122). That gradient is not vignetting: it is a low desk lamp off the arm's
# left-front, whose specular pool the camera sees in the near-left corner. The
# lamp position is the mirror of the camera about the desk plane through that
# pool, so the highlight lands where the dataset puts it.
#
# The other presets are domain-randomisation handles, NOT claims about any real
# bench: use them to check the policy is not keying on the exposure it was
# trained under.
_LIGHTING_PRESETS: dict[str, _LightingPreset] = {
    "bench": _LightingPreset(0.42, 0.168, 0.52, (0.25, -0.23, 0.30), 0.185, (1.0, 1.0, 1.02)),
    "bright": _LightingPreset(0.60, 0.30, 0.40, (0.25, -0.23, 0.45), 0.30, (1.0, 1.0, 1.0)),
    "dim": _LightingPreset(0.20, 0.10, 0.40, (0.25, -0.23, 0.28), 0.09, (1.0, 0.99, 1.0)),
    "warm": _LightingPreset(0.40, 0.17, 0.52, (0.25, -0.23, 0.30), 0.19, (1.10, 1.0, 0.86)),
    "cool": _LightingPreset(0.40, 0.17, 0.55, (0.25, -0.23, 0.30), 0.19, (0.90, 0.97, 1.14)),
    "overhead": _LightingPreset(0.30, 0.45, 0.10, (0.0, -0.26, 0.95), 0.20, (1.0, 1.0, 1.0)),
    "sidelit": _LightingPreset(0.14, 0.50, 0.55, (-0.45, -0.28, 0.22), 0.08, (1.0, 0.99, 0.97)),
}


@dataclass(frozen=True)
class EraserSceneOptions:
    """Every dimension and pose the ``so101_eraser`` scene exposes.

    Lengths are metres, angles are degrees where suffixed ``_deg``, radians
    otherwise. The arm base sits at the world origin on the desk top (z = 0).

    Attributes:
        desk_size_xy: Desk top dimensions ``(X, Y)``. Big enough that the
            overhead camera sees only desk, matching the real bench's white
            table filling the frame.
        desk_thickness: Half-thickness of the desk slab.
        desk_rgba: Desk top colour. Default is the real bench's white melamine.
        robot_base_xyz: World position of the SO-101 ``base`` body.
        robot_base_yaw_deg: Yaw applied to the base body. ``0`` keeps the arm
            facing -Y (the upstream MJCF convention).
        eraser_offset_y: Distance IN FRONT of the base (toward -Y) at which the
            eraser spawns. Default 0.258 m — the FK grasp mean over the 25
            teleop episodes.
        eraser_offset_x: Lateral offset of the eraser from the base centreline
            (+X is the arm's left / image left). Default 0.024 m, same source.
        tape_offset_x: Offset of the tape square from the eraser, measured
            along the arm's RIGHT (= image right = world -X). Default 0.084 m
            — the FK release mean, independently confirmed by the frames (the
            two objects are 129.3 px apart at 1490 px/m).
        tape_offset_y: Offset of the tape square from the eraser along ±Y.
            Zero: the frames put both objects on the same image row.
        eraser_size: Eraser half-extents ``(X, Y, Z)`` — a 42 × 17 × 10 mm
            block at the default. Sized so the RENDERED object matches the
            frames' own principal extents (4.0 × 2.2 cm against a real
            4.2 × 2.2 cm): the apparent width includes the visible side face,
            so a taller box overshoots even at the right footprint.
        eraser_rgba: Colour of the ERASER BODY — the white rubber, exposed at
            the un-sleeved end. Brighter than the desk in the real frames
            (RGB 143/146/149 against a desk reading 132).
        eraser_sleeve_rgba: Colour of the paper SLEEVE that wraps the rest of
            the block. Dark navy: this is what the overview camera mostly sees
            (RGB 54/55/65 on a desk reading 140) and why the object reads as a
            dark blob from above despite being a white eraser.
        eraser_sleeve_fraction: Fraction of the block's length the sleeve
            covers, measured from the +X end. ``0`` removes the sleeve and
            leaves a plain white block.
        eraser_mass: Eraser mass, kg.
        eraser_yaw_deg: Spawn yaw of the block, degrees. Default -135° matches
            the frames: long axis running lower-left to upper-right in the
            overview image, with the exposed white end toward the near left.
        jaw_rgba: Colour for the two gripper jaw meshes, or ``None`` to keep
            the upstream off-white. Defaults to near-black: the real rig prints
            its jaws in black filament, and they occupy the bottom third of
            every wrist frame.
        tape_size: Blue tape half-extents ``(X, Y, Z)``. The tape is a STATIC
            geom welded to the desk: real tape does not slide.
        tape_rgba: Tape colour, matched to the frames (RGB 26/55/103).
        front_camera_pos: World position of the fixed overview camera, mounted
            above and slightly behind the arm looking down and forward, so the
            arm hardware sits at the bottom edge of the frame.
        front_camera_target: Look-at point of the front camera.
        front_camera_fovy: Vertical FoV of the front camera, degrees.
        wrist_camera_pos_local: Wrist-camera position in the ``gripper`` body
            frame. Defaults match ``robots/so101_follower`` ``sim_placement``.
        wrist_camera_target_local: Wrist-camera look-at in the same frame.
        wrist_camera_up_local: Wrist-camera up vector (the phone rig is
            mounted inverted).
        wrist_camera_fovy: Wrist-camera vertical FoV, degrees.
        home_qpos_rad: Arm home pose (5 joints, radians) + gripper as a
            NORMALISED ``[0, 1]`` opening in the last slot — the mixed-units
            joint-channel contract the SO-101 manifest and the eraser rSkill's
            ``starting_pose`` both use. Default is that rSkill's
            ``starting_pose`` (the teleop operator's real home).
        spawn_jitter_m: Uniform ± jitter applied to the eraser spawn XY on
            ``reset``. Defaults to 0.013 m — the per-episode spread the
            operator actually produced. Set to 0 for a fixed layout.
        spawn_yaw_jitter_deg: Uniform ± yaw jitter on the eraser spawn.
        place_xy_tol_m: Success tolerance — how far the eraser centre may sit
            from the tape centre in XY and still count as placed.
        joint_units: ``"degrees"`` (default) makes the env emit proprio and
            accept actions in the LeRobot servo-degree convention the eraser
            checkpoint was trained in; ``"radians"`` keeps MuJoCo-native units.
        joint_offsets_deg: Per-joint calibration offset, degrees mode only:
            ``lerobot_deg = joint_signs * mujoco_deg + joint_offsets_deg``.
            IDENTITY by default — and identity is CORRECT for this checkpoint:
            its rSkill manifest records the operator home as
            ``[-7.21, -102.68, 95.12, 56.70, 6.15]`` LeRobot degrees and
            declares ``starting_pose`` as the plain ``radians()`` of those same
            numbers, i.e. this dataset was recorded in the ``so101_new_calib``
            zero the MJCF already uses.
        joint_signs: Per-joint sign, degrees mode only. Entries are ±1.
        lighting: Name of a :data:`_LIGHTING_PRESETS` entry. ``"bench"`` is
            calibrated against the dataset frames; the rest (``bright``,
            ``dim``, ``warm``, ``cool``, ``overhead``, ``sidelit``) are
            domain-randomisation handles, not other real benches.
        lighting_scale: Multiplier on every intensity in the preset — a
            continuous exposure knob on top of the named condition.
        lighting_jitter: Per-reset randomisation of lamp intensity (±fraction)
            and lamp XY (±0.2 m at 1.0). ``0`` (default) keeps ``bench``
            bit-identical across resets.
        extra_metadata: Free-form strings forwarded to the scene metadata.
    """

    desk_size_xy: tuple[float, float] = (2.40, 2.00)
    desk_thickness: float = 0.02
    desk_rgba: tuple[float, float, float, float] = (0.88, 0.88, 0.89, 1.0)

    robot_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base_yaw_deg: float = 0.0

    # MEASURED, not guessed. Forward kinematics on all 25 teleop episodes: the
    # frame where the gripper closes is the eraser, the frame where it reopens
    # is the tape. TCP (`gripper` site) means over N=25:
    #     grasp    ( +0.024, -0.258)  std (0.014, 0.013)
    #     release  ( -0.060, -0.267)  std (0.024, 0.011)
    # So the eraser sits 25.8 cm in front of the base and 2.4 cm to the arm's
    # LEFT, and the tape is 8.4 cm to the arm's RIGHT of it (NOT 5 cm) at the
    # same depth. `spawn_jitter_m` defaults to the observed grasp spread.
    eraser_offset_y: float = 0.258
    eraser_offset_x: float = 0.024
    tape_offset_x: float = 0.084
    tape_offset_y: float = 0.0

    # Sized + coloured off the dataset's own frames. At the fitted camera the
    # image scale is 15.0 px/cm, which puts the block at ~4.4 × 2.6 cm and the
    # tape at ~4.2 × 3.0 cm.
    #
    # TWO-TONE, and the split matters: the wrist view resolves the label well
    # enough to read "PLASTIC ERASER" — it is a WHITE rubber eraser in a dark
    # navy paper sleeve covering roughly the far two thirds. From overhead the
    # sleeve dominates, which is why the object reads as a dark blob there
    # (RGB 54/55/65 on a desk reading 140) while the exposed rubber reads
    # BRIGHTER than the desk in the wrist view (143/146/149 vs 132).
    eraser_size: tuple[float, float, float] = (0.021, 0.0085, 0.005)
    eraser_rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.96, 1.0)
    eraser_sleeve_rgba: tuple[float, float, float, float] = (0.35, 0.355, 0.41, 1.0)
    eraser_sleeve_fraction: float = 0.68
    eraser_mass: float = 0.015
    # Long axis points away from the camera, canted ~15° off the depth axis.
    eraser_yaw_deg: float = -135.0

    # The real rig prints its two jaw fingers in BLACK filament while the
    # upstream MJCF paints the whole arm off-white. The jaws fill the bottom
    # third of every wrist frame, so this is the single biggest appearance gap
    # on the camera the policy uses for the grasp. ``None`` leaves the upstream
    # colours alone.
    jaw_rgba: tuple[float, float, float, float] | None = (0.11, 0.11, 0.12, 1.0)

    tape_size: tuple[float, float, float] = (0.021, 0.015, 0.0004)
    tape_rgba: tuple[float, float, float, float] = (0.17, 0.355, 0.65, 1.0)

    # FITTED to the dataset frames, not eyeballed — and then deliberately
    # shifted. The object-cluster-only PnP has a depth/height ambiguity (it
    # first landed the camera 28 cm above the desk at fovy 53.1°, visibly too
    # close); refitting against the MOVING ARM's silhouette across multiple
    # frames broke the ambiguity and put the camera at (-0.063, 0.138, 0.638),
    # fovy 28.8° — high behind the arm, matching the real overview shot's
    # composition. The default below keeps that height/target/fovy but slides
    # the camera to x = -0.210 (the arm's right): at the dataset x the follower
    # arm occludes the gripper + eraser during the grasp, which the real bench's
    # viewpoint does not; a lateral sweep showed -0.21 m clears the occlusion
    # while keeping the eraser, tape and arm in frame, and going farther right
    # adds nothing. Set front_camera_pos x back to -0.063 for the strict
    # dataset-composition fit.
    front_camera_pos: tuple[float, float, float] = (-0.210, 0.138, 0.638)
    front_camera_target: tuple[float, float, float] = (-0.080, -0.304, 0.0)
    front_camera_fovy: float = 28.8

    # POSED against the dataset's own wrist video at matched joint poses. An
    # earlier tape-only PnP fit (26.6 px reprojection over 49 frames) matched
    # the WORLD objects but put the jaws almost entirely out of frame — the
    # real rig's two black fingers rise from the bottom edge, fill the bottom
    # third, and frame the eraser between their tips at grasp, which is the
    # dominant close-range cue the checkpoint's grasp phase saw in training.
    # The tape sits ~60-70 deg off the jaw axis, so that fit was insensitive
    # to exactly the translation the jaw framing depends on. This pose is
    # constructed in the gripper frame instead — behind/above the jaw base
    # (jaw axis = local -z, tips at z = -0.104), aimed past the tips, rolled
    # so the fingers enter bottom-center — and validated side-by-side against
    # real home/approach/grasp frames of episode 0: jaw tips ~55-70%% height,
    # eraser above the tips on approach, tape entering at the left edge, all
    # matching the real composition. The desk-object reprojection stays
    # consistent with the real frames at those poses (checked visually, not
    # re-fit numerically).
    wrist_camera_pos_local: tuple[float, float, float] = (0.0, 0.05, -0.02)
    wrist_camera_target_local: tuple[float, float, float] = (0.0, -0.04, -0.31)
    wrist_camera_up_local: tuple[float, float, float] = (0.0, 0.9551, -0.2964)
    wrist_camera_fovy: float = 52.0

    home_qpos_rad: tuple[float, ...] = (-0.1258, -1.7453, 1.5708, 0.9897, 0.1074, 0.0195)

    spawn_jitter_m: float = 0.013
    spawn_yaw_jitter_deg: float = 15.0
    place_xy_tol_m: float = 0.02

    lighting: str = "bench"
    lighting_scale: float = 1.0
    lighting_jitter: float = 0.0

    joint_units: str = "degrees"
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    extra_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def eraser_xy(self) -> tuple[float, float]:
        """World XY of the eraser spawn — ``eraser_offset_y`` in front of the base."""
        return (
            self.robot_base_xyz[0] + self.eraser_offset_x,
            self.robot_base_xyz[1] - self.eraser_offset_y,
        )

    @property
    def tape_xy(self) -> tuple[float, float]:
        """World XY of the tape square — offset to the arm's RIGHT (world -X)."""
        ex, ey = self.eraser_xy
        return (ex - self.tape_offset_x, ey + self.tape_offset_y)

    @property
    def lighting_preset(self) -> _LightingPreset:
        """The resolved lighting condition, with ``lighting_scale`` applied.

        Raises:
            ROSConfigError: If ``lighting`` names no known preset.
        """
        try:
            preset = _LIGHTING_PRESETS[self.lighting]
        except KeyError:
            raise ROSConfigError(
                f"so101_eraser: unknown lighting preset {self.lighting!r}; "
                f"valid presets: {sorted(_LIGHTING_PRESETS)!r}.",
            ) from None
        k = max(0.0, self.lighting_scale)
        return preset._replace(
            ambient=preset.ambient * k,
            lamp_diffuse=preset.lamp_diffuse * k,
            lamp_specular=preset.lamp_specular * k,
            fill_diffuse=preset.fill_diffuse * k,
        )


def _rgba(c: tuple[float, float, float, float]) -> str:
    return " ".join(str(v) for v in c)


def _render_desk(opts: EraserSceneOptions) -> str:
    """Desk slab + the two-light rig, as a worldbody snippet.

    The desk top is flush with z = 0 so the arm base — which the composer
    anchors at ``robot_base_xyz`` — rests on it. The desk carries a specular
    term because the real bench's finish does: the lamp's reflection in it is
    the single most prominent feature of the training frames after the objects.

    Light names are stable (``desk_lamp`` / ``desk_fill``) because the rollout
    re-writes their intensities on ``reset`` when ``lighting_jitter`` is on.
    """
    dx, dy = opts.desk_size_xy
    t = opts.desk_thickness
    bx, by, _ = opts.robot_base_xyz
    light = opts.lighting_preset
    tint = light.tint

    def _scaled(level: float) -> str:
        return " ".join(f"{level * c:.4f}" for c in tint)

    desk = (
        f'<geom name="desk" type="box" pos="{bx} {by - dy / 4.0} {-t}" '
        f'size="{dx / 2.0} {dy / 2.0} {t}" rgba="{_rgba(opts.desk_rgba)}" '
        f'friction="1 0.01 0.001"/>'
    )
    lx, ly, lz = light.lamp_pos
    lamp = (
        f'<light name="desk_lamp" pos="{bx + lx} {by + ly} {lz}" '
        f'dir="{-lx} {-ly} {-lz}" diffuse="{_scaled(light.lamp_diffuse)}" '
        f'specular="{_scaled(light.lamp_specular)}" castshadow="true"/>'
    )
    # The fill sits high and out to the arm's LEFT, not straight overhead: the
    # real frames are brightest along the left edge at every depth, which a
    # centred fill cannot reproduce (it leaves the far corners too dark).
    fill = (
        f'<light name="desk_fill" pos="{bx + 0.32} {by - 0.42} 0.95" '
        f'dir="-0.30 0.18 -1" diffuse="{_scaled(light.fill_diffuse)}" '
        'specular="0 0 0" castshadow="false"/>'
    )
    return "\n        ".join([desk, lamp, fill])


def _inject_visual_headlight(xml: str, opts: EraserSceneOptions) -> str:
    """Replace MuJoCo's default headlight with the preset's ambient term.

    MuJoCo's built-in headlight (diffuse 0.4, specular 0.5, camera-aligned)
    stacks with this scene's desk lights and clips every upward-facing surface
    to white — which erases both the desk's exposure gradient and the eraser's
    top face, the object the policy has to find. Replacing it with a pure
    ambient term is what lets the preset control the exposure.

    ``so101_box``'s equivalent helper is not reusable here: it bails out when
    the string ``"<visual"`` appears anywhere in the file, and the upstream
    SO-101 MJCF contains ``<default class="visual">``, so the injection is
    silently skipped and the default headlight survives.
    """
    if re.search(r"<visual[\s>]", xml):
        return xml
    light = opts.lighting_preset
    ambient = " ".join(f"{light.ambient * c:.4f}" for c in light.tint)
    # The skybox is not decoration: the desk is a finite plane, so any camera
    # looking at or above the horizon (the wrist camera does, at fovy 90) sees
    # pure black past its edge. A pale gradient stands in for the room the real
    # bench's wrist camera actually sees; it tracks the ambient level so a dim
    # preset does not leave a brightly lit wall behind a dark desk.
    sky = 0.55 + 0.9 * light.ambient
    headlight = (
        "\n  <visual>\n"
        f'    <headlight ambient="{ambient}" diffuse="0 0 0" specular="0 0 0"/>\n'
        '    <quality shadowsize="4096"/>\n'
        "  </visual>\n"
        "  <asset>\n"
        '    <texture name="desk_skybox" type="skybox" builtin="gradient" '
        f'rgb1="{sky:.3f} {sky:.3f} {sky + 0.02:.3f}" '
        f'rgb2="{sky * 0.8:.3f} {sky * 0.8:.3f} {sky * 0.82:.3f}" '
        'width="256" height="256"/>\n'
        "  </asset>"
    )
    body, n = re.subn(r"(<mujoco\b[^>]*>)", r"\1" + headlight, xml, count=1)
    if n != 1:
        raise ROSConfigError(
            "so101_eraser: cannot find the opening <mujoco> tag to inject scene "
            "lighting — upstream MJCF structure changed.",
        )
    return body


# The two materials the SO-101's jaw fingers use: the moving jaw's own mesh and
# the wrist-roll follower, which carries the static jaw.
_JAW_MATERIALS = ("moving_jaw_so101_v1_material", "wrist_roll_follower_so101_v1_material")


def _recolour_jaws(xml: str, rgba: tuple[float, float, float, float] | None) -> str:
    """Repaint the jaw materials, matching the real rig's black-printed fingers.

    Rewrites the ``rgba`` attribute on the two ``<material>`` declarations
    rather than the geoms, so both the visual and collision copies of each jaw
    mesh follow. A no-op when ``rgba`` is ``None``.

    Raises:
        ROSConfigError: If a jaw material is missing — an upstream rename would
            otherwise silently leave the jaws the wrong colour.
    """
    if rgba is None:
        return xml
    for name in _JAW_MATERIALS:
        xml, n = re.subn(
            rf'(<material\s+name="{name}"[^>]*\brgba=")[^"]+(")',
            rf"\g<1>{_rgba(rgba)}\g<2>",
            xml,
            count=1,
        )
        if n != 1:
            raise ROSConfigError(
                f"so101_eraser: material {name!r} not found in the upstream SO-101 MJCF, "
                "so the jaws cannot be recoloured. Set backend_options.jaw_rgba to null "
                "to keep the upstream colours, or update _JAW_MATERIALS.",
            )
    return xml


def _render_front_camera(opts: EraserSceneOptions) -> str:
    """The fixed overview camera, emitted under BOTH names it is known by.

    ``front`` is the checkpoint's own input-feature suffix, which is what the
    sim tier's ``scene.cameras`` asks for. ``top`` is the name
    ``robots/so101_follower`` gives the same physical camera, and it is what
    ``deploy sim`` renders: the HAL's camera rig splices a camera for every
    manifest sensor whose name is MISSING from the MJCF, so without this alias
    a deploy run would quietly ignore the fitted pose and rig the manifest's
    nominal straight-down one instead. Same pose, same fovy, two names — the
    renderer only ever evaluates the one it is asked for.
    """
    pos = opts.front_camera_pos
    quat = look_at_quat_wxyz(pos, opts.front_camera_target)
    common = (
        f'pos="{pos[0]} {pos[1]} {pos[2]}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
        f'fovy="{opts.front_camera_fovy}" mode="fixed"/>'
    )
    return f'<camera name="front" {common}\n        <camera name="top" {common}'


def _render_eraser(opts: EraserSceneOptions) -> str:
    """The eraser: a white block wearing a navy paper sleeve.

    Two geoms in one free body. The white block is the whole object and the
    only COLLIDER; the sleeve is a visual-only shell (``contype``/``conaffinity``
    zero) covering ``eraser_sleeve_fraction`` of the length from the +X end and
    standing 0.6 mm proud so it neither z-fights with the block underneath
    nor casts shadow acne onto it.
    Contacts and inertia therefore stay exactly those of the plain block — the
    sleeve is appearance only, which is the point: it is what the overview
    camera sees.

    The spawn pose is baked into the BODY element, not just written by
    ``reset``. The sim tier does call ``reset`` (and re-rolls the jitter there),
    but ``openral deploy sim`` has no rollout at all: it hands this MJCF to a
    bare ``MujocoArmHAL`` and steps it. A body left at the MJCF default would
    spawn its freejoint at the world origin — inside the robot base — and
    physics would eject the eraser off the desk before the first frame, leaving
    the deploy twin with a tape square and nothing to pick up. (Observed
    exactly that on the first live deploy run.)

    Raises:
        ROSConfigError: If the sleeve fraction is not in ``[0, 1]``.
    """
    sx, sy, sz = opts.eraser_size
    m = opts.eraser_mass
    f = opts.eraser_sleeve_fraction
    if not 0.0 <= f <= 1.0:
        raise ROSConfigError(
            f"so101_eraser: eraser_sleeve_fraction must be in [0, 1]; got {f}.",
        )
    ex, ey = opts.eraser_xy
    quat = yaw_to_quat_wxyz(math.radians(opts.eraser_yaw_deg))
    body = (
        f'<body name="eraser" pos="{ex} {ey} {sz}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">\n'
        '          <freejoint name="eraser_joint"/>\n'
        f'          <inertial pos="0 0 0" mass="{m}" '
        f'diaginertia="{m * ((2 * sy) ** 2 + (2 * sz) ** 2) / 12.0} '
        f"{m * ((2 * sx) ** 2 + (2 * sz) ** 2) / 12.0} "
        f'{m * ((2 * sx) ** 2 + (2 * sy) ** 2) / 12.0}"/>\n'
        f'          <geom name="eraser_geom" type="box" size="{sx} {sy} {sz}" '
        f'rgba="{_rgba(opts.eraser_rgba)}" friction="1.2 0.02 0.002"/>\n'
    )
    if f > 0.0:
        # The shell stands proud on ALL FIVE covered faces, the +X cap
        # included: leaving that one cap coplanar with the block's own +X face
        # z-fights, and the near end of the object is exactly what the overview
        # camera looks at.
        proud = 0.0006
        x_start = sx - 2.0 * sx * f  # the sleeve's open edge, where white shows
        x_end = sx + proud  # past the block's own cap
        body += (
            f'          <geom name="eraser_sleeve" type="box" '
            f'pos="{(x_end + x_start) / 2.0} 0 0" '
            f'size="{(x_end - x_start) / 2.0} {sy + proud} {sz + proud}" '
            f'rgba="{_rgba(opts.eraser_sleeve_rgba)}" '
            'contype="0" conaffinity="0"/>\n'
        )
    return body + "        </body>"


def _render_tape(opts: EraserSceneOptions) -> str:
    """The blue tape square: a static geom + a site marking the drop target."""
    sx, sy, sz = opts.tape_size
    tx, ty = opts.tape_xy
    geom = (
        f'<geom name="tape" type="box" pos="{tx} {ty} {sz}" size="{sx} {sy} {sz}" '
        f'rgba="{_rgba(opts.tape_rgba)}" contype="0" conaffinity="0"/>'
    )
    site = (
        f'<site name="tape_center" type="sphere" pos="{tx} {ty} {2 * sz}" '
        'size="0.002" rgba="1 0 0 0.05" group="3"/>'
    )
    return "\n        ".join([geom, site])


def compose_so101_eraser_mjcf(
    options: EraserSceneOptions | None = None,
    robot_description: RobotDescription | None = None,
) -> tuple[str, Path]:
    """Return ``(composed_xml, output_mjcf_path)`` for the so101_eraser scene.

    The output is written next to the upstream SO-101 MJCF so its
    ``meshdir="assets"`` resolves at compile time without copying meshes.

    Args:
        options: Scene options; ``None`` uses every :class:`EraserSceneOptions`
            default.
        robot_description: Robot whose manifest ``assets.mjcf`` provides the
            base arm MJCF. Must share the so_arm101 splice anchors.

    Returns:
        ``(xml, output_path)`` — the composed MJCF, also written to disk.

    Raises:
        ROSConfigError: If a splice anchor (``<body name="base">``,
            ``<body name="gripper">``, ``</worldbody>``) is missing from the
            upstream MJCF.
    """
    opts = options if options is not None else EraserSceneOptions()

    if robot_description is None:
        from robot_descriptions import so_arm101_mj_description

        upstream_path = Path(so_arm101_mj_description.MJCF_PATH)
    else:
        upstream_path = _resolve_robot_mjcf(robot_description)

    body = _inject_visual_headlight(upstream_path.read_text(), opts)
    body = _recolour_jaws(body, opts.jaw_rgba)
    body = _reanchor_robot_base(body, opts.robot_base_xyz, opts.robot_base_yaw_deg)
    body = _splice_wrist_camera(
        body,
        opts.wrist_camera_pos_local,
        opts.wrist_camera_target_local,
        opts.wrist_camera_up_local,
        opts.wrist_camera_fovy,
    )

    snippet = (
        f"        {_render_desk(opts)}\n"
        f"        {_render_front_camera(opts)}\n"
        f"        {_render_tape(opts)}\n"
        f"        {_render_eraser(opts)}\n"
    )
    body, n = re.subn(r"(</worldbody>)", f"{snippet}      \\1", body, count=1)
    if n != 1:
        raise ROSConfigError(
            "so101_eraser: cannot find </worldbody> in the upstream SO-101 MJCF — "
            "composer cannot splice the desk scene in.",
        )

    output_path = upstream_path.parent / "so101_eraser_generated.xml"
    output_path.write_text(body)
    return body, output_path


def compose_so101_eraser_deploy_mjcf(**overrides: Any) -> tuple[str, Path]:
    """``DeployScene.composition`` entry point for ``openral deploy sim``.

    The SO-101 is a ``bare_twin_sim`` robot: ``deploy sim`` does NOT attach to
    the sim-tier rollout, it builds a MuJoCo twin straight off the robot
    manifest's MJCF. Declaring this composer in a deploy scene is what puts the
    desk, the eraser and the tape into that twin — otherwise the stack boots
    against a bare arm on an empty plane and the policy sees nothing.

    Differs from :func:`compose_so101_eraser_mjcf` only in its return shape:
    the HAL node contract is ``(xml, meshdir)`` and it writes the XML to
    ``meshdir.parent``, so the mesh directory is returned rather than the file.

    Args:
        **overrides: ``scene.backend_options``-style keys, validated through the
            same parser the sim tier uses, so a deploy scene and a sim scene
            cannot drift in how they spell an option.

    Returns:
        ``(xml, meshdir)`` — the composed MJCF and the SO-101 mesh directory.
    """
    opts = _options_from_backend_options(dict(overrides))
    xml, path = compose_so101_eraser_mjcf(opts)
    return xml, path.parent / "assets"


def _options_from_backend_options(raw: dict[str, Any] | None) -> EraserSceneOptions:
    """Build :class:`EraserSceneOptions` from ``scene.backend_options``.

    Unknown keys are rejected loudly so YAML typos surface immediately.
    """
    raw = dict(raw or {})
    valid = {f.name for f in EraserSceneOptions.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ROSConfigError(
            f"so101_eraser: unknown scene.backend_options keys {sorted(unknown)!r}; "
            f"valid keys: {sorted(valid)!r}.",
        )

    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "extra_metadata":
            if not isinstance(value, dict):
                raise ROSConfigError(
                    f"so101_eraser: scene.backend_options.extra_metadata must be a dict; "
                    f"got {type(value).__name__}",
                )
            parsed[key] = {str(k): str(v) for k, v in value.items()}
        elif key == "joint_units":
            units = str(value).lower()
            if units not in ("radians", "degrees"):
                raise ROSConfigError(
                    f"so101_eraser: scene.backend_options.joint_units must be "
                    f"'radians' or 'degrees'; got {value!r}",
                )
            parsed[key] = units
        elif key == "lighting":
            name = str(value).lower()
            if name not in _LIGHTING_PRESETS:
                raise ROSConfigError(
                    f"so101_eraser: unknown lighting preset {value!r}; "
                    f"valid presets: {sorted(_LIGHTING_PRESETS)!r}.",
                )
            parsed[key] = name
        elif value is None:
            # Only the optional overrides accept null; everything else is a
            # number or a vector and a null there is a YAML mistake.
            if key != "jaw_rgba":
                raise ROSConfigError(
                    f"so101_eraser: scene.backend_options.{key} cannot be null.",
                )
            parsed[key] = None
        elif isinstance(value, (list, tuple)):
            parsed[key] = tuple(float(v) for v in value)
        else:
            parsed[key] = float(value)

    for key in ("home_qpos_rad", "joint_offsets_deg", "joint_signs"):
        vec = parsed.get(key)
        if vec is not None and len(vec) != _SO101_ARM_DOF:
            raise ROSConfigError(
                f"so101_eraser: scene.backend_options.{key} must be a "
                f"{_SO101_ARM_DOF}-vector (one per joint); got {vec!r}",
            )
    if any(s not in (1.0, -1.0) for s in parsed.get("joint_signs", ())):
        raise ROSConfigError(
            f"so101_eraser: scene.backend_options.joint_signs entries must be +1 or -1; "
            f"got {parsed['joint_signs']!r}",
        )
    for key in ("robot_base_xyz", "front_camera_pos", "front_camera_target"):
        vec = parsed.get(key)
        if vec is not None and len(vec) != _XYZ_LEN:
            raise ROSConfigError(
                f"so101_eraser: scene.backend_options.{key} must be a 3-vector; got {vec!r}",
            )
    return EraserSceneOptions(**parsed)


@dataclass
class _So101EraserRollout:
    """Rollout for the so101_eraser scene."""

    scene: SceneSpec
    task: TaskSpec
    options: EraserSceneOptions
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _arm_actuator_ids: list[int]
    _arm_qpos_addrs: list[int]
    # Action clip bounds (radians). The upstream <position> actuators declare
    # no ctrlrange, so actuator_ctrlrange is the [0, 0] "unlimited" sentinel and
    # clipping to it would pin every command to zero — same trap as so101_box.
    _arm_joint_ranges: NDArray[np.float64]
    _eraser_qpos_addr: int
    _eraser_body_id: int
    _tape_site_id: int
    # Lighting-jitter state: the lamp's light id and the as-composed intensity /
    # position of every light, so each reset randomises around the preset
    # rather than compounding the previous episode's roll.
    _lamp_light_id: int
    _light_baselines: dict[int, dict[str, NDArray[np.float64]]]
    _instruction: str
    _max_steps: int
    _render_height: int
    _render_width: int
    _joint_units: str = "degrees"
    _joint_offsets_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(_SO101_ARM_DOF, dtype=np.float64)
    )
    _joint_signs: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(_SO101_ARM_DOF, dtype=np.float64)
    )
    _settle_steps: int = 50
    _step_count: int = 0
    _renderer_rgb: Any = None  # mujoco.Renderer — lazy
    _last_front_rgb: NDArray[np.uint8] | None = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ---------------------------------------------------------------- SimRollout

    def reset(self, seed: int | None = None) -> Observation:
        import mujoco

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        mujoco.mj_resetData(self._model, self._data)
        home = self._home_qpos_rad()
        for qa, aid, q in zip(self._arm_qpos_addrs, self._arm_actuator_ids, home, strict=True):
            self._data.qpos[qa] = q
            self._data.ctrl[aid] = q

        jitter = self.options.spawn_jitter_m
        ex, ey = self.options.eraser_xy
        if jitter:
            ex += float(self._rng.uniform(-jitter, jitter))
            ey += float(self._rng.uniform(-jitter, jitter))
        yaw_j = np.radians(self.options.spawn_yaw_jitter_deg)
        yaw = np.radians(self.options.eraser_yaw_deg)
        if yaw_j:
            yaw += float(self._rng.uniform(-yaw_j, yaw_j))
        self._write_freejoint(
            qpos_addr=self._eraser_qpos_addr,
            xyz=(ex, ey, self.options.eraser_size[2]),
            yaw=float(yaw),
        )
        self._apply_lighting_jitter()

        mujoco.mj_forward(self._model, self._data)
        for _ in range(self._settle_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self._observation()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        import mujoco

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (_SO101_ARM_DOF,):
            raise ROSConfigError(
                f"so101_eraser expects a {_SO101_ARM_DOF}-D joint-position action; "
                f"got shape {a.shape}",
            )
        if self._joint_units == "degrees":
            cmd: NDArray[np.float64] = np.radians(self._joint_signs * (a - self._joint_offsets_deg))
            # The gripper channel is NOT degrees: lerobot SO-ARM datasets store
            # it normalised [0, 100] over the jaw's calibrated travel. Mapping
            # it through radians() like the arm joints lands ~10° open at the
            # closed end (MJCF jaw range is [-10°, +100°]) — exactly the
            # grasp-critical regime. Map the fraction onto the jaw range, the
            # same [0, 1]-style surface the deploy HAL exposes.
            g_lo, g_hi = self._arm_joint_ranges[-1]
            cmd[-1] = g_lo + np.clip(a[-1] / 100.0, 0.0, 1.0) * (g_hi - g_lo)
        else:
            cmd = a
        lo = self._arm_joint_ranges[:, 0]
        hi = self._arm_joint_ranges[:, 1]
        self._data.ctrl[self._arm_actuator_ids] = np.clip(cmd, lo, hi)

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        success = self._check_placed()
        info: dict[str, Any] = {}
        if self.task.success_key is not None:
            info[self.task.success_key] = success
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=bool(success),
            truncated=self._step_count >= self._max_steps,
            info=info,
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self._last_front_rgb is None:
            return None
        return self._last_front_rgb.copy()

    def close(self) -> None:
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = None

    def mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Expose the model + data to ``openral sim run --view``."""
        return self._model, self._data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (monotonic within an episode)."""
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts — SO-101 = 6 joint targets."""
        return _SO101_ARM_DOF

    # ------------------------------------------------------------- observation

    def _observation(self) -> Observation:
        front = self._render_named_rgb("front")
        wrist = self._render_named_rgb("wrist")
        self._last_front_rgb = front
        # Keys are the bare MuJoCo camera names, matching ``scene.cameras`` —
        # ``front``/``wrist`` are already the checkpoint's own feature suffixes,
        # so the rSkill's camera1/camera2 aliases pass through untouched.
        return {
            "images": {"front": front, "wrist": wrist},
            "state": self._proprio_state(),
            "task": self._instruction,
        }

    def _proprio_state(self) -> NDArray[np.float32]:
        """6-D joint proprioception in the policy's expected units."""
        qpos = np.asarray(
            [float(self._data.qpos[a]) for a in self._arm_qpos_addrs],
            dtype=np.float64,
        )
        if self._joint_units == "degrees":
            lerobot_deg: NDArray[np.float64] = (
                self._joint_signs * np.degrees(qpos) + self._joint_offsets_deg
            )
            # Inverse of step()'s gripper mapping: jaw angle → lerobot [0, 100].
            # degrees(qpos) would report the manifest home (lerobot 1.95) as
            # -7.9 — below the normalizer's observed minimum.
            g_lo, g_hi = self._arm_joint_ranges[-1]
            lerobot_deg[-1] = (qpos[-1] - g_lo) / (g_hi - g_lo) * 100.0
            return np.asarray(lerobot_deg, dtype=np.float32)
        return np.asarray(qpos, dtype=np.float32)

    def _render_named_rgb(self, camera_name: str) -> NDArray[np.uint8]:
        import mujoco

        if self._renderer_rgb is None:
            self._renderer_rgb = mujoco.Renderer(
                self._model,
                height=self._render_height,
                width=self._render_width,
            )
        self._renderer_rgb.update_scene(self._data, camera=camera_name)
        return np.asarray(self._renderer_rgb.render(), dtype=np.uint8).copy()

    # ------------------------------------------------------------------ poses

    def _home_qpos_rad(self) -> list[float]:
        """Home pose in MuJoCo radians, one entry per arm joint.

        The manifest's mixed-units joint contract: the 5 arm channels are
        radians, the gripper channel is a normalised ``[0, 1]`` opening. The
        gripper is mapped onto its MJCF joint range here.
        """
        home = list(self.options.home_qpos_rad)
        lo, hi = self._arm_joint_ranges[-1]
        home[-1] = float(lo + np.clip(home[-1], 0.0, 1.0) * (hi - lo))
        return [
            float(np.clip(q, self._arm_joint_ranges[i, 0], self._arm_joint_ranges[i, 1]))
            for i, q in enumerate(home)
        ]

    def _apply_lighting_jitter(self) -> None:
        """Re-roll the lamp intensity + position for this episode.

        Domain randomisation around whichever preset the scene selected: a
        no-op at ``lighting_jitter == 0`` (the default), so the calibrated
        ``bench`` exposure stays bit-identical unless randomisation is asked
        for. Lights are mutated on the MODEL — MuJoCo reads ``light_diffuse`` /
        ``light_pos`` per frame, so no recompile is needed.
        """
        j = self.options.lighting_jitter
        if not j or self._lamp_light_id < 0:
            return
        gain = float(self._rng.uniform(max(0.0, 1.0 - j), 1.0 + j))
        for lid, base in self._light_baselines.items():
            self._model.light_diffuse[lid] = base["diffuse"] * gain
            self._model.light_specular[lid] = base["specular"] * gain
        # Slide the lamp around the desk so the specular pool is not always in
        # the same corner; ±20 cm at full jitter.
        base_pos = self._light_baselines[self._lamp_light_id]["pos"]
        offset = self._rng.uniform(-0.2 * j, 0.2 * j, size=2)
        self._model.light_pos[self._lamp_light_id, :2] = base_pos[:2] + offset

    def _write_freejoint(
        self,
        *,
        qpos_addr: int,
        xyz: tuple[float, float, float],
        yaw: float,
    ) -> None:
        """Write a 7-D freejoint slot (position + pure-yaw quaternion)."""
        self._data.qpos[qpos_addr + 0] = xyz[0]
        self._data.qpos[qpos_addr + 1] = xyz[1]
        self._data.qpos[qpos_addr + 2] = xyz[2]
        self._data.qpos[qpos_addr + 3] = float(np.cos(yaw / 2.0))
        self._data.qpos[qpos_addr + 4] = 0.0
        self._data.qpos[qpos_addr + 5] = 0.0
        self._data.qpos[qpos_addr + 6] = float(np.sin(yaw / 2.0))

    # ----------------------------------------------------------- success check

    def _check_placed(self) -> bool:
        """True iff the eraser is resting on the blue tape square.

        Two conditions, AND-ed: the eraser centre is within
        ``place_xy_tol_m`` of the tape centre in XY, and it is resting at
        tape height (not held in the air or tumbled off the desk).
        """
        eraser_xyz = np.asarray(self._data.xpos[self._eraser_body_id], dtype=np.float64)
        tape_xyz = np.asarray(self._data.site_xpos[self._tape_site_id], dtype=np.float64)
        if float(np.linalg.norm(eraser_xyz[:2] - tape_xyz[:2])) > self.options.place_xy_tol_m:
            return False
        # Resting height: the eraser's own half-height above the tape surface,
        # plus 5 mm of settle slack. Anything higher is still in the gripper.
        rest_z = float(tape_xyz[2] + self.options.eraser_size[2])
        return bool(eraser_xyz[2] <= rest_z + 0.005)


@SCENES.register("so101_eraser", fixed_robot="so101_follower")
def build_so101_eraser_scene(env_cfg: SimEnvironment) -> _So101EraserRollout:
    """Build the so101_eraser rollout from a composed :class:`SimEnvironment`.

    ``env_cfg.scene.backend_options`` drives every dimension and pose via
    :func:`_options_from_backend_options`. The MJCF is composed once at build
    time and reused across ``reset()`` calls. The base arm MJCF comes from the
    robot manifest's ``assets.mjcf``, like every other native SO-101 scene.
    """
    import mujoco

    from openral_sim.factory import make_robot  # reason: defer to avoid import cycle

    options = _options_from_backend_options(env_cfg.scene.backend_options)
    description = make_robot(env_cfg)
    _, output_path = compose_so101_eraser_mjcf(options, robot_description=description)
    model = mujoco.MjModel.from_xml_path(str(output_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    arm_actuator_ids: list[int] = []
    arm_qpos_addrs: list[int] = []
    arm_joint_ranges: list[tuple[float, float]] = []
    for k in range(1, _SO101_ARM_DOF + 1):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(k))
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(k))
        if aid < 0 or jid < 0:
            raise ROSConfigError(
                f"so101_eraser: SO-101 actuator/joint {k!r} missing in compiled model "
                f"(actuator_id={aid}, joint_id={jid}). Composer must be updated.",
            )
        arm_actuator_ids.append(aid)
        arm_qpos_addrs.append(int(model.jnt_qposadr[jid]))
        lo, hi = (float(x) for x in model.jnt_range[jid])
        arm_joint_ranges.append((lo, hi))

    def _resolve_named(kind: int, name: str) -> int:
        idx = mujoco.mj_name2id(model, kind, name)
        if idx < 0:
            raise ROSConfigError(
                f"so101_eraser: required {name!r} missing in compiled model (kind={kind}).",
            )
        return int(idx)

    eraser_body = _resolve_named(mujoco.mjtObj.mjOBJ_BODY, "eraser")
    eraser_joint = _resolve_named(mujoco.mjtObj.mjOBJ_JOINT, "eraser_joint")
    tape_site = _resolve_named(mujoco.mjtObj.mjOBJ_SITE, "tape_center")
    lamp_light = _resolve_named(mujoco.mjtObj.mjOBJ_LIGHT, "desk_lamp")
    light_baselines = {
        lid: {
            "diffuse": np.asarray(model.light_diffuse[lid], dtype=np.float64).copy(),
            "specular": np.asarray(model.light_specular[lid], dtype=np.float64).copy(),
            "pos": np.asarray(model.light_pos[lid], dtype=np.float64).copy(),
        }
        for lid in range(model.nlight)
    }

    return _So101EraserRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        options=options,
        _model=model,
        _data=data,
        _arm_actuator_ids=arm_actuator_ids,
        _arm_qpos_addrs=arm_qpos_addrs,
        _arm_joint_ranges=np.asarray(arm_joint_ranges, dtype=np.float64),
        _joint_units=options.joint_units,
        _joint_offsets_deg=np.asarray(options.joint_offsets_deg, dtype=np.float64),
        _joint_signs=np.asarray(options.joint_signs, dtype=np.float64),
        _eraser_qpos_addr=int(model.jnt_qposadr[eraser_joint]),
        _eraser_body_id=eraser_body,
        _tape_site_id=tape_site,
        _lamp_light_id=lamp_light,
        _light_baselines=light_baselines,
        _instruction=env_cfg.task.instruction or _INSTRUCTION,
        _max_steps=int(env_cfg.task.max_steps or _DEFAULT_MAX_STEPS),
        _render_height=int(env_cfg.scene.observation_height or _DEFAULT_RENDER_HEIGHT),
        _render_width=int(env_cfg.scene.observation_width or _DEFAULT_RENDER_WIDTH),
        _rng=np.random.default_rng(env_cfg.seed or 0),
    )
