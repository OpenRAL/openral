"""Hermetic unit tests for the Bucket-2 pure conversion functions.

Loads ``bucket2_markers.py`` by filesystem path (no ament package
install required) mirroring the approach in ``test_foxglove_launch.py``.

Only the pure, ROS-free function is exercised here:
  - ``occupied_voxel_centers`` — OccupancyVoxels → list[(cx, cy, cz)]

No ROS context, no mocks, no stubs (CLAUDE.md §1.11).  All inputs are
real numeric values; all assertions are on real numeric outputs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the module by path so this test runs under plain pytest without the
# ament package installed (same pattern as test_foxglove_launch.py).
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent.parent
_MODULE_PATH = _PKG_DIR / "openral_foxglove_bringup" / "bucket2_markers.py"


def _load_bucket2() -> object:
    import sys

    spec = importlib.util.spec_from_file_location("_bucket2_markers", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module, the standard importlib pattern.
    sys.modules["_bucket2_markers"] = mod
    spec.loader.exec_module(mod)
    return mod


_b2 = _load_bucket2()
occupied_voxel_centers = _b2.occupied_voxel_centers  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# occupied_voxel_centers — OccupancyVoxels → list[(cx, cy, cz)]
# ---------------------------------------------------------------------------


class TestOccupiedVoxelCenters:
    def test_2x2x2_grid_two_occupied(self) -> None:
        """2×2×2 grid with voxels (0,0,0) and (1,1,1) occupied.

        Index formula: idx = x + size_x*(y + size_y*z)
          (0,0,0) → 0 + 2*(0 + 2*0) = 0
          (1,1,1) → 1 + 2*(1 + 2*1) = 7
        Centers at resolution=0.1, origin=(0,0,0):
          (0,0,0) → (0.05, 0.05, 0.05)
          (1,1,1) → (0.15, 0.15, 0.15)
        """
        size_x, size_y, size_z = 2, 2, 2
        occupancy = [0] * 8
        occupancy[0] = 1  # voxel (0,0,0)
        occupancy[7] = 1  # voxel (1,1,1)

        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0),
            resolution=0.1,
            size=(size_x, size_y, size_z),
            occupancy=occupancy,
        )

        assert len(centers) == 2
        c_set = {(round(cx, 9), round(cy, 9), round(cz, 9)) for cx, cy, cz in centers}
        assert (0.05, 0.05, 0.05) in c_set
        assert (0.15, 0.15, 0.15) in c_set

    def test_center_offset_is_half_resolution(self) -> None:
        """Single voxel (0,0,0) with non-unit resolution and non-zero origin."""
        occupancy = [0] * (3 * 3 * 3)
        # Voxel (1, 2, 0): idx = 1 + 3*(2 + 3*0) = 1 + 6 = 7
        occupancy[7] = 255

        centers = occupied_voxel_centers(
            origin=(10.0, 20.0, 30.0),
            resolution=0.5,
            size=(3, 3, 3),
            occupancy=occupancy,
        )

        assert len(centers) == 1
        cx, cy, cz = centers[0]
        assert _approx(cx, 10.0 + 1.5 * 0.5)  # 10.75
        assert _approx(cy, 20.0 + 2.5 * 0.5)  # 21.25
        assert _approx(cz, 30.0 + 0.5 * 0.5)  # 30.25

    def test_the_grids_orientation_places_the_voxel(self) -> None:
        """The lattice is the OctoMap's, so its axes are not `base_frame`'s.

        Drawing a voxel without the grid's rotation puts the obstacle somewhere
        it is not on every dashboard the operator judges a stop by.
        """
        occupancy = [0, 1]  # voxel (1, 0, 0)
        quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # +90° about z
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0),
            resolution=0.1,
            size=(2, 1, 1),
            occupancy=occupancy,
            orientation_xyzw=quat,
        )
        assert len(centers) == 1
        cx, cy, cz = centers[0]
        assert _approx(cx, -0.05)
        assert _approx(cy, 0.15)
        assert _approx(cz, 0.05)

    def test_an_unset_orientation_is_refused_not_read_as_identity(self) -> None:
        """All-zeros is what an unset field carries, and it is not a rotation."""
        with pytest.raises(ValueError, match="unit quaternion"):
            occupied_voxel_centers(
                origin=(0.0, 0.0, 0.0),
                resolution=0.1,
                size=(1, 1, 1),
                occupancy=[1],
                orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
            )

    def test_all_free_returns_empty_list(self) -> None:
        occupancy = [0] * (2 * 2 * 2)
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=0.05, size=(2, 2, 2), occupancy=occupancy
        )
        assert centers == []

    def test_empty_grid_returns_empty_list(self) -> None:
        """Zero-size grid (no voxels) → empty list, no crash."""
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=0.1, size=(0, 0, 0), occupancy=[]
        )
        assert centers == []

    def test_wrong_occupancy_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="occupancy length"):
            occupied_voxel_centers(
                origin=(0.0, 0.0, 0.0),
                resolution=0.1,
                size=(2, 2, 2),
                occupancy=[0] * 5,  # needs 8
            )

    def test_row_major_x_fastest_ordering(self) -> None:
        """Verify x-fastest indexing: (1,0,0)=idx1, (0,1,0)=idx2."""
        # 3×2×1 grid: size_x=3, size_y=2, size_z=1
        # (1,0,0) → 1 + 3*(0 + 2*0) = 1
        # (0,1,0) → 0 + 3*(1 + 2*0) = 3
        occupancy = [0] * 6
        occupancy[1] = 1  # voxel (1,0,0)
        occupancy[3] = 1  # voxel (0,1,0)

        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=1.0, size=(3, 2, 1), occupancy=occupancy
        )

        assert len(centers) == 2
        c_set = {(round(cx, 9), round(cy, 9), round(cz, 9)) for cx, cy, cz in centers}
        # (1,0,0) → center (1.5, 0.5, 0.5)
        assert (1.5, 0.5, 0.5) in c_set
        # (0,1,0) → center (0.5, 1.5, 0.5)
        assert (0.5, 1.5, 0.5) in c_set


class TestOccupiedVoxelCentersFromTheWire:
    def test_a_numpy_occupancy_array_matches_the_list_form(self) -> None:
        """The node hands the message's occupancy over as an ndarray, never a list copy:
        the grid is ~10^6 cells at several Hz, and the list copy plus a Python loop
        pinned a Thor core (2026-10-02)."""
        import numpy as np

        occupancy = np.zeros(4 * 3 * 2, dtype=np.uint8)
        occupancy[[0, 5, 23]] = 1
        from_array = occupied_voxel_centers((1.0, 2.0, 3.0), 0.02, (4, 3, 2), occupancy)
        from_list = occupied_voxel_centers((1.0, 2.0, 3.0), 0.02, (4, 3, 2), occupancy.tolist())
        assert from_array == from_list
        assert len(from_array) == 3
        last = (1.0 + 3.5 * 0.02, 2.0 + 2.5 * 0.02, 3.0 + 1.5 * 0.02)
        assert from_array[-1] == pytest.approx(last)


# ---------------------------------------------------------------------------
# attachment_markers — AttachmentState → MarkerArray
# ---------------------------------------------------------------------------

attachment_markers = _b2.attachment_markers  # type: ignore[attr-defined]
parse_attach_link_tf_frames = _b2.parse_attach_link_tf_frames  # type: ignore[attr-defined]

# The OpenArm cell's vendor TF name for the manifest's attach link (scene tf_frames).
_TF_FRAMES = {"openarm_right_link7": "openarm_right_ee_base_link"}
_BASE = "openarm_base"


def _msgs() -> tuple[object, object, object]:
    """``(openral_msgs.msg, geometry_msgs.msg, visualization_msgs.msg)``, or skip without ROS."""
    return (
        pytest.importorskip("openral_msgs.msg", reason="needs a sourced openral_msgs install"),
        pytest.importorskip("geometry_msgs.msg", reason="needs ROS 2 sourced"),
        pytest.importorskip("visualization_msgs.msg", reason="needs ROS 2 sourced"),
    )


def _pose(xyz: tuple[float, float, float], quat_xyzw: tuple[float, ...] = (0, 0, 0, 1)) -> object:
    _, gm, _ = _msgs()
    p = gm.Pose()  # type: ignore[attr-defined]
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat_xyzw
    return p


def _primitive(
    shape: int, dims: list[float], xyz: tuple[float, float, float] = (0, 0, 0)
) -> object:
    om, _, _ = _msgs()
    prim = om.AttachedCollisionPrimitive()  # type: ignore[attr-defined]
    prim.shape_type = shape
    prim.shape_dimensions = dims
    prim.pose_in_object = _pose(xyz)
    return prim


def _object(**fields: object) -> object:
    om, _, _ = _msgs()
    obj = om.AttachedCollisionObject()  # type: ignore[attr-defined]
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def _region(xyz: tuple[float, float, float], half: tuple[float, float, float]) -> object:
    om, gm, _ = _msgs()
    region = om.PlaceRegion()  # type: ignore[attr-defined]
    region.frame_id = _BASE
    region.pose = _pose(xyz)
    region.half_extents = gm.Vector3(x=half[0], y=half[1], z=half[2])  # type: ignore[attr-defined]
    region.evidence_ref = "head_seg+voxels:cell:restock_box"
    return region


def _state(objects: list[object] | None = None) -> object:
    om, _, _ = _msgs()
    state = om.AttachmentState()  # type: ignore[attr-defined]
    state.header.frame_id = _BASE
    state.header.stamp.sec = 42
    state.objects = objects or []
    return state


def _adds(array: object, ns: str) -> list[object]:
    return [m for m in array.markers if m.ns == ns]  # type: ignore[attr-defined]


class TestAttachmentMarkers:
    def test_an_empty_state_is_one_deleteall(self) -> None:
        """A detach publishes an empty set: the viewer must drop every marker it drew."""
        _, _, vm = _msgs()
        array = attachment_markers(_state(), _TF_FRAMES)
        assert len(array.markers) == 1
        assert array.markers[0].action == vm.Marker.DELETEALL  # type: ignore[attr-defined]

    def test_a_region_payload_box_rides_the_attach_links_tf_frame(self) -> None:
        """The grasp-target leg's handover: one box, posed pose_in_link ∘ pose_in_object,
        on the vendor TF frame the manifest link is renamed to."""
        om, _, vm = _msgs()
        mk = vm.Marker  # type: ignore[attr-defined]
        quarter_z = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # +90° yaw
        obj = _object(
            object_id="cell:restock_box",
            attach_link="openarm_right_link7",
            pose_in_link=_pose((0.0, 0.0, 0.1), quarter_z),
            primitives=[
                _primitive(
                    om.AttachedCollisionPrimitive.SHAPE_BOX, [0.03, 0.02, 0.05], (0.02, 0, 0)
                )  # type: ignore[attr-defined]
            ],
            evidence_kind="grasp_target_region",
            evidence_ref="grasp_target_region:cell:restock_box:seg@1",
        )
        array = attachment_markers(_state([obj]), _TF_FRAMES)

        assert array.markers[0].action == mk.DELETEALL
        (cube,) = _adds(array, "attached")
        assert cube.type == mk.CUBE and cube.action == mk.ADD
        assert cube.header.frame_id == "openarm_right_ee_base_link"
        assert cube.header.stamp.sec == 42
        assert cube.frame_locked
        assert (cube.scale.x, cube.scale.y, cube.scale.z) == pytest.approx((0.06, 0.04, 0.10))
        # (0.02, 0, 0) in the object frame, yawed +90°, lifted 0.1 → (0, 0.02, 0.1).
        p = cube.pose.position
        assert (p.x, p.y, p.z) == pytest.approx((0.0, 0.02, 0.1), abs=1e-12)
        o = cube.pose.orientation
        assert abs(o.z * quarter_z[2] + o.w * quarter_z[3]) == pytest.approx(1.0)
        assert (cube.color.r, cube.color.g, cube.color.b) == pytest.approx((1.0, 0.55, 0.0))
        assert 0.0 < cube.color.a < 1.0
        (label,) = _adds(array, "attached_label")
        assert label.type == mk.TEXT_VIEW_FACING
        assert label.text == "restock_box · grasp_target_region"
        assert label.header.frame_id == "openarm_right_ee_base_link"

    def test_a_released_record_frozen_on_the_base_frame_is_labelled(self) -> None:
        """``freeze_released_attachment``'s record: attached to the base, ``|frozen_release``,
        support attested — drawn where it was let go, tagged so."""
        om, _, vm = _msgs()
        obj = _object(
            object_id="cell:restock_box",
            attach_link=_BASE,
            pose_in_link=_pose((0.45, -0.2, 0.3)),
            primitives=[_primitive(om.AttachedCollisionPrimitive.SHAPE_BOX, [0.03, 0.03, 0.03])],  # type: ignore[attr-defined]
            evidence_kind="vision_segmentation",
            evidence_ref="vision_segmentation:wrist_right@7|frozen_release",
            support_contact_valid=True,
        )
        array = attachment_markers(_state([obj]), _TF_FRAMES)
        (cube,) = _adds(array, "attached")
        assert cube.header.frame_id == _BASE  # unmapped: its own frame
        p = cube.pose.position
        assert (p.x, p.y, p.z) == pytest.approx((0.45, -0.2, 0.3))
        assert (cube.color.r, cube.color.g, cube.color.b) == pytest.approx((0.0, 0.8, 1.0))
        (label,) = _adds(array, "attached_label")
        assert label.text == "restock_box · vision_segmentation · support · released"
        assert array.markers[0].action == vm.Marker.DELETEALL  # type: ignore[attr-defined]

    def test_capsule_and_sphere_primitives(self) -> None:
        """A capsule is a cylinder plus end caps along its local +Z; a sphere is one sphere."""
        om, _, vm = _msgs()
        mk = vm.Marker  # type: ignore[attr-defined]
        acp = om.AttachedCollisionPrimitive  # type: ignore[attr-defined]
        obj = _object(
            object_id="sim:mug",
            attach_link="openarm_right_link7",
            pose_in_link=_pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),  # 180° about x: +Z → -Z
            primitives=[
                _primitive(acp.SHAPE_CAPSULE, [0.01, 0.08], (0.0, 0.0, 0.1)),
                _primitive(acp.SHAPE_SPHERE, [0.02]),
            ],
            evidence_kind="sim_geom_distance",
        )
        shapes = _adds(attachment_markers(_state([obj]), _TF_FRAMES), "attached")
        assert [m.type for m in shapes] == [mk.CYLINDER, mk.SPHERE, mk.SPHERE, mk.SPHERE]
        cyl, cap_lo, cap_hi, ball = shapes
        assert (cyl.scale.x, cyl.scale.z) == pytest.approx((0.02, 0.08))
        assert cyl.pose.position.z == pytest.approx(-0.1)
        assert sorted([cap_lo.pose.position.z, cap_hi.pose.position.z]) == pytest.approx(
            [-0.14, -0.06]
        )
        assert ball.scale.x == pytest.approx(0.04)
        # Uncurated evidence kinds share the "other" colour.
        assert (ball.color.r, ball.color.g, ball.color.b) == pytest.approx((0.8, 0.3, 1.0))
        assert len({m.id for m in shapes}) == len(shapes)

    def test_grasp_and_place_regions(self) -> None:
        """An active grasp region (green) and a retracted place region (grey), each a box,
        12 wireframe edges and a target label in the region's own frame."""
        _, _, vm = _msgs()
        mk = vm.Marker  # type: ignore[attr-defined]
        state = _state()
        state.grasp_declaration_valid = True  # type: ignore[attr-defined]
        g = state.grasp_declaration  # type: ignore[attr-defined]
        g.target_id, g.active, g.region_valid = "cell:restock_box", True, True
        g.region = _region((0.5, -0.2, 0.1), (0.04, 0.03, 0.05))
        state.place_declaration_valid = True  # type: ignore[attr-defined]
        pl = state.place_declaration  # type: ignore[attr-defined]
        pl.target_id, pl.active, pl.region_valid = "surface:shelf_2", False, True
        pl.region = _region((0.6, 0.1, 0.4), (0.2, 0.1, 0.01))

        array = attachment_markers(state, _TF_FRAMES)
        for name, rgb, target in (
            ("grasp", (0.1, 0.9, 0.2), "grasp: cell:restock_box"),
            ("place", (0.6, 0.6, 0.6), "place: surface:shelf_2 (inactive)"),
        ):
            box, edges = _adds(array, f"{name}_region")
            assert box.type == mk.CUBE and edges.type == mk.LINE_LIST
            assert box.header.frame_id == edges.header.frame_id == _BASE
            assert (box.color.r, box.color.g, box.color.b) == pytest.approx(rgb)
            assert len(edges.points) == 24  # 12 edges
            for a, b in zip(edges.points[::2], edges.points[1::2], strict=True):
                assert (
                    sum(abs(pa - pb) > 1e-12 for pa, pb in ((a.x, b.x), (a.y, b.y), (a.z, b.z)))
                    == 1
                )
            (label,) = _adds(array, f"{name}_region_label")
            assert label.text == target
        grasp_box = _adds(array, "grasp_region")[0]
        assert (grasp_box.scale.x, grasp_box.scale.y, grasp_box.scale.z) == pytest.approx(
            (0.08, 0.06, 0.1)
        )
        assert _adds(array, "grasp_region_label")[0].pose.position.z == pytest.approx(0.18)

    def test_a_declaration_without_a_measured_region_draws_nothing(self) -> None:
        """Dispatch's own declaration carries no region; nothing is invented for it."""
        state = _state()
        state.grasp_declaration_valid = True  # type: ignore[attr-defined]
        state.grasp_declaration.target_id = "cell:restock_box"  # type: ignore[attr-defined]
        state.grasp_declaration.active = True  # type: ignore[attr-defined]
        assert len(attachment_markers(state, _TF_FRAMES).markers) == 1

    def test_a_malformed_primitive_is_refused(self) -> None:
        om, _, _ = _msgs()
        obj = _object(
            object_id="x",
            attach_link=_BASE,
            pose_in_link=_pose((0, 0, 0)),
            primitives=[_primitive(om.AttachedCollisionPrimitive.SHAPE_BOX, [0.03])],  # type: ignore[attr-defined]
        )
        with pytest.raises(ValueError, match="dimension"):
            attachment_markers(_state([obj]))


class TestParseAttachLinkTfFrames:
    def test_the_scene_strings_and_the_ros_default(self) -> None:
        assert parse_attach_link_tf_frames(
            ["openarm_left_link7=openarm_left_ee_base_link", "", "a=b"]
        ) == {"openarm_left_link7": "openarm_left_ee_base_link", "a": "b"}
        assert parse_attach_link_tf_frames([""]) == {}

    @pytest.mark.parametrize("entries", [["nolink"], ["=frame"], ["link="], ["a=b", "a=c"]])
    def test_malformed_maps_are_refused(self, entries: list[str]) -> None:
        with pytest.raises(ValueError):
            parse_attach_link_tf_frames(entries)


def test_module_docstring_examples_run() -> None:
    """Every ``Example`` in the module's docstrings executes (CLAUDE.md §2)."""
    import doctest

    results = doctest.testmod(_b2, verbose=False)  # type: ignore[arg-type]
    assert results.attempted > 0
    assert results.failed == 0
