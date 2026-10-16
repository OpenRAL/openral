# SPDX-License-Identifier: Apache-2.0
"""Generic scene inputs: the one RGB-D frame contract and the derived safety declarations.

* ``openral_sim.depth_frames``: every backend's registered RGB-D frame goes through
  ``depth_frame`` (Isaac obs, the MuJoCo bridge).
* ``openral_sim.backends.mujoco_geometry`` / ``tools/derive_scene_inputs.py`` /
  ``openral_reasoner.grounding.ground_workspace``: static fixtures and the grasp
  declaration's boxes are derived from the sim, never required by hand.

Real manifests and the real OpenArm v2 MJCF (CLAUDE.md §1.11); the RoboCasa and Isaac
ends run in the sim tier (``tests/sim/test_panda_mobile_hal_robocasa_scene_inputs.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from openral_core import GraspDeclaration, RobotDescription
from openral_core.exceptions import ROSConfigError, ROSReasonerInvalidPlan
from openral_reasoner.grounding import ground_workspace
from openral_sim.depth_frames import depth_frame

_REPO = Path(__file__).resolve().parents[2]


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "derive_scene_inputs", _REPO / "tools" / "derive_scene_inputs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDepthFrameContract:
    def test_a_rendered_frame_round_trips(self) -> None:
        f = depth_frame(
            np.array([[1.0, np.nan], [-2.0, 3.0]]),
            np.zeros((2, 2, 3)),
            [300.0, 300.0, 1.0, 1.0],
            np.eye(4),
        )
        assert f["depth"].tolist() == [[1.0, 0.0], [0.0, 3.0]]  # no-return sentinel
        assert f["rgb"].dtype == np.uint8 and f["k"].shape == (4,)

    @pytest.mark.parametrize(
        ("depth", "rgb", "k", "pose"),
        [
            (np.ones((2, 2)), np.zeros((3, 2, 3)), [1, 1, 1, 1], np.eye(4)),
            (np.ones((2, 2)), np.zeros((0,)), [1, 1, 1, 1], np.eye(4)),
            (np.ones((2, 2)), np.zeros((0, 0, 4)), [1, 1, 1, 1], np.eye(4)),
            (np.ones(4), None, [1, 1, 1, 1], np.eye(4)),
            (np.ones((2, 2)), None, [1, 1, 1], np.eye(4)),
            (np.ones((2, 2)), None, [1, 1, 1, 1], np.eye(3)),
        ],
    )
    def test_off_contract_frames_are_refused(self, depth, rgb, k, pose) -> None:  # type: ignore[no-untyped-def]  # reason: parametrized fixtures
        with pytest.raises(ROSConfigError):
            depth_frame(depth, rgb, k, pose)


class TestStaticReplyFrame:
    def test_a_fixed_base_with_no_base_body_is_refused_not_exported_in_world(self) -> None:
        """Without a body backing base_frame the walls would land offset by the base's pose."""
        mujoco = pytest.importorskip("mujoco")
        from types import SimpleNamespace

        from openral_core import RobotDescription

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='mount' pos='1 2 0'/>"
            "<body name='shelf' pos='1 0 0'><geom type='box' size='.1 .2 .3'/></body>"
            "</worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(mujoco_handles=lambda: (model, data))
        desc = RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))
        with pytest.raises(ROSConfigError, match="no MJCF body backs"):
            _tool()._static_reply(env, desc)  # type: ignore[attr-defined]
        # The world frame on request (a mobile base's odom) is still fine.
        reply, base_id = _tool()._static_reply(env, desc, world=True)  # type: ignore[attr-defined]
        assert base_id == 0 and len(reply["boxes"]) == 1


class TestMujocoStaticColliders:
    def test_the_robot_is_never_a_static_fixture(self) -> None:
        """OpenArm's base is welded to the world, yet it is the robot, not a fixture."""
        mujoco = pytest.importorskip("mujoco")
        from openral_hal.depth_cloud import robot_self_body_ids
        from openral_sim.backends.mujoco_geometry import static_collider_boxes

        try:
            from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

            mjcf = ensure_openarm_v2_mjcf()
        except Exception as exc:  # reason: the MJCF is a cached clone; offline hosts skip
            pytest.skip(f"OpenArm v2 MJCF unavailable: {exc}")
        model = mujoco.MjModel.from_xml_path(mjcf)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        desc = RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml"))
        robot = robot_self_body_ids(model, [j.sim_joint_name for j in desc.joints])
        assert robot
        welded_robot = [
            b for b in robot if int(model.body_weldid[b]) == 0 and int(model.body_geomnum[b])
        ]
        assert welded_robot, "the fixed base must be a welded robot body for this test to bite"
        everything = static_collider_boxes(model, data, base_body_id=0)["boxes"]
        fixtures = static_collider_boxes(model, data, base_body_id=0, exclude_body_ids=robot)
        robot_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in robot}
        assert any(b["path"].split("/")[0] in robot_names for b in everything)
        assert not any(b["path"].split("/")[0] in robot_names for b in fixtures["boxes"])


class TestDerivedDeclarations:
    # A pick-and-place cell's source tote (static colliders, base frame) and one shelf slot,
    # from the Isaac export the hand-written scene was built from.
    TOTE: ClassVar[dict[str, tuple[list[float], list[float]]]] = {
        "SourceToteCollision/Floor": ([0.1937, -0.0059, -0.58705], [0.1191, 0.1759, 0.0045]),
        "SourceToteCollision/Rim_Front": ([0.0496, -0.00595, -0.45135], [0.003, 0.20335, 0.004]),
        "SourceToteCollision/Rim_Back": ([0.3378, -0.00595, -0.45135], [0.003, 0.20335, 0.004]),
        "SourceToteCollision/Rim_Left": ([0.1937, 0.1944, -0.45135], [0.1471, 0.003, 0.004]),
        "SourceToteCollision/Rim_Right": ([0.1937, -0.2063, -0.45135], [0.1471, 0.003, 0.004]),
        "StagingBoard": ([0.210012, -0.034918, -0.596321], [0.05, 0.408818, 0.00925]),
    }

    def _colliders(self) -> list[dict[str, object]]:
        return [
            {
                "path": f"/World/environment/{p}",
                "center_m": c,
                "rotation": np.eye(3),
                "half_extents_m": h,
            }
            for p, (c, h) in self.TOTE.items()
        ]

    def test_the_object_rests_in_the_tote_not_on_the_board_under_it(self) -> None:
        tool = _tool()
        carton = (0.20, 0.05, -0.582, 0.32, 0.14, -0.462)
        bound, group = tool.source_fixture(carton, self._colliders())  # type: ignore[attr-defined]
        assert group == "/World/environment/SourceToteCollision"
        assert bound == pytest.approx((0.0466, -0.2093, -0.59155, 0.3408, 0.1974, -0.44735))

    def test_nothing_static_under_the_object(self) -> None:
        tool = _tool()
        assert tool.source_fixture((2.0, 2.0, 0.0, 2.1, 2.1, 0.1), self._colliders()) is None  # type: ignore[attr-defined]

    def test_workspace_is_the_padded_hull_of_tote_and_slot(self) -> None:
        tool = _tool()
        tote, _ = tool.source_fixture((0.20, 0.05, -0.582, 0.32, 0.14, -0.462), self._colliders())  # type: ignore[attr-defined]
        slot = (0.339, 0.0342, -0.4438, 0.715, 0.1579, -0.3238)  # lower_3, stage -> base
        ws = ground_workspace(
            tote, slot, anchor_xyz=(0.26, 0.095, -0.52), base_frame="openarm_base",
            pad_m=0.02, headroom_m=0.10, evidence_ref="sim:tote+lower_3",
        )  # fmt: skip
        lo = np.subtract(ws.pose.xyz, ws.half_extents)
        hi = np.add(ws.pose.xyz, ws.half_extents)
        assert lo == pytest.approx([0.0266, -0.2293, -0.61155])
        assert hi == pytest.approx([0.735, 0.2174, -0.2038])
        # It passes the declaration's own workspace caps.
        GraspDeclaration(
            target_id="cell:stock_object", contact_links=("openarm_left_finger_pair",),
            timeout_s=70.0, stamp_ns=0, workspace_box=ws,
        )  # fmt: skip

    def test_a_counter_sized_source_is_cut_to_the_cap_around_the_target(self) -> None:
        counter = (-1.0, -0.3, 0.8, 1.5, 0.3, 0.9)  # a 2.5 m kitchen counter top
        ws = ground_workspace(
            counter, None, anchor_xyz=(0.6, 0.0, 0.95), base_frame="base_link",
            pad_m=0.02, headroom_m=0.10, evidence_ref="sim:counter",
        )  # fmt: skip
        assert ws.half_extents[0] == pytest.approx(GraspDeclaration.MAX_WORKSPACE_HALF_EXTENT_M)
        assert ws.pose.xyz[0] == pytest.approx(0.6)

    def test_a_room_sized_workspace_is_refused(self) -> None:
        with pytest.raises(ROSReasonerInvalidPlan, match="cap"):
            ground_workspace(
                (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0), None, anchor_xyz=(0.0, 0.0, 0.0),
                base_frame="base_link", pad_m=0.0, headroom_m=0.0, evidence_ref="sim:room",
            )  # fmt: skip


class TestArenaMapleTableSupport:
    """Arena's maple table: a dynamic rigid body the banana rests on, grounded as its support.

    The colliders are the Isaac 6.1 sidecar's export of
    ``isaac:Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/scenes/maple_table.usda``
    as measured on Spark (2026-10-16), in the DROID scene's ``panda_link0`` (the robot sits at
    the world origin): the table top and four legs carry ``CollisionAPI`` under the
    enabled, non-kinematic ``RigidBodyAPI`` of ``/World/table`` (Arena keeps it dynamic,
    ``MapleTableRobolab``), the ground plane is static. The banana's bound is the
    ``scene_ground_truth`` of ``scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml``.
    """

    TABLE = "/World/environment/table/table_01"
    # (lo, hi) of each collider's USD world bound.
    PARTS: ClassVar[dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]] = {
        "top": ((0.19859, -0.477937, -0.026999), (0.898591, 0.522063, 0.003001)),
        "leg_0": ((0.808591, 0.402063, -0.696999), (0.848591, 0.442063, -0.026999)),
        "leg_1": ((0.808591, -0.397937, -0.696999), (0.848591, -0.357937, -0.026999)),
        "leg_2": ((0.248591, 0.402063, -0.696999), (0.288591, 0.442063, -0.026999)),
        "leg_3": ((0.248591, -0.397937, -0.696999), (0.288591, -0.357937, -0.026999)),
    }
    BANANA_CENTRE: ClassVar[tuple[float, float, float]] = (0.509671, -0.115792, 0.021202)
    BANANA_HALF: ClassVar[list[float]] = [0.054504, 0.089225, 0.018602]
    # t3's stop cell (left inner finger vs the table top, PhysX 8.0 mm clear).
    STOP_CELL: ClassVar[list[float]] = [0.4425, -0.0675, 0.0075]

    @staticmethod
    def _box(path: str, lo: tuple[float, ...], hi: tuple[float, ...], **kw: object) -> dict:  # type: ignore[type-arg]  # reason: the sidecar's JSON entry
        lo_a, hi_a = np.asarray(lo), np.asarray(hi)
        return {
            "path": path,
            "kind": "box",
            "collision_enabled": True,
            "center_m": ((lo_a + hi_a) / 2).tolist(),
            "rotation": np.eye(3).tolist(),
            "half_extents_m": ((hi_a - lo_a) / 2).tolist(),
            **kw,
        }

    def _reply(self) -> dict:  # type: ignore[type-arg]  # reason: the sidecar's JSON reply
        boxes = [
            self._box(f"{self.TABLE}/{name}", lo, hi, dynamic=True)
            for name, (lo, hi) in self.PARTS.items()
        ]
        ground = ((-1.0, -1.0, -0.697), (1.0, 1.0, -0.697))
        boxes.append(self._box("/World/environment/GroundPlane/CollisionPlane", *ground))
        return {"frame": "base_frame", "boxes": boxes}

    def _banana(self) -> tuple[float, float, float, float, float, float]:
        c, h = np.asarray(self.BANANA_CENTRE), np.asarray(self.BANANA_HALF)
        lo, hi = c - h, c + h
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def test_the_banana_rests_on_the_table_top_not_the_floor(self) -> None:
        bound, group = _tool().source_fixture(self._banana(), self._reply()["boxes"])  # type: ignore[attr-defined]
        assert group == self.TABLE
        lo, hi = self.PARTS["top"]
        assert bound == pytest.approx((*lo, *hi)), "the legs below the top are not the support"

    def test_the_workspace_is_the_table_top_and_holds_the_t3_stop_cell(self) -> None:
        bound, group = _tool().source_fixture(self._banana(), self._reply()["boxes"])  # type: ignore[attr-defined]
        ws = ground_workspace(
            bound, None, anchor_xyz=self.BANANA_CENTRE, base_frame="panda_link0",
            pad_m=0.02, headroom_m=0.10, evidence_ref=f"sim:{group}",
        )  # fmt: skip
        GraspDeclaration(
            target_id="obj:banana", contact_links=("robotiq_finger_pair",), timeout_s=120.0,
            stamp_ns=0, workspace_box=ws,
        )  # fmt: skip
        lo = np.subtract(ws.pose.xyz, ws.half_extents)
        hi = np.add(ws.pose.xyz, ws.half_extents)
        assert np.all(lo <= self.STOP_CELL) and np.all(hi >= self.STOP_CELL)
        assert lo[2] == pytest.approx(self.PARTS["top"][0][2] - 0.02), "nothing below the top"

    def test_the_legs_would_have_stretched_the_workspace_past_the_cap(self) -> None:
        """What the support was before the below-top rule: the whole table, to the floor."""
        lo = (0.19859, -0.477937, -0.696999)
        hi = (0.898591, 0.522063, 0.003001)
        with pytest.raises(ROSReasonerInvalidPlan, match="cap"):
            ground_workspace(
                (*lo, *hi), None, anchor_xyz=self.BANANA_CENTRE, base_frame="panda_link0",
                pad_m=0.02, headroom_m=0.10, evidence_ref="sim:table",
            )  # fmt: skip

    def test_a_dynamic_fixture_is_never_declared_static_geometry(self, tmp_path: Path) -> None:
        import yaml

        out = tmp_path / "static.yaml"
        assert _tool().write_static(self._reply(), out, None) == 1  # type: ignore[attr-defined]
        entries = yaml.safe_load(out.read_text())["static_geometry"]
        assert [e["source"] for e in entries] == ["/World/environment/GroundPlane/CollisionPlane"]
