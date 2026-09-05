# SPDX-License-Identifier: Apache-2.0
"""The ``robocasa_fridge_drawer`` layout pin, checked against the real kernel.

``scenes/deploy/robocasa_fridge_drawer.yaml`` pins ``layout_ids: [47]``, and the
comment block above that pin carries the measurements it was chosen on — layout
30 stopping at ``-23.47 mm`` on ``panda_link2`` while layout 47 clears at
``+19.34 mm`` on ``panda_link1``. Until this file, **nothing executable held
that claim**: the pin could be moved, ``panda_link2``'s OBB loosened, or the
lowered kinematics shifted, and the only thing that would disagree was a YAML
comment. This is issue #102's third acceptance item — "a regression for the
nominal valid pose and a nearby genuinely colliding fixture pose" — with the
pair the start-state census
(:doc:`../../../docs/reference/robocasa-start-state-census`) identified.

Real kitchen, real manifest, real kernel binary, no mocks (CLAUDE.md §1.11):

1. RoboCasa composes the scene at a pinned layout, at the scene's own seed, and
   is read at reset with zero actions applied.
2. A 25 mm occupancy grid is built around the arm from that kitchen's own
   geometry, cell by cell, through the shipped
   :func:`openral_hal.sim_sensor_bridge.voxel_backing_record` — the same probe
   the E-stop evidence path uses to ask what backs a cell. A cell is occupied
   when a **solid** (collidable) world geom passes through it.
3. The real ``safety_kernel_node`` is launched from
   ``robots/panda_mobile/robot.yaml``, seeded with the reset configuration on
   ``/joint_states`` and that grid on ``/openral/world_voxels``, and asked to
   pass a zero ``CARTESIAN_DELTA`` chunk — the RoboCasa arm mode.

WHAT THIS GRID IS, AND WHAT IT IS NOT
-------------------------------------
This grid is built from true surfaces. The **live** grid is depth-camera
derived through octomap, and the scene file's own note applies unchanged: the
live map also holds non-collidable decoration and the octree->grid bridge
dilates whatever it holds, so **the live map stops more often than this one,
never less** (``docs/reference/world-map-fidelity.md``). Measured here, at
``world_voxel_margin_m = 0``:

    layout   this grid                      live octomap (scene file)
    ------   ----------------------------   -------------------------
       47    +44.55 mm  ``panda_link1``     +19.34 mm  ``panda_link1``
       30    +19.09 mm  ``panda_link2``     -23.47 mm  ``panda_link2``

Same dominant link on both layouts, same ordering, uniformly more generous.
**So layout 30 clears at zero margin here and stops on the live map, and this
file must not be read as clearing layout 30 for use.** The pin stays 47. What
is pinned instead is the part that does not depend on how dense the map is: 47
clears, 30 is the tighter of the two, and each is tight on the link the census
named. Reproducing the live verdict itself needs the whole deploy graph, which
is #102's separate end-to-end acceptance item, not this one.

The margin sweep is what makes the ordering quantitative. ``world_voxel_margin_m``
is the kernel's own standoff, so raising it walks a known distance in from the
surface: at ``0.020 m`` layout 30 trips and layout 47 does not, which is the
pin's criterion stated as a test rather than as a comment.

The all-occupied control is not decoration. A grid that never reached the
kernel, or landed in the wrong frame, would let *every* configuration through
and every clearance assertion here would pass vacuously — the failure mode #183
found in the Nav2 live tests. ``test_an_all_occupied_grid_is_refused`` fails if
that happens.

Gates: ROS_DISTRO + rclpy + openral_msgs on a sourced + colcon-built workspace,
plus MuJoCo and the RoboCasa kitchen backend.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
    ),
]
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

import numpy as np  # noqa: E402
from openral_core import (  # noqa: E402
    DeployScene,
    RobotDescription,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_hal.depth_cloud import robot_self_body_ids  # noqa: E402
from openral_hal.sim_sensor_bridge import voxel_backing_record  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    ee_link_index_from_collision_params,
    kernel_params_from_envelope,
)
from openral_sim.registry import SCENES  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_fridge_drawer.yaml"
_MANIFEST = str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")

#: The shipped pin, and the one it replaced on 2026-08-25.
_PIN_LAYOUT = 47
_RETIRED_LAYOUT = 30

#: The kernel's own world-voxel resolution in the deploy graph.
_RES = 0.025
#: Padding around the arm's own extent. The widest corner slop in this model is
#: ``panda_link4``'s 88.22 mm and a 25 mm cell adds 21.65 mm of half-diagonal,
#: so 200 mm leaves the nearest cell of every checked link inside the region.
_PAD = 0.20
#: Rays per face-fan side. ``voxel_backing_record`` casts ``3 * n**2`` per cell.
_RAYS_PER_AXIS = 2

#: The MJCF body the manifest's ``base_link`` denotes — the arm mount, 0.700 m
#: up the pedestal (ADR-0095). The kernel zeroes the base DoFs and evaluates the
#: arm in this frame, so the grid is published in it.
_BASE_FRAME_BODY = "robot0_link0"

#: A margin that separates the two layouts. Below the retired layout's clearance
#: on this grid (+19.09 mm) and above nothing else that matters; see the sweep in
#: the module docstring.
_SEPARATING_MARGIN_M = 0.020


def _compose(layout: int) -> Any:
    """Build and reset the fridge scene at ``layout``, the way ``openral deploy sim`` does."""
    deploy = DeployScene.model_validate(yaml.safe_load(_SCENE.read_text()))
    options = {
        **(deploy.scene.backend_options or {}),
        "ignore_done": True,
        "layout_ids": [layout],
    }
    scene = deploy.scene.model_copy(update={"backend_options": options})
    sim = SCENES.get(scene.id)(
        SimEnvironment(
            robot_id=deploy.robot_id or SCENES.fixed_robot(scene.id),
            scene=scene,
            task=TaskSpec(
                id=f"{scene.id}/_layout_pin_start_state",
                scene_id=scene.id,
                instruction="",
                max_steps=None,
                success_key=None,
            ),
            vla=VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_base"),
            base_pose=deploy.base_pose,
            seed=deploy.seed,
            n_episodes=1,
            record_video=False,
        )
    )
    sim.reset(seed=deploy.seed)
    return sim


class _StartState:
    """One layout's reset configuration plus the occupancy grid around its arm."""

    def __init__(self, layout: int) -> None:
        sim = _compose(layout)
        try:
            env = getattr(sim._env, "unwrapped", sim._env)
            model, data = env.sim.model._model, env.sim.data._data
            self.layout = int(env.layout_id)
            desc = RobotDescription.from_yaml(_MANIFEST)
            robot_bodies = robot_self_body_ids(
                model, [j.sim_joint_name for j in desc.joints if j.sim_joint_name]
            )

            # The reset configuration, read through the manifest's own sim
            # joint names rather than by guessing the robosuite prefix.
            self.joint_names = [j.name for j in desc.joints]
            self.positions: list[float] = []
            for joint in desc.joints:
                jid = int(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name or "")
                )
                assert jid >= 0, (
                    f"manifest joint {joint.name!r} maps to sim joint "
                    f"{joint.sim_joint_name!r}, which this kitchen does not have"
                )
                self.positions.append(float(data.qpos[model.jnt_qposadr[jid]]))

            self.origin, self.size, self.occupancy, self.occupied = _build_grid(
                model, data, robot_bodies
            )
        finally:
            sim.close()


def _build_grid(model: Any, data: Any, robot_bodies: frozenset[int]) -> tuple[Any, Any, Any, int]:
    """A base-frame 25 mm occupancy grid over the arm's neighbourhood.

    Bounded by the seven kernel-checked links' own bounding spheres plus
    :data:`_PAD`, so it holds the nearest cell of every link the kernel checks
    without rasterising a whole kitchen. Occupancy comes from
    :func:`~openral_hal.sim_sensor_bridge.voxel_backing_record`, one call per
    cell, and a cell counts as occupied only for a ``solid_world`` backing —
    the census's criterion, and the one the near-miss probes use.
    """
    base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _BASE_FRAME_BODY))
    assert base_id >= 0, f"{_BASE_FRAME_BODY} absent; the robot naming has drifted"
    base_t = np.asarray(data.xpos[base_id], dtype=np.float64)
    base_r = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)

    link_bodies = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot0_link{i}"))
        for i in range(1, 8)
    }
    corners: list[Any] = []
    for geom in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom]) not in link_bodies:
            continue
        centre = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        radius = float(model.geom_rbound[geom])
        corners += [centre - radius, centre + radius]
    assert corners, "no geoms on panda_link1..7; the collision model has drifted"

    in_base = base_r.T @ (np.asarray(corners).T - base_t[:, None])
    origin = np.floor((in_base.min(axis=1) - _PAD) / _RES) * _RES
    size = np.maximum(np.ceil((in_base.max(axis=1) + _PAD - origin) / _RES).astype(int), 1)

    total = int(size[0] * size[1] * size[2])
    occupancy = np.zeros(total, dtype=np.uint8)
    for index in range(total):
        record = voxel_backing_record(
            model,
            data,
            voxel_index=index,
            grid_origin=[float(v) for v in origin],
            grid_resolution=_RES,
            grid_size=[int(v) for v in size],
            robot_body_ids=robot_bodies,
            base_frame_body=_BASE_FRAME_BODY,
            rays_per_axis=_RAYS_PER_AXIS,
        )
        if "solid_world" in (record.get("classes") or []):
            occupancy[index] = 1
    return origin, size, occupancy, int(occupancy.sum())


def _kernel_params(desc: RobotDescription, margin_m: float) -> dict[str, object]:
    """The parameters ``sim_e2e.launch.py`` emits for this robot, at ``margin_m``."""
    collision = collision_params_from_description(desc)
    params: dict[str, object] = dict(kernel_params_from_envelope(compute_intersection(desc, None)))
    params.update(collision)
    params.update(
        {
            "world_voxel_enabled": True,
            "world_voxel_margin_m": float(margin_m),
            "world_voxel_deadline_ms": 5000.0,
            "world_voxel_max_cells": 2_000_000,
            "collision_joint_names": [j.name for j in desc.joints],
            "collision_base_dofs": [
                i for i, j in enumerate(desc.joints) if j.name in set(desc.base_joints or [])
            ],
            "collision_ee_link_index": ee_link_index_from_collision_params(collision),
            "collision_seed_dt_s": 0.0,
            "collision_state_deadline_ms": 5000.0,
        }
    )
    return params


def _kernel_verdict(
    state: _StartState, *, margin_m: float, all_occupied: bool = False
) -> dict[str, object]:
    """Publish ``state`` to a real kernel and return what it did with a zero chunk."""
    import rclpy
    from geometry_msgs.msg import Point, Quaternion
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    desc = RobotDescription.from_yaml(_MANIFEST)
    node_name = f"fridge_pin_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(desc, margin_m), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("fridge_pin_helper")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"

            safe: dict[str, Any] = {}
            failures: list[Any] = []
            estops: list[Any] = []
            helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 50
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            cand_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            voxel_pub = helper.create_publisher(
                OccupancyVoxels,
                "/openral/world_voxels",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
            js_pub = helper.create_publisher(
                JointState,
                "/joint_states",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 5.0
            while time.time() < deadline and cand_pub.get_subscription_count() < 1:
                executor.spin_once(timeout_sec=0.05)

            js = JointState()
            js.name = state.joint_names
            js.position = list(state.positions)

            grid = OccupancyVoxels()
            grid.header.frame_id = "base_link"
            grid.origin = Point(
                x=float(state.origin[0]), y=float(state.origin[1]), z=float(state.origin[2])
            )
            # An oriented lattice whose unset orientation is the all-zero
            # quaternion, which consumers refuse rather than read as identity.
            grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            grid.resolution = _RES
            grid.size_x, grid.size_y, grid.size_z = (int(v) for v in state.size)
            cells = state.occupancy
            grid.occupancy = (
                [1] * int(cells.size) if all_occupied else [int(v) for v in cells.tolist()]
            )

            warm = time.time() + 1.0
            while time.time() < warm:
                js.header.stamp = helper.get_clock().now().to_msg()
                grid.header.stamp = js.header.stamp
                js_pub.publish(js)
                voxel_pub.publish(grid)
                executor.spin_once(timeout_sec=0.02)

            chunk = ActionChunk()
            chunk.control_mode = 5  # CARTESIAN_DELTA, the robocasa arm mode
            chunk.horizon = 1
            chunk.n_dof = 6
            chunk.flat = [0.0] * 6
            chunk.rskill_id = "openral/fridge-layout-pin"
            chunk.trace_id = "start-state"

            end = time.time() + 5.0
            while time.time() < end and not estops:
                js.header.stamp = helper.get_clock().now().to_msg()
                grid.header.stamp = js.header.stamp
                js_pub.publish(js)
                voxel_pub.publish(grid)
                cand_pub.publish(chunk)
                executor.spin_once(timeout_sec=0.02)

            return {
                "estopped": bool(estops),
                "passed": "start-state" in safe,
                "evidence": json.loads(failures[-1].evidence_json) if failures else None,
            }
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)


@pytest.fixture(scope="module")
def pinned() -> _StartState:
    """The shipped ``layout_ids: [47]`` start state."""
    return _StartState(_PIN_LAYOUT)


@pytest.fixture(scope="module")
def retired() -> _StartState:
    """The ``layout_ids: [30]`` start state the pin replaced."""
    return _StartState(_RETIRED_LAYOUT)


def test_the_scene_still_pins_the_layout_this_file_measures() -> None:
    """A pin move must fail here, not silently invalidate every number above."""
    declared = yaml.safe_load(_SCENE.read_text())["scene"]["backend_options"]["layout_ids"]
    assert declared == [_PIN_LAYOUT], (
        f"{_SCENE.name} now pins {declared!r}. This file's measurements are for "
        f"layout {_PIN_LAYOUT}; re-measure before changing the constant."
    )


def test_the_pinned_start_state_clears_the_kernel(pinned: _StartState) -> None:
    """Layout 47 at reset: the kernel passes a zero chunk, having seen the kitchen."""
    assert pinned.layout == _PIN_LAYOUT
    assert pinned.occupied > 1000, (
        f"only {pinned.occupied} occupied cells — the grid is too empty to have "
        f"posed the question; a pass here would be vacuous"
    )
    verdict = _kernel_verdict(pinned, margin_m=0.0)
    assert not verdict["estopped"], (
        f"the shipped pin E-stops before the robot is commanded: {verdict['evidence']}"
    )
    assert verdict["passed"], "the chunk neither E-stopped nor reached /openral/safe_action"


def test_the_pinned_start_state_has_more_clearance_than_the_one_it_replaced(
    pinned: _StartState, retired: _StartState
) -> None:
    """At a 20 mm standoff layout 30 trips and layout 47 does not.

    This is the pin's criterion as a test. The tripping link is asserted too:
    the census puts ``panda_link2`` against the fridge's lower housing on this
    layout, and a primitive change that moved the binding link elsewhere would
    invalidate the reasoning the pin rests on even if the ordering survived.
    """
    assert retired.layout == _RETIRED_LAYOUT

    tight = _kernel_verdict(retired, margin_m=_SEPARATING_MARGIN_M)
    assert tight["estopped"], (
        f"layout {_RETIRED_LAYOUT} no longer trips at a {_SEPARATING_MARGIN_M * 1000:.0f} mm "
        f"standoff — it has gained clearance, and the pin's basis needs re-measuring"
    )
    evidence = tight["evidence"]
    assert evidence is not None and evidence["collision_kind"] == "world"
    assert evidence["link_a"] == "panda_link2", (
        f"layout {_RETIRED_LAYOUT} now binds on {evidence['link_a']}, not panda_link2 "
        f"as the start-state census measured"
    )
    assert str(evidence["link_b_or_object"]).startswith("voxel_")

    loose = _kernel_verdict(pinned, margin_m=_SEPARATING_MARGIN_M)
    assert not loose["estopped"], (
        f"layout {_PIN_LAYOUT} now trips at the same standoff that trips layout "
        f"{_RETIRED_LAYOUT}; the pin no longer buys the clearance it was chosen for: "
        f"{loose['evidence']}"
    )


def test_an_all_occupied_grid_is_refused(pinned: _StartState) -> None:
    """The control. Every clearance assertion above is vacuous if this fails.

    Same configuration, same frame, same lattice — only the occupancy differs.
    A grid that never reached the kernel, or that landed somewhere the arm is
    not, would pass a zero chunk here too.
    """
    verdict = _kernel_verdict(pinned, margin_m=0.0, all_occupied=True)
    assert verdict["estopped"], (
        "an entirely occupied grid did not E-stop — the world-voxel check is not "
        "reading this grid, so the clearances measured in this file mean nothing"
    )
    evidence = verdict["evidence"]
    assert evidence is not None and evidence["collision_kind"] == "world"
    assert str(evidence["link_a"]).startswith("panda_link")
    # #197: the verdict must name the configuration it was reached at, and the
    # kernel must have zeroed the base DoFs before evaluating the arm.
    joints = evidence["joint_positions_rad"]
    assert len(joints) == len(pinned.joint_names)
    assert joints[:3] == [0, 0, 0], f"base DoFs not zeroed: {joints[:3]}"
