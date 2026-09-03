"""#188 A/B — graded velocity scaling proved on the REAL kernel + REAL manifest.

The unit gtests pin the arithmetic on a synthetic one-link arm. This is the
question they cannot answer: *does the band do anything useful on the robot and
the control mode the stops actually came from?*

Setup is the deploy graph's, not a fixture's — the real ``safety_kernel_node``
binary, configured from ``robots/panda_mobile/robot.yaml`` exactly as
``sim_e2e.launch.py`` configures it (envelope + collision model +
``collision_base_dofs`` + ``collision_joint_names`` + ``collision_ee_link_index``),
driven with ``CARTESIAN_DELTA`` chunks, which is the robocasa arm mode.

A wall of occupied voxels is stepped toward the arm. The same approach is run
twice against the same geometry, and the only thing that differs is the band:

* **band armed** — the chunk stays ACCEPTED and is republished progressively
  slower as the wall closes.
* **band disabled** (``collision_scale_proximity_m: 0``) — every accepted chunk
  is republished byte-identical, which is the pre-#188 behaviour and the
  rollback contract.

Anything that differs between the two runs is the band and nothing else.

Writes ``outputs/graded-scaling/approach.mp4`` (and the per-step CSV beside it)
when ``imageio`` is importable, so the behaviour can be watched rather than
inferred from assertions.

CLAUDE.md §1.11 — real kernel binary, real manifest, real DDS, no mocks.
Gates: ROS_DISTRO + rclpy + openral_msgs on a sourced + colcon-built workspace.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import time
import uuid

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from openral_core import RobotDescription  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    ee_link_index_from_collision_params,
    kernel_params_from_envelope,
)

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_MANIFEST = "robots/panda_mobile/robot.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Base-relative grid, fine enough that the wall can be stepped in centimetres
# rather than in 25 mm voxel jumps.
_RES = 0.02
_ORIGIN = (-0.2, -0.6, -0.2)
_SX, _SY, _SZ = 50, 60, 70

#: Band under test. 0.10 m at k=20 reaches exp(-2)=0.135 at the margin, which
#: is above the 0.05 floor — so these numbers exercise the exponential itself,
#: not the clamp.
_BAND_M = 0.10
_BAND_K = 20.0
_BAND_FLOOR = 0.05

#: Wall positions, far → near, in metres along +x (base-relative).
#: Measured against this manifest: the home arm's forward-most collision
#: surface sits at x ≈ 0.2025 m in `base_link`, so 0.34 is clear of a 0.10 m
#: band and 0.22 is 17.5 mm from contact. The sweep deliberately stops short of
#: the latch at 0.20 — a latch is a different mechanism and has its own test.
_WALL_SWEEP = [0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22]

#: The commanded EE delta. Well inside any per-axis range and NOT normalized
#: (no `cartesian_delta_scale`), so published/commanded is a clean ratio.
_DELTA = 0.02


def _kernel_params(*, band_m: float) -> dict[str, object]:
    """Kernel params as `sim_e2e.launch.py` emits them, plus the #188 band."""
    desc = RobotDescription.from_yaml(_MANIFEST)
    collision = collision_params_from_description(desc)
    params: dict[str, object] = dict(kernel_params_from_envelope(compute_intersection(desc, None)))
    params.update(collision)
    params.update(
        {
            "world_voxel_enabled": True,
            "world_voxel_margin_m": 0.0,
            "world_voxel_deadline_ms": 5000.0,
            "world_voxel_max_cells": _SX * _SY * _SZ,
            "collision_joint_names": [j.name for j in desc.joints],
            "collision_base_dofs": [
                i for i, j in enumerate(desc.joints) if j.name in set(desc.base_joints or [])
            ],
            "collision_ee_link_index": ee_link_index_from_collision_params(collision),
            "collision_seed_dt_s": 0.0,
            "collision_state_deadline_ms": 5000.0,
            "collision_scale_proximity_m": band_m,
            "collision_scale_k": _BAND_K,
            "collision_scale_min": _BAND_FLOOR,
        }
    )
    return params


def _wall_grid(wall_x: float):  # ROS message type, imported lazily
    """An occupancy grid holding a thin wall slab whose near face is ``wall_x``.

    A slab rather than a filled half-space: geometrically identical for a
    distance test, and it keeps the occupied-cell count (and so the per-chunk
    check) flat as the resolution is refined.
    """
    from geometry_msgs.msg import Point, Quaternion
    from openral_msgs.msg import OccupancyVoxels

    grid = OccupancyVoxels()
    grid.header.frame_id = "base_link"
    grid.origin = Point(x=_ORIGIN[0], y=_ORIGIN[1], z=_ORIGIN[2])
    # A synthetic base-aligned lattice: the unset orientation is the all-zero
    # quaternion, which the kernel refuses rather than reading as identity.
    grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    grid.resolution = _RES
    grid.size_x, grid.size_y, grid.size_z = _SX, _SY, _SZ
    occ = bytearray(_SX * _SY * _SZ)
    for ix in range(_SX):
        cx = _ORIGIN[0] + (ix + 0.5) * _RES
        if not (wall_x <= cx < wall_x + 3.0 * _RES):
            continue
        for iy in range(_SY):
            for iz in range(_SZ):
                occ[ix + _SX * (iy + _SY * iz)] = 1
    grid.occupancy = list(occ)
    return grid


def _run_approach(*, band_m: float) -> list[dict[str, float]]:
    """Step the wall toward the arm; return one record per wall position."""
    import rclpy
    from openral_msgs.msg import ActionChunk, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    desc = RobotDescription.from_yaml(_MANIFEST)
    joint_names = [j.name for j in desc.joints]
    base_dofs = {jn for jn in (desc.base_joints or [])}

    node_name = f"safety_kernel_scale_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(band_m=band_m), node_name, isolated_domain_id())
    records: list[dict[str, float]] = []
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("graded_scaling_helper")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"

            safe: dict[str, ActionChunk] = {}
            estops: list[Empty] = []
            helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
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

            # Base parked away from the origin; arm at home. The kernel zeroes
            # the base dofs, so the arm is evaluated in the frame the grid is in.
            #
            # `panda_joint6` is held at the SRDF's own `ready` value rather than
            # zero: at an all-zero arm the Panda's link5 and link7 really
            # interpenetrate, by 5.65 mm at their own collision meshes, which
            # issue #191 made visible by retiring their ACM exemption. A real
            # self-collision would latch the kernel on step 1 and this test would
            # measure nothing.
            js = JointState()
            js.name = joint_names
            js.position = [
                5.0 if n in base_dofs else (1.571 if n == "panda_joint6" else 0.0)
                for n in joint_names
            ]

            for step, wall_x in enumerate(_WALL_SWEEP):
                grid = _wall_grid(wall_x)
                chunk = ActionChunk()
                chunk.control_mode = 5  # CARTESIAN_DELTA — the robocasa arm mode
                chunk.horizon = 1
                chunk.n_dof = 6
                chunk.flat = [_DELTA, 0.0, 0.0, 0.0, 0.0, 0.0]
                chunk.rskill_id = "openral/graded-scaling-approach"
                chunk.trace_id = f"wall{step}"

                # Land the seed + this wall position before the candidate: a
                # Cartesian chunk without a fresh state is refused fail-closed.
                warm = time.time() + 0.6
                while time.time() < warm:
                    now = helper.get_clock().now().to_msg()
                    js.header.stamp = now
                    grid.header.stamp = now
                    js_pub.publish(js)
                    voxel_pub.publish(grid)
                    executor.spin_once(timeout_sec=0.02)

                end = time.time() + 2.0
                while time.time() < end and chunk.trace_id not in safe and not estops:
                    now = helper.get_clock().now().to_msg()
                    js.header.stamp = now
                    grid.header.stamp = now
                    js_pub.publish(js)
                    voxel_pub.publish(grid)
                    cand_pub.publish(chunk)
                    executor.spin_once(timeout_sec=0.02)

                published = safe.get(chunk.trace_id)
                records.append(
                    {
                        "wall_x": wall_x,
                        "commanded": _DELTA,
                        "published": float(published.flat[0]) if published else float("nan"),
                        "scale": float(published.flat[0]) / _DELTA if published else float("nan"),
                        "accepted": 1.0 if published else 0.0,
                        "estopped": 1.0 if estops else 0.0,
                    }
                )
                if estops:
                    break
            return records
        finally:
            with contextlib.suppress(Exception):
                rclpy.shutdown()
    finally:
        terminate_kernel(proc)


def _write_video(armed: list[dict[str, float]], disabled: list[dict[str, float]]) -> Path | None:
    """Render the two approaches side by side as an mp4. Best-effort."""
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional viz dependency
        return None

    out_dir = _REPO_ROOT / "outputs" / "graded-scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "approach.mp4"

    frames = []
    for i in range(len(armed)):
        fig, (ax_scene, ax_plot) = plt.subplots(
            2, 1, figsize=(6.4, 6.4), height_ratios=[1, 1.1], dpi=100
        )
        wall = armed[i]["wall_x"]
        # Top: a schematic top-down view of the arm reach and the wall.
        ax_scene.axvspan(wall, 1.2, color="#B33A1F", alpha=0.35, label="occupied voxels")
        ax_scene.axvline(wall, color="#B33A1F", lw=2)
        ax_scene.plot([0.0], [0.0], marker="s", ms=14, color="#20242A", label="base_link")
        ax_scene.annotate(
            "",
            xy=(0.55, 0.0),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-", "lw": 8, "color": "#16697A"},
        )
        scale = armed[i]["scale"]
        ax_scene.annotate(
            "",
            xy=(0.55 + 4.0 * armed[i]["published"], 0.0),
            xytext=(0.55, 0.0),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": "#2E7D4F"},
        )
        ax_scene.set_xlim(-0.2, 1.2)
        ax_scene.set_ylim(-0.4, 0.4)
        ax_scene.set_yticks([])
        ax_scene.set_xlabel("x, base_link frame (m)")
        ax_scene.set_title(
            f"wall at {wall:.2f} m   ·   commanded {_DELTA:.3f}   ·   "
            f"published {armed[i]['published']:.4f}   ·   scale {scale:.3f}",
            fontsize=10,
        )
        ax_scene.legend(loc="upper left", fontsize=8)

        # Bottom: both runs' scale over the approach, cursor at the current step.
        xs = [r["wall_x"] for r in armed]
        ax_plot.plot(xs, [r["scale"] for r in armed], "-o", color="#16697A", label="band armed")
        ax_plot.plot(
            [r["wall_x"] for r in disabled],
            [r["scale"] for r in disabled],
            "-s",
            color="#9AA1AC",
            label="band disabled (rollback)",
        )
        ax_plot.axvline(wall, color="#B33A1F", ls="--", lw=1)
        ax_plot.set_xlim(max(xs) + 0.03, min(xs) - 0.03)  # near on the right
        ax_plot.set_ylim(0.0, 1.1)
        ax_plot.set_xlabel("wall distance from base_link (m)  →  closing")
        ax_plot.set_ylabel("published / commanded")
        ax_plot.grid(alpha=0.25)
        ax_plot.legend(loc="lower left", fontsize=8)
        fig.suptitle("#188 graded velocity scaling — real kernel, panda_mobile", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(path, frames, fps=2, macro_block_size=1)
    (out_dir / "approach.csv").write_text(
        "wall_x,commanded,published,scale,accepted,estopped\n"
        + "\n".join(
            f"{r['wall_x']},{r['commanded']},{r['published']},{r['scale']},"
            f"{r['accepted']},{r['estopped']}"
            for r in armed
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_graded_scaling_slows_a_real_approach_and_rolls_back_exactly() -> None:
    """The band slows an accepted approach; disabling it republishes verbatim."""
    armed = _run_approach(band_m=_BAND_M)
    disabled = _run_approach(band_m=0.0)

    assert armed, "the armed run produced no records"
    assert len(armed) == len(disabled) == len(_WALL_SWEEP), (
        "neither run may latch — the wall sweep stops short of contact; "
        f"armed={len(armed)} disabled={len(disabled)}"
    )
    assert all(r["accepted"] == 1.0 for r in armed), (
        "the band must never turn an accept into a drop"
    )
    assert all(r["estopped"] == 0.0 for r in armed), "slowing is not stopping"

    # Rollback: every chunk republished byte-identical with the band off.
    for record in disabled:
        assert record["published"] == pytest.approx(_DELTA, abs=0.0), (
            "collision_scale_proximity_m=0 must republish the chunk verbatim, "
            f"got {record['published']!r}"
        )

    # The band did something, and it did it in the right direction.
    scales = [r["scale"] for r in armed]
    assert min(scales) < 0.999, (
        f"the wall closes to within the band — at least one chunk must be slowed; scales={scales}"
    )
    assert scales[0] == pytest.approx(1.0), "the far end of the sweep is outside the band"
    # Monotone non-increasing as the wall closes: a scale that jitters up as the
    # obstacle gets nearer would be the chattering the band is watched for.
    for prev, nxt in itertools.pairwise(scales):
        assert nxt <= prev + 1e-9, f"scale rose while the wall closed: {scales}"
    assert min(scales) >= _BAND_FLOOR - 1e-9, "the floor must bound the slowdown"

    video = _write_video(armed, disabled)
    if video is not None:
        print(f"\ngraded-scaling approach video: {video}")
    print("\nwall_x  published  scale")
    for r in armed:
        print(f"{r['wall_x']:.2f}   {r['published']:.5f}   {r['scale']:.4f}")
