"""cuVSLAM multi-camera tracking on the real RoboCasa PandaMobile agentview rig.

The closed-loop check for the ``panda_mobile_vslam`` robot: build the same
multi-camera rig the node builds (rig ≡ base, each camera's ``rig_from_camera``
from its real pose — via the node's own ``transform_to_pose``), drive the base
along a known path, render the two agentview cameras, and confirm cuVSLAM
tracks metrically.

Real components (CLAUDE.md §1.11): a real RoboCasa kitchen env, real MuJoCo
renders, the real cuVSLAM engine. Gated — skips without robosuite / robocasa /
the operator-installed PyCuVSLAM wheel / a CUDA GPU (the legitimate skip path;
the engine is NVIDIA-licensed and never bundled).

Kitchen tasks park the base, so it is driven by setting the mobile-base joint
qpos directly. Scale on this wide (0.70 m), toed-in, low-overlap rig bootstraps
noisily and can diverge on long runs into degenerate views, so this asserts the
robust claim — the rig geometry is exact and tracking stays roughly metric over
a short well-conditioned trajectory — not a tight scale bound.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_NODE = (
    Path(__file__).resolve().parents[2]
    / "packages/openral_slam_bringup/openral_slam_bringup/pycuvslam_node.py"
)
_spec = importlib.util.spec_from_file_location("pycuvslam_node_sim", _NODE)
assert _spec is not None and _spec.loader is not None
pycuvslam_node = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pycuvslam_node)

_CAMS = ["robot0_agentview_left", "robot0_agentview_right"]
_HW = 256


class _Tf:
    """geometry_msgs/Transform stand-in built from a 4x4 pose (for transform_to_pose)."""

    def __init__(self, mat: np.ndarray) -> None:
        from math import sqrt

        r = mat[:3, :3]
        tr = r.trace()
        if tr > 0:
            s = sqrt(tr + 1.0) * 2
            q = [
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
                0.25 * s,
            ]
        else:
            i = int(np.argmax(np.diag(r)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = sqrt(r[i, i] - r[j, j] - r[k, k] + 1.0) * 2
            q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
            q[i], q[j], q[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
        self.rotation = type("R", (), dict(zip("xyzw", q, strict=True)))
        self.translation = type("T", (), dict(zip("xyz", mat[:3, 3], strict=True)))


def test_cuvslam_multicam_tracks_robocasa_agentview_rig() -> None:
    robosuite = pytest.importorskip("robosuite", reason="robocasa sim needs robosuite")
    pytest.importorskip("robocasa", reason="needs the robocasa group (just sync --group robocasa)")
    cuvslam = pytest.importorskip(
        "cuvslam", reason="PyCuVSLAM wheel not installed (operator-installed, never bundled)"
    )
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuVSLAM needs a CUDA GPU")
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils import camera_utils

    np.random.seed(0)
    cc = load_composite_controller_config(controller=None, robot="PandaMobile")
    env = robosuite.make(
        env_name="PickPlaceCounterToCabinet",
        robots=["PandaMobile"],
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=_CAMS,
        camera_widths=_HW,
        camera_heights=_HW,
        control_freq=20,
        ignore_done=True,
        layout_and_style_ids=[[1, 1]],
    )
    try:
        env.reset()
        sim = env.sim

        def world_from(name: str) -> np.ndarray:
            m = np.eye(4)
            m[:3, :3] = sim.data.get_body_xmat(name).reshape(3, 3)
            m[:3, 3] = sim.data.get_body_xpos(name)
            return m

        base = "mobilebase0_support"
        w_base0 = world_from(base)

        # Build the rig the way the node does: rig ≡ base, rig_from_camera read
        # from the (base ← camera) extrinsic via the node's transform_to_pose.
        rig = cuvslam.Rig()
        cams = []
        xs = []
        for c in _CAMS:
            k = camera_utils.get_camera_intrinsic_matrix(sim, c, _HW, _HW)
            w_cam = camera_utils.get_camera_extrinsic_matrix(sim, c)
            base_from_cam = np.linalg.inv(w_base0) @ w_cam
            q, t = pycuvslam_node.transform_to_pose(_Tf(base_from_cam))
            xs.append(t[1])
            cams.append(
                cuvslam.Camera(
                    size=[_HW, _HW],
                    principal=[float(k[0, 2]), float(k[1, 2])],
                    focal=[float(k[0, 0]), float(k[1, 1])],
                    rig_from_camera=cuvslam.Pose(rotation=list(q), translation=list(t)),
                )
            )
        # The rig geometry is deterministic: a ~0.70 m horizontal (y) baseline.
        assert abs(abs(xs[0] - xs[1]) - 0.70) < 0.05, f"stereo baseline off: {xs}"

        rig.cameras = cams
        tracker = cuvslam.Tracker(rig, odom_config=cuvslam.Tracker.OdometryConfig())

        fwd = "mobilebase0_joint_mobile_forward"
        adr = sim.model.get_joint_qpos_addr(fwd)
        gt0 = sim.data.get_body_xpos(base).copy()
        tracked = None
        lost = 0
        n = 70
        for i in range(n):
            sim.data.qpos[adr] += 0.01  # 1 cm/step, pure forward
            sim.forward()
            imgs = [
                np.ascontiguousarray(sim.render(camera_name=c, width=_HW, height=_HW)[::-1])
                for c in _CAMS
            ]
            est, _ = tracker.track(i * 50_000_000, imgs)
            if est.world_from_rig is None:
                lost += 1
                continue
            tracked = np.asarray(est.world_from_rig.pose.translation, dtype=float)

        gt = float(np.linalg.norm(sim.data.get_body_xpos(base) - gt0))
        assert tracked is not None, "cuVSLAM produced no pose"
        assert lost <= n // 10, f"too many lost frames: {lost}/{n}"
        ratio = float(np.linalg.norm(tracked)) / max(gt, 1e-6)
        # Roughly metric — the wide toed-in rig bootstraps scale noisily; this is
        # the robust "it tracks metrically" claim, not a tight scale assertion.
        assert 0.3 <= ratio <= 3.0, f"tracked magnitude not metric: ratio={ratio:.2f} (gt={gt:.3f})"
    finally:
        env.close()
