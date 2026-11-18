"""The robot console's Sim scene layer, HAL side (platform#133, #137, #139).

``write_sim_scene`` turns a ``scene_mesh(limit_bytes)`` reply (the Isaac sidecar's, or
``mujoco_scene_mesh`` on a live MuJoCo model) into ``/openral/viz/sim_scene``'s payload;
``SimAttachedHAL`` and ``MujocoArmHAL`` keep it as ``sim_scene`` once connected.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import RobotDescription
from openral_core.exceptions import ROSSafetyViolation

from openral_hal.mobile_base_bridge import describes_mobile_base
from openral_hal.ros_package_overlay import (
    SIM_SCENE_LIMIT_BYTES,
    SIM_SCENE_PACKAGE,
    sim_scene_package_root,
)

if TYPE_CHECKING:
    import mujoco

__all__ = ["mujoco_scene_mesh", "write_sim_scene"]

_WEEK_S = 7 * 24 * 3600.0


def write_sim_scene(scene_mesh: Callable[[int], dict[str, Any]]) -> dict[str, object]:
    """What ``/openral/viz/sim_scene`` carries: the static scene GLB's URI, or why not.

    Asks ``scene_mesh`` for the GLB at the env's reset pose and, once the reply checks
    out, writes it as ``<sha12>.glb`` into ``SIM_SCENE_PACKAGE`` (registered with
    foxglove_bridge by deploy_e2e). GLBs no deploy has written for a week are removed;
    a running deploy's file is never touched, since deploys may share the cache.
    ``{"uri", "bytes", "triangles"}`` or ``{"note"}``; an export failure is a note,
    never a failed connect: the layer is a viewer convenience (platform#133). A
    ``ROSSafetyViolation`` is never turned into a note. The export's summary line goes
    to structlog (``sim.scene_export``).
    """
    try:
        out = scene_mesh(SIM_SCENE_LIMIT_BYTES)
        structlog.get_logger(__name__).info("sim.scene_export", detail=out.get("log"))
        glb, triangles = out.get("glb"), out.get("triangles")
        if glb is None:
            return {"note": str(out.get("note") or "no static scene")}
        if not isinstance(glb, bytes) or type(triangles) is not int or triangles < 0:
            return {"note": "export failed: the sidecar sent a malformed scene"}
        name = f"{hashlib.sha256(glb).hexdigest()[:12]}.glb"
        root = sim_scene_package_root()
        now = time.time()
        for old in root.glob("*.glb"):
            if old.name != name and now - old.stat().st_mtime > _WEEK_S:
                old.unlink(missing_ok=True)  # ponytail: age-based; per-deploy dirs if needed
        dest = root / name
        if not dest.exists():
            tmp = root / f".{name}.{os.getpid()}.tmp"
            tmp.write_bytes(glb)
            tmp.replace(dest)
        os.utime(dest)  # in use again: keeps it out of the week-old sweep
    except ROSSafetyViolation:
        raise
    except Exception as exc:  # reason: viz only, see docstring
        return {"note": f"export failed: {type(exc).__name__}: {exc}"}
    uri = f"package://{SIM_SCENE_PACKAGE}/{name}"
    return {"uri": uri, "bytes": len(glb), "triangles": triangles}


def mujoco_scene_mesh(
    model: mujoco.MjModel, data: mujoco.MjData, description: RobotDescription, limit_bytes: int
) -> dict[str, Any]:
    """``scene_mesh`` for a live MuJoCo model (platform#139), in ``/openral/viz/sim_scene``'s frame.

    ``tools/scene_glb.py``'s ``mujoco_static_scene_glb``, found above this package. A
    mobile base's frame is its odometry frame, the MuJoCo world (``base_pose_6dof``
    publishes the base's world pose as ``odom → base_link``); a fixed arm's is its base
    frame: the MJCF body of that name, else the root body of the robot's actuated joints,
    else the world.
    """
    import mujoco  # noqa: PLC0415  # reason: sim only

    here = Path(__file__).resolve()
    found = (p / "tools" / "scene_glb.py" for p in here.parents)
    path = next((f for f in found if f.is_file()), None)
    spec = importlib.util.spec_from_file_location("scene_glb", path) if path else None
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no tools/scene_glb.py above {here}")
    scene_glb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene_glb)

    frame_from_world: NDArray[np.float64] = np.eye(4)
    if not describes_mobile_base(description):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, description.base_frame)
        if body < 0:
            joint_kinds = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
            roots = {
                int(model.body_rootid[int(model.jnt_bodyid[int(model.actuator_trnid[a, 0])])])
                for a in range(model.nu)
                if int(model.actuator_trntype[a]) in joint_kinds
            }
            body = roots.pop() if len(roots) == 1 else 0
        world_from_base = np.eye(4)
        world_from_base[:3, :3] = np.asarray(data.xmat[body]).reshape(3, 3)
        world_from_base[:3, 3] = data.xpos[body]
        frame_from_world = np.asarray(np.linalg.inv(world_from_base), dtype=np.float64)
    out: dict[str, Any] = scene_glb.mujoco_static_scene_glb(
        model, data, frame_from_world, limit_bytes
    )
    return out
