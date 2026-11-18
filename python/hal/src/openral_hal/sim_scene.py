"""The robot console's Sim scene layer, HAL side (platform#133, #137, #139).

``write_sim_scene`` turns a ``scene_mesh(limit_bytes)`` reply (the Isaac sidecar's, or
``mujoco_scene_mesh`` on a live MuJoCo model) into ``/openral/viz/sim_scene``'s payload.
``SimAttachedHAL`` and ``MujocoArmHAL`` expose it as ``export_sim_scene()``, which the
lifecycle node calls when it latches the topic (configure, and after ``reset_episode``),
so nothing is exported outside a deploy. ``SceneCache`` keeps one export per scene
source: a RoboCasa reset that composes a new kitchen (a new ``MjModel``) re-exports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
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

__all__ = ["SceneCache", "mujoco_scene_mesh", "odom_from_base", "write_sim_scene"]

_WEEK_S = 7 * 24 * 3600.0
_FAILED = "export failed"
_scene_glb: ModuleType | None = None


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

    Example:
        >>> write_sim_scene(lambda limit_bytes: {"note": "the scene has no static geometry"})
        {'note': 'the scene has no static geometry'}
    """
    try:
        out = scene_mesh(SIM_SCENE_LIMIT_BYTES)
        structlog.get_logger(__name__).info("sim.scene_export", detail=out.get("log"))
        glb, triangles = out.get("glb"), out.get("triangles")
        if glb is None:
            return {"note": str(out.get("note") or "no static scene")}
        if not isinstance(glb, bytes) or type(triangles) is not int or triangles < 0:
            return {"note": f"{_FAILED}: the exporter sent a malformed scene"}
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
        return {"note": f"{_FAILED}: {type(exc).__name__}: {exc}"}
    uri = f"package://{SIM_SCENE_PACKAGE}/{name}"
    return {"uri": uri, "bytes": len(glb), "triangles": triangles}


class SceneCache:
    """One ``write_sim_scene`` result per scene source, re-exported when the source changes.

    ``key`` identifies the static scene (the live ``MjModel``, or one constant for an
    Isaac stage); a failed export is retried on the next call.

    Example:
        >>> calls = []
        >>> cache = SceneCache()
        >>> mesh = lambda limit_bytes: calls.append(1) or {"note": "nothing static"}
        >>> cache.get("kitchen-12", mesh) == cache.get("kitchen-12", mesh)
        True
        >>> _ = cache.get("kitchen-7", mesh)
        >>> len(calls)
        2
    """

    def __init__(self) -> None:
        """An empty cache: the first ``get`` exports."""
        self._key: object = None
        self._scene: dict[str, object] | None = None

    def get(self, key: object, scene_mesh: Callable[[int], dict[str, Any]]) -> dict[str, object]:
        """The cached payload for ``key``, exporting through ``scene_mesh`` when it is stale."""
        failed = self._scene is not None and str(self._scene.get("note", "")).startswith(_FAILED)
        if self._scene is None or failed or self._key is not key:
            self._key, self._scene = key, write_sim_scene(scene_mesh)
        return self._scene


def odom_from_base(hal: Any) -> NDArray[np.float64]:  # noqa: ANN401  # reason: any HAL, duck-typed like MobileBaseBridge
    """The base frame's pose in ``/openral/viz/sim_scene``'s frame, as ``MobileBaseBridge`` sees it.

    Identity for a fixed base (the scene is in ``base_frame``); for a mobile base, the
    ``odom → base`` pose the bridge publishes: ``base_pose_6dof()`` when the HAL has one,
    else the planar ``base_pose``.

    Example:
        >>> from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION
        >>> class Base:
        ...     description = PANDA_MOBILE_DESCRIPTION
        ...     base_pose = (1.0, 2.0, 0.0)
        >>> odom_from_base(Base())[:3, 3].tolist()
        [1.0, 2.0, 0.0]
    """
    out = np.eye(4)
    if not describes_mobile_base(hal.description):
        return out
    getter = getattr(hal, "base_pose_6dof", None)
    pose = getter() if callable(getter) else None
    if pose is not None:
        (px, py, pz), (qx, qy, qz, qw) = pose
        out[:3, :3] = _quat_xyzw_matrix(qx, qy, qz, qw)
        out[:3, 3] = (px, py, pz)
        return out
    x, y, yaw = hal.base_pose
    c, s = np.cos(yaw), np.sin(yaw)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:2, 3] = (x, y)
    return out


def _quat_xyzw_matrix(x: float, y: float, z: float, w: float) -> NDArray[np.float64]:
    n = float(np.sqrt(x * x + y * y + z * z + w * w)) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _load_scene_glb() -> ModuleType:
    """``tools/scene_glb.py`` (found above this package), loaded once per process."""
    global _scene_glb  # noqa: PLW0603  # reason: a read-only module, loaded once
    if _scene_glb is None:
        here = Path(__file__).resolve()
        found = (p / "tools" / "scene_glb.py" for p in here.parents)
        path = next((f for f in found if f.is_file()), None)
        spec = importlib.util.spec_from_file_location("scene_glb", path) if path else None
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"no tools/scene_glb.py above {here}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _scene_glb = module
    return _scene_glb


def mujoco_scene_mesh(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    description: RobotDescription,
    limit_bytes: int,
    odom_from_base_link: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """``scene_mesh`` for a live MuJoCo model (platform#139), in ``/openral/viz/sim_scene``'s frame.

    ``tools/scene_glb.py``'s ``mujoco_static_scene_glb``. The base body is the one the
    sim's ``world → base_frame`` TF is measured on (``resolve_base_frame_body_name``),
    else the root body of the robot's actuated joints. The scene lands in
    ``odom_from_base_link · base_from_world`` (``odom_from_base``; identity, the default,
    for a fixed arm's ``base_frame``); with no base body it stays in the MuJoCo world.

    Example:
        >>> import mujoco
        >>> from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION
        >>> m = mujoco.MjModel.from_xml_string(
        ...     "<mujoco><worldbody><geom type='box' size='1 1 1'/></worldbody></mujoco>"
        ... )
        >>> d = mujoco.MjData(m)
        >>> mujoco.mj_forward(m, d)
        >>> mujoco_scene_mesh(m, d, PANDA_MOBILE_DESCRIPTION, 10_000_000)["triangles"]
        12
    """
    import mujoco  # noqa: PLC0415  # reason: sim only

    from openral_hal.depth_cloud import (  # noqa: PLC0415  # reason: sim only
        resolve_base_frame_body_name,
    )

    name = resolve_base_frame_body_name(model, description=description)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
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
    odom = np.eye(4) if odom_from_base_link is None or body == 0 else odom_from_base_link
    frame_from_world = np.asarray(odom @ np.linalg.inv(world_from_base), dtype=np.float64)
    out: dict[str, Any] = _load_scene_glb().mujoco_static_scene_glb(
        model, data, frame_from_world, limit_bytes
    )
    return out
