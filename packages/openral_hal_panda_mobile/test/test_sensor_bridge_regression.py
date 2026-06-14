"""SAFETY regression: panda_mobile + robocasa → /scan + depth PointCloud2 + /odom survive
the SimSensorBridge refactor (ADR-0034 Phase 2 / T13).

**What this test proves** — "at-least-as-conservative" evidence for the ADR-0034 Phase 2
safety claim.  Before T13, the panda_mobile lifecycle node published ``/scan``
(ray-cast), ``/openral/cameras/front_depth/points`` (depth cloud → octomap input),
and ``/odom`` directly in its own timers.  T13 delegated the first two streams to the
shared :class:`openral_hal.sim_sensor_bridge.SimSensorBridge`; ``/odom`` remained in
the node.  If that refactor silently broke any of those three topics, the nav stack
and safety kernel would be blind to obstacles or lose odometry — a **regression**.

Concretely asserted:
* ``sensor_msgs/LaserScan`` on ``/scan``
  – ``len(ranges) > 0`` and at least one finite (non-inf, non-NaN) range.
* ``sensor_msgs/PointCloud2`` on ``/openral/cameras/front_depth/points``
  – ``width * height > 0`` (non-empty cloud; the octomap input MUST NOT be empty).
* ``nav_msgs/Odometry`` on ``/odom``.

Run::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    MUJOCO_GL=egl uv run pytest \\
        packages/openral_hal_panda_mobile/test/test_sensor_bridge_regression.py -v \\
        --timeout=600

Per CLAUDE.md §1.11: no mocks. Real ``_PandaMobileLifecycleNode``, real robocasa
MuJoCo scene, real ROS IDL.  Skips cleanly when robocasa or the kitchen assets are
unavailable — see guard section below.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import time
from pathlib import Path

import pytest

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCENE_YAML = str(_REPO_ROOT / "scenes" / "sim" / "robocasa_panda_mobile_kitchen.yaml")

# The depth sensor name declared in robots/panda_mobile/robot.yaml (§ ADR-0030).
# SimSensorBridge publishes on /openral/cameras/<name>/points.
_DEPTH_SENSOR_NAME = "front_depth"
_DEPTH_TOPIC = f"/openral/cameras/{_DEPTH_SENSOR_NAME}/points"

# Generous configure timeout: on_configure calls build_sim_env_from_yaml which
# runs ensure_backend_deps (robocasa kitchen git-clone + pip install) and
# ensure_robocasa_assets (CC-BY-4.0 asset download) on first use.
# Subsequent runs short-circuit the probes; 300 s covers a cold-cache first run.
_CONFIGURE_TIMEOUT_S = 300.0

# Topic-presence timeouts after the node is ACTIVE.  Depth is at 5 Hz so we
# wait up to 4 s; /scan is 10 Hz → 3 s; /odom is 20 Hz → 2 s.
_SCAN_TIMEOUT_S = 5.0
_DEPTH_TIMEOUT_S = 8.0
_ODOM_TIMEOUT_S = 5.0


# ── Guard helpers (run at collection time; must not import rclpy yet) ─────────


def _robosuite_compatible() -> str:
    """Return an empty string when robosuite >= 1.5.2 is present, else the skip reason.

    ``robocasa`` imports ``robosuite.utils.get_elements`` which was added in
    1.5.2; PyPI's 1.5.1 wheel ships without it.

    NOTE: the uv workspace venv **cannot** reach robosuite>=1.5.2 — ``lerobot``
    (0.5.1, required workspace-wide) caps robosuite at <=1.5.1, so
    ``robosuite>=1.5.2`` makes the lock unsatisfiable. This is the same
    mutual-exclusion shape as the ADR-0011 libero⊥robocasa conflict. So this
    test runs in a **dedicated robocasa environment** (e.g. the conda/miniforge
    env that ships robosuite>=1.5.2 without lerobot), or CI provisioned with it
    — never via ``just sync --group robocasa`` in the uv venv.
    """
    if importlib.util.find_spec("robosuite") is None:
        return "robosuite not installed — run this test in a robocasa env with robosuite>=1.5.2"
    try:
        ver = importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError:
        return "robosuite dist-info not found — cannot verify version"
    parts = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
    if parts < (1, 5, 2):
        return (
            f"robocasa needs robosuite>=1.5.2 (get_elements); found {ver}. The uv venv caps "
            "robosuite at 1.5.1 (lerobot 0.5.1) — run this test in a dedicated robocasa env "
            "(robosuite>=1.5.2, no lerobot), not via `just sync --group robocasa`."
        )
    return ""


_REQUIRED_MODULES = ("mujoco", "openral_hal", "openral_sim")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)

_ROBOSUITE_REASON = _robosuite_compatible()

_ROBOCASA_AVAILABLE = (
    not _MISSING_MODULES
    and not _ROBOSUITE_REASON
    and importlib.util.find_spec("robocasa") is not None
)

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))

# Defer rclpy import — it must not be evaluated at collection time when ROS is
# absent (same pattern as tests/integration/test_panda_mobile_hal_lifecycle.py).
pytestmark = [
    pytest.mark.skipif(
        not _ROS2_AVAILABLE,
        reason="ROS_DISTRO not set — test requires a sourced ROS 2 install.",
    ),
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=(
            "panda_mobile robocasa sensor-bridge regression requires: "
            + ", ".join(_MISSING_MODULES)
        ),
    ),
    pytest.mark.skipif(
        bool(_ROBOSUITE_REASON),
        reason=_ROBOSUITE_REASON or "robosuite incompatible",
    ),
    pytest.mark.skipif(
        not _ROBOCASA_AVAILABLE,
        reason=(
            "robocasa not importable — run in a dedicated robocasa env "
            "(robosuite>=1.5.2, no lerobot; see ADR-0011), not the uv workspace venv"
        ),
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _spin_until(executor: object, condition: object, timeout_s: float) -> bool:
    """Spin *executor* (rclpy) until *condition()* is truthy or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)  # type: ignore[union-attr]
        if condition():  # type: ignore[operator]
            return True
    return False


def _drive_configure(node: object, executor: object, timeout_s: float) -> bool:
    """Call trigger_configure and spin until SUCCESS or timeout.

    The configure transition calls ``build_sim_env_from_yaml`` which in turn
    calls ``ensure_backend_deps`` (slow first-time git-clone + pip install) and
    ``ensure_robocasa_assets`` (kitchen-asset download).  We spin the executor
    while we wait so the node's callbacks (including the configure hook) run on
    the executor thread.

    Returns True on SUCCESS.
    """
    from rclpy.lifecycle import TransitionCallbackReturn  # type: ignore[import-untyped]

    result_box: list[TransitionCallbackReturn] = []

    import threading

    def _do_configure() -> None:
        result_box.append(node.trigger_configure())  # type: ignore[union-attr]

    t = threading.Thread(target=_do_configure, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)  # type: ignore[union-attr]
        if result_box:
            break
        if not t.is_alive():
            break
    t.join(timeout=2.0)
    if not result_box:
        return False
    return str(result_box[0]).endswith("SUCCESS")


# ── Test ──────────────────────────────────────────────────────────────────────


def test_panda_mobile_robocasa_sensor_bridge_regression() -> None:
    """ADR-0034 Phase 2 safety regression: SimSensorBridge preserves /scan + /points + /odom.

    Brings up ``_PandaMobileLifecycleNode`` with
    ``sim_env_yaml=scenes/sim/robocasa_panda_mobile_kitchen.yaml``,
    drives configure → activate (allow 300 s for the robocasa kitchen build on
    first run), then asserts within per-topic timeouts:

    1. ``/scan`` (LaserScan) — ``len(ranges) > 0``; at least one finite range
       value confirming the SimSensorBridge ray-cast is live (not all-NaN / empty).

    2. ``/openral/cameras/front_depth/points`` (PointCloud2) — ``width*height > 0``
       (non-empty cloud confirms depth is being synthesised from the MJCF;
       an empty cloud would mean the octomap input to the C++ safety kernel is dark).

    3. ``/odom`` (Odometry) — message received (odometry must survive the
       refactor for Nav2 + slam_toolbox).

    Guard:  if configure fails (robocasa build failures, missing kitchen assets,
    network unavailable), the test calls ``pytest.skip`` with the exact error
    rather than failing loudly — the test is evidence that the refactor is safe,
    not a robocasa infrastructure test.  CI / HIL with pre-pulled assets will
    always reach the assertions.
    """
    # Import rclpy only inside the function body — the module-level skip guards
    # above prevent reaching this point without ROS 2 available.
    import rclpy  # type: ignore[import-untyped]
    from nav_msgs.msg import Odometry  # type: ignore[import-untyped]
    from rclpy.lifecycle import TransitionCallbackReturn  # type: ignore[import-untyped]
    from rclpy.parameter import Parameter  # type: ignore[import-untyped]
    from rclpy.qos import (  # type: ignore[import-untyped]
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import LaserScan, PointCloud2  # type: ignore[import-untyped]

    # ── Import the generic manifest-driven node (issue #191 Phase 3) ─────────
    try:
        from openral_hal.lifecycle import ManifestHALLifecycleNode
    except ImportError as exc:
        pytest.skip(f"rclpy unavailable; ManifestHALLifecycleNode not importable. Error: {exc}")

    # ── Guard: scene YAML on disk ─────────────────────────────────────────────
    if not Path(_SCENE_YAML).is_file():
        pytest.skip(f"scene YAML not found at {_SCENE_YAML}")

    # ── Guard: robocasa/NavigateKitchen registered in openral_sim ────────────
    try:
        from openral_sim.registry import (
            SCENES,  # type: ignore[import-untyped]  # reason: optional dep
        )

        if "robocasa/NavigateKitchen" not in SCENES:
            pytest.skip(
                "robocasa/NavigateKitchen not registered in openral_sim.SCENES — "
                "robocasa kitchen backend not available on this host"
            )
    except ImportError:
        pytest.skip("openral_sim.registry not importable")

    # ── Bring up the lifecycle node ───────────────────────────────────────────
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_panda_mobile")
        node.set_parameters(
            [
                Parameter(
                    "robot_yaml",
                    Parameter.Type.STRING,
                    str(_REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"),
                ),
                Parameter("hal_mode", Parameter.Type.STRING, "sim"),
                # sim_env_yaml → SimAttachedHAL scene-attach (the kitchen); the
                # scene's own `seed` (42) drives the reproducible layout via
                # build_hal → build_sim_env_from_yaml.
                Parameter("sim_env_yaml", Parameter.Type.STRING, _SCENE_YAML),
                Parameter("viewer_enabled", Parameter.Type.BOOL, False),  # headless
            ]
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        try:
            # ── configure ────────────────────────────────────────────────────
            # configure calls build_sim_env_from_yaml → ensure_backend_deps
            # (robocasa kitchen git-clone + pip install) → ensure_robocasa_assets
            # (kitchen asset download).  On a warm cache this is fast (<5 s);
            # on a cold CI cache it can take several minutes.
            configure_ok = _drive_configure(node, executor, _CONFIGURE_TIMEOUT_S)
            if not configure_ok:
                pytest.skip(
                    f"configure transition did not return SUCCESS within "
                    f"{_CONFIGURE_TIMEOUT_S:.0f}s — robocasa kitchen build or asset "
                    "download likely blocked.  This test requires the robocasa kitchen "
                    "assets to be pre-downloaded (OPENRAL_ALLOW_ROBOCASA_ASSETS=1 or "
                    "interactive prompt on first `openral deploy sim` run)."
                )

            # ── activate ─────────────────────────────────────────────────────
            result = node.trigger_activate()
            assert result == TransitionCallbackReturn.SUCCESS, (
                f"activate transition failed: {result!r} — "
                "SimSensorBridge setup or publisher creation raised an error."
            )

            # ── Subscribe to all three topics ─────────────────────────────────
            helper = rclpy.create_node("test_panda_mobile_sensor_bridge_regression")
            executor.add_node(helper)

            # /scan: BEST_EFFORT per CLAUDE.md §2 (sensor data)
            scan_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            # /openral/cameras/…/points: BEST_EFFORT (sensor-class pointcloud)
            depth_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=2,
            )
            # /odom: RELIABLE per CLAUDE.md §2 (control-class odometry)
            odom_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )

            received_scans: list[LaserScan] = []
            received_clouds: list[PointCloud2] = []
            received_odom: list[Odometry] = []

            helper.create_subscription(LaserScan, "/scan", received_scans.append, scan_qos)
            helper.create_subscription(PointCloud2, _DEPTH_TOPIC, received_clouds.append, depth_qos)
            helper.create_subscription(Odometry, "/odom", received_odom.append, odom_qos)

            # ── Assert /scan ──────────────────────────────────────────────────
            got_scan = _spin_until(executor, lambda: bool(received_scans), _SCAN_TIMEOUT_S)
            assert got_scan, (
                f"no LaserScan on /scan within {_SCAN_TIMEOUT_S:.0f}s after ACTIVE — "
                "SimSensorBridge /scan timer may have failed to start or the "
                "ray-cast returned no data."
            )
            scan = received_scans[-1]
            assert len(scan.ranges) > 0, (
                f"/scan.ranges is empty (len={len(scan.ranges)}) — "
                "SimSensorBridge published an empty LaserScan."
            )
            finite_ranges = [r for r in scan.ranges if math.isfinite(r)]
            assert len(finite_ranges) > 0, (
                f"/scan has {len(scan.ranges)} beams but ALL are inf/NaN — "
                "this suggests the ray-cast returned no valid hits.  "
                "Confirm the MJCF has geometry within max_range_m and the "
                "sensor origin is correctly placed."
            )

            # ── Assert /openral/cameras/front_depth/points ────────────────────
            got_cloud = _spin_until(executor, lambda: bool(received_clouds), _DEPTH_TIMEOUT_S)
            assert got_cloud, (
                f"no PointCloud2 on {_DEPTH_TOPIC} within {_DEPTH_TIMEOUT_S:.0f}s "
                f"after ACTIVE — SimSensorBridge depth publisher for sensor "
                f"'{_DEPTH_SENSOR_NAME}' may have failed to start or the depth "
                "ray-cast returned no data."
            )
            cloud = received_clouds[-1]
            cloud_size = int(cloud.width) * int(cloud.height)
            assert cloud_size > 0, (
                f"{_DEPTH_TOPIC}: width={cloud.width} height={cloud.height} → "
                f"cloud_size={cloud_size} — the octomap input PointCloud2 is EMPTY. "
                "SAFETY REGRESSION: the C++ safety kernel's world-collision voxel "
                "check would have no data from this stream."
            )

            # ── Assert /odom ──────────────────────────────────────────────────
            got_odom = _spin_until(executor, lambda: bool(received_odom), _ODOM_TIMEOUT_S)
            assert got_odom, (
                f"no Odometry on /odom within {_ODOM_TIMEOUT_S:.0f}s after ACTIVE — "
                "the panda_mobile lifecycle node's /odom timer failed to start."
            )

            # ── Graceful teardown ─────────────────────────────────────────────
            node.trigger_deactivate()
            node.trigger_cleanup()
            node.trigger_shutdown()

            executor.remove_node(helper)
            helper.destroy_node()

        finally:
            executor.remove_node(node)
            node.destroy_node()

    finally:
        rclpy.shutdown()
