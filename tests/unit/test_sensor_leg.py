"""Unit tests for the real-mode camera leg (``openral_rskill_ros.sensor_leg``).

The leg is what `openral deploy run` uses to open every deploy-bound
``SensorSpec`` (robot manifest ∪ ``DeployScene.sensors``) and publish each
camera onto the WorldState image topics
(``/openral/cameras/<name>/image``). Real components per CLAUDE.md §1.11:
the GStreamer test uses a real ``videotestsrc`` pipeline (headless, no
camera hardware); ROS-touching paths skip when ``rclpy`` isn't importable
(CI runners without a sourced ROS install).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    SensorDeployBinding,
    SensorReaderBackend,
    SensorReaderConfig,
    SensorSpec,
)
from openral_rskill_ros.sensor_leg import (
    SensorLeg,
    _publish_rate_hz,
    merge_deploy_sensors,
    open_deploy_sensor_readers,
)


def _spec(name: str, *, binding: SensorDeployBinding | None, rate_hz: float = 0.0) -> SensorSpec:
    return SensorSpec(
        name=name,
        modality="rgb",
        frame_id=name,
        rate_hz=rate_hz,
        deploy_binding=binding,
    )


def test_publish_rate_binding_fps_wins_over_spec_rate() -> None:
    """The binding's fps beats the spec's declared rate and the 10 Hz default."""
    spec = _spec(
        "top",
        binding=SensorDeployBinding(backend_params={"device": "/dev/video0", "fps": 15}),
        rate_hz=30.0,
    )
    assert _publish_rate_hz(spec) == 15.0


def test_publish_rate_falls_back_to_spec_rate_then_10hz() -> None:
    """No binding fps → the spec's rate_hz; neither → the WorldState-friendly 10 Hz."""
    spec = _spec("top", binding=SensorDeployBinding(), rate_hz=30.0)
    assert _publish_rate_hz(spec) == 30.0
    bare = _spec("top", binding=SensorDeployBinding(), rate_hz=0.0)
    assert _publish_rate_hz(bare) == 10.0


def test_merge_scene_entry_wins_on_name_collision() -> None:
    """A same-named DeployScene entry is that robot sensor's deploy binding —
    the manifest copy is dropped so the device is never double-opened."""
    manifest = [_spec("top", binding=None), _spec("wrist", binding=None)]
    scene = [
        _spec("top", binding=SensorDeployBinding(backend_params={"device": "/dev/video0"})),
        _spec("overhead", binding=SensorDeployBinding(backend_params={"device": "/dev/video2"})),
    ]
    merged = merge_deploy_sensors(manifest, scene)
    names = [s.name for s in merged]
    assert sorted(names) == ["overhead", "top", "wrist"]
    top = next(s for s in merged if s.name == "top")
    assert top.deploy_binding is not None  # the scene's bound copy survived


def test_unbound_specs_are_skipped() -> None:
    """Committed reference manifests leave deploy_binding unset — the leg skips them."""
    leg = open_deploy_sensor_readers([_spec("top", binding=None)])
    assert leg.readers == []
    assert leg.publishers == []


def test_empty_leg_close_is_idempotent() -> None:
    """close() on an empty/already-closed leg never raises."""
    leg = SensorLeg()
    leg.close()
    leg.close()
    assert leg.readers == []
    assert leg.publishers == []


def test_multi_camera_ros_publishers_start_only_after_explicit_lifecycle_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ROS entities are prepared before background publishing is allowed."""
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY
    from openral_sensors.ros_publisher import SensorRosPublisher

    events: list[tuple[str, str]] = []

    class _Reader:
        def __init__(self, sensor_id: str) -> None:
            self.sensor_id = sensor_id

        def open(self) -> None:
            events.append(("open", self.sensor_id))

        def close(self) -> None:
            events.append(("close", self.sensor_id))

    def build_reader(config: SensorReaderConfig) -> _Reader:
        return _Reader(config.sensor_id)

    backend = SensorReaderBackend.OPENCV_THREAD
    monkeypatch.setitem(SENSOR_BACKEND_REGISTRY, backend.value, build_reader)
    monkeypatch.setattr(
        SensorRosPublisher,
        "prepare",
        lambda self: events.append(("prepare", self._reader.sensor_id)),
    )
    monkeypatch.setattr(
        SensorRosPublisher,
        "start",
        lambda self: events.append(("start", self._reader.sensor_id)),
    )
    monkeypatch.setattr(SensorRosPublisher, "stop", lambda self: None)

    binding = SensorDeployBinding(backend=backend)
    leg = open_deploy_sensor_readers(
        [
            _spec("front", binding=binding),
            _spec("wrist", binding=binding),
        ],
        ros_node=object(),
    )
    try:
        assert events == [
            ("open", "front"),
            ("open", "wrist"),
            ("prepare", "front"),
            ("prepare", "wrist"),
        ]
        leg.start()
        assert events == [
            ("open", "front"),
            ("open", "wrist"),
            ("prepare", "front"),
            ("prepare", "wrist"),
            ("start", "front"),
            ("start", "wrist"),
        ]
    finally:
        leg.close()


def test_gstreamer_testsrc_leg_publishes_frames(tmp_path: Path) -> None:
    """Full leg over a real videotestsrc pipeline: open → ROS tee → frame → close.

    Skips without PyGObject (gi) or rclpy — the same gates the production
    factory enforces. Runs in a SUBPROCESS with the production import
    order (GStreamer backend → Gst.init → rclpy), because ``rclpy.Node()``
    segfaults inside Fast-DDS thread setup when rclpy was imported before
    ``Gst.init()`` (see ``openral_runner/backends/gstreamer/reader.py``
    PR I/8 note) — and the pytest process may already have rclpy loaded
    from an earlier test.
    """
    pytest.importorskip("gi", reason="PyGObject (gstreamer extra) not installed")
    pytest.importorskip("rclpy", reason="ROS 2 not sourced")

    probe = """
# Production import order (mirrors scripts/runtime_node): gi + Gst.init()
# FIRST — before numpy/pydantic/rclpy. Under Fast-DDS, rclpy.Node()
# segfaults when numpy or pydantic were imported before Gst.init()
# (x86 Ubuntu 24.04 / system PyGObject / ROS Jazzy; Cyclone unaffected).
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import time

import rclpy
from openral_core import SensorDeployBinding, SensorSpec
from openral_rskill_ros.sensor_leg import open_deploy_sensor_readers

rclpy.init()
spec = SensorSpec(
    name="testcam",
    modality="rgb",
    frame_id="testcam",
    rate_hz=10.0,
    deploy_binding=SensorDeployBinding(
        backend="gstreamer",
        backend_params={"source": "testsrc", "width": 320, "height": 240, "fps": 10},
    ),
)
# Phase 3: the shared aggregator receives frames straight from
# the reader (no ROS hop). Subclass the REAL aggregator only to observe
# the write (super() still runs) — no behaviour is faked.
from openral_core import RobotDescription
from openral_world_state import WorldStateAggregator

description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")

class _CountingAggregator(WorldStateAggregator):
    def __init__(self, desc):
        super().__init__(desc)
        self.image_writes = []

    def update_image_frame(self, sensor_name, frame):
        super().update_image_frame(sensor_name, frame)
        self.image_writes.append((sensor_name, frame))

aggregator = _CountingAggregator(description)
leg = open_deploy_sensor_readers([spec], aggregator=aggregator)
try:
    leg.start()
    assert len(leg.readers) == 1, leg.readers
    # The direct-aggregator pump is registered as a publisher-shaped pump;
    # the sensor is recorded for WorldState's direct_image_frame_sensors.
    assert leg.direct_sensors == ["testcam"], leg.direct_sensors
    assert len(leg.publishers) == 1, leg.publishers
    deadline = time.time() + 10.0
    frame = None
    while time.time() < deadline:
        try:
            frame = leg.readers[0].read_latest(max_age_ms=2000)
        except Exception:
            frame = None
        if frame is not None:
            break
        time.sleep(0.1)
    assert frame is not None, "videotestsrc produced no frame within 10 s"

    # The pump delivered the same frames in-process, pixels intact.
    deadline = time.time() + 10.0
    while time.time() < deadline and not aggregator.image_writes:
        time.sleep(0.1)
    assert aggregator.image_writes, "aggregator pump wrote no frame within 10 s"
    written_name, written_frame = aggregator.image_writes[0]
    assert written_name == "testcam"
    # CPU hosts deliver inline data; the DeepStream tier delivers a zero-copy
    # NVMM handle (SensorFrame's data/handle exclusivity) — both are the point.
    assert written_frame.data is not None or written_frame.handle is not None
    assert written_frame.width == 320

    # And the WorldState side of the contract: a BEST_EFFORT subscriber
    # on the leg's topic (same profile world_state requests) receives a
    # sensor_msgs/Image from the in-pipeline ROS tee.
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image

    got = []
    sub_node = rclpy.create_node("probe_subscriber")
    sub_node.create_subscription(
        Image,
        "/openral/cameras/testcam/image",
        got.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        ),
    )
    deadline = time.time() + 10.0
    while time.time() < deadline and not got:
        rclpy.spin_once(sub_node, timeout_sec=0.2)
    sub_node.destroy_node()
    assert got, "no Image arrived on /openral/cameras/testcam/image within 10 s"
    assert got[0].height == 240 and got[0].width == 320, (got[0].height, got[0].width)
finally:
    leg.close()
    rclpy.try_shutdown()
assert leg.readers == []
print("SENSOR_LEG_PROBE_OK")
"""
    import os
    import subprocess
    import sys

    pkg_dir = Path(__file__).resolve().parents[2] / "packages" / "openral_rskill_ros"
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = f"{pkg_dir}{os.pathsep}{env_vars.get('PYTHONPATH', '')}"
    result = subprocess.run(  # reason: sys.executable with a fixed -c script
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        env=env_vars,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "SENSOR_LEG_PROBE_OK" in result.stdout
