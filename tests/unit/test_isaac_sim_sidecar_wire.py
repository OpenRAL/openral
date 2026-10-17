"""Wire round-trip tests for the Isaac Sim sidecar backend.

The Isaac Sim scene adapter (``openral_sim.backends.isaac_sim``) talks to an
out-of-process Isaac Lab sidecar over ZMQ REQ/REP framed by msgpack — the same
transport the RLDX-1 policy adapter uses. Booting real Isaac Sim costs a ~50 GB
install + an RTX GPU + tens-of-seconds of Omniverse Kit startup, so these tests
pin the openral-side wire (codec + obs unwrapping + StepResult marshalling)
against a **real in-process ZMQ REP echo server** — the legitimate
process/network boundary double allowed by CLAUDE.md §1.11. No Isaac, no GPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pytest

zmq = pytest.importorskip("zmq", reason="sidecar-wire group (pyzmq) not installed")
msgpack = pytest.importorskip("msgpack", reason="sidecar-wire group (msgpack) not installed")

from openral_core import RobotDescription, SceneSpec, SimEnvironment, TaskSpec  # noqa: E402
from openral_core.schemas import PhysicsBackend  # noqa: E402
from openral_hal.sim_attached import SimAttachedHAL  # noqa: E402
from openral_sim.backends.isaac_sim import _FRAME_KEYS, _IsaacSimSidecar  # noqa: E402
from openral_sim.sidecar import (  # noqa: E402
    SidecarClient,
    decode_ndarray,
    encode_ndarray,
)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeSidecar:
    """A real ZMQ REP server speaking the sidecar protocol, on a thread.

    This is the network-boundary double: it exercises the exact msgpack +
    ndarray framing the real Isaac sidecar uses, so the openral-side codec and
    obs unwrapping are tested for real, not mocked.
    """

    def __init__(
        self,
        port: int,
        obs_height: int = 64,
        obs_width: int = 48,
        sim_dt_per_tick_s: float | None = None,
        null_time_base: bool = False,
        obs_extra: dict[str, Any] | None = None,
        sim_time_ns: int | None = None,
        tick_ns: int = 0,
    ) -> None:
        self.port = port
        self.obs_height = obs_height
        self.obs_width = obs_width
        # Sim seconds per step the ping reports (the real sidecar derives it from
        # the manifest control rate); None = a sidecar predating the field.
        self.sim_dt_per_tick_s = sim_dt_per_tick_s
        # Send the field as an explicit null (a scene that never resolved its tick).
        self.null_time_base = null_time_base
        # Observation keys merged over the default obs (e.g. ``image_time_ns``), and
        # the ``sim_time_ns`` reset/step replies carry (None = left out).
        self.obs_extra = obs_extra or {}
        self.sim_time_ns = sim_time_ns
        # Sim time each step advances (needs ``sim_time_ns``), and the ``frames``
        # field of every step request (None = left out, i.e. "render").
        self.tick_ns = tick_ns
        self.frames_requests: list[str | None] = []
        self.last_action: npt.NDArray[Any] | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._stop = threading.Event()

    def _obs(self) -> dict[str, Any]:
        return {
            "images": {
                "camera1": np.full((self.obs_height, self.obs_width, 3), 7, dtype=np.uint8),
            },
            "state": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "task": "lift the cube",
            **self.obs_extra,
        }

    def _serve(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://127.0.0.1:{self.port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=100)):
                continue
            raw = sock.recv()
            req = msgpack.unpackb(raw, object_hook=decode_ndarray, raw=False)
            endpoint = req["endpoint"]
            data = req.get("data", {})
            if endpoint == "ping":
                reply: dict[str, Any] = {"ok": True, "action_dim": 7}
                if self.sim_dt_per_tick_s is not None or self.null_time_base:
                    reply["sim_dt_per_tick_s"] = self.sim_dt_per_tick_s
            elif endpoint == "reset":
                reply = {"observation": self._obs()}
                if self.sim_time_ns is not None:
                    reply["sim_time_ns"] = self.sim_time_ns
            elif endpoint == "step":
                self.last_action = np.asarray(data["action"], dtype=np.float32)
                frames = data.get("frames")
                self.frames_requests.append(frames)
                obs = self._obs()
                if frames == "none":  # the real sidecar renders nothing and sends no frames
                    obs = {k: v for k, v in obs.items() if k not in _FRAME_KEYS}
                if self.sim_time_ns is not None:
                    self.sim_time_ns += self.tick_ns
                reply = {
                    "observation": obs,
                    "reward": 1.5,
                    "terminated": False,
                    "truncated": True,
                    "info": {"is_success": True},
                }
                if self.sim_time_ns is not None:
                    reply["sim_time_ns"] = self.sim_time_ns
            elif endpoint == "close":
                reply = {"ok": True}
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
            sock.send(msgpack.packb(reply, default=encode_ndarray, use_bin_type=True))
        sock.close(linger=0)
        ctx.term()

    def __enter__(self) -> _FakeSidecar:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


@pytest.fixture
def sidecar() -> Iterator[_FakeSidecar]:
    with _FakeSidecar(_free_port()) as s:
        yield s


def _make_rollout(port: int, scene: SceneSpec, task: TaskSpec) -> _IsaacSimSidecar:
    client = SidecarClient(
        name="isaac",
        host="127.0.0.1",
        port=port,
        timeout_ms=5_000,
        # Generous: the fake answers in milliseconds, but its thread can be slow to
        # start on a loaded host (1.0 s failed right after the Isaac sim suite on spark).
        boot_timeout_s=5.0,
        launch_argv=["/bin/false"],  # never used — server already up
        auto_spawn=False,
    )
    client.connect()  # pings the already-running fake; no spawn
    return _IsaacSimSidecar(scene=scene, task=task, _client=client)


def _scene_task(h: int = 64, w: int = 48) -> tuple[SceneSpec, TaskSpec]:
    scene = SceneSpec(
        id="isaac_sim",
        backend=PhysicsBackend.ISAACSIM,
        observation_height=h,
        observation_width=w,
    )
    task = TaskSpec(
        id="isaac_sim/lift_cube",
        scene_id="isaac_sim",
        instruction="lift the cube",
        success_key="is_success",
        max_steps=100,
    )
    return scene, task


class TestNdarrayCodec:
    """The msgpack ndarray sentinel codec must round-trip dtype + shape."""

    @pytest.mark.parametrize(
        "arr",
        [
            np.zeros((8, 6, 3), dtype=np.uint8),
            np.asarray([0.1, -2.0, 3.5], dtype=np.float32),
            np.arange(12, dtype=np.int64).reshape(3, 4),
        ],
    )
    def test_round_trip(self, arr: npt.NDArray[Any]) -> None:
        packed = msgpack.packb({"a": arr}, default=encode_ndarray, use_bin_type=True)
        out = msgpack.unpackb(packed, object_hook=decode_ndarray, raw=False)
        np.testing.assert_array_equal(out["a"], arr)
        assert out["a"].dtype == arr.dtype


class TestSidecarRollout:
    """End-to-end client↔server over a real ZMQ socket — no Isaac."""

    def test_connect_pings_existing_sidecar(self, sidecar: _FakeSidecar) -> None:
        # auto_spawn=False + a bogus launch_argv means connect() can only
        # succeed by pinging the already-running fake — a real spawn attempt
        # would try to exec "/bin/false" and fail. Assert on what that ping
        # actually returned (the wire round trip, not just "no exception").
        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            assert rollout.action_dim == 7
        finally:
            rollout.close()

    def test_action_dim_queried_from_ping(self, sidecar: _FakeSidecar) -> None:
        # SimAttachedHAL._probe_env_action_dim reads env.action_dim (deploy sim);
        # the sidecar ping carries it (the fake answers 7).
        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            assert rollout.action_dim == 7
            assert rollout.action_dim == 7  # cached; no second failure
        finally:
            rollout.close()

    def test_sim_dt_per_tick_s_comes_from_the_ping(self) -> None:
        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 30) as s:
            rollout = _make_rollout(s.port, scene, task)
            try:
                assert rollout.sim_dt_per_tick_s == pytest.approx(1 / 30)
                assert rollout.action_dim == 7  # same cached ping
            finally:
                rollout.close()

    def test_a_null_time_base_fails_closed(self) -> None:
        # The field present but null: the scene never resolved its tick. Reading it
        # as an old sidecar would skip the HAL guard, so the rollout refuses.
        from openral_core.exceptions import ROSConfigError

        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), null_time_base=True) as s:
            rollout = _make_rollout(s.port, scene, task)
            try:
                with pytest.raises(ROSConfigError, match="reports no time base"):
                    _ = rollout.sim_dt_per_tick_s
            finally:
                rollout.close()

    def test_sim_dt_per_tick_s_is_none_for_a_sidecar_without_the_field(
        self, sidecar: _FakeSidecar
    ) -> None:
        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            assert rollout.sim_dt_per_tick_s is None
        finally:
            rollout.close()

    def test_reset_returns_eval_shaped_observation(self, sidecar: _FakeSidecar) -> None:
        scene, task = _scene_task(h=64, w=48)
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            obs = rollout.reset(seed=0)
            assert set(obs) >= {"images", "state", "task"}
            assert obs["images"]["camera1"].shape == (64, 48, 3)
            assert obs["images"]["camera1"].dtype == np.uint8
            np.testing.assert_array_almost_equal(obs["state"], [0.1, 0.2, 0.3])
            assert obs["task"] == "lift the cube"
        finally:
            rollout.close()

    def test_step_marshals_action_and_unpacks_stepresult(self, sidecar: _FakeSidecar) -> None:
        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            rollout.reset(seed=0)
            action = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0], dtype=np.float32)
            result = rollout.step(action)
            # Action crossed the wire intact (ndarray codec).
            assert sidecar.last_action is not None
            np.testing.assert_array_almost_equal(sidecar.last_action, action)
            # StepResult fields unpacked with correct types.
            assert result.reward == pytest.approx(1.5)
            assert result.terminated is False
            assert result.truncated is True
            assert result.info["is_success"] is True
        finally:
            rollout.close()

    def test_render_returns_last_frame(self, sidecar: _FakeSidecar) -> None:
        scene, task = _scene_task(h=64, w=48)
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            rollout.reset(seed=0)
            frame = rollout.render()
            assert frame is not None
            assert frame.shape == (64, 48, 3)
            assert int(frame[0, 0, 0]) == 7
        finally:
            rollout.close()

    def test_auto_spawn_disabled_raises_when_no_sidecar(self) -> None:
        from openral_core.exceptions import ROSConfigError

        client = SidecarClient(
            name="isaac",
            host="127.0.0.1",
            port=_free_port(),  # nothing listening
            timeout_ms=500,
            boot_timeout_s=1.0,
            launch_argv=["/bin/false"],
            auto_spawn=False,
        )
        with pytest.raises(ROSConfigError, match="did not answer ping"):
            client.connect()


class TestSceneRegistration:
    """The factory registers under the canonical id as a free-axis scene."""

    def test_registered_free_axis(self) -> None:
        # Import side-effect registers the backend.
        import openral_sim.backends  # noqa: F401
        from openral_sim.registry import SCENES

        assert "isaac_sim" in SCENES
        assert SCENES.fixed_robot("isaac_sim") is None


class TestMockActionDimByLayout:
    """The mock policy default action dim is layout-, not scene.id-, determined.

    Isaac scenes share scene.id="isaac_sim", but their width depends on the
    robot (11-D panda_mobile, 16-D OpenArm). Keying only on scene.id would feed
    one robot's width to another.
    """

    def _env(self, layout: str | None, action_dim: int | None = None) -> SimEnvironment:
        from openral_core import VLASpec

        opts: dict[str, object] = {} if layout is None else {"layout": layout}
        if action_dim is not None:
            opts["action_dim"] = action_dim
        scene = SceneSpec(id="isaac_sim", backend=PhysicsBackend.ISAACSIM, backend_options=opts)
        task = TaskSpec(id="isaac_sim/t", scene_id="isaac_sim", instruction="x", max_steps=10)
        return SimEnvironment(
            robot_id="franka_panda",
            scene=scene,
            task=task,
            vla=VLASpec(id="random", weights_uri="none"),
        )

    def test_layout_width_comes_from_the_sidecar_not_a_table(self) -> None:
        # The robot-determined width is read from the live sidecar's
        # `action_dim` by SimRunner, not guessed here.
        from openral_sim.policies.mock import _resolve_action_dim

        assert _resolve_action_dim(self._env("manifest")) is None
        assert _resolve_action_dim(self._env(None)) is None

    def test_explicit_override_wins(self) -> None:
        from openral_sim.policies.mock import _resolve_action_dim

        assert _resolve_action_dim(self._env("manifest", action_dim=9)) == 9


class TestHalTimeBaseGuard:
    """``SimAttachedHAL.connect`` holds the env's tick to the manifest's control rate.

    The real OpenArm manifest (30 Hz ``action_spec.control_freq_hz``) wrapped
    around the sidecar rollout, over the real wire (issue #355).
    """

    @staticmethod
    def _openarm() -> RobotDescription:
        from pathlib import Path

        here = Path(__file__).resolve()
        root = next(p for p in here.parents if (p / "robots").is_dir())
        return RobotDescription.from_yaml(str(root / "robots" / "openarm" / "robot.yaml"))

    def _hal(self, port: int) -> SimAttachedHAL:
        scene, task = _scene_task()
        return SimAttachedHAL(_make_rollout(port, scene, task), self._openarm())

    def test_one_control_period_per_step_is_accepted_and_recorded(self) -> None:
        from openral_core.schemas import ClockOrigin

        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 30) as s:
            hal = self._hal(s.port)
            try:
                hal.connect()
                assert hal._sim_dt_per_tick_s == pytest.approx(1 / 30)
                # The sidecar reports no sim clock on reset, so the authority
                # stays host-wall; the tick is still what clock_authority
                # would stamp once the clock is live.
                assert hal.clock_authority().origin is ClockOrigin.HOST_WALL
            finally:
                hal.env.close()

    def test_a_physics_step_per_tick_is_refused(self) -> None:
        from openral_core.exceptions import ROSConfigError

        # 1/60 s per step under a 30 Hz runner: the pre-#355 behaviour, 2x speed.
        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 60) as s:
            hal = self._hal(s.port)
            try:
                with pytest.raises(ROSConfigError, match=r"0\.016667 s.*30 Hz"):
                    hal.connect()
            finally:
                hal.env.close()

    def test_a_nan_tick_is_refused(self) -> None:
        from openral_core.exceptions import ROSConfigError

        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=float("nan")) as s:
            hal = self._hal(s.port)
            try:
                with pytest.raises(ROSConfigError, match="One step must be one control period"):
                    hal.connect()
            finally:
                hal.env.close()

    def test_a_manifest_without_a_rate_cannot_be_held_to_the_tick(self) -> None:
        from openral_core.exceptions import ROSConfigError

        silent = self._openarm().model_copy(update={"action_spec": None})
        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 30) as s:
            hal = SimAttachedHAL(_make_rollout(s.port, scene, task), silent)
            try:
                with pytest.raises(ROSConfigError, match=r"no action_spec\.control_freq_hz"):
                    hal.connect()
            finally:
                hal.env.close()

    def test_a_sidecar_without_the_field_is_not_checked(self, sidecar: _FakeSidecar) -> None:
        hal = self._hal(sidecar.port)
        try:
            hal.connect()  # back-compat: older protocol, nothing to hold it to
            assert hal._sim_dt_per_tick_s is None
        finally:
            hal.env.close()


# One 30 Hz control tick, and an image 4 ticks behind the state (issue #361).
_TICK_NS = 33_333_333
_STATE_NS = 2_000_000_000
_IMAGE_NS = _STATE_NS - 4 * _TICK_NS


class TestImageTime:
    """The sim time each camera frame shows crosses the wire and reaches the ROS stamp.

    Isaac's RTX pipeline delivers a frame ~4 steps after the physics it shows, so
    the sidecar reports per camera the sim time its frame shows (issue #361). Real
    OpenArm manifest, real ``SimAttachedHAL`` and ``SimSensorBridge`` on a real
    rclpy node; the sidecar is the ZMQ boundary double.
    """

    _FRAMES: ClassVar[dict[str, Any]] = {
        "images": {"top": np.full((64, 48, 3), 9, dtype=np.uint8)},
        "image_time_ns": {"top": _IMAGE_NS, "head_zed": _IMAGE_NS - _TICK_NS},
        "depth_points": {"head_zed": np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32)},
        "depth_frames": {
            "head_zed": {
                "depth": np.ones((4, 6), dtype=np.float32),
                "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                "k": np.asarray([3.0, 3.0, 3.0, 2.0]),
                "optical_in_base": np.eye(4),
            }
        },
    }

    def _hal(self, port: int) -> SimAttachedHAL:
        scene, task = _scene_task()
        return SimAttachedHAL(_make_rollout(port, scene, task), TestHalTimeBaseGuard._openarm())

    def test_frame_times_cross_the_wire(self) -> None:
        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), obs_extra=self._FRAMES) as s:
            rollout = _make_rollout(s.port, scene, task)
            try:
                assert rollout.reset(seed=0)["image_time_ns"] == {
                    "top": _IMAGE_NS,
                    "head_zed": _IMAGE_NS - _TICK_NS,
                }
            finally:
                rollout.close()

    def test_a_sidecar_without_frame_times_reports_none(self, sidecar: _FakeSidecar) -> None:
        hal = self._hal(sidecar.port)
        try:
            hal.connect()
            assert hal._last_obs is not None
            assert "image_time_ns" not in hal._last_obs
            assert hal.read_image_times() == {}
        finally:
            hal.env.close()

    def test_hal_frame_times_share_the_sim_clock_offset(self) -> None:
        with _FakeSidecar(_free_port(), obs_extra=self._FRAMES, sim_time_ns=_STATE_NS) as s:
            hal = self._hal(s.port)
            try:
                hal.connect()
                hal._sim_time_offset_ns = 7 * 10**9  # as after an earlier episode
                times = hal.read_image_times()
                # Same domain as sim_time_ns, so the difference is the frame's age.
                now_ns = hal.sim_time_ns()
                assert now_ns is not None
                assert now_ns - times["top"] == 4 * _TICK_NS
                assert times["head_zed"] == 7 * 10**9 + _IMAGE_NS - _TICK_NS
            finally:
                hal.env.close()

    def test_bridge_stamps_each_frame_with_the_time_it_shows(self) -> None:
        rclpy = pytest.importorskip("rclpy")  # default reason matches the declared `ros` gap
        from openral_core import CameraTopicKind, camera_topic
        from openral_hal.sim_sensor_bridge import SimSensorBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, PointCloud2
        from tf2_msgs.msg import TFMessage

        with _FakeSidecar(_free_port(), obs_extra=self._FRAMES, sim_time_ns=_STATE_NS) as s:
            hal = self._hal(s.port)
            rclpy.init()
            node = rclpy.create_node("test_isaac_frame_stamps")
            bridge = SimSensorBridge(
                node=node, hal=hal, description=hal.description, viewer_enabled=False
            )
            got: dict[str, int] = {}

            def keep(key: str) -> Any:
                def cb(msg: Any) -> None:
                    header = msg.transforms[0].header if isinstance(msg, TFMessage) else msg.header
                    got[key] = header.stamp.sec * 10**9 + header.stamp.nanosec

                return cb

            sensor = qos_profile_sensor_data
            for key, kind, topic in (
                ("image", Image, camera_topic("top")),
                ("colour", Image, camera_topic("head_zed")),
                ("depth", Image, camera_topic("head_zed", CameraTopicKind.DEPTH_IMAGE)),
                ("cloud", PointCloud2, camera_topic("head_zed", CameraTopicKind.POINTS)),
                ("tf", TFMessage, "/tf"),
            ):
                node.create_subscription(kind, topic, keep(key), sensor)
            try:
                hal.connect()
                # The camera and depth legs of ``setup()`` only: the rest needs the
                # colcon-built openral_msgs, which the unit tier does not have.
                bridge._setup_cameras()
                bridge._setup_depth()
                before = node.get_clock().now().nanoseconds
                bridge._publish_images()
                bridge._publish_depth_clouds_from_obs()
                after = node.get_clock().now().nanoseconds
                deadline = time.monotonic() + 5.0
                while len(got) < 5 and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                assert set(got) == {"image", "colour", "depth", "cloud", "tf"}, got
                # Each frame stamped its age back from the publish, not at it: the
                # RGB image 4 ticks (133 ms), the depth camera's streams 5.
                age = 4 * _TICK_NS
                assert before - age <= got["image"] <= after - age
                depth_age = age + _TICK_NS
                for key in ("colour", "depth", "cloud"):
                    assert before - depth_age <= got[key] <= after - depth_age, key
                # The optical TF carries the pose measured now, so it keeps now.
                assert before <= got["tf"] <= after
            finally:
                bridge.teardown()
                node.destroy_node()
                rclpy.shutdown()
                hal.env.close()

    def test_on_the_sim_clock_each_frame_is_stamped_with_its_sim_time(self) -> None:
        """Issue #418: on ``use_sim_time`` the stamp is the frame time itself.

        The age form (node now minus ``sim_now - frame_time``) was early whenever the
        node's clock trailed the HAL's newest sim time, and the robot self-filter then
        posed the arm before the instant the cloud shows.
        """
        rclpy = pytest.importorskip("rclpy")  # default reason matches the declared `ros` gap
        from openral_core import CameraTopicKind, camera_topic
        from openral_hal.sim_sensor_bridge import SimSensorBridge
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, PointCloud2

        with _FakeSidecar(_free_port(), obs_extra=self._FRAMES, sim_time_ns=_STATE_NS) as s:
            hal = self._hal(s.port)
            rclpy.init()
            # Sim time with no /clock yet: the node's now (0) trails the sim by seconds.
            node = rclpy.create_node(
                "test_isaac_frame_sim_stamps",
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            bridge = SimSensorBridge(
                node=node, hal=hal, description=hal.description, viewer_enabled=False
            )
            got: dict[str, int] = {}

            def keep(key: str) -> Any:
                def cb(msg: Any) -> None:
                    got[key] = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

                return cb

            for key, kind, topic in (
                ("image", Image, camera_topic("top")),
                ("cloud", PointCloud2, camera_topic("head_zed", CameraTopicKind.POINTS)),
            ):
                node.create_subscription(kind, topic, keep(key), qos_profile_sensor_data)
            try:
                hal.connect()
                bridge._setup_cameras()
                bridge._setup_depth()
                times = hal.read_image_times()
                bridge._publish_images()
                bridge._publish_depth_clouds_from_obs()
                deadline = time.monotonic() + 5.0
                while len(got) < 2 and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                assert got == {"image": times["top"], "cloud": times["head_zed"]}, (got, times)
            finally:
                bridge.teardown()
                node.destroy_node()
                rclpy.shutdown()
                hal.env.close()


class TestJointEfforts:
    """The sidecar's measured joint efforts cross the wire into ``/joint_states.effort``.

    Real OpenArm manifest and real ``SimAttachedHAL``; the sidecar is the ZMQ
    boundary double. A sidecar without the key publishes no effort, not zeros.
    """

    def test_efforts_reach_read_state(self) -> None:
        scene, task = _scene_task()
        desc = TestHalTimeBaseGuard._openarm()
        n = len(desc.joints)
        eff = np.linspace(-6.0, 6.0, n).astype(np.float32)
        extra = {"joint_positions": np.zeros(n, dtype=np.float32), "joint_efforts": eff}
        with _FakeSidecar(_free_port(), obs_extra=extra) as s:
            hal = SimAttachedHAL(_make_rollout(s.port, scene, task), desc)
            try:
                hal.connect()
                np.testing.assert_allclose(hal.read_state().effort, eff, atol=1e-6)
            finally:
                hal.env.close()

    def test_a_sidecar_without_efforts_publishes_none(self, sidecar: _FakeSidecar) -> None:
        scene, task = _scene_task()
        hal = SimAttachedHAL(
            _make_rollout(sidecar.port, scene, task), TestHalTimeBaseGuard._openarm()
        )
        try:
            hal.connect()
            assert hal.read_state().effort == []
        finally:
            hal.env.close()


class TestFramesOnDemand:
    """A step can skip rendering; the rollout then reuses its last frames.

    Rendering is most of an Isaac step's cost, so ``sim run`` renders the policy's
    observation only when it infers and the deploy-sim HAL only when a camera publish
    is due. Real wire, real OpenArm manifest and ``SimAttachedHAL``; the sidecar is
    the ZMQ boundary double, which like the real one sends no frames on a
    ``frames="none"`` step.
    """

    _FRAMES: ClassVar[dict[str, Any]] = {
        "images": {"top": np.full((8, 6, 3), 9, dtype=np.uint8)},
        "image_time_ns": {"top": _IMAGE_NS},
    }

    def test_a_none_step_reuses_the_last_frames_and_is_one_shot(self) -> None:
        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), obs_extra=self._FRAMES) as s:
            rollout = _make_rollout(s.port, scene, task)
            try:
                rollout.reset(seed=0)
                rollout.step(np.zeros(7, dtype=np.float32))
                rollout.set_frames_next_step("none")
                obs = rollout.step(np.zeros(7, dtype=np.float32)).observation
                rollout.step(np.zeros(7, dtype=np.float32))
                rollout.set_frames_next_step("fresh")
                rollout.step(np.zeros(7, dtype=np.float32))
            finally:
                rollout.close()
        # The frames the reply left out are the previous step's, with their own time.
        np.testing.assert_array_equal(obs["images"]["top"], self._FRAMES["images"]["top"])
        assert obs["image_time_ns"] == {"top": _IMAGE_NS}
        assert s.frames_requests == [None, "none", None, "fresh"]

    def test_an_unknown_mode_is_refused(self, sidecar: _FakeSidecar) -> None:
        from openral_core.exceptions import ROSConfigError

        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
        try:
            with pytest.raises(ROSConfigError, match="frames must be one of"):
                rollout.set_frames_next_step("sometimes")
        finally:
            rollout.close()

    def test_hal_renders_once_per_frame_period(self) -> None:
        # 30 Hz ticks, a 10 Hz camera: a fresh frame every third step, starting now.
        with _FakeSidecar(
            _free_port(),
            sim_dt_per_tick_s=1 / 30,
            obs_extra=self._FRAMES,
            sim_time_ns=_STATE_NS,
            tick_ns=_TICK_NS,
        ) as s:
            scene, task = _scene_task()
            hal = SimAttachedHAL(
                _make_rollout(s.port, scene, task), TestHalTimeBaseGuard._openarm()
            )
            try:
                hal.connect()
                hal.set_frame_period_s(0.1)
                for _ in range(7):
                    assert hal.idle_step()
            finally:
                hal.env.close()
        assert s.frames_requests[-7:] == ["fresh", "none", "none", "fresh", "none", "none", "fresh"]
        # The reused frames still reach the camera publisher.
        assert set(hal.read_images()) == {"top"}

    def test_hal_without_a_frame_period_renders_every_step(self) -> None:
        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 30, sim_time_ns=_STATE_NS) as s:
            scene, task = _scene_task()
            hal = SimAttachedHAL(
                _make_rollout(s.port, scene, task), TestHalTimeBaseGuard._openarm()
            )
            try:
                hal.connect()
                for _ in range(3):
                    assert hal.idle_step()
            finally:
                hal.env.close()
        assert s.frames_requests[-3:] == [None, None, None]


class TestSimRunnerFramesOnDemand:
    """``sim run`` asks for fresh frames exactly when the policy will infer.

    A real ``ChunkedExecutor`` (chunk of 4, synchronous) behind the policy and the
    real rollout over the wire; the video is off, so every other step renders nothing.
    """

    def test_fresh_before_each_inference_none_between(self) -> None:
        from types import SimpleNamespace

        from openral_rskill.executor import ChunkedExecutor
        from openral_sim.sim_runner import SimRunner

        executor = ChunkedExecutor(chunk_fn=lambda _b: [[0.0]] * 4, chunk_size=4, prefetch_at=0)
        executor.start()
        policy = SimpleNamespace(_chunk_executor=executor)
        scene, task = _scene_task()
        with _FakeSidecar(_free_port(), sim_dt_per_tick_s=1 / 30) as s:
            runner = SimRunner.__new__(SimRunner)  # only the frame-request seam is exercised
            runner._env = _make_rollout(s.port, scene, task)
            runner._policy = policy  # type: ignore[assignment]  # reason: executor-only stand-in
            runner._recorder = None
            runner._env_cfg = SimpleNamespace(record_video=False)  # type: ignore[assignment]  # reason: only record_video is read
            try:
                runner._env.reset(seed=0)
                for step in range(8):
                    runner._step_idx = step
                    executor.select_action({})
                    runner._request_frames_for_next_step()
                    runner._env.step(np.zeros(7, dtype=np.float32))
            finally:
                runner._env.close()
        assert s.frames_requests == ["none", "none", "none", "fresh"] * 2
