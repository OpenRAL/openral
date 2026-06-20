"""Wire round-trip tests for the Isaac Sim sidecar backend (ADR-0045).

The Isaac Sim scene adapter (:mod:`openral_sim.backends.isaac_sim`) talks to an
out-of-process Isaac Lab sidecar over ZMQ REQ/REP framed by msgpack — the same
transport the RLDX-1 policy adapter uses. Booting real Isaac Sim costs a ~50 GB
install + an RTX GPU + tens-of-seconds of Omniverse Kit startup, so these tests
pin the openral-side wire (codec + obs unwrapping + StepResult marshalling)
against a **real in-process ZMQ REP echo server** — the legitimate
process/network boundary double allowed by CLAUDE.md §1.11. No Isaac, no GPU.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np
import pytest

zmq = pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
msgpack = pytest.importorskip("msgpack", reason="isaacsim group (msgpack) not installed")

from openral_core import SceneSpec, TaskSpec  # noqa: E402
from openral_core.schemas import PhysicsBackend  # noqa: E402
from openral_sim.backends.isaac_sim import _IsaacSimSidecar  # noqa: E402
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

    def __init__(self, port: int, obs_height: int = 64, obs_width: int = 48) -> None:
        self.port = port
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.last_action: np.ndarray | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._stop = threading.Event()

    def _obs(self) -> dict:
        return {
            "images": {
                "camera1": np.full((self.obs_height, self.obs_width, 3), 7, dtype=np.uint8),
            },
            "state": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "task": "lift the cube",
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
                reply: dict = {"ok": True, "action_dim": 7}
            elif endpoint == "reset":
                reply = {"observation": self._obs()}
            elif endpoint == "step":
                self.last_action = np.asarray(data["action"], dtype=np.float32)
                reply = {
                    "observation": self._obs(),
                    "reward": 1.5,
                    "terminated": False,
                    "truncated": True,
                    "info": {"is_success": True},
                }
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
        boot_timeout_s=1.0,
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
    def test_round_trip(self, arr: np.ndarray) -> None:
        packed = msgpack.packb({"a": arr}, default=encode_ndarray, use_bin_type=True)
        out = msgpack.unpackb(packed, object_hook=decode_ndarray, raw=False)
        np.testing.assert_array_equal(out["a"], arr)
        assert out["a"].dtype == arr.dtype


class TestSidecarRollout:
    """End-to-end client↔server over a real ZMQ socket — no Isaac."""

    def test_connect_pings_existing_sidecar(self, sidecar: _FakeSidecar) -> None:
        scene, task = _scene_task()
        rollout = _make_rollout(sidecar.port, scene, task)
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

    Isaac layouts share scene.id="isaac_sim", but lift_cube is 8-D (joint-delta)
    and bowl_plate is the LIBERO 7-D OSC-pose delta. Keying only on scene.id
    silently fed an 8-D action into the 7-D bowl_plate scene.
    """

    def _env(self, layout: str | None, action_dim: int | None = None):
        from openral_core import SimEnvironment, VLASpec

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

    def test_lift_cube_and_default_are_8d(self) -> None:
        from openral_sim.policies.mock import _resolve_action_dim

        assert _resolve_action_dim(self._env("lift_cube")) == 8
        assert _resolve_action_dim(self._env(None)) == 8

    def test_bowl_plate_is_7d(self) -> None:
        from openral_sim.policies.mock import _resolve_action_dim

        assert _resolve_action_dim(self._env("bowl_plate")) == 7

    def test_explicit_override_wins(self) -> None:
        from openral_sim.policies.mock import _resolve_action_dim

        assert _resolve_action_dim(self._env("bowl_plate", action_dim=9)) == 9
