"""The Isaac sidecar's request loop and lockstep (``tools/isaac_sidecar.py``, openral-pro#99).

The real ``_serve`` loop on a real ZMQ socket, with a stand-in for the Isaac scene (the
process boundary double CLAUDE.md §1.11 allows): no Isaac, no GPU.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

zmq = pytest.importorskip("zmq", reason="sidecar-wire group (pyzmq) not installed")
msgpack = pytest.importorskip("msgpack", reason="sidecar-wire group (msgpack) not installed")

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "isaac_sidecar.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("isaac_sidecar_lockstep_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sidecar = _load_tool()


class _Scene:
    """Counts world steps; each reply carries the step count and the action it applied.

    Any robot name reaches it (``robot``): the lockstep alone is under test here.
    """

    action_dim = 2
    sim_dt_per_tick_s = 0.1

    def __init__(self) -> None:
        self.steps = 0

    def robot(self, name: str) -> _Scene:
        return self

    def step_many(self, requests: dict[str, tuple[Any, str]]) -> dict[str, dict[str, Any]]:
        self.steps += 1
        return {
            name: {"step": self.steps, "action": np.asarray(action), "frames": frames}
            for name, (action, frames) in requests.items()
        }

    def render(self) -> dict[str, Any]:
        return {"steps": self.steps}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, _Scene]]:
    monkeypatch.setattr(sidecar, "_BARRIER_TIMEOUT_S", 0.5)
    scene = _Scene()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:  # a free port
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    kw = {"task": "t", "layout": "manifest", "environment": "", "spawn": [], "robot": "r"}
    thread = threading.Thread(
        target=sidecar._serve,
        args=(scene,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "sim_app": None,
            "objects": "",
            "robot_spec_hash": "",
            **kw,
        },
        daemon=True,
    )
    thread.start()
    url = f"tcp://127.0.0.1:{port}"
    yield url, scene
    _call(_req(url), "close")
    thread.join(timeout=5)


def _req(url: str) -> Any:
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    return sock


def _send(sock: Any, endpoint: str, data: dict[str, Any] | None = None) -> None:
    sock.send(
        msgpack.packb(
            {"endpoint": endpoint, "data": data or {}},
            default=sidecar._encode_ndarray,
            use_bin_type=True,
        )
    )


def _recv(sock: Any) -> dict[str, Any]:
    reply: dict[str, Any] = msgpack.unpackb(
        sock.recv(), object_hook=sidecar._decode_ndarray, raw=False
    )
    return reply


def _call(sock: Any, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    _send(sock, endpoint, data)
    return _recv(sock)


def test_one_robot_steps_as_before(served: tuple[str, _Scene]) -> None:
    url, scene = served
    client = _req(url)
    assert _call(client, "ping")["action_dim"] == 2
    for n in (1, 2, 3):
        reply = _call(client, "step", {"action": np.array([n, 0.0], np.float32)})
        assert reply["step"] == n
        assert reply["action"].tolist() == [n, 0.0]
    assert reply["frames"] == "render"
    assert _call(client, "nope")["error"] == "unknown endpoint 'nope'"
    assert scene.steps == 3


def test_a_step_waits_for_the_other_commanding_robot(served: tuple[str, _Scene]) -> None:
    url, scene = served
    a, b = _req(url), _req(url)
    assert _call(a, "step", {"robot": "a", "action": [1.0, 0.0]})["step"] == 1  # a alone
    _send(b, "step", {"robot": "b", "action": [2.0, 0.0]})  # a commands: b waits for it
    assert b.poll(200) == 0
    assert _call(_req(url), "render") == {"frame": {"steps": 1}}  # answered while b waits
    _send(a, "step", {"robot": "a", "action": [3.0, 0.0]})
    replies = [_recv(b), _recv(a)]
    assert sorted(r["action"][0] for r in replies) == [2.0, 3.0]
    assert scene.steps == 2  # one world advance answers both


def test_a_waiting_step_is_answered_when_the_other_robot_steps(served: tuple[str, _Scene]) -> None:
    url, scene = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0]})
    _send(a, "step", {"robot": "a", "action": [1.0, 0.0]})
    _send(b, "step", {"robot": "b", "action": [2.0, 0.0]})
    ra, rb = _recv(a), _recv(b)
    assert {ra["action"][0], rb["action"][0]} == {1.0, 2.0}
    assert scene.steps == 3


def test_an_idle_robot_never_holds_the_world_back(served: tuple[str, _Scene]) -> None:
    url, scene = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0], "idle": True})  # b holds
    for _ in range(3):  # a keeps commanding alone
        _call(a, "step", {"robot": "a", "action": [1.0, 0.0]})
    assert scene.steps == 5


def test_idle_takes_a_robot_out_of_the_barrier_at_once() -> None:
    """The production ``_Lockstep``, no wall clock: release without any timeout drop."""
    lock = sidecar._Lockstep(timeout_s=5.0)
    lock.submit("a", b"ra", {}, now=0.0)
    lock.submit("b", b"rb", {}, now=0.0)
    assert [r for r, *_ in lock.release()] == ["a", "b"]
    lock.submit("b", b"rb", {"idle": True}, now=0.1)  # b holds: still one period with a
    lock.submit("a", b"ra", {}, now=0.1)
    assert [r for r, *_ in lock.release()] == ["b", "a"]
    lock.submit("a", b"ra", {}, now=0.2)  # b sends nothing more
    assert lock.drop_late(now=0.2) == []  # nobody is late ...
    assert [r for r, *_ in lock.release()] == ["a"]  # ... and a goes on alone


def test_the_silence_clock_starts_at_the_reply() -> None:
    """A slow world advance is not a robot's silence."""
    lock = sidecar._Lockstep(timeout_s=5.0)
    lock.submit("a", b"ra", {}, now=0.0)
    lock.submit("b", b"rb", {}, now=0.0)
    lock.release()
    lock.replied(["a", "b"], now=10.0)  # the batch took 10 s
    lock.submit("b", b"rb", {}, now=10.1)
    assert lock.drop_late(now=12.0) == []  # a was answered 2 s ago
    assert lock.drop_late(now=15.5) == ["a"]


def test_a_refused_step_does_not_join_the_barrier(served: tuple[str, _Scene]) -> None:
    url, _ = served
    a, b = _req(url), _req(url)
    assert (
        "frames must be one of"
        in _call(b, "step", {"robot": "b", "action": [0.0], "frames": "x"})["error"]
    )
    assert "needs an action" in _call(b, "step", {"robot": "b"})["error"]
    for n in (1, 2):  # b never commanded: a never waits for it
        assert _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})["step"] == n
    assert sidecar._FRAME_MODES == scene_base.FRAME_MODES


def test_a_busy_socket_does_not_hold_a_waiting_robot_past_the_timeout(
    served: tuple[str, _Scene],
) -> None:
    url, _ = served
    a, b = _req(url), _req(url)
    spam = zmq.Context.instance().socket(zmq.DEALER)  # many requests in flight at once
    spam.setsockopt(zmq.LINGER, 0)
    spam.connect(url)
    render = msgpack.packb({"endpoint": "render", "data": {}}, use_bin_type=True)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0]})  # b commands, then goes silent
    _send(a, "step", {"robot": "a", "action": [1.0, 0.0]})  # waits for b ...
    deadline = time.monotonic() + 2.5
    while not a.poll(0) and time.monotonic() < deadline:
        for _ in range(200):  # ... while other requests keep the socket readable
            spam.send_multipart([b"", render])
        for _ in range(200):
            spam.recv_multipart()
    assert a.poll(0), "a's step was held past b's 0.5 s timeout"
    assert _recv(a)["action"][0] == 1.0
    spam.close()


def test_a_silent_robot_is_dropped_after_the_timeout(served: tuple[str, _Scene]) -> None:
    url, _ = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0]})  # b commands, then goes silent
    reply = _call(a, "step", {"robot": "a", "action": [1.0, 0.0]})  # waits 0.5 s, then goes on
    assert reply["action"][0] == 1.0
    assert _call(a, "step", {"robot": "a", "action": [2.0, 0.0]})["step"] == 4  # no wait now


# ── several robots in one world: the real IsaacSceneBase.step_many behind _serve ──────


def _load_scene_base() -> Any:
    path = _TOOL_PATH.parent / "_isaac_scene_base.py"
    spec = importlib.util.spec_from_file_location("isaac_scene_base_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scene_base = _load_scene_base()


class _World:
    """Isaac's ``World`` as the scenes use it: counts physics steps, renders and resets."""

    def __init__(self) -> None:
        self.steps = 0
        self.renders = 0
        self.resets = 0
        self.current_time = 0.0

    def step(self, render: bool = False) -> None:
        self.steps += 1
        self.renders += int(render)
        self.current_time += 0.05

    def render(self) -> None:
        self.renders += 1

    def reset(self) -> None:
        self.resets += 1
        self.current_time = 0.0


class _Robot(scene_base.IsaacSceneBase):  # type: ignore[misc,name-defined]  # reason: tools module
    """One robot of the shared world: its state is the last action applied to it."""

    warmup_steps = 2

    def __init__(self, name: str, world: _World, action_dim: int) -> None:
        super().__init__(obs_height=1, obs_width=1, instruction="", success_key="s", max_steps=9)
        self.name = name
        self._world = world
        self.action_dim = action_dim
        self.physics_substeps = 2
        self._physics_dt_s = 0.05
        self.applied: list[list[float]] = []
        self.substeps = 0  # physics steps this robot's hooks ran before
        self.world_resets_seen = 0
        self.overlap: list[list[str]] = []

    def _apply_action(self, action: Any) -> None:
        self.applied.append([float(v) for v in action])

    def _before_substep(self) -> None:
        self.substeps += 1

    def _after_world_reset(self) -> None:
        self.world_resets_seen += 1

    def _images(self) -> dict[str, Any]:
        return {"camera1": np.zeros((1, 1, 3), np.uint8)}

    def _state(self) -> Any:
        return np.asarray(self.applied[-1] if self.applied else [], np.float32)

    def _reward_terminated(self) -> tuple[float, bool]:
        return 0.0, False

    def footprint_overlaps(self) -> list[list[str]]:
        return self.overlap


@pytest.fixture
def two_robots(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, _World, _Robot, _Robot]]:
    monkeypatch.setattr(sidecar, "_BARRIER_TIMEOUT_S", 0.5)
    world = _World()
    south, north = _Robot("south", world, 2), _Robot("north", world, 3)
    peers = {"south": south, "north": north}
    south._peers = north._peers = peers  # what IsaacManifestScene.build_all does
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    robots = [{"name": n, "robot_spec_hash": n, "spawn_pose": [0.0] * 4} for n in peers]
    thread = threading.Thread(
        target=sidecar._serve,
        args=(south,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "sim_app": None,
            "task": "t",
            "layout": "manifest",
            "environment": "",
            "spawn": [],
            "robot": "",
            "objects": "",
            "robot_spec_hash": "h",
            "robots": robots,
        },
        daemon=True,
    )
    thread.start()
    url = f"tcp://127.0.0.1:{port}"
    yield url, world, south, north
    _call(_req(url), "close")
    thread.join(timeout=5)


def test_two_robots_released_together_advance_the_world_once(
    two_robots: tuple[str, _World, _Robot, _Robot],
) -> None:
    url, world, south, north = two_robots
    s, n = _req(url), _req(url)
    _call(s, "step", {"robot": "south", "action": [0.0, 0.0]})  # south alone: one period
    _send(n, "step", {"robot": "north", "action": [0.0, 0.0, 0.0]})  # north waits for south
    _call(s, "step", {"robot": "south", "action": [0.0, 0.0]})
    _recv(n)
    assert world.steps == 4  # two periods: south alone, then both together
    _send(s, "step", {"robot": "south", "action": [1.0, 2.0], "frames": "none"})
    assert s.poll(200) == 0  # north commands: south waits for it
    _send(n, "step", {"robot": "north", "action": [3.0]})  # short: padded with NaN (hold)
    rs, rn = _recv(s), _recv(n)
    assert world.steps == 6  # ONE more control period (2 substeps) for both robots
    assert south.substeps == north.substeps == 6  # every robot's hooks ran on every step
    assert rs["observation"]["state"].tolist() == [1.0, 2.0]
    assert rn["observation"]["state"][0] == 3.0 and np.isnan(rn["observation"]["state"][1:]).all()
    assert "images" not in rs["observation"] and "images" in rn["observation"]  # its own frames
    assert rs["sim_time_ns"] == rn["sim_time_ns"] == 300_000_000  # one clock


def test_each_endpoint_answers_for_the_robot_it_names(
    two_robots: tuple[str, _World, _Robot, _Robot],
) -> None:
    url, _world, _south, _north = two_robots
    c = _req(url)
    assert _call(c, "ping", {"robot": "south"})["action_dim"] == 2
    ping = _call(c, "ping", {"robot": "north"})
    assert ping["action_dim"] == 3 and ping["ok"] is True
    assert [r["name"] for r in ping["robots"]] == ["south", "north"]
    # A client of another robot set is answered with the identity that tells it apart.
    stranger = _call(c, "ping")
    assert stranger["ok"] is False and stranger["action_dim"] is None
    assert stranger["robot_spec_hash"] == "h"
    assert "no robot 'west'" in _call(c, "step", {"robot": "west", "action": [0.0]})["error"]
    assert "no robot 'west'" in _call(c, "render", {"robot": "west"})["error"]


def test_a_second_robots_reset_never_rewinds_the_shared_world(
    two_robots: tuple[str, _World, _Robot, _Robot],
) -> None:
    url, world, south, north = two_robots
    s, n = _req(url), _req(url)
    assert "observation" in _call(s, "reset", {"robot": "south"})
    assert world.resets == 1 and south.world_resets_seen == north.world_resets_seen == 1
    t = _call(s, "step", {"robot": "south", "action": [1.0, 1.0]})["sim_time_ns"]
    reply = _call(n, "reset", {"robot": "north"})  # north's graph starts while south runs
    assert world.resets == 1  # south keeps its world
    assert reply["observation"]["state"].size == 0  # north's own observation: no action yet
    assert reply["sim_time_ns"] == t  # the running world's clock
    time.sleep(0.6)  # south silent past the barrier timeout
    _call(n, "reset", {"robot": "north"})
    _call(s, "reset", {"robot": "south"})  # south's graph restarts too
    assert world.resets == 1  # neither graph's /clock ever goes back
    assert _call(s, "step", {"robot": "south", "action": [0.0, 0.0]})["sim_time_ns"] > t


def test_base_footprint_overlaps_are_logged_once_and_never_sent_in_a_step(
    two_robots: tuple[str, _World, _Robot, _Robot],
) -> None:
    url, _world, south, _north = two_robots
    s = _req(url)
    _call(s, "step", {"robot": "south", "action": [0.0, 0.0]})
    assert _call(s, "footprint_overlaps") == {"overlaps": [], "events": []}
    south.overlap = [["north", "south"]]
    for _ in range(2):
        reply = _call(s, "step", {"robot": "south", "action": [0.0, 0.0]})
    assert "overlaps" not in reply and "overlaps" not in reply["info"]
    got = _call(s, "footprint_overlaps")
    assert got["overlaps"] == [["north", "south"]]
    assert [e["robots"] for e in got["events"]] == [["north", "south"]]  # one event per start


def test_robot_clients_of_a_shared_sidecar(two_robots: tuple[str, _World, _Robot, _Robot]) -> None:
    """The real client side (``SidecarClient.robot``, ``_IsaacSimSidecar``) against two robots."""
    from openral_core import SceneSpec, TaskSpec
    from openral_sim.backends.isaac_sim import _IsaacSimSidecar
    from openral_sim.sidecar import SidecarClient

    url, world, south, _north = two_robots
    port = int(url.rsplit(":", 1)[1])
    clients = {
        n: SidecarClient(
            name="isaac",
            host="127.0.0.1",
            port=port,
            timeout_ms=3000,
            boot_timeout_s=1.0,
            launch_argv=[],
            auto_spawn=False,
            robot=n,
        )
        for n in ("south", "north")
    }
    for c in clients.values():
        c.connect()
    assert clients["north"].call("ping")["action_dim"] == 3  # its own robot's
    env = _IsaacSimSidecar(
        scene=SceneSpec(id="isaac_sim", backend="isaacsim"),
        task=TaskSpec(id="t", scene_id="isaac_sim", instruction=""),
        _client=clients["north"],
    )
    env.set_idle_next_step()  # what SimAttachedHAL.idle_step does before its hold step
    assert env._step_options() == {"idle": True}
    assert env._step_options() == {}  # one-shot
    clients["north"].call("step", {"action": [0.0, 0.0, 0.0], "idle": True})
    for _ in range(3):  # north holds: south never waits for it
        clients["south"].call("step", {"action": [1.0, 0.0]})
    assert south.applied[-1] == [1.0, 0.0] and world.steps == 8
    env.close()  # one robot leaving a shared sidecar must not end it for the others
    assert clients["south"].call("ping")["ok"] is True
    clients["south"].close()


def test_a_robot_without_its_own_physics_callback_takes_the_shared_world_step(
    two_robots: tuple[str, _World, _Robot, _Robot],
) -> None:
    """Seen on q5090: the first robot's callback never reported; the world's step is one."""
    _url, _world, south, north = two_robots
    south._observed_physics_step_dt_s, north._observed_physics_step_dt_s = None, 0.05
    assert south._observed_sim_dt_per_tick_s == pytest.approx(0.1)  # 2 substeps
    north._observed_physics_step_dt_s = None
    assert south._observed_sim_dt_per_tick_s is None  # nothing measured: still fails closed


def test_holding_robots_step_the_world_together() -> None:
    """With none commanding, one advance per round: each robot's steps stay one period apart.

    Stepped one by one, three holding robots put three periods between one robot's joint
    states and its self-filter dropped every cloud (openral-pro#99).
    """
    lock = sidecar._Lockstep(timeout_s=5.0, idle_round_s=1.0, idle_max_wait_s=0.25)
    hold = {"idle": True}
    lock.submit("a", b"a", hold, now=0.0)
    assert [x for x, *_ in lock.release(now=0.0)] == ["a"]  # alone: at once
    lock.submit("b", b"b", hold, now=0.01)
    lock.submit("c", b"c", hold, now=0.02)
    assert lock.release(now=0.02) == []  # a held a moment ago: the round waits for it
    lock.submit("a", b"a", hold, now=0.03)
    assert [x for x, *_ in lock.release(now=0.03)] == ["b", "c", "a"]
    lock.submit("a", b"a", hold, now=0.1)
    lock.submit("b", b"b", hold, now=0.11)
    assert lock.release(now=0.12) == []  # c held a moment ago: the round waits for it
    assert lock.wait_s(now=0.12) == pytest.approx(0.23)
    lock.submit("c", b"c", hold, now=0.13)
    assert [x for x, *_ in lock.release(now=0.13)] == ["a", "b", "c"]  # one advance
    lock.submit("a", b"a", hold, now=0.2)
    assert lock.release(now=0.3) == []
    assert [x for x, *_ in lock.release(now=0.46)] == ["a"]  # b and c late: the round ran out
    lock.submit("a", b"a", hold, now=2.0)  # b and c held over a second ago: out of the round
    assert [x for x, *_ in lock.release(now=2.0)] == ["a"]


def test_a_holders_round_clock_restarts_at_its_reply() -> None:
    """A slow advance must not age a holding robot out of the next round."""
    lock = sidecar._Lockstep(timeout_s=5.0, idle_round_s=1.0, idle_max_wait_s=0.25)
    hold = {"idle": True}
    lock.submit("a", b"a", hold, now=0.0)
    lock.submit("b", b"b", hold, now=0.0)
    assert [x for x, *_ in lock.release(now=0.0)] == ["a", "b"]
    lock.replied(["a", "b"], now=2.0)  # the advance took 2 s
    lock.submit("a", b"a", hold, now=2.1)
    assert lock.release(now=2.1) == []  # b was answered 0.1 s ago: still in the round
