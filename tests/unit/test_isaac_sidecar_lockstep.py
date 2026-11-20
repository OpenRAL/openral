"""The Isaac sidecar's request loop and lockstep (``tools/isaac_sidecar.py``, openral-pro#99).

The real ``_serve`` loop on a real ZMQ socket, with a stand-in for the Isaac scene (the
process boundary double CLAUDE.md §1.11 allows): no Isaac, no GPU.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
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
    """Counts world steps; each reply carries the step count and the action it applied."""

    action_dim = 2
    sim_dt_per_tick_s = 0.1

    def __init__(self) -> None:
        self.steps = 0

    def step(self, action: Any, frames: str) -> dict[str, Any]:
        self.steps += 1
        return {"step": self.steps, "action": np.asarray(action), "frames": frames}

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
    return msgpack.unpackb(sock.recv(), object_hook=sidecar._decode_ndarray, raw=False)


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
    assert scene.steps == 3


def test_a_waiting_step_is_answered_when_the_other_robot_steps(served: tuple[str, _Scene]) -> None:
    url, scene = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0]})
    _send(a, "step", {"robot": "a", "action": [1.0, 0.0]})
    _send(b, "step", {"robot": "b", "action": [2.0, 0.0]})
    ra, rb = _recv(a), _recv(b)
    assert {ra["action"][0], rb["action"][0]} == {1.0, 2.0}
    assert scene.steps == 4


def test_an_idle_robot_never_holds_the_world_back(served: tuple[str, _Scene]) -> None:
    url, scene = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0], "idle": True})  # b holds
    for _ in range(3):  # a keeps commanding alone
        _call(a, "step", {"robot": "a", "action": [1.0, 0.0]})
    assert scene.steps == 5


def test_a_silent_robot_is_dropped_after_the_timeout(served: tuple[str, _Scene]) -> None:
    url, _ = served
    a, b = _req(url), _req(url)
    _call(a, "step", {"robot": "a", "action": [0.0, 0.0]})
    _call(b, "step", {"robot": "b", "action": [0.0, 0.0]})  # b commands, then goes silent
    reply = _call(a, "step", {"robot": "a", "action": [1.0, 0.0]})  # waits 0.5 s, then goes on
    assert reply["action"][0] == 1.0
    assert _call(a, "step", {"robot": "a", "action": [2.0, 0.0]})["step"] == 4  # no wait now
