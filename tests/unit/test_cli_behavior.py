"""BEHAVIOR policy-server protocol and CLI coverage."""

from __future__ import annotations

import msgpack
import numpy as np
from fastapi.testclient import TestClient
from openral_cli.behavior import _create_behavior_app
from openral_cli.main import app
from openral_core import PhysicsBackend, SceneSpec, SimEnvironment, TaskSpec, VLASpec
from openral_sim import make_policy
from openral_sim.policy import PolicyAdapter
from typer.testing import CliRunner

_CAMERAS = {
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}


def _zero_policy() -> PolicyAdapter:
    env = SimEnvironment(
        robot_id="r1pro",
        scene=SceneSpec(
            id="behavior_challenge",
            backend=PhysicsBackend.ISAACSIM,
            cameras=["head", "left_wrist", "right_wrist"],
        ),
        task=TaskSpec(
            id="behavior/turning_on_radio",
            scene_id="behavior_challenge",
            instruction="turn on the radio",
        ),
        vla=VLASpec(id="zero", weights_uri="mock://behavior", extra={"action_dim": 23}),
    )
    return make_policy(env)


def _pack(value: object) -> bytes:
    def _encode(item: object) -> object:
        if isinstance(item, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": item.tobytes(),
                b"dtype": item.dtype.str,
                b"shape": item.shape,
            }
        return item

    return msgpack.packb(value, default=_encode)


def _unpack(value: bytes) -> object:
    def _decode(item: dict[object, object]) -> object:
        if b"__ndarray__" not in item:
            return item
        return np.ndarray(
            buffer=item[b"data"],
            dtype=np.dtype(item[b"dtype"]),
            shape=item[b"shape"],
        )

    return msgpack.unpackb(value, object_hook=_decode)


def test_behavior_serve_help() -> None:
    result = CliRunner().invoke(app, ["behavior", "serve", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--rskill", "--task", "--host", "--port", "--state-dim", "--action-dim"):
        assert flag in result.output


def test_behavior_websocket_protocol_round_trip() -> None:
    policy = _zero_policy()
    behavior_app = _create_behavior_app(
        policy,
        task="turning_on_radio",
        instruction="turn on the radio",
        state_dim=61,
        action_dim=23,
    )
    with TestClient(behavior_app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        with client.websocket_connect("/") as websocket:
            metadata = _unpack(websocket.receive_bytes())
            assert isinstance(metadata, dict)
            assert metadata["task"] == "turning_on_radio"
            assert metadata["action_dim"] == 23

            websocket.send_bytes(_pack({"reset": True}))
            observation: dict[str, object] = {
                "robot_r1::proprio": np.zeros(61, dtype=np.float32),
            }
            for key in _CAMERAS:
                observation[key] = np.zeros((8, 8, 4), dtype=np.uint8)
            websocket.send_bytes(_pack(observation))

            reply = _unpack(websocket.receive_bytes())
            assert isinstance(reply, dict)
            np.testing.assert_array_equal(reply["action"], np.zeros(23, dtype=np.float32))

    policy.close()
