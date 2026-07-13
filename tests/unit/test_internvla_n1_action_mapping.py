"""Unit tests for the InternVLA-N1 nav adapter — twist mapping + adapter logic.

No model, no sidecar process: the pure discrete-action→twist mapping is tested
directly, and the adapter is driven against a fake in-process ZMQ client
(sidecar is a process/network boundary, so a fake here is the sanctioned double
under CLAUDE.md §1.11 — not a mock of OpenRAL logic).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_server_module() -> Any:
    """Import tools/_internvla_n1_server.py by path (not a package)."""
    path = _REPO_ROOT / "tools" / "_internvla_n1_server.py"
    spec = importlib.util.spec_from_file_location("_internvla_n1_server", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FWD = 0.25
_TURN = math.radians(15.0)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        ([1], (_FWD, 0.0, False)),  # forward
        ([2], (0.0, _TURN, False)),  # turn-left → +yaw
        ([3], (0.0, -_TURN, False)),  # turn-right → -yaw
        ([0], (0.0, 0.0, True)),  # STOP latches, zero twist
        ([1, 0], (0.0, 0.0, True)),  # STOP anywhere → stop
        ([5], (0.0, 0.0, False)),  # look-down → no base motion
        ([], (0.0, 0.0, False)),  # empty plan → hold
    ],
)
def test_discrete_action_to_twist(actions: list[int], expected: tuple[float, float, bool]) -> None:
    srv = _load_server_module()
    got = srv.discrete_action_to_twist(actions, forward_mps=_FWD, turn_radps=_TURN)
    assert got[0] == pytest.approx(expected[0])
    assert got[1] == pytest.approx(expected[1])
    assert got[2] is expected[2]


class _FakeClient:
    """In-process stand-in for SidecarClient — records calls, scripts replies."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((endpoint, data))
        if endpoint == "step":
            return self._replies.pop(0)
        return {"ok": True}

    def require(self, reply: dict[str, Any], key: str) -> Any:
        return reply[key]

    def close(self) -> None:
        pass


class _FakeDa3:
    """In-process stand-in for the DA3 depth client — records the RGB it saw."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def infer(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.calls.append((int(rgb.shape[0]), int(rgb.shape[1])))
        return np.full(rgb.shape[:2], 2.5, dtype=np.float32)  # metric metres

    def close(self) -> None:
        pass


def _make_adapter(client: _FakeClient, da3: Any = None) -> Any:
    import sys

    for p in (_REPO_ROOT / "python").glob("*/src"):
        sys.path.insert(0, str(p))
    from openral_sim.policies.internvla_n1 import _InternVLAN1Adapter

    return _InternVLAN1Adapter(
        spec=None, device="cuda", _client=client, _camera_key="camera1", _da3=da3
    )


def _obs() -> dict[str, Any]:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    return {"images": {"camera1": rgb}}


def test_adapter_maps_twist_into_body_twist_row() -> None:
    client = _FakeClient([{"twist": [0.25, -0.1], "stop": False}])
    adapter = _make_adapter(client)
    action = adapter.step(_obs(), "go to the kitchen")
    assert action.shape == (6,)
    assert action[0] == pytest.approx(0.25)  # forward → vx
    assert action[5] == pytest.approx(-0.1)  # yaw → wz
    assert action[1] == action[2] == action[3] == action[4] == 0.0


def test_adapter_latches_stop_and_returns_zero_thereafter() -> None:
    client = _FakeClient([{"twist": [0.0, 0.0], "stop": True}])
    adapter = _make_adapter(client)
    first = adapter.step(_obs(), "go to the kitchen")
    assert np.all(first == 0.0)
    # After STOP the adapter must NOT call the sidecar again — the base holds.
    second = adapter.step(_obs(), "go to the kitchen")
    assert np.all(second == 0.0)
    step_calls = [c for c in client.calls if c[0] == "step"]
    assert len(step_calls) == 1  # only the first step hit the model


def test_adapter_reset_clears_stop_latch() -> None:
    client = _FakeClient([{"twist": [0.0, 0.0], "stop": True}])
    adapter = _make_adapter(client)
    adapter.step(_obs(), "go")
    adapter.reset()
    assert adapter._stopped is False


def test_adapter_requires_the_rgb_camera_slot() -> None:
    from openral_core.exceptions import ROSConfigError

    client = _FakeClient([{"twist": [0.1, 0.0], "stop": False}])
    adapter = _make_adapter(client)
    obs = _obs()
    del obs["images"]["camera1"]  # no RGB for the resolved camera slot
    with pytest.raises(ROSConfigError, match="camera"):
        adapter.step(obs, "go")


def test_adapter_sources_depth_from_da3_when_present() -> None:
    client = _FakeClient([{"twist": [0.1, 0.0], "stop": False}])
    da3 = _FakeDa3()
    adapter = _make_adapter(client, da3=da3)
    adapter.step(_obs(), "go")
    # DA3 was called with the RGB frame, and its metric depth reached the sidecar.
    assert da3.calls == [(224, 224)]
    step_data = next(d for ep, d in client.calls if ep == "step")
    assert step_data is not None
    assert float(step_data["depth"].mean()) == pytest.approx(2.5)


def test_adapter_sends_unit_depth_placeholder_when_da3_disabled() -> None:
    # OPENRAL_INTERNVLA_N1_DEPTH=none path (_da3 is None): DualVLN's nextdit
    # System-1 ignores depth, so a unit-metre placeholder is sent (nonzero, so
    # the sidecar's broken-sensor guard accepts it).
    client = _FakeClient([{"twist": [0.1, 0.0], "stop": False}])
    adapter = _make_adapter(client, da3=None)
    adapter.step(_obs(), "go")
    step_data = next(d for ep, d in client.calls if ep == "step")
    assert step_data is not None
    depth = step_data["depth"]
    assert depth.shape == (224, 224)
    assert float(depth.min()) == 1.0 and float(depth.max()) == 1.0
