"""A render on the bridge's thread and the HAL's teardown never overlap.

``SimSensorBridge`` calls ``MujocoArmHAL.read_images`` from its camera timer on the
executor's spin thread, while a lifecycle ``on_cleanup`` calls ``disconnect`` on
another thread. Closing a renderer's GL context in the middle of its ``render()`` is a
use-after-free: the docker-build live suite segfaulted in ``Renderer.render`` (OSMesa)
while ``on_cleanup`` closed it. Real SO-100 manifest + real MuJoCo; skips only without
an offscreen GL context.
"""

from __future__ import annotations

import threading
import time

import pytest
from _renderer_probe import requires_renderer
from openral_core import RobotDescription
from openral_hal import build_hal

pytest.importorskip("mujoco")


def _connected_rendering_arm() -> object:
    from openral_hal._mujoco_arm import MujocoArmHAL

    hal = build_hal(RobotDescription.from_yaml("robots/so100_follower/robot.yaml"), mode="sim")
    assert isinstance(hal, MujocoArmHAL)
    hal.connect()
    if not hal.read_images():
        hal.disconnect()
        pytest.skip("offscreen MuJoCo render produced no frames (no GL context)")
    return hal


@requires_renderer
def test_disconnect_waits_for_an_in_flight_render() -> None:
    """While a render holds the lock, ``disconnect`` blocks; it completes once released."""
    hal = _connected_rendering_arm()
    lock = hal._render_lock  # type: ignore[attr-defined]  # reason: MujocoArmHAL internal
    done = threading.Event()

    def teardown() -> None:
        hal.disconnect()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
        done.set()

    with lock:  # stands in for a render in progress
        t = threading.Thread(target=teardown)
        t.start()
        assert not done.wait(0.3), "disconnect closed the renderers under a live render"
    t.join(timeout=10.0)
    assert done.is_set()
    assert hal.read_images() == {}  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface


@requires_renderer
def test_disconnect_during_a_render_loop_is_clean() -> None:
    """A camera loop racing ``disconnect`` neither crashes nor renders a closed context."""
    hal = _connected_rendering_arm()
    stop = threading.Event()
    errors: list[BaseException] = []

    def camera_timer() -> None:
        while not stop.is_set():
            try:
                hal.read_images()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
            except BaseException as exc:  # reason: surfaced to the main thread
                errors.append(exc)
                return

    t = threading.Thread(target=camera_timer)
    t.start()
    time.sleep(0.2)
    hal.disconnect()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    time.sleep(0.1)
    stop.set()
    t.join(timeout=10.0)
    assert not errors, errors
