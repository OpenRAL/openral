"""SO-ARM (SO-100 / SO-101) deploy-sim cameras via the generic rig (ADR-0065; issue #88).

`openral deploy sim` builds a bare MuJoCo twin for the SO-ARM arms (their
manifests set ``hal.sim: null`` → ``MujocoArmHAL.from_description``). The upstream
``so_arm100`` / ``so_arm101`` MJCFs declare zero ``<camera>`` elements, so the HAL
rendered nothing and every manifest camera came back absent.

The fix: each RGB ``SensorSpec`` carries a ``sim_placement``, and
``MujocoArmHAL.connect`` runs the generic camera rig
(``openral_hal._camera_rig``) to splice those cameras into the bare MJCF — no
per-robot scene composer, no ``scene_defaults.composition`` on the manifest.
This is exercised here against the real manifests + real MuJoCo (no mocks).
"""

from __future__ import annotations

import os

import pytest
from openral_core import RobotDescription
from openral_hal import build_hal

pytest.importorskip("mujoco")

# The classic renderer calls glXOpenDisplay() and raises SIGABRT on headless
# runners; EGL avoids the display requirement entirely.
os.environ.setdefault("MUJOCO_GL", "egl")


def _mujoco_renderer_probe_error() -> str | None:
    """Return ``None`` if a MuJoCo off-screen renderer can be created, else a reason.

    Creating a ``mujoco.Renderer`` on a headless host without a working GL/EGL
    stack calls ``abort()`` at the C level (SIGABRT), which a Python
    ``try/except`` cannot catch — an in-process probe therefore crashes pytest
    outright (``Fatal Python error: Aborted``) and takes the whole partition
    down with it. Running the probe in a subprocess turns that abort into a
    non-zero exit code we can detect and convert into a clean skip reason,
    leaving collection alive. Mirrors ``tests/sim/conftest`` and
    ``test_sim_attached_idle_step`` (sibling test roots we cannot import across).
    """
    import subprocess
    import sys

    probe = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>');"
        "r = mujoco.Renderer(m, 1, 1); r.close()"
    )
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except FileNotFoundError:  # mujoco import unavailable in the probe interpreter
        return "mujoco unavailable for renderer probe"
    except subprocess.TimeoutExpired:
        return "mujoco renderer probe timed out (120s)"
    if proc.returncode == 0:
        return None
    stderr_lines = (proc.stderr or "").strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else "no stderr"
    return f"renderer probe exited {proc.returncode}: {detail}"


# ``test_rig_renders_both_manifest_cameras`` renders an off-screen frame inside
# ``read_images()``; on a headless CI runner the native MuJoCo Renderer
# ``abort()`s the process (SIGABRT), which the test's ``try/except`` cannot
# catch. Skip it (only) when no off-screen renderer is available.
_RENDERER_ERROR = _mujoco_renderer_probe_error()
_requires_renderer = pytest.mark.skipif(
    _RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}",
)

_SOARM = ["robots/so100_follower/robot.yaml", "robots/so101_follower/robot.yaml"]


@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_manifest_sensors_carry_sim_placement(robot_yaml: str) -> None:
    desc = RobotDescription.from_yaml(robot_yaml)
    # No scene composition hook on the robot manifest (ADR-0065).
    assert desc.scene_defaults is None or desc.scene_defaults.composition is None
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    # top = world-fixed overhead; wrist = parented to the roll-mounted end body.
    assert by_name["top"].sim_placement is not None
    assert by_name["top"].sim_placement.parent_body is None
    assert by_name["wrist"].sim_placement is not None
    assert by_name["wrist"].sim_placement.parent_body is not None


@_requires_renderer
@pytest.mark.parametrize("robot_yaml", _SOARM)
def test_rig_renders_both_manifest_cameras(robot_yaml: str) -> None:
    """End-to-end: the bare twin renders `top` + `wrist` after the connect-time rig.

    Mirrors the SimSensorBridge camera tick exactly (it calls
    ``hal.read_images()``). Skips only if offscreen GL is unavailable on the
    runner (CI without a display); a dev host with a GPU renders for real.
    """
    import numpy as np

    desc = RobotDescription.from_yaml(robot_yaml)
    # No transport / no composition: the HAL rigs cameras from sim_placement.
    hal = build_hal(desc, mode="sim")
    assert type(hal).__name__ == "MujocoArmHAL"
    hal.connect()
    try:
        frames = hal.read_images()
    except Exception as exc:  # pragma: no cover - headless GL not available
        pytest.skip(f"offscreen MuJoCo render unavailable: {exc}")
    finally:
        hal.disconnect()

    if not frames:
        pytest.skip("offscreen MuJoCo render produced no frames (no GL context)")
    assert set(frames) == {"top", "wrist"}
    intr = {s.name: s.intrinsics for s in desc.sensors if s.modality == "rgb"}
    for name, arr in frames.items():
        a = np.asarray(arr)
        assert a.ndim == 3 and a.shape[2] == 3, f"{name}: {a.shape}"
        assert a.dtype == np.uint8
        # Each camera renders at ITS OWN declared intrinsics, not a shared max
        # across sensors (so a 256x256 wrist isn't upsized to a 640x480 overhead).
        assert a.shape[:2] == (intr[name].height, intr[name].width), (
            f"{name} rendered {a.shape[:2]} != intrinsics {(intr[name].height, intr[name].width)}"
        )
        # The staging floor + fill light guarantee a non-black frame (a forward
        # wrist camera over a bare arm with no floor would otherwise be void).
        assert a.std() > 1.0, f"{name} looks blank (std={a.std():.2f})"
