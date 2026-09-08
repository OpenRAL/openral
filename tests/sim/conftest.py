"""Sim test session bootstrap.

Pre-stubs the broken ``lerobot.policies.groot.modeling_groot`` module before any test
imports ``lerobot.policies``. See ``openral_rskill._lerobot_compat``.

Pins every sim test to its own DDS domain — see the module-level ``os.environ`` block below
for why that is a correctness requirement here, and the one setting that must NOT be added
alongside it.

Exposes ``compose_sim_env``, the test-side equivalent of ``openral sim run``'s
``_load_or_build_env``: on-disk YAMLs under ``scenes/sim/`` and ``scenes/benchmark/`` are
``SimScene`` / ``BenchmarkScene`` shapes, and the runtime ``SimEnvironment`` is
composed by the CLI from a ``SimScene`` plus a loaded rSkill manifest (never loaded from
YAML directly) — tests must compose the same way. ``load_scene_strict`` accepts a
``BenchmarkScene`` YAML transparently when ``expected=SimScene``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import openral_rskill._lerobot_compat  # noqa: F401
import pytest

from tests.sim.safety._kernel_subprocess import isolated_domain_id

# Confine every sim test to its own DDS graph, before any test module imports rclpy — pytest
# loads this conftest first, the only moment early enough to matter.
#
# CORRECTNESS requirement, not hygiene: a test on the default domain sees every other ROS
# process on the host (domain 0 is a subnet multicast) — another worktree's leftover nodes, a
# live `openral deploy sim`, a real robot. `SimSensorBridge._on_attachment_state_applied` arms
# its voxel-update wait from `count_publishers("/openral/world_voxels")`, so a foreign
# publisher changes the verdict, not just adds noise. Measured on `q-laptop`:
# `test_bridge_masks_multiple_attached_objects_without_reset` failed against seven orphaned
# `octomap_voxel_bridge` graphs left by another worktree, and passed on domains 91 and 92.
#
# Lives here (not in that test) because every other rclpy-using sim test already isolates via
# `start_kernel`; this was the one file that didn't. One place, so it can't be repeated.
#
# `setdefault` keeps an operator's own export authoritative — same posture as
# `openral_cli._dds_scope.confine_sim_scope` and
# `packages/openral_safety_watchdog/test/conftest.py`. `isolated_domain_id` is per-PID (from
# the kernel helpers, where the convention already lives), so concurrent pytest runs don't
# collide either.
#
# The domain, and ONLY the domain — do NOT also set `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`
# here, however tempting the symmetry with `confine_sim_scope`: it was tried and it HANGS
# every kernel-subprocess test. `start_kernel` spawns `safety_kernel_node` as a separate
# process the in-process test node must discover, and under that range they never find each
# other — measured as `test_kernel_fridge_layout_pin_start_state.py` blocked past 20 minutes
# at ~93% idle CPU with no kernel process alive, against 91 s with the domain alone.
# `confine_sim_scope` can set it because it launches a whole graph into one environment; this
# conftest straddles a process boundary the launcher does not own, so it cannot.
os.environ.setdefault("ROS_DOMAIN_ID", str(isolated_domain_id()))
from openral_core import (
    BenchmarkMetadata,
    SimEnvironment,
    SimScene,
    VLASpec,
    load_scene_strict,
)


def compose_sim_env(
    config_path: Path,
    rskill_uri: str,
    *,
    robot_id: str | None = None,
    n_episodes: int = 1,
    max_steps: int | None = None,
) -> SimEnvironment:
    """Compose a ``SimEnvironment`` from a ``SimScene`` YAML + rSkill URI.

    Mirrors ``openral_sim.cli._load_or_build_env`` so the sim
    tests exercise the same composition the production CLI uses.

    Args:
        config_path: Path to the scene/task YAML (e.g.
            ``scenes/benchmark/pusht.yaml``). Accepts a ``SimScene``
            or ``BenchmarkScene`` shape.
        rskill_uri: Bare rSkill reference (name, path, or dir) to the manifest.
        robot_id: Optional explicit robot id when the scene does not
            hard-fix one (LIBERO / MetaWorld / PushT / Aloha all do).
        n_episodes: Override ``n_episodes`` on the composed config.
        max_steps: Override ``task.max_steps``. ``None`` keeps the YAML
            value.

    Returns:
        A composed ``SimEnvironment`` ready for ``SimRunner``.
    """
    from openral_rskill.loader import load_rskill_manifest
    from openral_sim.registry import SCENES

    scene_env = load_scene_strict(str(config_path), SimScene)

    fixed = SCENES.fixed_robot(scene_env.scene.id)
    resolved_robot = fixed or robot_id or scene_env.robot_id
    if resolved_robot is None:
        raise RuntimeError(
            f"scene {scene_env.scene.id!r} has no fixed robot; pass `robot_id=` "
            f"to compose_sim_env()."
        )

    manifest = load_rskill_manifest(rskill_uri)

    vla_spec = VLASpec(
        id=manifest.model_family,
        weights_uri=rskill_uri,
        device="auto",
        extra=dict(manifest.policy_extras),
    )
    task = scene_env.task
    if max_steps is not None:
        task = task.model_copy(update={"max_steps": max_steps})
    # `SimScene.metadata` is `dict | BenchmarkMetadata`; `SimEnvironment.metadata`
    # is strictly `dict`. Flatten when a typed `BenchmarkMetadata` was provided
    # so the runtime composition stays serialisable.
    raw_meta = scene_env.metadata
    metadata: dict[str, object] = (
        raw_meta.model_dump() if isinstance(raw_meta, BenchmarkMetadata) else dict(raw_meta)
    )
    return SimEnvironment(
        robot_id=resolved_robot,
        scene=scene_env.scene,
        task=task,
        vla=vla_spec,
        seed=scene_env.seed,
        n_episodes=n_episodes,
        record_video=scene_env.record_video,
        save_dir=scene_env.save_dir,
        metadata=metadata,
    )


def mujoco_renderer_probe_error() -> str | None:
    """Return ``None`` if a MuJoCo off-screen renderer can be created, else a reason.

    Creating a ``mujoco.Renderer`` on a headless host without a working
    GL/EGL stack calls ``abort()`` at the C level (SIGABRT), which a Python
    ``try/except`` cannot catch — an in-process probe therefore crashes
    pytest *collection* outright (``Fatal Python error: Aborted``) and takes
    the whole partition down with it. Running the probe in a subprocess
    turns that abort into a non-zero exit code we can detect and convert
    into a clean skip reason, leaving collection alive.

    Module-level ``skipif`` markers call this once per renderer-using sim
    test module; the ~1 s subprocess cost is paid only at collection.
    """
    import os
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
    except subprocess.TimeoutExpired:
        return "mujoco renderer probe timed out (120s)"
    if proc.returncode == 0:
        return None
    stderr_lines = (proc.stderr or "").strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else "no stderr"
    return f"renderer probe exited {proc.returncode}: {detail}"


def _libero_robosuite_conflict() -> bool:
    """True when an installed robosuite (>=1.5) blocks the LIBERO 1.4.x runtime.

    A >=1.5 robosuite (e.g. provisioned by a robocasa install) makes LIBERO
    unprovisionable here — the ``libero`` dependency group cannot downgrade
    robosuite. Skip cleanly rather than go red; on a clean runner robosuite
    is absent, so the ``libero`` group install supplies 1.4.x and it runs.
    """
    import importlib.metadata as _md

    if importlib.util.find_spec("robosuite") is None:
        return False
    try:
        return not _md.version("robosuite").startswith("1.4")
    except _md.PackageNotFoundError:
        return False


def _sidecar_python_available() -> bool:
    """Whether the Isaac Sim sidecar venv (or an operator override) is provisioned."""
    override = os.environ.get("OPENRAL_ISAAC_SIDECAR_PYTHON")
    if override:
        return Path(override).is_file()
    default = Path.home() / ".cache" / "openral" / "isaac-sidecar" / ".venv" / "bin" / "python"
    return default.is_file()


def _repo_root() -> Path:
    """Walk up from *this file* to the directory holding ``robots/`` + ``pyproject.toml``.

    Callers historically walked up from their own ``__file__``; since both
    conftest.py and every caller live under the same repo tree, walking up
    from here lands on the same root.
    """
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError("could not locate repo root from test file")


def _robotwin_obs() -> dict[str, object]:
    """A synthetic 3-camera RoboTwin-shaped observation for the aloha_agilex (14-DoF) rig."""
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return {
        "images": {"camera1": img, "camera2": img.copy(), "camera3": img.copy()},
        "state": np.zeros(14, dtype=np.float32),
    }


@pytest.fixture()
def connected_hal(hal: Any) -> Any:
    """Connect/disconnect wrapper generic over whichever concrete ``hal`` fixture is in scope."""
    hal.connect()
    yield hal
    hal.disconnect()


def _robocasa_unavailable() -> str:
    """Empty string if RoboCasa's kitchen fork is installed and active, else why not."""
    if importlib.util.find_spec("robocasa") is None:
        return "robocasa not installed"
    from openral_sim._deps import _has_robocasa_kitchen

    return "" if _has_robocasa_kitchen() else "RoboCasa kitchen fork is not active"


_PANDA_MOBILE_ROBOT = Path(__file__).resolve().parents[2] / "robots" / "panda_mobile" / "robot.yaml"
_BAGUETTE_SCENE = (
    Path(__file__).resolve().parents[2] / "scenes" / "deploy" / "robocasa_baguette.yaml"
)


@pytest.fixture
def hal() -> Any:
    """A connected panda_mobile HAL attached to the real baguette scene."""
    from openral_core import RobotDescription
    from openral_hal import build_hal

    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE_ROBOT))
    built = build_hal(desc, mode="sim", sim_env_yaml=str(_BAGUETTE_SCENE))
    built.connect()
    try:
        yield built
    finally:
        built.disconnect()


@pytest.fixture(scope="module")
def scene_env(_scene_config: Path) -> Any:
    """Load *_scene_config* as a ``BenchmarkScene``, skipping if absent."""
    from openral_core import BenchmarkScene, load_scene_strict

    if not _scene_config.exists():
        pytest.skip(f"sim config not found at {_scene_config}")
    return load_scene_strict(str(_scene_config), BenchmarkScene)
