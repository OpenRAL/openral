"""Sim test session bootstrap.

Pre-stubs the broken ``lerobot.policies.groot.modeling_groot`` module before
any test imports ``lerobot.policies``. See ``openral_rskill._lerobot_compat``.

Also exposes :func:`compose_sim_env` — the test-side equivalent of
``openral sim run``'s ``_load_or_build_env`` helper. The on-disk YAMLs under
``scenes/sim/`` and ``scenes/benchmark/`` are :class:`SimScene` /
:class:`BenchmarkScene` shapes (scene + task only); the runtime
:class:`SimEnvironment` is composed by the CLI from a :class:`SimScene`
plus a loaded rSkill manifest. ``SimEnvironment.from_yaml`` was removed in
``feat(core,sim): SceneEnvironment + openral sim run --rskill, no legacy``;
tests must compose the same way the CLI does. ``load_scene_strict`` accepts a
``BenchmarkScene`` YAML transparently when ``expected=SimScene``.
"""

from __future__ import annotations

from pathlib import Path

import openral_rskill._lerobot_compat  # noqa: F401
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
    """Compose a :class:`SimEnvironment` from a ``SimScene`` YAML + rSkill URI.

    Mirrors :func:`openral_sim.cli._load_or_build_env` so the sim
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
        A composed :class:`SimEnvironment` ready for :class:`SimRunner`.
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
