r"""``openral sim run`` driver — Typer-based.

Canonical invocation::

    openral sim run --config scenes/sim/robocasa_pnp.yaml \
               --rskill rskills/pi05-robocasa365-human300-nf4

The YAML carries the scene + task (and an optional robot_id for free-axis
scenes); the policy is supplied entirely via ``--rskill``. The CLI loads
the YAML as a `SimScene` (strict — `openral sim run` accepts SimScene only;
DeployScene and BenchmarkScene YAMLs are rejected with a redirect message),
the manifest via `openral_rskill.loader.load_rskill_manifest`, and composes
them into the runtime `SimEnvironment` that's driven by `SimRunner`
(ADR-0010 amendment 1: sim and hardware share the same `InferenceRunner`
Protocol).

The sim runner accepts bare rSkill references (name, path, or HF repo id).
The command prints a per-episode summary and exits 0 on success, 1 on any
loader / runtime / config error.

The old ``--scene / --task / --vla`` form is gone (no back-compat per
ADR amendment in the feature/more_sims branch). YAMLs that still carry
a ``vla:`` block fail loud with an actionable error pointing at the new
``--rskill`` flag.

See ``docs/tutorials/sim/create-a-sim-environment.md`` for an end-to-end
authoring guide (YAML, new robot manifest, custom scene / policy adapters).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from openral_core import (
    SimEnvironment,
    SimScene,
    VLASpec,
    load_scene_strict,
)
from openral_core.exceptions import ROSConfigError, ROSError

from openral_sim.sim_runner import SimRunner

__all__ = ["main", "sim_app", "sim_run_app"]


# ── Typer apps ────────────────────────────────────────────────────────────────
#
# Two Typer objects + one leaf command are exposed:
#   * ``sim_run_app`` — the leaf command that actually runs the rollout
#     (``invoke_without_command=True``, callback takes --config/--robot/...).
#   * ``sim_app`` — the public ``openral sim`` group that mounts ``sim_run_app``
#     under ``name="run"`` and the ``sim_list`` registry-printer under
#     ``name="list"``. Canonical invocations: ``openral sim run --config X`` and
#     ``openral sim list``.
#   * ``sim_list`` — ``@sim_app.command("list")`` callback that prints the
#     three sim registries (scenes / policies / robots) and exits.

sim_run_app = typer.Typer(
    name="run",
    help=(
        "Run a sim eval composed from a SimScene YAML (--config, scenes/sim/) "
        "and an rSkill manifest (--rskill)."
    ),
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)


@sim_run_app.callback(invoke_without_command=True)
def _sim_run_callback(
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Path to a SimScene YAML (scene + task + optional robot_id). "
            "Strict: DeployScene / BenchmarkScene YAMLs are rejected with "
            "a redirect to `openral deploy sim` / `openral benchmark scene`."
        ),
    ),
    rskill: str | None = typer.Option(
        None,
        "--rskill",
        help=(
            "rSkill reference — a bare name ('smolvla-libero'), a "
            "path ('rskills/smolvla-libero'), or an HF Hub repo id "
            "('OpenRAL/rskill-smolvla-libero'). Required. Resolves to the "
            "policy manifest; the policy adapter is selected from the "
            "manifest's model_family field. Run `openral sim list` for "
            "paste-able strings."
        ),
    ),
    robot: str | None = typer.Option(
        None,
        "--robot",
        help=(
            "robot_id for free-axis scenes. Rejected on scenes with a "
            "fixed_robot (LIBERO, MetaWorld, PushT, ALOHA, RoboCasa)."
        ),
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help=(
            "Override the YAML's task.id (preserves instruction / max_steps "
            "/ success_key). Useful for sweeping a task suite without "
            "editing the YAML."
        ),
    ),
    instruction: str | None = typer.Option(
        None, "--instruction", help="Override the natural-language task instruction."
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override task.max_steps."),
    n_episodes: int | None = typer.Option(None, "--n-episodes", help="Override n_episodes."),
    n_action_steps: int | None = typer.Option(
        None,
        "--n-action-steps",
        help=("Override the chunk-replay cadence. spec_extra > manifest > checkpoint chunk_size."),
    ),
    seed: int | None = typer.Option(None, "--seed", help="Override the global seed."),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device override for the policy (cpu, cuda:0, mps, auto).",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Directory to write the JSON summary (and any adapter artefacts).",
    ),
    save_video: str | None = typer.Option(
        None,
        "--save-video",
        help=(
            "Write a 3-panel debug MP4 after each episode (rSkill input | "
            "rollout view | joint positions plot). Pass a directory, a "
            "filename ending in .mp4, or the empty string to default to "
            "'example_videos/<config-stem>[_ep<i>].mp4'. Setting this flag "
            "also enables per-step frame capture for the run."
        ),
    ),
    video_style: str = typer.Option(
        "debug",
        "--video-style",
        help=(
            "Style of the --save-video output. 'debug' (default) writes the "
            "3-panel montage (rSkill input | rollout | joint plot). 'world' "
            "writes a clean single-view MP4 of just the simulated viewer "
            "(world render), named <scene>_<rskill>_<success|fail>.mp4 and "
            "logged to videos.json — for website hero clips where overlays "
            "are rendered by the page, not burned into pixels."
        ),
    ),
    video_size: int = typer.Option(
        1024,
        "--video-size",
        help=(
            "Square edge (px) for --video-style world output. Frames are "
            "center-cropped to a square and resized to this size. Source "
            "sharpness is bounded by the scene's native render resolution; "
            "this does not change the policy's observation resolution."
        ),
    ),
    dataset_out: Path | None = typer.Option(
        None,
        "--dataset-out",
        help=(
            "ADR-0019: write a LeRobotDataset v3.0 to PATH as the sim "
            "runs. Every episode (success or failure) becomes rows in "
            "the dataset; meta/info.json carries the per-dataset success "
            "rate. Path MUST NOT pre-exist (lerobot v3 refuses to write "
            "into a populated root)."
        ),
    ),
    dataset_repo_id: str | None = typer.Option(
        None,
        "--dataset-repo-id",
        help=(
            "Repo id for the produced dataset (e.g. "
            "openral/dataset-pick-cube). Lands in meta/info.json; not "
            "pushed to HF Hub by `openral sim run` (PR5's `openral dataset push` "
            "owns publishing). Defaults to openral/dataset-<robot_id>."
        ),
    ),
    dataset_license: str = typer.Option(
        "CC-BY-4.0",
        "--dataset-license",
        help=(
            "SPDX license string for the produced dataset. Default "
            "matches the official LeRobot convention. PII-bearing "
            "datasets MUST set a more restrictive license; the consent "
            "prompt at `openral dataset push` (PR5) enforces this."
        ),
    ),
    view: bool | None = typer.Option(
        None,
        "--view/--no-view",
        help=(
            "Open a passive mujoco.viewer window and stream the rollout in "
            "real time. Default: on when a display is available and the scene "
            "is MuJoCo-backed; otherwise auto-disabled with a WARNING. Pass "
            "--view to require a window (errors loud if unsupported), or "
            "--no-view to force offscreen. Incompatible with MUJOCO_GL=egl."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging.",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help=(
            "Boot `openral dashboard` as a child process, point OTel at it, "
            "and shut it down on exit. Convenience for live-viewing a "
            "single sim run; equivalent to running `openral dashboard` in "
            "another terminal and exporting OTEL_EXPORTER_OTLP_ENDPOINT "
            "by hand."
        ),
    ),
    dashboard_port: int = typer.Option(
        4318,
        "--dashboard-port",
        help="Port for the spawned dashboard when --dashboard is set.",
    ),
) -> None:
    """Top-level eval entry point — collects argv into the args namespace and dispatches."""
    args = SimpleNamespace(
        config=config,
        rskill=rskill,
        robot=robot,
        task=task,
        instruction=instruction,
        max_steps=max_steps,
        n_episodes=n_episodes,
        n_action_steps=n_action_steps,
        seed=seed,
        device=device,
        save_dir=save_dir,
        save_video=_resolve_save_video(save_video),
        video_style=video_style,
        video_size=video_size,
        dataset_out=dataset_out,
        dataset_repo_id=dataset_repo_id,
        dataset_license=dataset_license,
        view=view,
        verbose=verbose,
    )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # Observability is owned by the top-level ``openral`` callback
    # (``openral_cli.main:_root``), which opens the ``cli.command`` root
    # span and registers OTel shutdown via ``atexit``. Calling
    # ``shutdown_observability`` here would drain the providers *before*
    # the root span exits during click context teardown, dropping the
    # cli.command export.
    # attached_dashboard is a no-op when enabled=False (no spawn, no
    # FastAPI/uvicorn imports) so we wrap unconditionally. The helper
    # handles spawn → re-configure_observability → shutdown drain →
    # SIGINT child as one ``with`` block.
    from openral_observability.dashboard import attached_dashboard

    with attached_dashboard(enabled=dashboard, port=dashboard_port):
        rc = _run(args)
    raise typer.Exit(code=rc)


def _resolve_save_video(raw: str | None) -> Path | None:
    """Map the ``--save-video`` Typer string to ``Path | None`` semantics.

    argparse's ``nargs="?"`` with a ``const`` value can't be expressed
    directly in Typer; instead we accept a string and treat the empty string
    (`--save-video=`) and the bare flag (`--save-video` with no `=`) as
    "default directory" (``example_videos/``). Any other value is taken as
    a path verbatim.
    """
    if raw is None:
        return None
    if raw == "":
        return Path("example_videos")
    return Path(raw)


def _load_or_build_env(args: SimpleNamespace) -> SimEnvironment:
    """Compose a SimEnvironment from --config (scene+task YAML) and --rskill (manifest)."""
    if args.config is None:
        raise ROSConfigError(
            "--config FILE.yaml is required. The YAML carries the scene + task "
            "(and optional robot_id for free-axis scenes); the policy is supplied "
            "via --rskill rskills/<id>."
        )
    if args.rskill is None:
        raise ROSConfigError(
            "--rskill <ref> is required (e.g. --rskill smolvla-libero or "
            "--rskill rskills/<id>). Run `openral sim list` for the "
            "in-tree rSkill catalogue."
        )

    scene_env = load_scene_strict(str(args.config), SimScene)

    # Robot guard runs FIRST -- it depends only on the scene, not on the
    # manifest. Surfacing this error before the (potentially slow / network-
    # backed) manifest load gives a fast, accurate failure when the user
    # passed --robot on a fixed-robot scene or left robot_id in the YAML.
    from openral_sim.registry import SCENES

    fixed = SCENES.fixed_robot(scene_env.scene.id)
    if fixed is not None:
        if args.robot is not None:
            raise ROSConfigError(
                f"scene {scene_env.scene.id!r} hard-fixes the physics robot to "
                f"{fixed!r}; --robot must not be passed."
            )
        if scene_env.robot_id is not None:
            raise ROSConfigError(
                f"scene {scene_env.scene.id!r} hard-fixes the physics robot to "
                f"{fixed!r}; drop `robot_id: {scene_env.robot_id}` from the YAML."
            )
        if scene_env.base_pose is not None:
            raise ROSConfigError(
                f"scene {scene_env.scene.id!r} hard-fixes the physics robot to "
                f"{fixed!r}; `base_pose:` is honoured by free-axis scenes only "
                "and must be dropped from the YAML."
            )
        resolved_robot: str = fixed
    else:
        chosen = args.robot or scene_env.robot_id
        if not chosen:
            raise ROSConfigError(
                f"scene {scene_env.scene.id!r} does not hard-fix a robot; "
                "set `robot_id:` in the YAML or pass --robot <robot_id>."
            )
        resolved_robot = chosen

    # Now load the manifest -- robot guard above already rejected the
    # cheap-to-detect error cases, so any failure here is genuinely
    # about the manifest itself. Bare names (e.g. `smolvla-libero` or
    # `OpenRAL/rskill-smolvla-libero`) are accepted directly.
    from openral_rskill.loader import _validate_skill_ref, load_rskill_manifest

    rskill_uri = _validate_skill_ref(args.rskill)
    manifest = load_rskill_manifest(rskill_uri)

    # Compose the runtime VLASpec from the manifest + CLI overrides. The
    # adapter id is the manifest's `model_family` — historically a separate
    # `policy_id` field carried the same value, but every in-tree skill set
    # them equal and the duplication has been removed.
    extra: dict[str, object] = {}
    if args.n_action_steps is not None:
        extra["n_action_steps"] = args.n_action_steps
    vla_spec = VLASpec(
        id=manifest.model_family,
        weights_uri=rskill_uri,
        device=args.device or "auto",
        extra=extra,
    )

    # Apply task-level overrides (--task, --instruction, --max-steps).
    task = scene_env.task
    if args.task is not None:
        task = task.model_copy(update={"id": args.task})
    if args.instruction is not None:
        task = task.model_copy(update={"instruction": args.instruction})
    if args.max_steps is not None:
        task = task.model_copy(update={"max_steps": args.max_steps})

    env = SimEnvironment(
        robot_id=resolved_robot,
        scene=scene_env.scene,
        task=task,
        vla=vla_spec,
        base_pose=scene_env.base_pose,
        seed=args.seed if args.seed is not None else scene_env.seed,
        n_episodes=(args.n_episodes if args.n_episodes is not None else scene_env.n_episodes),
        record_video=scene_env.record_video or args.save_video is not None,
        save_dir=str(args.save_dir) if args.save_dir is not None else scene_env.save_dir,
        metadata=scene_env.metadata,
    )

    # Structured summary of what was actually composed -- gives the
    # user a single log line confirming the manifest's per-checkpoint
    # contract was picked up. Mirrors `pi05_prequantized_loaded`'s
    # style: one line, machine-grep-able fields.
    import structlog

    log = structlog.get_logger("ral.sim.compose")
    log.info(
        "sim_run_composed",
        config=str(args.config),
        rskill=args.rskill,
        manifest=manifest.name,
        manifest_version=manifest.version,
        model_family=manifest.model_family,
        scene_id=scene_env.scene.id,
        scene_fixed_robot=fixed,
        resolved_robot=resolved_robot,
        task_id=task.id,
        max_steps=task.max_steps,
        manifest_image_preprocessing=(
            manifest.image_preprocessing.model_dump(mode="json")
            if manifest.image_preprocessing is not None
            else None
        ),
        manifest_state_contract=(
            manifest.state_contract.model_dump(mode="json")
            if manifest.state_contract is not None
            else None
        ),
        manifest_n_action_steps=manifest.n_action_steps,
        manifest_runtime=manifest.runtime.value,
        manifest_quantization_dtype=manifest.quantization.dtype.value,
        cli_vla_extra=extra,
        seed=env.seed,
        n_episodes=env.n_episodes,
    )
    return env


def main(argv: list[str] | None = None) -> int:
    """Run the sim CLI; return ``0`` on success, ``1`` on config error."""
    try:
        result = sim_run_app(args=argv, standalone_mode=False)
    except SystemExit as exc:  # reason: Click's --help path raises SystemExit(0)
        return int(exc.code or 0)
    if isinstance(result, int):
        return result
    return 0


# ── Public Typer group: openral sim ────────────────────────────────────────────────


sim_app = typer.Typer(
    name="sim",
    help=(
        "Sim eval — free-axis (robot × scene × task × rSkill) rollouts.\n"
        "\n"
        "Examples:\n"
        "  openral sim list                        # discover scenes / rskills / robots\n"
        "  openral sim run --config FILE.yaml \\\n"
        "             --rskill rskills/NAME  # canonical config-driven entry\n"
        "\n"
        "Run `openral sim COMMAND --help` for the full flag table on each subcommand."
    ),
    no_args_is_help=True,
    add_completion=False,
)
sim_app.add_typer(sim_run_app, name="run")


def _discover_sim_configs() -> list[Path]:
    """Return every ``scenes/**/*.yaml`` (recursive) under the repo root.

    Walks ``scenes/benchmark/`` (canonical paper-reproduction targets),
    ``scenes/sim/`` (richer hand-authored sims), and ``scenes/deploy/``
    (digital-twin deploy fixtures). Read-only filesystem walk; safe to
    call without any sim dependencies. Sorted by relative path so the
    listing is deterministic.
    """
    from openral_rskill.loader import _find_repo_root_from

    repo_root = _find_repo_root_from(Path(__file__))
    if repo_root is None:
        return []
    scenes_root = repo_root / "scenes"
    if not scenes_root.is_dir():
        return []
    return sorted(scenes_root.rglob("*.yaml"))


@sim_app.command("list")
def sim_list() -> None:
    """List every sim config under ``scenes/**/*.yaml``.

    Each entry is a paste-able ``--config`` path for ``openral sim run``. No
    rollout, no OTel span, no GPU — safe to run on any host.
    """
    from openral_rskill.loader import _find_repo_root_from

    repo_root = _find_repo_root_from(Path(__file__))
    configs = _discover_sim_configs()
    if not configs:
        print("<none>")
        return
    for cfg in configs:
        rel = cfg.relative_to(repo_root) if repo_root else cfg
        print(rel)


def _resolve_view(flag: bool | None) -> tuple[bool, bool]:
    """Resolve the tri-state ``--view/--no-view`` flag.

    Returns ``(view, strict_view)`` where ``view`` is whether the runner
    should attempt to open the viewer, and ``strict_view`` is whether
    failures (missing handles, no display) should raise instead of falling
    back to offscreen.

    Tri-state semantics:
      * ``None``  → user said nothing: auto-on if a display is available
        and ``MUJOCO_GL`` is not ``egl``; non-strict (silent fallback).
      * ``True``  → user passed ``--view``: required; strict.

        When ``MUJOCO_GL=egl`` is set (typically because a ``just
        sim-*`` recipe hard-codes it for CI-headless runs), the user's
        explicit ``--view`` wins: we override ``MUJOCO_GL=glfw``
        in-process before mujoco is imported, so the rollout actually
        opens a window. A WARNING records the override so the operator
        sees why their env var was clobbered.

        When ``DISPLAY`` is unset on linux there is no display to draw
        on; ``--view`` raises :class:`ROSConfigError` rather than
        silently degrading to offscreen (the original "attempt anyway"
        path swallowed the request).
      * ``False`` → user passed ``--no-view``: off.
    """
    log = logging.getLogger(__name__)
    if flag is False:
        return (False, False)
    if flag is True:
        # Linux + no DISPLAY is unrecoverable; raise instead of silently
        # degrading to offscreen. macOS / WSLg / other platforms expose
        # a display through different mechanisms, so this guard only
        # fires when we can be confident no window will appear.
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            from openral_core.exceptions import ROSConfigError

            raise ROSConfigError(
                "--view requires a graphical display, but DISPLAY is unset "
                "on this linux host. Run from a session with a window "
                "system attached, or drop --view to use offscreen rendering."
            )
        # MUJOCO_GL=egl forces mujoco onto a headless EGL backend that
        # cannot mount a passive viewer window. The user explicitly
        # asked for --view, so override the env var in-process before
        # mujoco gets imported by the scene factory. This must happen
        # before _build_env_and_policy fires; _resolve_view is called
        # from the CLI entry point well ahead of any mujoco import.
        if os.environ.get("MUJOCO_GL") == "egl":
            os.environ["MUJOCO_GL"] = "glfw"
            log.warning(
                "MUJOCO_GL=egl is incompatible with --view; overriding to "
                "MUJOCO_GL=glfw in-process so the viewer can open. Unset "
                "MUJOCO_GL in your shell to silence this override, or pass "
                "--no-view to keep egl for offscreen rendering."
            )
        return (True, True)
    # flag is None: auto.
    headless = os.environ.get("MUJOCO_GL") == "egl" or (
        sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    )
    if headless:
        return (False, False)
    return (True, False)


def _maybe_build_recorder(args: SimpleNamespace, env_cfg: SimEnvironment) -> Any | None:
    """Build a :class:`openral_dataset.RolloutRecorder` when ``--dataset-out`` was set.

    Returns ``None`` (no recorder) when the flag was absent. Returns
    ``None`` (with a logged warning) when ``openral_dataset`` /
    ``lerobot`` are not importable — the sim continues, just without
    dataset emission. Constructing the recorder raises
    :class:`openral_core.exceptions.ROSConfigError` when the robot
    manifest isn't usable for dataset binding; we let that propagate
    so the user sees the failure clearly.
    """
    if args.dataset_out is None:
        return None

    # Resolve the robot manifest. Sim envs auto-load it from the
    # ROBOTS registry; we follow the same path so the recorder's
    # observation_spec / action_spec match what the env presents.
    from openral_sim.registry import ROBOTS

    # ROBOTS.get returns a factory (callable that builds a fresh
    # RobotDescription on each call) — matches the existing
    # openral_sim.factory.make_robot pattern at factory.py:77.
    robot = ROBOTS.get(env_cfg.robot_id)()

    # Lazy imports: openral_dataset is itself an optional dep for the
    # sim path (sim works fine without it). If the import fails the
    # user gets a structured warning, not a crash.
    try:
        from openral_dataset import LeRobotDatasetSink, RolloutRecorder
    except ImportError as exc:
        import structlog as _structlog

        _structlog.get_logger("ral.sim.dataset").warning(
            "openral_dataset_not_installed",
            error=str(exc),
            dataset_out=str(args.dataset_out),
            hint="install openral-dataset to use --dataset-out",
        )
        return None

    # fps: pulled from the robot's action_spec.control_freq_hz (the
    # authoritative source), with 30.0 fallback matching the workspace
    # WorldStateAggregator default.
    fps: float = (
        float(robot.action_spec.control_freq_hz)
        if robot.action_spec is not None and robot.action_spec.control_freq_hz
        else 30.0
    )

    # ADR-0019: the sim-side state / action contract belongs on the
    # per-checkpoint rSkill manifest (state_contract / action_contract),
    # not on the physical RobotDescription (ADR-0007 split). Load the
    # manifest and pass its contracts as overrides to the sink so the
    # right dims are used even when the robot manifest is sim-agnostic
    # (Franka 8-D for LIBERO vs 16-D for RoboCasa, same robot.yaml).
    state_shape_override: tuple[int, ...] | None = None
    action_dim_override: int | None = None
    if args.rskill is not None:
        from openral_rskill.loader import _validate_skill_ref, load_rskill_manifest

        try:
            uri = _validate_skill_ref(args.rskill)
            manifest = load_rskill_manifest(uri)
        except Exception:
            manifest = None
        if manifest is not None:
            if manifest.state_contract is not None and manifest.state_contract.dim:
                state_shape_override = (int(manifest.state_contract.dim),)
            if manifest.action_contract is not None:
                action_dim_override = int(manifest.action_contract.dim)

    # Graceful degradation: the sink's construction can still raise
    # ROSConfigError when neither manifest contract nor RobotDescription
    # spec is declared. The user keeps their sim run; they just need to
    # add state_contract + action_contract to the rSkill manifest (or
    # observation_spec + action_spec to the robot manifest for hardware).
    from openral_core.exceptions import ROSConfigError as _ROSConfigError

    # ADR-0019: sim renders all cameras at the scene's resolution
    # (potentially different from the physical sensor's intrinsics).
    # Pass that as a uniform per-camera shape override.
    camera_shape_override: tuple[int, int] = (
        int(env_cfg.scene.observation_height),
        int(env_cfg.scene.observation_width),
    )

    try:
        sink = LeRobotDatasetSink(
            root=args.dataset_out,
            robot=robot,
            fps=fps,
            repo_id=args.dataset_repo_id,
            license=args.dataset_license,
            state_shape=state_shape_override,
            action_dim=action_dim_override,
            camera_shape=camera_shape_override,
        )
    except _ROSConfigError as exc:
        import structlog as _structlog

        _structlog.get_logger("ral.sim.dataset").warning(
            "dataset_sink_construction_failed",
            error=str(exc),
            robot_id=env_cfg.robot_id,
            dataset_out=str(args.dataset_out),
            hint=(
                "the dataset sink could not be constructed. Sim continues "
                "without --dataset-out. Common causes: lerobot missing; "
                "missing state_contract / action_contract on the rSkill "
                "manifest; missing intrinsics on a RobotDescription "
                "sensor with vla_feature_key set."
            ),
        )
        return None
    return RolloutRecorder(
        robot=robot,
        task_string=env_cfg.task.instruction,
        fps=fps,
        sinks=[sink],
        repo_id=args.dataset_repo_id,
    )


def _run(args: SimpleNamespace) -> int:
    """Body of `main` after argv parsing and observability setup."""
    try:
        env_cfg = _load_or_build_env(args)
    except (ROSError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  openral sim run")
    print(f"  robot : {env_cfg.robot_id}")
    print(f"  scene : {env_cfg.scene.id}  [{env_cfg.scene.backend.value}]")
    print(f"  task  : {env_cfg.task.id}")
    print(f"  vla   : {env_cfg.vla.id}  ({env_cfg.vla.weights_uri})")
    print(f"  seed  : {env_cfg.seed}  episodes={env_cfg.n_episodes}")
    print("=" * 60)

    view, strict_view = _resolve_view(args.view)

    # ADR-0019: build a RolloutRecorder + LeRobotDatasetSink when
    # --dataset-out was passed. The recorder is fanned out alongside
    # the existing _EpisodeBuffer — buffer drives in-memory video /
    # benchmark JSON, recorder drives the durable LeRobotDataset v3.
    recorder = _maybe_build_recorder(args, env_cfg)
    # ``args.instruction`` is ``None`` unless the user passed ``--instruction``.
    # Thread it as an explicit override so it wins over a scene's per-episode
    # ``obs["task"]`` language (custom BDDL ``:language``, RoboCasa sampled
    # object); without this the flag was silently ignored on such scenes.
    runner = SimRunner(
        env_cfg,
        view=view,
        strict_view=strict_view,
        instruction_override=args.instruction,
        recorder=recorder,
    )
    # Upper bound: each episode is at most ``max_steps`` step-ticks plus a
    # leading reset-tick; ``SimRunner._should_terminate`` stops the loop
    # once ``n_episodes`` complete, so ``max_ticks`` is just a ceiling.
    _max = env_cfg.task.max_steps if env_cfg.task.max_steps is not None else 1000
    max_ticks = env_cfg.n_episodes * (_max + 1)
    try:
        runner.activate()
        run_result = runner.run(max_ticks=max_ticks)
        results = runner.episode_results
    except ROSError as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        return 1
    finally:
        runner.deactivate()

    if run_result.trace_id is not None:
        print(f"  trace_id: {run_result.trace_id}")

    successes = sum(1 for r in results if r.success)
    print()
    for i, r in enumerate(results):
        print(f"  ep{i}: {r.summary()}")
    print()
    print(f"  success_rate: {successes}/{len(results)} = {successes / len(results):.0%}")

    if env_cfg.save_dir is not None:
        out_dir = Path(env_cfg.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": env_cfg.model_dump(mode="json"),
            "episodes": [
                {
                    "success": r.success,
                    "steps": r.steps,
                    "total_reward": r.total_reward,
                    "mean_step_latency_ms": r.mean_step_latency_ms,
                    "max_step_latency_ms": r.max_step_latency_ms,
                }
                for r in results
            ],
            "success_rate": successes / len(results),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out_dir / 'summary.json'}")

    if args.save_video is not None:
        _write_videos(args, results, env_cfg)

    # Success rate is reported but not gated on; CI / users decide thresholds via the JSON summary.
    return 0


def _write_videos(
    args: SimpleNamespace,
    results: list[Any],
    env_cfg: SimEnvironment,
) -> None:
    """Dispatch ``--save-video`` to the debug or world writer per ``--video-style``."""
    style = getattr(args, "video_style", "debug")
    if style == "world":
        _write_website_videos(args, results, env_cfg)
        return
    if style != "debug":
        raise ROSConfigError(f"--video-style must be 'debug' or 'world'; got {style!r}.")
    _write_debug_videos(args, results, env_cfg)


def _write_debug_videos(
    args: SimpleNamespace,
    results: list[Any],
    env_cfg: SimEnvironment,
) -> None:
    """Write one 3-panel debug MP4 per episode under ``args.save_video``.

    Resolution rules:
      * filename ending in ``.mp4`` → exact path (single episode only).
      * anything else → directory; one file per episode named
        ``<config-stem>[_ep<i>].mp4`` inside it.

    See ``openral_sim._video:save_episode_mp4`` for the panel layout —
    this is the single source of truth for example videos; CLI and
    example scripts both call it.
    """
    # Local import keeps imageio / matplotlib out of the openral sim run start
    # path when no video is requested.
    from openral_sim._video import save_episode_mp4

    target = args.save_video
    is_file = target.suffix == ".mp4"
    out_dir = target.parent if is_file else target
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        stem = Path(args.config).stem
    else:
        stem = f"{env_cfg.robot_id}_{env_cfg.vla.id}"

    title = f"{env_cfg.vla.id} on {env_cfg.task.id} (robot={env_cfg.robot_id})"

    for i, r in enumerate(results):
        if not r.frames and not r.vla_input_frames:
            print(f"  ep{i}: no frames captured (record_video off?)")
            continue
        if is_file:
            path = target if len(results) == 1 else out_dir / f"{target.stem}_ep{i}.mp4"
        else:
            path = out_dir / (f"{stem}.mp4" if len(results) == 1 else f"{stem}_ep{i}.mp4")
        out = save_episode_mp4(r, path, title=title)
        print(f"  wrote {out}")


def _write_website_videos(
    args: SimpleNamespace,
    results: list[Any],
    env_cfg: SimEnvironment,
) -> None:
    """Write clean single-view world MP4s + a ``videos.json`` manifest.

    Files are named ``<scene>_<rskill>_<success|fail>.mp4`` (``_ep<i>`` inserted
    for multi-episode runs) under the ``--save-video`` directory, plus a
    ``videos.json`` manifest. See ``openral_sim._website_video:write_world_videos``.
    """
    from openral_sim._website_video import write_world_videos

    target = args.save_video
    out_dir = target.parent if target.suffix == ".mp4" else target
    write_world_videos(
        results,
        out_dir,
        scene=env_cfg.scene.id,
        rskill=Path(args.rskill).name if args.rskill else env_cfg.vla.id,
        section=Path(args.config).parent.name if args.config is not None else "sim",
        size=int(getattr(args, "video_size", 1024)),
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
