"""Build an :class:`openral_sim.SimRollout` from a SimScene/DeployScene YAML path.

ADR-0025 Stage 3 — shared helper for any HAL ROS lifecycle node that
wants to flip into ``SimAttachedHAL`` mode via a ``sim_env_yaml`` ROS
parameter. The lookup path is:

1. Resolve ``sim_env_yaml`` to an absolute path (walking parents of
   this source file when the path is relative).
2. Load it as either a :class:`~openral_core.SimScene` (``openral sim
   run --config``) or a :class:`~openral_core.DeployScene` (``openral
   deploy sim --config``; ADR-0041). For DeployScene the HAL
   synthesises a noop :class:`~openral_core.TaskSpec` since the HAL
   drives ``env.step`` directly and never consults the task's
   ``id`` / ``instruction`` / ``max_steps`` / ``success_key``.
   :class:`~openral_core.BenchmarkScene` YAMLs are rejected with a
   redirect message — those belong to ``openral benchmark scene``.
3. Wrap the scene + task in a :class:`~openral_core.SimEnvironment`
   with a dummy :class:`~openral_core.VLASpec` (the lifecycle node
   drives ``env.step`` directly via
   :meth:`SimAttachedHAL.send_action`; the VLA is never invoked).
4. Look up the scene's factory in :data:`openral_sim.SCENES` and
   instantiate the env.

Generic across robots — the only per-HAL piece is the
``robot_id_fallback`` argument, defaulting to ``None`` so callers
provide their own (e.g. the panda_mobile lifecycle node passes
``"panda_mobile"`` for robocasa-shaped YAMLs that have no
``robot_id:`` field).

Most backends ignore ``task.id`` entirely (so101, robocasa, native
MjSpec), so the synthesised task carries an inert ``_hal_deploy_noop``
suffix. The index-parsing suites (LIBERO ``"<suite>/<int>"``) are the
exception: they have no taskless floor — each suite task *is* a distinct
MuJoCo scene — so :func:`_synthesise_deploy_task_id` synthesises a valid
concrete index (``0``) for them instead. Deploy-sim is env-only and never
reads the task's success criterion, so booting task ``0``'s floor is the
correct continuous-operation twin; the reasoner picks the rSkill at
runtime (ADR-0034 / ADR-0036).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml as _yaml
from openral_core import (
    BenchmarkScene,
    DeployScene,
    SimEnvironment,
    SimScene,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

if TYPE_CHECKING:
    from openral_sim.rollout import SimRollout

__all__ = ["build_sim_env_from_yaml"]

# Scene-id prefixes whose backend factory parses ``task.id`` as
# ``"<suite>/<int>"`` (``openral_sim.backends.libero._parse_task_id``).
# For these the deploy-promoted noop task must carry a *valid* integer
# index — there is no taskless floor — so the LIBERO env boots task 0's
# scene. All other backends ignore ``task.id`` and tolerate the inert
# ``_hal_deploy_noop`` suffix.
_INDEX_PARSING_SCENE_PREFIXES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _synthesise_deploy_task_id(scene_id: str) -> str:
    """Build the noop deploy task id for ``scene_id``.

    Index-parsing suites (LIBERO) require ``"<suite>/<int>"`` and reject a
    string suffix, so they get task index ``0``; every other backend gets
    the inert ``"<suite>/_hal_deploy_noop"`` it never reads.
    """
    if any(
        scene_id == prefix or scene_id.startswith(f"{prefix}/")
        for prefix in _INDEX_PARSING_SCENE_PREFIXES
    ):
        return f"{scene_id}/0"
    return f"{scene_id}/_hal_deploy_noop"


def _load_scene_for_hal(path: str) -> SimScene:
    """Load ``path`` as SimScene; upcast DeployScene by synthesising a TaskSpec.

    The HAL only ever needs ``scene`` / ``robot_id`` / ``base_pose`` /
    ``seed`` from the YAML; the task block is plumbing the
    :class:`SimEnvironment` schema requires but the HAL never invokes
    (see module docstring). When the YAML omits ``task:`` (DeployScene),
    a noop :class:`TaskSpec` is synthesised so the HAL boots without the
    operator having to author a fake task.
    """
    raw_obj = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ROSConfigError(f"{path}: YAML root must be a mapping, got {type(raw_obj).__name__}")
    raw: dict[str, object] = raw_obj

    # BenchmarkScene is a superset of SimScene; reject it explicitly so the
    # eval-only contract (n_episodes / metadata / honest_scope) doesn't slip
    # into the HAL path silently. Same redirect as loaders._load_as_sim.
    try:
        BenchmarkScene.model_validate(raw)
    except ValidationError:
        pass  # Not a BenchmarkScene; continue with the SimScene / DeployScene path.
    else:
        raise ROSConfigError(
            f"{path}: this YAML is a BenchmarkScene (has n_episodes, seed, and "
            "metadata). The HAL only accepts SimScene (``openral sim run``) or "
            "DeployScene (``openral deploy sim``). Drive the benchmark via "
            f"`openral benchmark scene --config {path}` instead."
        )

    if "task" in raw:
        try:
            return SimScene.model_validate(raw)
        except ValidationError as exc:
            raise ROSConfigError(f"{path}: not a valid SimScene: {exc}") from exc

    # No task: must be a DeployScene. Validate strictly, then synthesise the
    # noop TaskSpec the HAL never reads.
    try:
        deploy = DeployScene.model_validate(raw)
    except ValidationError as exc:
        raise ROSConfigError(f"{path}: not a valid DeployScene: {exc}") from exc

    noop_task = TaskSpec(
        id=_synthesise_deploy_task_id(deploy.scene.id),
        scene_id=deploy.scene.id,
        instruction="",
        max_steps=None,
        success_key=None,
    )
    # Re-validate via SimScene so the downstream code path is identical.
    # ``metadata`` is intentionally omitted: ``DeployScene`` has no such field
    # (only ``SimScene`` / ``BenchmarkScene`` carry one), and the SimScene
    # default — ``Field(default_factory=dict)`` → ``{}`` — is the correct
    # contract for a deploy-promoted SimScene (no benchmark provenance to
    # carry; the HAL never reads ``metadata`` anyway).
    return SimScene(
        scene=deploy.scene,
        task=noop_task,
        robot_id=deploy.robot_id,
        base_pose=deploy.base_pose,
    )


def _maybe_force_ignore_done(scene_env: SimScene) -> SimScene:
    """Force ``ignore_done=True`` for robocasa scenes only (deploy-sim continuous).

    ``ignore_done`` is a robocasa-only knob: robocasa is the only backend that
    reads ``opts.ignore_done`` (LIBERO/so100 hardcode it; native MjSpec
    backends such as ``tabletop_push`` strictly REJECT unknown
    ``backend_options`` keys). Injecting it only for robocasa scenes keeps
    deploy-sim's continuous stepping from tripping robocasa's
    terminated-episode guard without breaking strict-validation backends.
    Non-robocasa scenes are returned unchanged. (§1.15 fix.)
    """
    if not scene_env.scene.id.startswith("robocasa"):
        return scene_env
    backend_options = dict(scene_env.scene.backend_options or {})
    if backend_options.get("ignore_done") is True:
        return scene_env
    backend_options["ignore_done"] = True
    return scene_env.model_copy(
        update={"scene": scene_env.scene.model_copy(update={"backend_options": backend_options})}
    )


def build_sim_env_from_yaml(
    sim_env_yaml: str,
    *,
    robot_id_fallback: str | None = None,
) -> tuple[SimRollout, int | None]:
    """Resolve a SimScene YAML path to a live :class:`SimRollout`.

    Args:
        sim_env_yaml: Path to a SimScene **or** DeployScene YAML on disk
            (BenchmarkScene is rejected). Relative paths are resolved by
            walking parents of this source file looking for a match —
            ROS parameter values are cwd-naïve so the lifecycle node
            can't rely on the operator's invoke directory. DeployScene
            YAMLs (env-only, no ``task:``) gain a synthesised noop
            :class:`TaskSpec` so the SimEnvironment schema is satisfied;
            see :func:`_load_scene_for_hal`.
        robot_id_fallback: Robot id to plug into the constructed
            :class:`SimEnvironment` when the YAML omits ``robot_id``
            (robocasa-shaped fixtures forbid it). ``None`` means
            "trust the YAML / SCENES registry" and the loader will
            raise if neither source supplies one.

    Returns:
        ``(env, seed)`` — the instantiated :class:`SimRollout` env and
        the YAML's ``seed`` field (``None`` for DeployScene; SimScene
        defaults to 0). The caller is responsible for plumbing the seed
        into ``env.reset(seed=...)`` (via :class:`SimAttachedHAL`'s
        ``env_reset_seed`` kwarg) so that the deploy_sim and sim_run
        paths reach the same episode of the same scene from the same YAML.

    Raises:
        ROSConfigError: when the YAML can't be located, is a
            BenchmarkScene, is neither a valid SimScene nor DeployScene,
            the scene id isn't registered in :data:`openral_sim.SCENES`,
            or the schema validators reject the loaded fields.
    """
    yaml_path = Path(sim_env_yaml)
    if not yaml_path.is_absolute():
        cursor = Path(__file__).resolve()
        for parent in cursor.parents:
            candidate = parent / sim_env_yaml
            if candidate.is_file():
                yaml_path = candidate
                break
    if not yaml_path.is_file():
        raise ROSConfigError(
            f"build_sim_env_from_yaml: sim_env_yaml={sim_env_yaml!r} not found "
            f"(resolved to {yaml_path})."
        )
    scene_env = _load_scene_for_hal(str(yaml_path))

    scene_env = _maybe_force_ignore_done(scene_env)

    from openral_sim.registry import SCENES  # noqa: PLC0415  # reason: optional dep

    scene_id = scene_env.scene.id
    if scene_id not in SCENES:
        raise ROSConfigError(
            f"build_sim_env_from_yaml: scene id {scene_id!r} is not registered in "
            f"openral_sim.SCENES. Available: {sorted(SCENES)}."  # type: ignore[call-overload]  # reason: _Registry is iterable and yields comparable str keys at runtime
        )

    # Resolve robot_id from the YAML, the SCENES fixed_robot registry,
    # or the caller-supplied fallback. Schema requires the field but
    # robocasa-shaped YAMLs forbid it — same fallback chain
    # `openral_cli.deploy_sim._load_scene_robot_id` uses.
    robot_id = scene_env.robot_id or SCENES.fixed_robot(scene_id) or robot_id_fallback
    if robot_id is None:
        raise ROSConfigError(
            f"build_sim_env_from_yaml: cannot resolve robot_id for scene "
            f"{scene_id!r}; pass robot_id_fallback or set scene.robot_id."
        )

    # Dummy VLA — required by the SimEnvironment schema but never
    # invoked from the HAL lifecycle path (the HAL drives env.step
    # directly via SimAttachedHAL.send_action).
    dummy_vla = VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_base")
    sim_env = SimEnvironment(
        robot_id=robot_id,
        scene=scene_env.scene,
        task=scene_env.task,
        vla=dummy_vla,
        # ADR-0002 Amendment 3: free-axis scenes (openarm_tabletop_pnp) read
        # the mandatory mounting pose off ``SimEnvironment.base_pose``. The
        # YAML carries it on the ``SimScene``; propagate it through so
        # the loader composes the same env the direct factory path does.
        # ``None`` (most scenes) leaves backend behaviour unchanged.
        base_pose=scene_env.base_pose,
        seed=scene_env.seed,
        n_episodes=1,
        record_video=False,
    )
    return SCENES.get(scene_id)(sim_env), scene_env.seed
