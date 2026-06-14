"""``openral dataset`` Typer app — ADR-0019 PR5.

Subcommands:

* ``push <root>`` — upload a local LeRobotDataset v3 to the HF Hub.
  Always ``private=True``; a typed consent prompt discloses the PII risks
  (faces / room layouts / biometric joint trajectories) before any
  network call. ``--yes`` or ``OPENRAL_DATASET_CONSENT=1`` skips the
  prompt for CI / scripted invocations; ``--dry-run`` short-circuits
  the consent gate entirely and just validates the dataset.

The PR4 ``from-bag`` subcommand will be added in the rosbag2 converter
PR. ``push`` ships first because it has no ROS dependency surface and
unblocks the dataset-flywheel demo regardless of rosbag2 readiness.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

import structlog
import typer
from openral_core.exceptions import ROSConfigError, ROSError
from rich.console import Console
from rich.panel import Panel

from openral_cli._hf_publish import IGNORE_PATTERNS, ensure_private, resolve_token

__all__ = ["dataset_app"]

_log = structlog.get_logger(__name__)
_console = Console()

# Env-var override for the consent prompt. Lets CI runs / scripted
# uploads avoid the typer.confirm() interactive prompt.
_CONSENT_ENV_VAR: Final[str] = "OPENRAL_DATASET_CONSENT"


dataset_app = typer.Typer(
    name="dataset",
    help=(
        "Publish or convert OpenRAL datasets (ADR-0019).\n"
        "\n"
        "Commands:\n"
        "  push      — upload a local LeRobotDataset v3 to the HF Hub (private; consent-gated).\n"
        "  from-bag  — convert an mcap rosbag2 written by Rosbag2Sink into a LeRobotDataset v3."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# ── from-bag subcommand (PR4) ────────────────────────────────────────────────


@dataset_app.command("from-bag")
def from_bag_command(
    bag_path: Path = typer.Argument(
        ...,
        help="Input .mcap bag written by openral_dataset.Rosbag2Sink (ADR-0019 PR3).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    robot_yaml: Path = typer.Option(
        ...,
        "--robot",
        help=(
            "Path to a RobotDescription robot.yaml — used to bind feature "
            "shapes (state_shape from ObservationSpec, dim from ActionSpec, "
            "camera keys from SensorSpec.vla_feature_key). Must match the "
            "robot used at record time."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_root: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Destination LeRobotDataset v3 root directory. Must NOT pre-exist "
            "(lerobot v3 refuses to write into a populated root)."
        ),
        resolve_path=True,
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help=(
            "Override the resulting dataset's repo_id. Defaults to openral/dataset-<robot_name>."
        ),
    ),
    license_str: str = typer.Option(
        "CC-BY-4.0",
        "--license",
        help="SPDX license string for the produced dataset.",
    ),
    fps: float | None = typer.Option(
        None,
        "--fps",
        help=(
            "Frames-per-second for the produced dataset. Defaults to "
            "robot.action_spec.control_freq_hz or 30.0."
        ),
    ),
) -> None:
    """Convert an mcap rosbag2 written by `Rosbag2Sink` into a LeRobotDataset v3.

    Walks the bag's `/openral/episode` markers to segment episodes,
    replays each tick through `RolloutRecorder` → `LeRobotDatasetSink`,
    and produces an on-disk v3 dataset ready for `openral dataset push`.

    Per CLAUDE.md §1.11 — this command uses real `mcap` and real
    `lerobot.datasets.LeRobotDataset.create` end-to-end; no mocks.
    """
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSError
    from openral_dataset.converter import Rosbag2ToLeRobotConverter

    try:
        robot = RobotDescription.from_yaml(str(robot_yaml))
    except (OSError, ROSError) as exc:
        _console.print(f"[red]robot.yaml error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        summary = Rosbag2ToLeRobotConverter.from_bag(
            bag_path=bag_path,
            robot=robot,
            output_root=output_root,
            repo_id=repo_id,
            license=license_str,
            fps=fps,
        )
    except ROSError as exc:
        _console.print(f"[red]conversion error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(
        f"[green]converted:[/green] {summary.output_root}\n"
        f"  episodes : {summary.n_episodes} "
        f"({summary.n_success} success / {summary.n_episodes - summary.n_success} failure)\n"
        f"  frames   : {summary.n_frames}\n"
        f"  repo_id  : {summary.repo_id}\n"
        f"  publish  : openral dataset push {summary.output_root} --yes"
    )


# ── push subcommand ──────────────────────────────────────────────────────────


_CONSENT_PROMPT_BODY = """
You are about to upload a robot dataset to the Hugging Face Hub.

Dataset    : {repo_id}
Local root : {root}
Visibility : PRIVATE (mandatory — OpenRAL never publishes public datasets automatically)
Episodes   : {n_episodes}
Frames     : {n_frames}
Cameras    : {cameras}
License    : {license}

This dataset may contain PERSONAL DATA:
  - faces, bystanders, or other identifiable people in camera frames
  - room layouts / posters / screens that disclose location or identity
  - voice prints if audio sensors were recorded
  - joint trajectories that may identify the operator (biometric)

You are responsible for:
  - obtaining consent from anyone visible in camera frames
  - the license you set ({license}) — PII-bearing data needs a stricter license
  - regulatory compliance (GDPR / CCPA / equivalent) in your jurisdiction

The dataset's `meta/info.json` already records the license, episode count,
and success rate. The HF Hub repo will mirror them.
""".strip()


def _read_info_json(root: Path) -> dict[str, object]:
    """Read ``meta/info.json`` from a local LeRobotDataset v3 root.

    Raises:
        ROSConfigError: When the file is missing or unparseable. A clean
            ``openral dataset push <root>`` failure here means the user
            pointed at a non-dataset path; we surface that loudly.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ROSConfigError(
            f"{root} does not look like a LeRobotDataset v3 root "
            f"(missing meta/info.json). "
            "Did you mean to pass the directory written by "
            "`openral sim run --dataset-out <path>`?"
        )
    try:
        parsed = json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        raise ROSConfigError(f"meta/info.json at {info_path} is not valid JSON: {exc!s}") from exc
    if not isinstance(parsed, dict):
        raise ROSConfigError(
            f"meta/info.json at {info_path} is not a JSON object (got {type(parsed).__name__})"
        )
    return parsed


def _camera_keys_from_info(info: dict[str, object]) -> list[str]:
    """Extract ``observation.images.*`` feature keys from a v3 info.json."""
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(k for k in features if k.startswith("observation.images."))


def _confirm_consent(repo_id: str, root: Path, info: dict[str, object], yes: bool) -> None:
    """Display the PII / consent disclosure and require an explicit confirmation.

    Acceptable confirmations:
    - ``--yes`` flag passed to ``push``
    - ``OPENRAL_DATASET_CONSENT=1`` env var set
    - Interactive prompt where the user retypes the repo_id

    Raises:
        ROSConfigError: On any refusal / mismatch / non-interactive
            invocation without the override.
    """
    if yes:
        _log.info("dataset.push.consent.flag_override", repo_id=repo_id)
        return
    if os.environ.get(_CONSENT_ENV_VAR) == "1":
        _log.info("dataset.push.consent.env_override", repo_id=repo_id)
        return

    metadata = info.get("metadata", {})
    license_str = (
        metadata.get("license", "CC-BY-4.0") if isinstance(metadata, dict) else "CC-BY-4.0"
    )
    cameras = _camera_keys_from_info(info)
    body = _CONSENT_PROMPT_BODY.format(
        repo_id=repo_id,
        root=str(root),
        n_episodes=info.get("total_episodes", "?"),
        n_frames=info.get("total_frames", "?"),
        cameras=", ".join(cameras) or "(none)",
        license=license_str,
    )
    _console.print(Panel(body, title="Consent required", border_style="yellow"))

    if not sys.stdin.isatty():
        raise ROSConfigError(
            "non-interactive stdin: pass --yes or set OPENRAL_DATASET_CONSENT=1 "
            "to skip the consent prompt in CI / scripted runs"
        )

    prompt = f"Type the dataset repo_id ({repo_id!r}) to confirm:"
    typed = typer.prompt(prompt, default="", show_default=False)
    if typed.strip() != repo_id:
        raise ROSConfigError(f"consent refused: typed {typed!r} did not match repo_id {repo_id!r}")


def _resolve_repo_id(root: Path, info: dict[str, object], cli_repo_id: str | None) -> str:
    """Resolve the target repo_id, preferring CLI override → meta/info.json → default.

    Returns:
        A non-empty ``<org>/<name>`` string. Raises if the resolved id
        is malformed (missing ``/`` separator).

    Raises:
        ROSConfigError: When no repo_id can be resolved or the format
            is invalid.
    """
    if cli_repo_id is not None:
        repo_id = cli_repo_id
    else:
        metadata = info.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("repo_id"):
            repo_id = str(metadata["repo_id"])
        else:
            raise ROSConfigError(
                f"no repo_id found in {root}/meta/info.json[metadata][repo_id]; "
                "pass --repo-id <org>/<name> on the command line"
            )
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ROSConfigError(f"invalid repo_id {repo_id!r}: must be of the form '<org>/<name>'")
    return repo_id


@dataset_app.command("push")
def push_command(
    root: Path = typer.Argument(
        ...,
        help=(
            "Local LeRobotDataset v3 root directory. Must contain meta/info.json "
            "(typically produced by `openral sim run --dataset-out <path>`)."
        ),
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help=(
            "HF Hub repo id (e.g. openral/dataset-pick-cube). Overrides "
            "meta/info.json[metadata][repo_id]; required when info.json "
            "doesn't carry one."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help=(
            "Skip the interactive consent prompt. Required for non-TTY runs "
            "(or set OPENRAL_DATASET_CONSENT=1 in the environment)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Validate the dataset and resolve the consent gate, but do "
            "NOT contact the HF Hub. Useful for CI smoke checks."
        ),
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help=(
            "HF token with 'repo.write' scope. Defaults to $HF_TOKEN / "
            "$HUGGINGFACE_HUB_TOKEN env var."
        ),
    ),
    commit_message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Commit message for the upload. Defaults to a structured auto-message.",
    ),
) -> None:
    """Upload a local LeRobotDataset v3 to the HF Hub (private; consent-gated).

    Reads ``meta/info.json`` to determine the default repo_id and the
    fields shown in the consent prompt (episode count, frame count,
    camera keys, license). The local file system is the source of
    truth — the HF Hub repo created here is a mirror.

    The repo is **always** created with ``private=True``. A safety gate
    re-fetches the repo metadata after creation and aborts if the API
    reports it as public. PR0 / ADR-0019 §3 covers the "discard vs
    persist + tag" decision; this command is the one place where
    user consent gates whether dataset persistence becomes dataset
    publication.
    """
    try:
        info = _read_info_json(root)
        resolved_repo_id = _resolve_repo_id(root, info, repo_id)
        _confirm_consent(resolved_repo_id, root, info, yes=yes)
    except ROSError as exc:
        _console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        _console.print(
            f"[green]dry-run OK[/green] — would publish {resolved_repo_id!r} from {root}"
        )
        return

    try:
        resolved_token = resolve_token(token)
    except ROSError as exc:
        _console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _console.print(
            "[red]config error:[/red] huggingface_hub is not installed. "
            "Run `uv pip install huggingface_hub` and retry."
        )
        raise typer.Exit(code=1) from exc

    api = HfApi(token=resolved_token)

    try:
        repo_url = api.create_repo(
            repo_id=resolved_repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
    except Exception as exc:  # HF SDK raises many specific types
        err_str = str(exc)
        if "403" in err_str or "rights" in err_str.lower():
            _console.print(
                "[red]token insufficient:[/red] HF token does not have "
                "'repo.write' scope. Regenerate with Write permission at "
                "https://huggingface.co/settings/tokens."
            )
        else:
            _console.print(f"[red]create_repo failed:[/red] {err_str}")
        raise typer.Exit(code=1) from exc

    try:
        ensure_private(api, resolved_repo_id, repo_type="dataset")
    except ROSError as exc:
        _console.print(f"[red]privacy gate:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    message = commit_message or (
        f"chore(dataset): publish {resolved_repo_id} "
        f"({info.get('total_episodes', '?')} episodes, "
        f"{info.get('total_frames', '?')} frames)"
    )
    try:
        api.upload_folder(
            folder_path=str(root),
            repo_id=resolved_repo_id,
            repo_type="dataset",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=message,
        )
    except Exception as exc:
        _console.print(f"[red]upload failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(
        f"[green]published:[/green] {repo_url} (private)\n"
        f"  load via:  LeRobotDataset({resolved_repo_id!r})"
    )
