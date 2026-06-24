"""tools/rskill_publisher.py — Package and publish a local rSkill directory to the HF Hub.

Usage
-----
::

    # Dry-run (validates manifest, no upload):
    uv run python tools/rskill_publisher.py rskills/smolvla-libero/

    # Publish as private (the default):
    uv run python tools/rskill_publisher.py rskills/smolvla-libero/ --publish

    # Publish as PUBLIC (only allowed for commercially-licensed weights):
    uv run python tools/rskill_publisher.py rskills/rtdetr-coco-r18/ --publish --public

    # Pin upstream weights to a specific commit SHA and then publish:
    uv run python tools/rskill_publisher.py rskills/smolvla-libero/ --bump-revision --publish

Design constraints (CLAUDE.md §7.2, §9, §12)
--------------------------------------------
- Private by default. ``--public`` opts into a public repo, but ONLY for skills
  whose license permits commercial use (``RSkillManifest.is_commercial_use_allowed``):
  the tool refuses to make a non-commercial-licensed skill (e.g. NVIDIA GR00T /
  LocateAnything) public, enforcing the license-lineage posture (§9). The chosen
  visibility is re-verified against the API after ``create_repo`` and the upload
  aborts on a mismatch (so a pre-existing repo of the wrong visibility is caught).
- Requires an HF token with ``repo.write`` scope.  Read-only tokens produce a
  clear error message rather than a generic 403.
- Validates the manifest against :class:`openral_core.schemas.RSkillManifest`
  before touching the network.
- Runs :func:`openral_cli._rskill_doc_validator.validate_rskill_docs` as a hard
  gate (CLAUDE.md §6.4): the upload is refused if ``README.md`` is missing,
  too short, missing canonical sections, or contains unresolved template
  sentinels (``TEMPLATE_ORG`` / ``TODO:`` / …), and likewise if the manifest
  description, ``paper_url``, ``weights_uri``, or ``source_repo`` are still at
  their template placeholders. The same report prints in dry-run mode so
  authors see what to fix without attempting an upload.
- Uses ``HfApi.upload_folder`` with ``ignore_patterns`` to exclude
  non-distributable files (.env, *.pyc, __pycache__, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the workspace Python packages are importable when run as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "core" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "python" / "cli" / "src"))

import structlog  # noqa: E402 — after sys.path fixup

# De-duped per CLAUDE.md §1.13 — ADR-0019 PR5 lifted these helpers into
# the openral_cli._hf_publish module so `openral dataset push` and this
# tool share one canonical token/scope/ignore-patterns surface.
from openral_cli._hf_publish import IGNORE_PATTERNS as _IGNORE_PATTERNS  # noqa: E402
from openral_cli._hf_publish import ensure_private as _ensure_private_shared  # noqa: E402
from openral_cli._hf_publish import resolve_token as _resolve_token_shared  # noqa: E402
from openral_cli._rskill_doc_validator import (  # noqa: E402
    DocValidationReport,
    format_report,
    validate_rskill_docs,
)
from openral_core.exceptions import ROSConfigError  # noqa: E402

log = structlog.get_logger(__name__)

# Files that must be present in a valid rSkill directory.
_REQUIRED_FILES = ["rskill.yaml"]


def public_visibility_error(manifest: RSkillManifest, public: bool) -> str | None:  # type: ignore[name-defined]  # noqa: F821
    """Return an error message if ``--public`` is not allowed, else ``None``.

    Enforces the license-lineage posture (CLAUDE.md §9): a non-commercial-licensed
    skill (e.g. NVIDIA GR00T / LocateAnything) must not be published to a public
    repo. Commercially-licensed weights (Apache-2.0 / MIT / NVIDIA Open Model …)
    may be public. Pure (no network) so the guard is unit-testable and fails fast
    before any HF call.

    Args:
        manifest: The validated rSkill manifest.
        public: Whether ``--public`` was requested.

    Returns:
        ``None`` when the request is allowed; otherwise an explanatory string.
    """
    if public and not manifest.is_commercial_use_allowed:
        return (
            f"--public refused: {manifest.name!r} has a non-commercial license "
            f"({manifest.license.value}); only commercially-licensed weights may be made "
            "public (CLAUDE.md §9). Publish privately instead (drop --public)."
        )
    return None


def _validate_docs(skill_dir: Path, manifest: RSkillManifest) -> DocValidationReport:  # type: ignore[name-defined]  # noqa: F821
    """Run the README + manifest documentation validator.

    Prints the rendered report to stdout (so authors see the same output
    in both dry-run and ``--publish`` paths) and returns the structured
    report so the caller can decide whether to abort.

    Args:
        skill_dir: Path to the rSkill directory.
        manifest: Already-loaded manifest from :func:`_validate_manifest`.

    Returns:
        The :class:`openral_cli._rskill_doc_validator.DocValidationReport`.
    """
    report = validate_rskill_docs(skill_dir, manifest)
    print()
    print(format_report(report))
    return report


def _resolve_token(token_arg: str | None) -> str:
    """Resolve the HF token via the shared helper, exiting on missing token.

    Wraps :func:`openral_cli._hf_publish.resolve_token` to preserve the
    script's ``sys.exit(1)`` contract — Typer commands raise
    :class:`ROSConfigError` and let the dispatcher handle the exit code,
    but this stand-alone script needs the legacy SystemExit behaviour.
    """
    try:
        return _resolve_token_shared(token_arg)
    except ROSConfigError as exc:
        log.error(
            "rskill_publisher.no_token",
            hint="Set HF_TOKEN env var or pass --token <your_token>",
            error=str(exc),
        )
        sys.exit(1)


def _validate_manifest(skill_dir: Path) -> RSkillManifest:  # type: ignore[name-defined]  # noqa: F821
    """Load and validate the rskill.yaml manifest.

    Args:
        skill_dir: Path to the local rSkill directory.

    Returns:
        Validated :class:`openral_core.schemas.RSkillManifest`.

    Raises:
        SystemExit: If the manifest is missing or fails Pydantic validation.
    """
    from openral_core.schemas import RSkillManifest

    manifest_path = skill_dir / "rskill.yaml"
    if not manifest_path.exists():
        log.error("rskill_publisher.missing_manifest", path=str(manifest_path))
        sys.exit(1)

    try:
        manifest = RSkillManifest.from_yaml(str(manifest_path))
    except Exception as exc:
        log.error("rskill_publisher.invalid_manifest", error=str(exc))
        sys.exit(1)

    log.info(
        "rskill_publisher.manifest_ok",
        name=manifest.name,
        version=manifest.version,
        license=manifest.license.value,
        role=manifest.role,
        embodiment_tags=manifest.embodiment_tags,
    )
    return manifest


def _validate_task_space(manifest: RSkillManifest, skill_dir: Path) -> None:  # type: ignore[name-defined]  # noqa: F821
    """ADR-0071 Phase 2 — warn-only cross-layer task-space check at publish time.

    For an actuating rSkill (one carrying an ``action_contract``), build its
    :class:`openral_core.TaskSpace` and run :func:`task_space_compatible`
    (``hal_mode="sim"``) against every in-tree ``robots/<id>/robot.yaml`` whose
    ``embodiment_tags`` the skill targets. Emits a warning per incompatible
    (skill, robot) pair — catching slot end-effector-name mismatches and
    joint-width overruns (the class of bug ADR-0071's sweep surfaced) before the
    manifest reaches the Hub. **Never fails the publish** — Phase 4 makes this
    gate blocking.

    Args:
        manifest: The validated rSkill manifest.
        skill_dir: The local ``rskills/<name>`` directory (its great-grandparent
            is the repo root holding ``robots/``).
    """
    from openral_core import RobotDescription, TaskSpace, task_space_compatible

    if manifest.action_contract is None:
        return  # detector / vlm / reward / ros_action — no task space to check.
    robots_dir = skill_dir.resolve().parent.parent / "robots"
    if not robots_dir.is_dir():
        log.info("rskill_publisher.task_space_no_robots_dir", path=str(robots_dir))
        return
    tags = set(manifest.embodiment_tags or [])
    matched = 0
    for robot_yaml in sorted(robots_dir.glob("*/robot.yaml")):
        try:
            robot = RobotDescription.from_yaml(str(robot_yaml))
        except Exception:  # reason: a malformed sibling robot must not block this skill's publish
            continue
        if not (tags & set(robot.capabilities.embodiment_tags or [])):
            continue
        matched += 1
        space = TaskSpace.from_action_contract(manifest.action_contract, robot)
        match = task_space_compatible(space, robot, hal_mode="sim")
        if not match.ok:
            log.warning(
                "rskill_publisher.task_space_incompatible",
                skill=manifest.name,
                robot=robot.name,
                reasons=match.reasons,
                note="ADR-0071 Phase 2 — warn-only, not yet blocking",
            )
    if matched == 0:
        log.info(
            "rskill_publisher.task_space_no_matching_robot",
            skill=manifest.name,
            embodiment_tags=sorted(tags),
        )


def _bump_revision(manifest_path: Path, weights_uri_base: str, token: str) -> str:
    """Resolve the latest commit SHA for the upstream weights and patch rskill.yaml.

    Fetches the HEAD commit SHA of the upstream HF repo encoded in
    ``weights_uri`` (strips the ``hf://`` prefix) and rewrites the
    ``weights_uri`` field to pin it.

    Args:
        manifest_path: Path to ``rskill.yaml``.
        weights_uri_base: The bare ``hf://<repo_id>`` URI without a SHA.
        token: HF API token.

    Returns:
        The new pinned URI string.

    Raises:
        SystemExit: If the upstream repo cannot be reached.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("rskill_publisher.missing_hf_hub")
        sys.exit(1)

    repo_id = weights_uri_base.removeprefix("hf://")
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except Exception as exc:
        log.error("rskill_publisher.bump_revision_failed", repo_id=repo_id, error=str(exc))
        sys.exit(1)

    sha = info.sha
    pinned_uri = f"hf://{repo_id}@{sha}"

    # Patch the YAML file in-place (simple string replace; preserves comments).
    text = manifest_path.read_text(encoding="utf-8")
    old_uri = weights_uri_base
    if old_uri not in text:
        log.warning(
            "rskill_publisher.bump_revision_noop",
            note="weights_uri not found verbatim in rskill.yaml — skipping patch.",
        )
        return weights_uri_base

    text = text.replace(old_uri, pinned_uri, 1)
    manifest_path.write_text(text, encoding="utf-8")
    log.info("rskill_publisher.revision_pinned", pinned_uri=pinned_uri, sha=sha)
    return pinned_uri


def _ensure_private(api: HfApi, repo_id: str) -> None:  # type: ignore[name-defined]  # noqa: F821
    """Assert the repo is private; abort the script (sys.exit) if it is not.

    Wraps :func:`openral_cli._hf_publish.ensure_private` for the script's
    legacy ``sys.exit(1)`` contract; the shared helper raises
    :class:`ROSConfigError` instead.
    """
    try:
        _ensure_private_shared(api, repo_id, repo_type="model")
    except ROSConfigError as exc:
        log.error("rskill_publisher.not_private", repo_id=repo_id, error=str(exc))
        sys.exit(1)
    log.info("rskill_publisher.privacy_verified", repo_id=repo_id, private=True)


def _ensure_public(api: HfApi, repo_id: str) -> None:  # type: ignore[name-defined]  # noqa: F821
    """Assert the repo is public; abort the script (sys.exit) if it is private.

    The ``--public`` counterpart of :func:`_ensure_private` — catches the case
    where ``create_repo(exist_ok=True)`` reused a pre-existing *private* repo, so
    a ``--public`` publish never silently lands in a private repo (or vice
    versa).
    """
    try:
        info = api.model_info(repo_id)
    except Exception as exc:
        log.error("rskill_publisher.visibility_check_failed", repo_id=repo_id, error=str(exc))
        sys.exit(1)
    if getattr(info, "private", False):
        log.error(
            "rskill_publisher.not_public",
            repo_id=repo_id,
            hint="A repo with this id already exists and is PRIVATE; refusing to "
            "upload public content into it. Delete/rename it or drop --public.",
        )
        sys.exit(1)
    log.info("rskill_publisher.visibility_verified", repo_id=repo_id, private=False)


def _publish(
    skill_dir: Path,
    manifest: RSkillManifest,  # type: ignore[name-defined]  # noqa: F821
    token: str,
    *,
    public: bool = False,
) -> str:
    """Create an HF Hub repo (private by default) and upload the rSkill directory.

    Steps:
    1. Create (or reuse) the repo with ``private = not public``.
    2. Re-fetch repo metadata and abort if the visibility does not match the
       requested mode (safety gate — catches a reused repo of the wrong kind).
    3. Upload all files from ``skill_dir`` (excluding ``_IGNORE_PATTERNS``).

    Args:
        skill_dir: Local rSkill directory containing ``rskill.yaml``.
        manifest: Validated manifest (name used as the HF repo ID).
        token: HF API token with ``repo.write`` scope.
        public: When ``True`` create/verify a **public** repo; otherwise private.
            The caller (:func:`main`) only passes ``True`` after the license gate.

    Returns:
        The HF Hub URL of the published repo.

    Raises:
        SystemExit: On any API error or if the visibility check fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("rskill_publisher.missing_hf_hub")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = manifest.name
    private = not public

    # ── 1. Create repo (private unless --public) ───────────────────────────────
    log.info("rskill_publisher.creating_repo", repo_id=repo_id, private=private)
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except Exception as exc:
        err_str = str(exc)
        if "403" in err_str or "rights" in err_str.lower():
            log.error(
                "rskill_publisher.token_insufficient",
                error=err_str,
                hint=(
                    "The HF token does not have 'repo.write' scope. "
                    "Generate a new token at https://huggingface.co/settings/tokens "
                    "with 'Write' permission (or fine-grained with repo.write)."
                ),
            )
        else:
            log.error("rskill_publisher.create_repo_failed", error=err_str)
        sys.exit(1)

    # ── 2. Visibility safety gate ──────────────────────────────────────────────
    if public:
        _ensure_public(api, repo_id)
    else:
        _ensure_private(api, repo_id)

    # ── 3. Upload files ────────────────────────────────────────────────────────
    log.info("rskill_publisher.uploading", skill_dir=str(skill_dir), repo_id=repo_id)
    try:
        api.upload_folder(
            folder_path=str(skill_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=_IGNORE_PATTERNS,
            commit_message=f"chore: publish rSkill {manifest.name} v{manifest.version}",
        )
    except Exception as exc:
        log.error("rskill_publisher.upload_failed", error=str(exc))
        sys.exit(1)

    url = str(repo_url)
    log.info("rskill_publisher.published", url=url, repo_id=repo_id, private=private)
    return url


def main() -> None:
    """Entry point for the skill publisher CLI.

    Example:
        >>> # python tools/rskill_publisher.py rskills/smolvla-libero/ --publish
    """
    parser = argparse.ArgumentParser(
        description="Validate and publish a local rSkill directory to the HF Hub "
        "(private by default; --public for commercially-licensed weights).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "skill_dir",
        type=Path,
        help="Path to the rSkill directory (must contain rskill.yaml).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Actually create the HF Hub repo and upload files (default: dry-run only).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help=(
            "Create/verify a PUBLIC repo instead of private. Only allowed for "
            "commercially-licensed weights (refused for non-commercial skills, §9)."
        ),
    )
    parser.add_argument(
        "--bump-revision",
        action="store_true",
        default=False,
        help=(
            "Resolve the latest commit SHA of the upstream weights repo and "
            "pin it in rskill.yaml before publishing."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token with repo.write scope. Defaults to $HF_TOKEN env var.",
    )
    args = parser.parse_args()

    skill_dir: Path = args.skill_dir.resolve()
    if not skill_dir.is_dir():
        log.error("rskill_publisher.not_a_directory", path=str(skill_dir))
        sys.exit(1)

    # ── Validate manifest ──────────────────────────────────────────────────────
    manifest = _validate_manifest(skill_dir)

    # ── Cross-layer task-space check (ADR-0071 Phase 2, warn-only) ──────────────
    _validate_task_space(manifest, skill_dir)

    # ── Validate README + manifest documentation (CLAUDE.md §6.4) ──────────────
    # Runs in both dry-run and --publish paths so authors see the same report
    # without having to attempt an upload first. The dry-run prints it as
    # informational; --publish treats any error-severity issue as fatal.
    doc_report = _validate_docs(skill_dir, manifest)

    # ── License gate for --public (CLAUDE.md §9) ───────────────────────────────
    # Checked before the dry-run return + before any network call, so `--public`
    # on a non-commercial skill fails fast with a clear reason.
    public_error = public_visibility_error(manifest, args.public)
    if public_error:
        log.error("rskill_publisher.public_refused", error=public_error)
        sys.exit(1)

    visibility = "public" if args.public else "private"

    if not args.publish and not args.bump_revision:
        if doc_report.is_valid:
            print(f"\n[dry-run] rSkill '{manifest.name}' v{manifest.version} is valid.")
            print(f"Pass --publish to upload to HF Hub ({visibility}).")
            print("Pass --bump-revision to pin the upstream weights SHA first.")
        else:
            print(
                f"\n[dry-run] rSkill '{manifest.name}' v{manifest.version} "
                "is NOT publish-ready — fix the errors above first."
            )
            sys.exit(1)
        return

    if not doc_report.is_valid:
        log.error(
            "rskill_publisher.docs_incomplete",
            errors=len(doc_report.errors),
            warnings=len(doc_report.warnings),
            hint="See rskills/template/README.md for the canonical layout.",
        )
        sys.exit(1)

    token = _resolve_token(args.token)

    # ── Optionally pin revision ────────────────────────────────────────────────
    if args.bump_revision:
        weights_uri = manifest.weights_uri
        if not weights_uri.startswith("hf://") or "@" in weights_uri:
            log.warning(
                "rskill_publisher.already_pinned_or_not_hf",
                weights_uri=weights_uri,
                note="Skipping --bump-revision; URI is already pinned or not an hf:// URI.",
            )
        else:
            _bump_revision(skill_dir / "rskill.yaml", weights_uri, token)
            # Reload manifest after patching
            manifest = _validate_manifest(skill_dir)

    # ── Publish ────────────────────────────────────────────────────────────────
    if args.publish:
        url = _publish(skill_dir, manifest, token, public=args.public)
        print(f"\nPublished: {url}")
        print(f"Repo is private: {not args.public}")
        print("\nInstall with:")
        print(f"  ral skill install hf://{manifest.name}")
        print("  # or in Python:")
        print(f'  rSkill.from_pretrained("{manifest.name}")')


if __name__ == "__main__":
    main()
