"""The ``openral rskill install`` code path, shared with a deploy's rSkill auto-install.

``openral rskill install`` and :func:`install_listed_rskills` (run at ``openral deploy
run`` / ``deploy sim`` start) fetch the manifest with :func:`fetch_rskill_manifest`,
read its licence against :data:`PERMISSIVE_RSKILL_LICENSES`, and install through
``rSkill.from_pretrained``, which downloads the snapshot and writes the registry
(``~/.local/share/openral/rskills.json``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final

import huggingface_hub
from openral_core import RSKILLS_ENV, ROSError, allowed_rskills_from_env
from openral_core.schemas import RSkillManifest
from openral_rskill.loader import rSkill

#: Licences ``openral rskill install`` accepts without ``--yes`` (no interactive prompt).
PERMISSIVE_RSKILL_LICENSES: Final[frozenset[str]] = frozenset({"apache-2.0", "mit", "bsd"})

#: Set to ``0`` to turn the deploy-start rSkill auto-install off (``--no-rskill-install``).
RSKILL_AUTO_INSTALL_ENV: Final[str] = "OPENRAL_RSKILL_AUTO_INSTALL"


def fetch_rskill_manifest(hub_id: str, revision: str | None = None) -> RSkillManifest:
    """Download and parse only ``rskill.yaml`` from ``hub_id`` (the licence is read before weights).

    Raises whatever ``hf_hub_download`` or ``RSkillManifest.from_yaml`` raises.
    """
    # Looked up at call time, so a test can double the Hub at its network boundary.
    path = huggingface_hub.hf_hub_download(
        repo_id=hub_id, filename="rskill.yaml", revision=revision
    )
    return RSkillManifest.from_yaml(path)


def _hub_status(exc: BaseException | None) -> int | None:
    """The Hub HTTP status behind ``exc`` (or the error it wraps), if any."""
    while exc is not None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
        exc = exc.__cause__ or exc.__context__
    return None


def install_listed_rskills(
    rskill_ids: list[str], *, echo: Callable[[str], None]
) -> tuple[list[str], list[str]]:
    """Install each listed rSkill that isn't installed yet, one at a time; never raises.

    The robot's openral.com rSkill list (``OPENRAL_RSKILLS``) reaches a deploy before
    the reasoner starts; this gets the listed rSkills onto the robot without anyone
    running ``openral rskill install`` there. Only permissive licences install: a
    proprietary or non-commercial rSkill needs an operator's acceptance, so it is
    skipped with the command to run by hand (never ``--yes`` / ``--non-commercial``
    on the operator's behalf). A gated or private repo (Hub 401/403), an unreachable
    Hub, or a bad manifest is said and skipped. One line per rSkill.

    Args:
        rskill_ids: Hub repo ids (as listed; compared case-insensitively).
        echo: Where each line goes.

    Returns:
        ``(installed, skipped)`` repo ids.
    """
    try:
        have = {e.repo_id.casefold() for e in rSkill.list_installed()}
    except ROSError as exc:
        echo(f"rskill auto-install: skipped, can't read the rSkill registry: {exc}")
        return [], list(rskill_ids)
    installed: list[str] = []
    skipped: list[str] = []
    for hub_id in dict.fromkeys(rskill_ids):  # ordered, de-duplicated
        if hub_id.casefold() in have:
            echo(f"rskill {hub_id}: already installed")
            continue
        by_hand = f"openral rskill install {hub_id}"
        echo(f"rskill {hub_id}: installing …")
        try:
            manifest = fetch_rskill_manifest(hub_id)
            licence = manifest.license.value
            if licence not in PERMISSIVE_RSKILL_LICENSES:
                echo(
                    f"rskill {hub_id}: skipped, licence {licence} needs your acceptance; "
                    f"run `{by_hand}`"
                )
                skipped.append(hub_id)
                continue
            pkg = rSkill.from_pretrained(hub_id)
        except Exception as exc:  # reason: an rSkill must never block the robot's start
            if _hub_status(exc) in (401, 403):
                echo(
                    f"rskill {hub_id}: skipped, no access; request access on Hugging Face "
                    f"(https://huggingface.co/{hub_id}), then run `{by_hand}`"
                )
            else:
                echo(f"rskill {hub_id}: skipped, {type(exc).__name__}: {exc}; run `{by_hand}`")
            skipped.append(hub_id)
            continue
        have.add(hub_id.casefold())
        installed.append(hub_id)
        echo(f"rskill {hub_id}: installed v{pkg.manifest.version}")
    return installed, skipped


def auto_install_rskills(*, echo: Callable[[str], None]) -> None:
    """The deploy-start hook: install the robot's listed rSkills unless turned off.

    Reads the list after the account settings are applied (``OPENRAL_RSKILLS``);
    ``OPENRAL_RSKILL_AUTO_INSTALL=0`` turns it off.
    """
    if os.environ.get(RSKILL_AUTO_INSTALL_ENV, "1").strip() == "0":
        return
    if allowed_rskills_from_env() is None:
        return
    # As listed, not casefolded: the Hub id's spelling names the cache dir and registry row.
    ids = [x.strip() for x in os.environ[RSKILLS_ENV].split(",") if x.strip()]
    install_listed_rskills(ids, echo=echo)
