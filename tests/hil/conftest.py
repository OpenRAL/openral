"""Shared helpers for ``tests/hil/``."""

from __future__ import annotations

from pathlib import Path


def _can_links_up(can_links: tuple[str, ...]) -> bool:
    """True if every link in *can_links* is up (per ``enumerate_can_interfaces``)."""
    from openral_cli.autodetect import enumerate_can_interfaces

    up = {i.name for i in enumerate_can_interfaces() if i.is_up}
    return set(can_links) <= up


def _installed_slot_rskill(embodiment: str) -> Path | None:
    """Manifest path of an installed rSkill that drives *embodiment* by slots.

    Some cells run a policy that is not distributable — a customer task, or a
    checkpoint whose weights cannot be published — so it is installed from a
    private Hub repo (``openral rskill install <repo-id>``) rather than shipped
    in ``rskills/``. A test that needs such a policy's ``action_contract.slots``
    therefore resolves it through the local registry instead of the source
    tree, and skips where the cell has no matching install.

    Returns the first installed manifest tagged for *embodiment* whose action
    contract declares slots, or ``None`` when there is none. A registry that
    cannot be read counts as "none" — these are lab-gated tests, and an
    unreadable registry must skip them, never fail them.

    Args:
        embodiment: Embodiment tag to require, e.g. ``"openarm"``.

    Returns:
        Path to the installed ``rskill.yaml``, or ``None``.

    Example:
        >>> _installed_slot_rskill("no_such_embodiment") is None
        True
    """
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import RSkillManifest
    from openral_rskill.loader import rSkill

    try:
        entries = rSkill.list_installed()
    except (ROSConfigError, ValueError):  # reason: corrupt registry -> skip, never fail
        return None
    for entry in entries:
        if embodiment not in entry.embodiment_tags:
            continue
        candidate = Path(entry.manifest_path)
        if not candidate.is_file():
            continue
        try:
            manifest = RSkillManifest.from_yaml(str(candidate))
        except (ROSConfigError, ValueError):  # reason: a stale row is not a usable policy
            continue
        contract = manifest.action_contract
        if contract is not None and contract.slots:
            return candidate
    return None
