"""Vendor the Anvil OpenARM 2.0 MJCF from ``bensonlee5/anvil-openarm-mujoco``.

The Anvil OpenARM 2.0 (https://docs.anvil.bot/introduction/openarm-2.0)
is the standard OpenArm v2 with two documented range deltas (J1 clamped
to +/-135 deg, J6 radial deviation widened to -45..+70 deg) plus the
wrist support bracket that enables the extra deviation.  No upstream
package
ships that variant: ``robot_descriptions`` carries only the stock
enactic assets, so this module maintains a pinned clone of the
``anvil-openarm-mujoco`` workspace, whose generated
``models/anvil_openarm_bimanual.xml`` applies the Anvil deltas (and the
visual-only CAD bracket meshes) on top of the upstream
``enactic/openarm_mujoco`` v2 files.  The pattern mirrors
:mod:`openral_hal._openarm_v2_assets` (the enactic v2 out-pin), with one
extra step: the generated MJCF references the stock link meshes from the
``upstream/openarm_mujoco`` git submodule (the bracket STL lives in the
anvil repo's own ``models/assets/``), so the clone must also initialise
that submodule.
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_clone import fetch_pinned_clone

__all__ = ["ensure_anvil_openarm_v2_mjcf"]

# Pinned to anvil-openarm-mujoco `fix/bracket-attachment-and-keyframe-ctrl`
# — manual-CAD wrist bracket (real STL from the user hardware CAD in
# `models/assets/`, aluminum + fastener materials, geoms hosted on the
# link5 forearm bodies) on top of main's URDF-generation work.  Bump
# when the generator or the local Anvil spec changes; keep a pinned SHA
# (never a branch name) so the sim contract is reproducible — but note
# a squash-merge of the source branch would orphan this commit, so
# re-pin to the main merge commit once the PR lands.
_ANVIL_PINNED_SHA: str = "1ea898f7a849e9621b834d2dfa4139c552ab14bb"
_ANVIL_REPO_URL: str = "https://github.com/bensonlee5/anvil-openarm-mujoco.git"
_ANVIL_MJCF_REL: str = "models/anvil_openarm_bimanual.xml"
# The generated MJCF's meshdir points into this submodule (the pristine
# enactic/openarm_mujoco checkout); it must be initialised for the model
# to compile.
_ANVIL_MESH_SUBMODULE: str = "upstream/openarm_mujoco"


def _cache_dir() -> Path:
    """OpenRAL cache root for the Anvil OpenARM clone.

    Honours ``$OPENRAL_CACHE_DIR`` for tests / CI; falls back to
    ``~/.cache/openral`` — the same convention
    :mod:`openral_hal._openarm_v2_assets` uses.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "anvil_openarm_v2"


def ensure_anvil_openarm_v2_mjcf() -> str:
    """Return the on-disk path to the Anvil 2.0 bimanual MJCF, fetching if needed.

    Idempotent: subsequent calls reuse the cached clone.  The repo is
    checked out at :data:`_ANVIL_PINNED_SHA` and its mesh submodule is
    initialised, so the in-tree sim contract doesn't drift with either
    upstream.

    Raises:
        ROSConfigError: When ``git`` is not on the PATH, the clone /
            checkout / submodule init fails, or the expected MJCF path
            is missing inside the resulting tree.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    repo_dir = cache / _ANVIL_PINNED_SHA

    mjcf = repo_dir / _ANVIL_MJCF_REL
    meshes = repo_dir / _ANVIL_MESH_SUBMODULE / "v2" / "assets"
    if mjcf.is_file() and meshes.is_dir():
        return str(mjcf)

    # Staged clone + atomic rename (shared with the enactic v2 fetcher). The
    # MJCF's meshdir points into the pristine enactic submodule, so it is
    # initialised in staging; its gitdir/worktree links are relative and
    # survive the rename.
    fetch_pinned_clone(
        _ANVIL_REPO_URL,
        _ANVIL_PINNED_SHA,
        repo_dir,
        submodule=_ANVIL_MESH_SUBMODULE,
        what="Anvil OpenARM 2.0",
    )

    if not mjcf.is_file() or not meshes.is_dir():
        raise ROSConfigError(
            f"Anvil OpenARM clone at {repo_dir!s} is missing the expected MJCF at "
            f"{_ANVIL_MJCF_REL} or the mesh submodule at {_ANVIL_MESH_SUBMODULE} — "
            "upstream may have moved the files."
        )
    return str(mjcf)
