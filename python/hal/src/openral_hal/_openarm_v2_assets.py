"""Vendor the upstream ``enactic/openarm_mujoco`` v2 MJCF assets.

The ``robot_descriptions`` package pins
``enactic/openarm_mujoco`` to commit ``cd30dd4`` — the v0.3 / v1 era,
before v2 landed.  v2 (PR #19 on master, commit ``45d4c29`` at the
time this file was written) ships a dramatically improved MJCF:
native ``<position>`` actuators on every joint with per-class PD
gains (``DM8009``: kp=230 kv=2.7, ``DM4340``: kp=190 kv=2.2,
``DM4310``: kp=30 kv=1.5, fingers: kp=30 kv=0.2), proper
``ctrlrange`` and ``forcerange``, symmetric left / right finger
gains, only 16 actuators total (one finger driver per side, the
second finger follows via an ``<equality>`` constraint).  That
collapses ~400 lines of software PD + workaround code in
:class:`openral_hal.OpenArmMujocoHAL` down to a trivial
write-target → write-ctrl mapping (CLAUDE.md §1.4 — don't write
abstractions you can delete by reading better upstream).

Until ``robot_descriptions`` bumps its pin past PR #19, this module
maintains a parallel clone under ``$OPENRAL_CACHE_DIR/openarm_v2/``
pinned to a known-good v2 SHA, and returns the bimanual MJCF path
from inside it.  The pattern mirrors
``python/sim/src/openral_sim/backends/so100_robosuite/_assets.py``
(the menagerie SO-100 wrapper).

When ``robot_descriptions`` adds a v2 entry, this module can be
removed and :mod:`openral_hal.openarm` can drop back to a clean
``from robot_descriptions import openarm_v2_mj_description`` —
tracked as a TODO at the call site.
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_clone import fetch_pinned_clone

__all__ = ["ensure_openarm_v2_mjcf"]

# Pinned to ``45d4c29`` on master — the latest v2 commit at the time
# this file was written ("add demopath (#23)").  v2 itself landed in
# PR #19 (commit ``0024877``).  Bump this when the upstream repo
# adds new v2 features we want; keep pinned (not ``master`` HEAD) so
# the sim contract is reproducible.
_OPENARM_V2_PINNED_SHA: str = "45d4c29fd108e6c1bd4c66cbf2322758c7940fe9"
_OPENARM_REPO_URL: str = "https://github.com/enactic/openarm_mujoco.git"
_OPENARM_V2_MJCF_REL: str = "v2/openarm_mujoco_v2/openarm_v20_bimanual.xml"


def _cache_dir() -> Path:
    """OpenRAL cache root for the OpenArm v2 clone.

    Honours ``$OPENRAL_CACHE_DIR`` for tests / CI; falls back to
    ``~/.cache/openral``.  The same convention the
    ``so100_robosuite`` asset cache uses (see
    ``python/sim/src/openral_sim/backends/so100_robosuite/_assets.py``).
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "openarm_v2"


def ensure_openarm_v2_mjcf() -> str:
    """Return the on-disk path to the v2 bimanual MJCF, fetching if needed.

    Idempotent: subsequent calls reuse the cached clone.  The repo
    is checked out at :data:`_OPENARM_V2_PINNED_SHA` so the
    in-tree sim contract doesn't drift with upstream master.

    Raises:
        ROSConfigError: When ``git`` is not on the PATH, the clone /
            fetch / checkout fails, or the expected MJCF path is
            missing inside the resulting tree.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    repo_dir = cache / _OPENARM_V2_PINNED_SHA

    mjcf = repo_dir / _OPENARM_V2_MJCF_REL
    if mjcf.is_file():
        return str(mjcf)

    # Staged clone + atomic rename (shared with the Anvil fetcher): shallow
    # clone of the pinned commit, mirroring the `robot_descriptions`
    # convention (small, no full history).
    fetch_pinned_clone(
        _OPENARM_REPO_URL,
        _OPENARM_V2_PINNED_SHA,
        repo_dir,
        what="OpenArm v2",
    )

    if not mjcf.is_file():
        raise ROSConfigError(
            f"OpenArm v2 clone at {repo_dir!s} is missing the expected MJCF "
            f"at {_OPENARM_V2_MJCF_REL} — upstream may have moved the file."
        )
    return str(mjcf)
