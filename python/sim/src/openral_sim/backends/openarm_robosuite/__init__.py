"""OpenArm v2 bimanual tabletop scene with robosuite OSC controllers.

Importing this package registers the ``openarm_tabletop_pnp`` scene id
in :data:`openral_sim.SCENES`. The heavy ``mujoco`` import in :mod:`.env`
is deferred to function bodies (`if TYPE_CHECKING: import mujoco` at
module level), so importing ``openral_sim`` — which the CLI does eagerly
via ``openral_sim.cli`` → ``SimRunner`` → ``backends`` registration —
stays free of mujoco / robosuite. The regression guard lives in
``tests/unit/test_cli_sim_run.py::test_bh_cli_import_is_light``.

Drive via::

    openral sim run --config scenes/sim/openarm_tabletop.yaml \
                    --rskill pi05-openarm-vision-nf4

See :mod:`._assets` for the MJCF composer and :mod:`.env` for the
:class:`SimRollout` implementation.
"""

from openral_sim.backends.openarm_robosuite import (
    env as _env,
)

__all__: list[str] = []
