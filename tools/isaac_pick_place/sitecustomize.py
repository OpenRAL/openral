"""Registers the scripted pick/place driver in the deploy runner process (``run_trial.sh``).

``run_trial.sh`` puts this directory on ``PYTHONPATH`` and sets ``PICK_PLACE_DRIVER=1``;
Python then imports this module at start-up in every process of the launched graph, and
only the runner process (``runtime_node``) registers the driver.
"""

import os
import sys

# ponytail: a sitecustomize hook, because the deploy runner resolves policy families only
# from openral_sim's registry and this driver must never ship in it; a policy entry-point
# group would replace this.
if os.environ.get("PICK_PLACE_DRIVER") == "1" and any("runtime_node" in a for a in sys.argv[:1]):
    try:
        import driver  # type: ignore[import-not-found]  # reason: this directory is on PYTHONPATH

        driver.register()
    except Exception as exc:  # reason: never break the runner; the goal then fails loudly
        print(f"[isaac_pick_place] driver registration failed: {exc!r}", file=sys.stderr)
