"""Compatibility alias: the pre-grasp target geometry now lives in ``openral_core``.

Moved to ``openral_core.perception_geometry`` (issue #418) so Layer 2 can import it;
import from there. This module only re-exports its public names.
"""

from openral_core.perception_geometry import *  # noqa: F403
